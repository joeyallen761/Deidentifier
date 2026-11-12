#!/usr/bin/env python3
"""Add audio from an original video to a silent video using FFmpeg."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


def restore_audio_only(
    silent_path: Path, original_path: Path, output_path: Path, *, require_audio: bool = True,
) -> None:
    """Keep the encoded video and fit original audio to its existing duration."""
    def streams(path: Path) -> list[dict]:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "stream=codec_type,start_time,duration,nb_frames", "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True,
        )
        return json.loads(result.stdout)["streams"]

    print("Reading video durations (without decoding frames)...", flush=True)
    silent_streams, original_streams = streams(silent_path), streams(original_path)
    silent_video = next((s for s in silent_streams if s["codec_type"] == "video"), None)
    original_video = next((s for s in original_streams if s["codec_type"] == "video"), None)
    if silent_video is None or original_video is None:
        raise ValueError("Both inputs must contain a video stream.")
    silent_count, original_count = silent_video.get("nb_frames"), original_video.get("nb_frames")
    if silent_count and original_count and silent_count != original_count:
        raise ValueError("Frame counts differ; use matching, untrimmed videos.")
    silent_duration = float(silent_video.get("duration", 0))
    original_duration = float(original_video.get("duration", 0))
    if silent_duration <= 0 or original_duration <= 0:
        raise ValueError("Could not read video durations for audio-only restoration.")
    audio_streams = [s for s in original_streams if s["codec_type"] == "audio"]
    if require_audio and not audio_streams:
        raise ValueError("The original video does not contain audio.")

    tempo = original_duration / silent_duration
    # Chain factors to stay within atempo's best-quality range of 0.5 to 2.
    remaining, filters = tempo, []
    while remaining < 0.5 or remaining > 2:
        factor = 0.5 if remaining < 0.5 else 2
        filters.append(f"atempo={factor}")
        remaining /= factor
    filters.append(f"atempo={remaining:.15g}")
    audio_options = []
    silent_start = float(silent_video.get("start_time", 0))
    original_start = float(original_video.get("start_time", 0))
    for index, audio in enumerate(audio_streams):
        # Preserve each track's starting offset while scaling the timeline.
        offset = silent_start + (float(audio.get("start_time", original_start)) - original_start) / tempo
        audio_options += [
            f"-filter:a:{index}",
            ",".join([*filters, f"asetpts=PTS-STARTPTS+({offset:.15g})/TB"]),
        ]
    print(
        f"Copying video; adjusting audio speed to {tempo:.6f}x with pitch preserved.\n"
        "This matches overall duration; variable frame timing may still cause local sync differences.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp_dir:
        combined_path = Path(temp_dir) / "combined.mp4"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-stats", "-copyts",
                "-i", str(silent_path), "-i", str(original_path),
                "-map", "0:v:0", "-map", "1:a" if require_audio else "1:a?",
                "-c:v", "copy", *audio_options, "-c:a", "aac", "-b:a", "192k", str(combined_path),
            ],
            check=True,
        )
        combined_path.replace(output_path)


def video_timing(path: Path) -> tuple[list[int], Fraction, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_frames", "-show_streams", "-show_entries",
            "frame=best_effort_timestamp,duration,pkt_duration:stream=time_base,start_pts,duration_ts",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    frames = data.get("frames", [])
    if not frames or any("best_effort_timestamp" not in frame for frame in frames):
        raise ValueError(f"Could not read video frame timestamps: {path}")
    timestamps = [int(frame["best_effort_timestamp"]) for frame in frames]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Video timestamps must be strictly increasing: {path}")
    stream = data["streams"][0]
    last_duration = int(frames[-1].get("duration", frames[-1].get("pkt_duration", 0)))
    if last_duration <= 0:
        last_duration = int(stream.get("start_pts", timestamps[0])) + int(stream.get("duration_ts", 0)) - timestamps[-1]
    if last_duration <= 0:
        raise ValueError(f"Could not determine the final frame duration: {path}")
    return timestamps, Fraction(stream["time_base"]), last_duration


def restore_audio(
    silent_path: Path, original_path: Path, output_path: Path, *, require_audio: bool = True,
    audio_only: bool = False,
) -> None:
    """Match frames by position, restore their timestamps, then add original audio."""
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"{executable} is required. Install FFmpeg and rerun this script.")

    silent_path, original_path, output_path = (
        path.resolve() for path in (silent_path, original_path, output_path)
    )
    if audio_only:
        restore_audio_only(silent_path, original_path, output_path, require_audio=require_audio)
        return
    print("Reading original and processed video timing...")
    silent_pts, silent_base, silent_last = video_timing(silent_path)
    original_pts, original_base, original_last = video_timing(original_path)
    if len(silent_pts) != len(original_pts):
        raise ValueError(
            f"Frame counts differ (processed: {len(silent_pts)}, original: {len(original_pts)}). "
            "Timing repair requires the same frames in the same order; use matching, untrimmed videos."
        )
    timing_matches = (
        all(a * silent_base == b * original_base for a, b in zip(silent_pts, original_pts))
        and silent_last * silent_base == original_last * original_base
    )

    with tempfile.TemporaryDirectory(dir=output_path.parent) as temp_dir:
        combined_path = Path(temp_dir) / "combined.mp4"
        video_options = ["-c:v", "copy"]
        if not timing_matches:
            print("Restoring original frame timing and re-encoding the processed video...")
            time_base = f"{original_base.numerator}/{original_base.denominator}"
            mapping_path = Path(temp_dir) / "timing.txt"
            mapping_path.write_text(
                "".join(f"{a} {b} {time_base}\n" for a, b in zip(silent_pts, original_pts)),
                encoding="utf-8",
            )
            # fsync assigns each processed frame its corresponding original PTS.
            # Disabling B-frames lets MP4 derive sample durations from these PTS;
            # setts supplies the original duration for the final sample as well.
            video_options = [
                "-vf", "fsync=file=timing.txt", "-fps_mode:v", "passthrough",
                "-enc_time_base:v", time_base,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-bf", "0",
                "-bsf:v", f"setts=duration={original_last}",
                "-video_track_timescale", str(original_base.denominator),
            ]
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-copyts",
                "-i", str(silent_path), "-i", str(original_path),
                "-map", "0:v:0", "-map", "1:a" if require_audio else "1:a?",
                *video_options, "-c:a", "aac", "-b:a", "192k", str(combined_path),
            ],
            check=True, cwd=temp_dir,
        )
        if not timing_matches:
            repaired_pts, repaired_base, repaired_last = video_timing(combined_path)
            if (
                len(repaired_pts) != len(original_pts)
                or any(a * repaired_base != b * original_base for a, b in zip(repaired_pts, original_pts))
                or repaired_last * repaired_base != original_last * original_base
            ):
                raise ValueError("The repaired video did not retain the original frame timing.")
        combined_path.replace(output_path)


def ask_video_path(prompt: str) -> Path:
    value = input(prompt).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        raise ValueError("A video path is required.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-only", action="store_true",
        help="copy video and stretch audio to its duration, preserving pitch; local sync may be approximate",
    )
    args = parser.parse_args()
    try:
        silent_path = ask_video_path("Silent/deidentified video file: ")
        original_path = ask_video_path("Original video file (with audio): ")
        output_path = silent_path.with_name(f"{silent_path.stem}_with_audio.mp4")
        if output_path.exists() or output_path == original_path:
            raise FileExistsError(f"Refusing to overwrite: {output_path}")

        print("Adding original audio to the silent video...")
        restore_audio(silent_path, original_path, output_path, audio_only=args.audio_only)
        print(f"Done. Wrote {output_path}")
        return 0
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print("Could not restore timing/audio. Check the error above. Exact timing repair requires FFmpeg with fsync and libx264.", file=sys.stderr)
        return 1
    except (OSError, ValueError, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
