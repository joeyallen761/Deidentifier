"""Integration checks using short, synthetic videos and real FFmpeg tools."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from restore_audio import restore_audio, video_timing


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class RestoreAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="restore timing ' ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original.mov"
        self.silent = self.root / "deidentified.mp4"
        self.output = self.root / "deidentified_with_audio.mp4"
        self.ffmpeg(
            "-f", "lavfi", "-i", "color=c=red:s=160x120:r=30:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=1.5",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-vf", r"settb=1/90000,setpts=if(lt(N\,15)\,N*3000\,45000+(N-15)*6000)",
            "-enc_time_base", "1/90000", "-fps_mode", "passthrough",
            "-c:v", "libx264", "-bf", "0", "-bsf:v", "setts=duration=3000",
            "-c:a", "pcm_s16le", str(self.original),
        )
        self.ffmpeg(
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=29:d=2",
            "-frames:v", "30", "-c:v", "libx264", str(self.silent),
        )

    def ffmpeg(self, *args):
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", *args],
            check=True, capture_output=True,
        ).stdout

    def timing(self, path):
        pts, base, last = video_timing(path)
        return [p * base for p in pts], last * base

    def test_cli_restores_each_timestamp_and_uses_processed_pixels(self):
        original_bytes = self.original.read_bytes()
        silent_bytes = self.silent.read_bytes()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "restore_audio.py")],
            input=f'"{self.silent}"\n"{self.original}"\n',
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.timing(self.output), self.timing(self.original))
        self.assertEqual(self.original.read_bytes(), original_bytes)
        self.assertEqual(self.silent.read_bytes(), silent_bytes)
        streams = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(self.output)],
            capture_output=True, text=True, check=True,
        ).stdout)["streams"]
        audio = [s for s in streams if s["codec_type"] == "audio"]
        self.assertEqual([s["codec_name"] for s in audio], ["aac", "aac"])
        pixels = self.ffmpeg(
            "-i", str(self.output), "-map", "0:v:0", "-frames:v", "1",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        )
        # The original is red; every processed pixel must remain blue.
        self.assertTrue(all(b > r + 100 for r, b in zip(pixels[0::3], pixels[2::3])))

    def test_mismatched_frame_counts_leave_no_output(self):
        shorter = self.root / "shorter.mp4"
        self.ffmpeg("-i", str(self.silent), "-frames:v", "20", "-c:v", "copy", str(shorter))
        with self.assertRaisesRegex(ValueError, "Frame counts differ"):
            restore_audio(shorter, self.original, self.output)
        self.assertFalse(self.output.exists())

    def test_matching_timing_copies_video_packets(self):
        restore_audio(self.silent, self.silent, self.output, require_audio=False)
        def video_hash(path):
            return self.ffmpeg("-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-")
        self.assertEqual(video_hash(self.output), video_hash(self.silent))
        self.assertEqual(self.timing(self.output), self.timing(self.silent))

    def test_in_place_processing_supports_original_without_audio(self):
        original_silent = self.root / "original_silent.mp4"
        self.ffmpeg("-i", str(self.original), "-map", "0:v:0", "-c:v", "copy", str(original_silent))
        restore_audio(self.silent, original_silent, self.silent, require_audio=False)
        self.assertEqual(self.timing(self.silent), self.timing(original_silent))

    def test_audio_only_stretches_audio_without_changing_video(self):
        original = self.root / "constant_rate.mov"
        self.ffmpeg(
            "-f", "lavfi", "-i", "color=c=red:s=160x120:r=30:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=1",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "libx264", "-c:a", "aac", str(original),
        )
        original_bytes, silent_bytes = original.read_bytes(), self.silent.read_bytes()
        with patch("restore_audio.video_timing", side_effect=AssertionError("Video decoding is too slow")):
            restore_audio(self.silent, original, self.output, audio_only=True)
        self.assertEqual(self.timing(self.output), self.timing(self.silent))
        def video_hash(path):
            return self.ffmpeg("-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-")
        self.assertEqual(video_hash(self.output), video_hash(self.silent))
        streams = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(self.output)],
            capture_output=True, text=True, check=True,
        ).stdout)["streams"]
        video_duration = float(next(s["duration"] for s in streams if s["codec_type"] == "video"))
        audio = [s for s in streams if s["codec_type"] == "audio"]
        self.assertEqual(len(audio), 2)
        self.assertTrue(all(abs(float(s["duration"]) - video_duration) < 0.04 for s in audio))
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(self.silent.read_bytes(), silent_bytes)

    def test_audio_only_missing_audio_leaves_no_output(self):
        with self.assertRaisesRegex(ValueError, "does not contain audio"):
            restore_audio(self.silent, self.silent, self.output, audio_only=True)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
