#!/usr/bin/env python3
"""Black out detected faces in a video using a local OpenCV YuNet model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np


MODEL_FILE = "face_detection_yunet_2023mar.onnx"
DETECTION_CONFIDENCE = 0.75
NMS_THRESHOLD = 0.3
TOP_K = 5000

EXPAND_X = 0.55
EXPAND_UP = 0.75
EXPAND_DOWN = 0.15

PERSIST_SECONDS = 1.0
MATCH_IOU_THRESHOLD = 0.15
MATCH_DISTANCE_FACTOR = 0.9
PROGRESS_EVERY_SECONDS = 10.0
MP4_CODECS = ("mp4v", "avc1", "H264")


def usage() -> None:
    print("Usage: python deanonymize.py movie.MOV", file=sys.stderr)


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_deidentified.mp4")


def make_detector(model_path: Path, frame_width: int, frame_height: int):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing face detector weights: {model_path}\n"
            "Download the model during setup, then rerun this script."
        )

    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "This OpenCV build does not include FaceDetectorYN. "
            "Install opencv-contrib-python."
        )

    return cv2.FaceDetectorYN_create(
        str(model_path),
        "",
        (frame_width, frame_height),
        DETECTION_CONFIDENCE,
        NMS_THRESHOLD,
        TOP_K,
    )


def open_video_writer(output_path: Path, fps: float, width: int, height: int):
    for codec in MP4_CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Writing {output_path} with codec {codec}")
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not open output video writer: {output_path}\n"
        f"Tried MP4 codecs: {', '.join(MP4_CODECS)}\n"
        "This usually means your OpenCV install cannot encode MP4 on this Mac."
    )


def clamp_rect(rect: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    return x1, y1, x2, y2


def expand_face_rect(face: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in face[:4]]
    return clamp_rect(
        (
            x - w * EXPAND_X,
            y - h * EXPAND_UP,
            x + w * (1.0 + EXPAND_X),
            y + h * (1.0 + EXPAND_DOWN),
        ),
        width,
        height,
    )


def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / float(area_a + area_b - intersection)


def rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = rect_center(a)
    bx, by = rect_center(b)
    return math.hypot(ax - bx, ay - by)


def rect_size(rect: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    return max(1.0, math.hypot(x2 - x1, y2 - y1))


def move_rect(rect: tuple[int, int, int, int], dx: float, dy: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    return clamp_rect((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)


def match_score(track: dict, detection: tuple[int, int, int, int]) -> float:
    iou = rect_iou(track["rect"], detection)
    distance = center_distance(track["rect"], detection)
    max_distance = max(rect_size(track["rect"]), rect_size(detection)) * MATCH_DISTANCE_FACTOR
    if iou >= MATCH_IOU_THRESHOLD:
        return iou + 1.0
    if distance <= max_distance:
        return 1.0 - distance / max_distance
    return -1.0


def update_tracks(
    tracks: list[dict],
    detections: list[tuple[int, int, int, int]],
    frame_index: int,
    max_missed_frames: int,
    width: int,
    height: int,
) -> list[dict]:
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))
    matches: list[tuple[float, int, int]] = []

    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = match_score(track, detection)
            if score >= 0:
                matches.append((score, track_index, detection_index))

    for _, track_index, detection_index in sorted(matches, reverse=True):
        if track_index not in unmatched_tracks or detection_index not in unmatched_detections:
            continue

        old_rect = tracks[track_index]["rect"]
        new_rect = detections[detection_index]
        old_cx, old_cy = rect_center(old_rect)
        new_cx, new_cy = rect_center(new_rect)
        tracks[track_index] = {
            "rect": new_rect,
            "velocity": (new_cx - old_cx, new_cy - old_cy),
            "last_seen": frame_index,
        }
        unmatched_tracks.remove(track_index)
        unmatched_detections.remove(detection_index)

    kept_tracks = []
    for track_index, track in enumerate(tracks):
        if track_index not in unmatched_tracks:
            kept_tracks.append(track)
            continue

        missed = frame_index - track["last_seen"]
        if missed <= max_missed_frames:
            dx, dy = track["velocity"]
            track["rect"] = move_rect(track["rect"], dx, dy, width, height)
            kept_tracks.append(track)

    for detection_index in unmatched_detections:
        kept_tracks.append({"rect": detections[detection_index], "velocity": (0.0, 0.0), "last_seen": frame_index})

    return kept_tracks


def black_out(frame: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    for x1, y1, x2, y2 in rects:
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = 0


def detect_faces(
    detector,
    frame: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []

    return [expand_face_rect(face, width, height) for face in faces]


def main() -> int:
    if len(sys.argv) != 2:
        usage()
        return 2

    input_path = Path(sys.argv[1]).expanduser()
    if not input_path.exists() or not input_path.is_file():
        print(f"Input video not found: {input_path}", file=sys.stderr)
        return 1

    output_path = output_path_for(input_path)
    if output_path.resolve() == input_path.resolve():
        print("Refusing to overwrite the input video.", file=sys.stderr)
        return 1
    if output_path.exists():
        print(f"Output already exists; refusing to overwrite: {output_path}", file=sys.stderr)
        return 1

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_FILE

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Could not open input video: {input_path}", file=sys.stderr)
        return 1

    writer = None
    try:
        ok, frame = cap.read()
        if not ok:
            print("Could not read the first frame from the input video.", file=sys.stderr)
            return 1

        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        detector = make_detector(model_path, width, height)
        writer = open_video_writer(output_path, fps, width, height)
        print(f"Detecting and writing at {width}x{height}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_missed_frames = max(1, int(round(fps * PERSIST_SECONDS)))
        progress_every = max(1, int(round(fps * PROGRESS_EVERY_SECONDS)))
        tracks: list[dict] = []
        frame_index = 0

        while ok:
            if frame.shape[1] != width or frame.shape[0] != height:
                print("Frame size changed unexpectedly; stopping.", file=sys.stderr)
                return 1

            detections = detect_faces(detector, frame, width, height)

            tracks = update_tracks(tracks, detections, frame_index, max_missed_frames, width, height)
            black_out(frame, [track["rect"] for track in tracks])
            writer.write(frame)

            if frame_index % progress_every == 0:
                if total_frames > 0:
                    percent = 100.0 * frame_index / total_frames
                    print(f"Processed frame {frame_index}/{total_frames} ({percent:.1f}%)")
                else:
                    print(f"Processed frame {frame_index}")

            ok, frame = cap.read()
            frame_index += 1

        print(f"Done. Wrote {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
