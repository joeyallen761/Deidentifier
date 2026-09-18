# Video De-identification

This script runs locally and blacks out detected faces in a video. It writes a new MP4 beside the input video and does not upload video anywhere.

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

Make sure the local face detector model is present in this directory:

```bash
face_detection_yunet_2023mar.onnx
```

If it is missing, download it once during setup:

```bash
curl -L -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Run

Run the script with the video path:

```bash
source .venv/bin/activate
python deidentify.py my_research_video.MOV
```

For example:

```bash
python deidentify.py TT1_FrontTableAngle_Video.MOV
```

The output is saved beside the input as:

```bash
TT1_FrontTableAngle_Video_deidentified.mp4
```

Audio is not preserved.
