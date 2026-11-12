# Video De-identification

This script runs locally and blacks out detected faces in a video. It writes a new MP4 beside the input video and does not upload video anywhere.

## Install

The working environment in this checkout is `.ritel_env`; OpenCV and NumPy are already installed there. Use the **Run** commands below to activate it.

For a new checkout, download or clone this repository, then create the environment and install dependencies from the project directory:

```bash
python3 -m venv .ritel_env
source .ritel_env/bin/activate      # macOS/Linux
python -m pip install -r requirements.txt
```

Activating an environment selects its Python interpreter; it does not install dependencies. If you use a different environment name, such as `.venv`, use that same environment for both installation and running the script.

Check that the active interpreter can import OpenCV:

```bash
python -c "import sys, cv2; print(sys.executable); print(cv2.__version__)"
```

Install [FFmpeg](https://ffmpeg.org/download.html) 6.1 or newer with the `fsync` filter and `libx264` encoder, and make sure `ffmpeg` and `ffprobe` are available on your PATH. On macOS with Homebrew, run `brew install ffmpeg`.

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
source .ritel_env/bin/activate
python deidentify.py my_research_video.MOV
```

You can also run the installed interpreter directly without activation:

```bash
.ritel_env/bin/python deidentify.py my_research_video.MOV
```

For example:

```bash
python deidentify.py TT1_FrontTableAngle_Video.MOV
```

The output is saved beside the input as:

```bash
TT1_FrontTableAngle_Video_deidentified.mp4
```

Existing output files are not overwritten. Move or rename a previous output before processing the same input again.

Original audio tracks are included in the output, encoded as AAC for MP4 compatibility. Videos without audio are also supported.

After face processing, the script checks the original frame timestamps and restores them if necessary. This prevents duration changes and audio drift when OpenCV writes fractional frame rates incorrectly, and preserves variable frame timing. Timing repair re-encodes the processed video as H.264 at CRF 18; matching timing allows the video to be copied without re-encoding. Reading and verifying frame timing adds processing time.

## Restore audio to an existing silent video

**I recommend the normal restoration mode for exact synchronization with the original video:**

```bash
python3 restore_audio.py
```

Enter the silent/deidentified video path, then the original video path when prompted. The script saves `<silent_video_stem>_with_audio.mp4` beside the silent video. It restores the original frame timestamps when timing differs, using the processed video frames, and adds all audio tracks from the original as AAC. Both input files are preserved, and existing output files are not overwritten. This standalone script requires only Python and FFmpeg, without OpenCV or the face detector model.

For corresponding videos with the same frames in the same order, the restored video has the same length and frame timing as the original, keeping audio synchronized without stretching speech. For TT1, this returns the video to the original 51:29 duration instead of leaving it at 53:14. This approach takes longer because it checks frame timestamps and re-encodes the video when timing needs repair.

Use corresponding, untrimmed videos with the same frames in the same order. Different frame counts are rejected because they cannot be repaired by matching frame positions. Existing files such as the 53:14 TT1 output can be repaired this way without rerunning face detection. If a previous `_with_audio.mp4` output exists, move or rename it before rerunning the script.

**I do not recommend audio-only restoration when accurate synchronization and undistorted audio matter.** It is available as a faster alternative:

```bash
python3 restore_audio.py --audio-only
```

This mode reads stream metadata, copies the encoded video, and stretches the original audio to match the silent video's duration while preserving pitch. Stretching changes speech speed and can introduce audible distortion; matching overall duration also does not guarantee local synchronization when the original has variable frame timing. For TT1, audio is slowed to approximately 0.9672x to match the existing 53:14 video, so the result retains a different length from the original recording. It avoids video decoding and encoding, shows FFmpeg progress, and preserves both input files.
