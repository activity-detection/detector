## Detector

An application for detecting objects and actions in video streams. It uses YOLO models (detection + pose) and LSTM for activity classification.

**Key Features**
- Object and person tracking (Ultralytics YOLO)
- Pose estimation and feature extraction
- Sequence classification (LSTM) for actions
- Sending clips to the backend

**Requirements**
- uv dependency manager
- Docker with GPU support (NVIDIA runtime)

Detailed dependencies are listed in `pyproject.toml`.

## Quick Start (Docker)

1. Fill in the `.env` file (an example file is included in the repo). The most important variables are described below.
2. Run using Compose (with GPU):

```bash
docker compose -f compose.yaml up --build
```

Or run without Compose (example):

```bash
docker run --gpus all --name detector \
    --mount type=bind,src=$(pwd)/clips,target=/clips \
    --mount type=bind,src=$(pwd)/vectors,target=/app/vectors,readonly \
	--mount type=bind,src=$(pwd)/examples,target=/app/examples,readonly \
    --env-file .env detector
```

## Configuration and environment variables
The configuration is loaded from the `.env` file. Key settings:

- `APP_MODE` - operating mode: `VIDEO` (file), `USB` (camera), `FOLDER` (image folder), `IMAGE` (single photo), `RTSP` (RTSP stream). Default is `VIDEO`.
- `SOURCE_PATH` - path to the video file / USB device number / folder path / image file.
- `ACTION_VECTORS_PATH` - path to the CSV file containing action vectors (e.g., `vectors/vectors.csv`).
- `BASE_MODEL_PATH` - path to the YOLO model for detection (default `models/yolo26m_main.pt`).
- `POSE_MODEL_PATH` - path to the pose model (default `models/yolo11m-pose.pt`).
- `LSTM_MODEL_PATH` (alias `LSTM_MODEL_PATH`) - the relative path to the LSTM weights file (set as `LSTM_MODEL_PATH` in `.env`; in the configuration, it is combined with `BASE_DIR`).
- `CLIP_FOLDER` - the directory where clips are saved if `SAVE_CLIPS=True`.
- `SAVE_CLIPS` - `True|False` - save clips locally.
- `UPLOAD_CLIPS` - `True|False` - upload clips to `BACKEND_UPLOAD_URL`.
- `BACKEND_UPLOAD_URL` - the backend endpoint URL for uploading clips.
- `CAMERA_USER`, `CAMERA_PASSWORD`, `CAMERA_IP`, `CAMERA_PORT`, `RTSP_PATH` - RTSP parameters (used when `APP_MODE=RTSP`).
- `BATCH_SIZE` - batch size of processed frames (default 8).

Many of the above variables have sample values in `compose.yaml`.

## Required Models and Files
- The `models/final_models/` folder should contains the required models (`yolo26m_main.pt`, `yolo11m-pose.pt`, `lstm_5class_1.pth`, etc.).
- The LSTM weights file is required-the application will terminate if it cannot find the LSTM model.
- The `vectors/vectors.csv` file (action vectors) is used by `Recorder` to load action classes.

## Operating Modes and Examples
- `APP_MODE=VIDEO` + `SOURCE_PATH=/path/to/video.mp4` - video file processing.
- `APP_MODE=RTSP` - configure camera parameters in `.env`.

## Saving and uploading clips
- If `SAVE_CLIPS=True`, clips with detections will be saved to `CLIP_FOLDER`.
- If `UPLOAD_CLIPS=True`, saved clips will be uploaded to `BACKEND_UPLOAD_URL`.


## Repository structure
- `main.py` - application entry point.
- `src/detector/` - implementation of the detector, configuration, sources, and recording.
- `models/` - folder containing models.
- `models/final_models/` - folder containing models that will be copied to Docker image.
- `vectors/vectors.csv` - file with action vectors.
- `compose.yaml`, `Dockerfile` - Docker/Compose configuration.
