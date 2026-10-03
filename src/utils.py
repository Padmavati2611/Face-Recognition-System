"""Application configuration, model retrieval, and image decoding."""

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("FACEVISION_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
DATABASE_PATH = Path(
    os.getenv("FACEVISION_DB_PATH", DATA_DIR / "facevision.db")
).expanduser()
MODELS_DIR = Path(
    os.getenv("FACEVISION_MODELS_DIR", PROJECT_ROOT / "models")
).expanduser()
MATCH_THRESHOLD = float(os.getenv("FACEVISION_MATCH_THRESHOLD", "0.42"))
COOLDOWN_SECONDS = int(os.getenv("FACEVISION_COOLDOWN_SECONDS", "3600"))

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}


def is_valid_model_payload(payload: bytes) -> bool:
    """Reject HTML or empty responses before saving a model file."""
    if len(payload) < 100_000:
        return False
    head = payload[:256].lower()
    return not (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"<html" in head
        or b"not found" in head
        or b"error" in head
        or b"access denied" in head
    )


def ensure_models() -> tuple[Path, Path]:
    """Download the public OpenCV Zoo models once; never upload face data."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [MODELS_DIR / filename for filename in MODEL_URLS]
    for path, urls in MODEL_URLS.items():
        file_path = MODELS_DIR / path
        if file_path.is_file() and file_path.stat().st_size > 100_000:
            continue
        temporary_path = file_path.with_suffix(file_path.suffix + ".download")
        downloaded = False
        last_error: Exception | None = None
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "FaceVision-AI"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                if not is_valid_model_payload(payload):
                    raise RuntimeError("The model download was incomplete or not an ONNX file.")
                temporary_path.write_bytes(payload)
                temporary_path.replace(file_path)
                downloaded = True
                break
            except Exception as exc:  # pragma: no cover - exercised by real downloads
                last_error = exc
                temporary_path.unlink(missing_ok=True)
        if not downloaded:
            raise RuntimeError(
                f"Could not download {file_path.name}. Check your network and retry: {last_error}"
            ) from last_error
    return paths[0], paths[1]


def decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This image could not be decoded. Try a JPG or PNG image.")
    return image