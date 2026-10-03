"""OpenCV YuNet detection and SFace embedding extraction."""

from pathlib import Path

import cv2
import numpy as np


class FaceEngine:
    """Detect faces and produce SFace feature vectors from BGR images."""

    def __init__(self, detector_path: Path, recognizer_path: Path) -> None:
        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(
            cv2, "FaceRecognizerSF_create"
        ):
            raise RuntimeError(
                "OpenCV face modules are missing. Install opencv-contrib-python."
            )
        if not detector_path.is_file() or not recognizer_path.is_file():
            raise FileNotFoundError(
                "Face models are missing. They are downloaded when recognition is opened."
            )
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_path), "", (320, 320), 0.85, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_path), "")

    def analyze(self, image: np.ndarray) -> list[dict[str, object]]:
        if image is None or image.size == 0:
            raise ValueError("The image is empty or could not be decoded.")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        results: list[dict[str, object]] = []
        if faces is None:
            return results
        for face in faces:
            aligned = self.recognizer.alignCrop(image, face)
            feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            results.append(
                {
                    "box": (x, y, box_width, box_height),
                    "embedding": feature,
                    "confidence": float(face[-1]),
                }
            )
        return results