from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    width: int
    height: int


_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        )
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Unable to load face detector: {cascade_path}")
        _FACE_CASCADE = cascade
    return _FACE_CASCADE


def detect_faces(image_rgb: np.ndarray) -> list[FaceRegion]:
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] == 0
        or image_rgb.shape[1] == 0
    ):
        raise ValueError("Expected a non-empty RGB image with three color channels.")

    scale = min(1.0, 960 / max(image_rgb.shape[:2]))
    if scale < 1:
        detection_image = cv2.resize(
            image_rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = image_rgb

    gray = cv2.cvtColor(detection_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    detections = _face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    inverse_scale = 1 / scale
    image_height, image_width = image_rgb.shape[:2]
    regions = []
    for x, y, width, height in detections:
        regions.append(
            FaceRegion(
                x=max(0, round(x * inverse_scale)),
                y=max(0, round(y * inverse_scale)),
                width=min(round(width * inverse_scale), image_width - round(x * inverse_scale)),
                height=min(round(height * inverse_scale), image_height - round(y * inverse_scale)),
            )
        )
    return regions


def crop_face_with_context(image_rgb: np.ndarray, face: FaceRegion) -> np.ndarray:
    x1, y1, x2, y2 = face_crop_bounds(image_rgb.shape, face)
    return image_rgb[y1:y2, x1:x2]


def face_crop_bounds(
    image_shape: tuple[int, ...],
    face: FaceRegion,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    margin_x = round(face.width * 0.2)
    margin_y = round(face.height * 0.2)
    x1 = max(0, face.x - margin_x)
    y1 = max(0, face.y - margin_y)
    x2 = min(width, face.x + face.width + margin_x)
    y2 = min(height, face.y + face.height + margin_y)
    return x1, y1, x2, y2
