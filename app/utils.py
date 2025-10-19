import os
import uuid
import cv2
import numpy as np


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def generate_filename(prefix: str = "", ext: str = ".png") -> str:
    return f"{prefix}{uuid.uuid4().hex}{ext}"


def save_image_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def bgr_image_to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


