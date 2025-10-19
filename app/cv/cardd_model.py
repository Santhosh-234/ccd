from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import numpy as np
import cv2


@dataclass
class DLResult:
    annotated_bgr: np.ndarray
    severity: str
    num_regions: int
    boxes: List[Tuple[int, int, int, int]]


_yolo_model = None  # lazy-loaded global


def _load_model() -> Optional[object]:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return None

    weights = os.getenv("YOLO_MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "car_damage.pt"))
    weights = os.path.abspath(weights)
    if not os.path.exists(weights):
        return None
    try:
        _yolo_model = YOLO(weights)
        return _yolo_model
    except Exception:
        return None


def _classify_severity(num_regions: int, area_ratio: float) -> str:
    if num_regions >= 6 or area_ratio >= 0.18:
        return "severe"
    if num_regions >= 3 or area_ratio >= 0.10:
        return "moderate"
    return "minor"


def detect_with_yolo(image_bytes: bytes, conf: float = 0.35) -> Optional[DLResult]:
    model = _load_model()
    if model is None:
        return None

    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    try:
        results = model.predict(source=bgr[:, :, ::-1], imgsz=768, conf=conf, verbose=False)
    except Exception:
        return None

    if not results:
        return None

    res = results[0]
    h, w = bgr.shape[:2]
    total_area = float(h * w)
    annotated = bgr.copy()
    boxes: List[Tuple[int, int, int, int]] = []
    area_sum = 0.0

    try:
        # Handle both detection and segmentation models
        if getattr(res, 'masks', None) is not None and res.masks is not None:
            masks = res.masks.data.cpu().numpy()  # (n, H, W) relative to input size
            # Resize masks to original image size
            for m in masks:
                m_resized = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < max(150, 0.0004 * total_area):
                        continue
                    x, y, bw, bh = cv2.boundingRect(c)
                    boxes.append((x, y, bw, bh))
                    area_sum += area
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        elif getattr(res, 'boxes', None) is not None and res.boxes is not None:
            for b in res.boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                bw, bh = x2 - x1, y2 - y1
                area = float(bw * bh)
                if area < max(1500, 0.0008 * total_area):
                    continue
                boxes.append((x1, y1, bw, bh))
                area_sum += area
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            return None
    except Exception:
        return None

    if not boxes:
        return None

    area_ratio = area_sum / total_area if total_area > 0 else 0.0
    severity = _classify_severity(len(boxes), area_ratio)

    w0 = annotated.shape[1]
    cv2.rectangle(annotated, (0, 0), (w0, 40), (0, 0, 0), thickness=-1)
    cv2.putText(annotated, f"YOLO damage: {severity} | regions: {len(boxes)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return DLResult(annotated_bgr=annotated, severity=severity, num_regions=len(boxes), boxes=boxes)


