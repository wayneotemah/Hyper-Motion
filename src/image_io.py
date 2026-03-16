from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


def save_bgr_image(image: np.ndarray, output_path: Path, quality: int = 95) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path, quality=quality)
    return output_path


def load_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.width, image.height


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bbox_from_points(points: Iterable[Tuple[float, float]], image_width: int, image_height: int) -> Optional[Tuple[float, float, float, float]]:
    valid = [(x, y) for x, y in points if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0]
    if not valid:
        return None

    xs = [pt[0] * image_width for pt in valid]
    ys = [pt[1] * image_height for pt in valid]
    return min(xs), min(ys), max(xs), max(ys)


def crop_signer_frame(
    frame: np.ndarray,
    points: Optional[Sequence[Tuple[float, float]]] = None,
    output_size: Optional[Tuple[int, int]] = None,
    padding_ratio: float = 0.2,
) -> np.ndarray:
    image_height, image_width = frame.shape[:2]
    bbox = _bbox_from_points(points or [], image_width, image_height)

    if bbox is None:
        crop_width = image_width * 0.55
        crop_height = image_height * 0.7
        center_x = image_width * 0.5
        center_y = image_height * 0.38
    else:
        min_x, min_y, max_x, max_y = bbox
        box_width = max(max_x - min_x, image_width * 0.18)
        box_height = max(max_y - min_y, image_height * 0.24)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        crop_width = box_width * (1.0 + padding_ratio * 2.2)
        crop_height = box_height * (1.0 + padding_ratio * 2.8)

    if output_size:
        target_width, target_height = output_size
        target_aspect = target_width / float(target_height)
        crop_aspect = crop_width / float(crop_height)
        if crop_aspect > target_aspect:
            crop_height = crop_width / target_aspect
        else:
            crop_width = crop_height * target_aspect

    half_w = crop_width / 2.0
    half_h = crop_height / 2.0
    left = _clamp(center_x - half_w, 0, image_width)
    top = _clamp(center_y - half_h, 0, image_height)
    right = _clamp(center_x + half_w, 0, image_width)
    bottom = _clamp(center_y + half_h, 0, image_height)

    left_i = int(round(left))
    top_i = int(round(top))
    right_i = int(round(right))
    bottom_i = int(round(bottom))
    cropped = frame[top_i:bottom_i, left_i:right_i]

    if output_size and cropped.size:
        cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
    return cropped
