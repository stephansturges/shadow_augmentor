from __future__ import annotations

from typing import Any

import numpy as np

from shadow_augmentor.models import YoloSample


def _boxes_yolo_to_xyxy_pixels(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = boxes.astype(np.float32, copy=False)
    x_center = boxes[:, 0] * image_width
    y_center = boxes[:, 1] * image_height
    widths = boxes[:, 2] * image_width
    heights = boxes[:, 3] * image_height
    x0 = x_center - (widths * 0.5)
    y0 = y_center - (heights * 0.5)
    x1 = x_center + (widths * 0.5)
    y1 = y_center + (heights * 0.5)
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)


def to_torchvision_detection_sample(sample: dict[str, Any]) -> dict[str, Any]:
    image = np.asarray(sample["image"])
    boxes = np.asarray(sample["boxes"], dtype=np.float32)
    class_ids = np.asarray(sample["class_ids"], dtype=np.int64)
    image_height, image_width = image.shape[:2]
    boxes_xyxy = _boxes_yolo_to_xyxy_pixels(boxes, image_width, image_height)

    return {
        "image": np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)),
        "target": {
            "boxes": boxes_xyxy,
            "labels": class_ids,
            "image_size": np.asarray([image_height, image_width], dtype=np.int64),
            "path": sample.get("path", ""),
        },
    }


def to_albumentations_sample(sample: dict[str, Any]) -> dict[str, Any]:
    boxes = np.asarray(sample["boxes"], dtype=np.float32)
    class_ids = np.asarray(sample["class_ids"], dtype=np.int64)
    return {
        "image": np.asarray(sample["image"]),
        "bboxes": [tuple(float(value) for value in box) for box in boxes],
        "class_labels": [int(value) for value in class_ids.tolist()],
        "bbox_format": "yolo",
        "path": sample.get("path", ""),
    }


def summarize_framework_views(sample: dict[str, Any]) -> dict[str, Any]:
    torchvision_sample = to_torchvision_detection_sample(sample)
    albumentations_sample = to_albumentations_sample(sample)
    source_sample = sample.get("sample")
    sample_path = source_sample.image_path if isinstance(source_sample, YoloSample) else sample.get("path", "")

    return {
        "path": str(sample_path),
        "torchvision": {
            "image_shape": list(torchvision_sample["image"].shape),
            "image_dtype": str(torchvision_sample["image"].dtype),
            "boxes_shape": list(torchvision_sample["target"]["boxes"].shape),
            "labels_shape": list(torchvision_sample["target"]["labels"].shape),
        },
        "albumentations": {
            "bbox_count": len(albumentations_sample["bboxes"]),
            "bbox_format": albumentations_sample["bbox_format"],
            "class_label_count": len(albumentations_sample["class_labels"]),
            "image_shape": list(albumentations_sample["image"].shape),
        },
    }
