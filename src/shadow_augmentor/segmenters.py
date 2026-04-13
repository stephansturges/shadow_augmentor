from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from shadow_augmentor.models import Point, YoloAnnotation, YoloSample


class BoxSegmenter(Protocol):
    def prepare_image(self, image: np.ndarray) -> None:
        ...

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        annotation: YoloAnnotation | None = None,
        sample: YoloSample | None = None,
    ) -> np.ndarray | Sequence[Point]:
        ...


@dataclass
class BBoxRectangleSegmenter:
    def prepare_image(self, image: np.ndarray) -> None:
        _ = image

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        annotation: YoloAnnotation | None = None,
        sample: YoloSample | None = None,
    ) -> np.ndarray:
        _ = image, annotation, sample
        x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = 1
        return mask


@dataclass
class SAMBoxPredictorAdapter:
    predictor: object
    multimask_output: bool = False

    def prepare_image(self, image: np.ndarray) -> None:
        set_image = getattr(self.predictor, "set_image", None)
        if callable(set_image):
            set_image(image)

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        annotation: YoloAnnotation | None = None,
        sample: YoloSample | None = None,
    ) -> np.ndarray:
        _ = image, annotation, sample
        predict = getattr(self.predictor, "predict", None)
        if not callable(predict):
            raise TypeError("Predictor must expose a callable `predict` method.")

        box = np.asarray(bbox_xyxy, dtype=np.float32)
        try:
            masks, scores, _ = predict(box=box[None, :], multimask_output=self.multimask_output)
        except TypeError:
            masks, scores, _ = predict(box=box, multimask_output=self.multimask_output)

        masks_array = np.asarray(masks)
        scores_array = np.asarray(scores)
        if masks_array.ndim == 2:
            return masks_array
        if masks_array.ndim < 3:
            raise ValueError("Predictor returned an unexpected mask tensor shape.")

        best_index = int(np.argmax(scores_array)) if scores_array.size else 0
        return masks_array[best_index]
