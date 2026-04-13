from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import warnings

import numpy as np

from shadow_augmentor.augment import ShadowAugmentor
from shadow_augmentor.models import CacheValidationResult, ShadowPolygon, YoloAnnotation, YoloSample
from shadow_augmentor.yolo import YoloDataset

RngFactory = Callable[[int, YoloSample], np.random.Generator]


@dataclass(frozen=True)
class TrainingSample:
    image: np.ndarray
    class_ids: np.ndarray
    boxes: np.ndarray
    shadow_polys: tuple[ShadowPolygon, ...]
    path: str
    sample: YoloSample

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "class_ids": self.class_ids,
            "boxes": self.boxes,
            "shadow_polys": self.shadow_polys,
            "path": self.path,
            "sample": self.sample,
        }


class ShadowAugmentedYoloDataset:
    def __init__(
        self,
        dataset: YoloDataset,
        split: str,
        augmenter: ShadowAugmentor | None = None,
        *,
        cache_validation: str = "error",
        preload_shadow_polys: bool = False,
        rng_factory: RngFactory | None = None,
    ) -> None:
        if cache_validation not in {"error", "warn", "ignore"}:
            raise ValueError("cache_validation must be `error`, `warn`, or `ignore`.")
        self.dataset = dataset
        self.split = split
        self.samples = dataset.get_samples(split)
        self.augmenter = augmenter
        self.cache_validation = cache_validation
        self.rng_factory = rng_factory
        self._shadow_polys_by_index: dict[int, tuple[ShadowPolygon, ...]] = {}
        self._cache_validation_by_index: dict[int, CacheValidationResult] = {}
        self._selected_class_ids: tuple[int, ...] | None = None

        if self.augmenter is not None and self.augmenter.config.selected_classes is not None:
            class_ids = tuple(sorted(dataset.resolve_class_ids(self.augmenter.config.selected_classes)))
            self._selected_class_ids = class_ids
            self.augmenter.set_selected_class_ids(class_ids)
        elif self.augmenter is not None:
            class_ids = tuple(sorted(dataset.resolve_class_ids(None)))
            self._selected_class_ids = class_ids
            self.augmenter.set_selected_class_ids(class_ids)

        if preload_shadow_polys:
            for index in range(len(self.samples)):
                sample = self.samples[index]
                annotations = self.dataset.load_annotations(sample)
                self._load_shadow_polys(index=index, sample=sample, annotations=annotations)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = self.dataset.load_image(sample)
        annotations = self.dataset.load_annotations(sample)
        shadow_polys = self._load_shadow_polys(index=index, sample=sample, annotations=annotations)

        if self.augmenter is not None:
            rng = self.rng_factory(index, sample) if self.rng_factory is not None else None
            image = self.augmenter(image, shadow_polys, rng=rng, annotations=annotations)

        class_ids = np.asarray([annotation.class_id for annotation in annotations], dtype=np.int64)
        boxes = np.asarray(
            [
                [
                    annotation.bbox.x_center,
                    annotation.bbox.y_center,
                    annotation.bbox.width,
                    annotation.bbox.height,
                ]
                for annotation in annotations
            ],
            dtype=np.float32,
        )
        training_sample = TrainingSample(
            image=image,
            class_ids=class_ids,
            boxes=boxes,
            shadow_polys=shadow_polys,
            path=str(sample.image_path),
            sample=sample,
        )
        return training_sample.to_dict()

    def _load_shadow_polys(
        self,
        *,
        index: int,
        sample: YoloSample,
        annotations: list[YoloAnnotation],
    ) -> tuple[ShadowPolygon, ...]:
        if index in self._shadow_polys_by_index:
            return self._shadow_polys_by_index[index]

        if self.cache_validation == "ignore":
            cache = self.dataset.load_shadow_poly_cache(sample)
            polygons = tuple(cache.polygons) if cache is not None else ()
            self._shadow_polys_by_index[index] = polygons
            return polygons

        validation = self.dataset.validate_shadow_poly_cache(
            sample,
            expected_selected_class_ids=self._selected_class_ids,
            selection_match_mode="subset",
        )
        self._cache_validation_by_index[index] = validation

        target_classes_present = self._target_classes_present(annotations)
        if validation.valid:
            polygons = tuple(validation.cache.polygons) if validation.cache is not None else ()
            self._shadow_polys_by_index[index] = polygons
            return polygons

        if target_classes_present:
            issue_summary = "; ".join(f"{issue.code}: {issue.message}" for issue in validation.issues)
            message = f"Invalid shadow polygon cache for `{sample.image_path}`. {issue_summary}"
            if self.cache_validation == "error":
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        self._shadow_polys_by_index[index] = ()
        return ()

    def _target_classes_present(self, annotations: list[YoloAnnotation]) -> bool:
        if self._selected_class_ids is None:
            return bool(annotations)
        selected = set(self._selected_class_ids)
        return any(annotation.class_id in selected for annotation in annotations)
