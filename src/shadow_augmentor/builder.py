from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence
import warnings

import numpy as np

from shadow_augmentor.constants import LIBRARY_VERSION
from shadow_augmentor.geometry import (
    bbox_to_polygon,
    mask_to_polygon,
    polygon_area_pixels,
    polygon_to_normalized,
)
from shadow_augmentor.models import (
    ShadowPolyCache,
    ShadowPolyIssue,
    ShadowPolygon,
    YoloAnnotation,
    YoloSample,
    build_cache_meta,
)
from shadow_augmentor.segmenters import BoxSegmenter
from shadow_augmentor.yolo import YoloDataset, _sha256_file


@dataclass(frozen=True)
class ShadowPolyBuildConfig:
    selected_classes: Sequence[int | str] | None = None
    splits: Sequence[str] | None = None
    cache_policy: str = "reuse_valid"
    on_segmenter_error: str = "raise"
    max_bbox_fallback_rate: float = 0.2
    min_polygon_area_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.cache_policy not in {"reuse_valid", "overwrite"}:
            raise ValueError("cache_policy must be `reuse_valid` or `overwrite`.")
        if self.on_segmenter_error not in {"raise", "bbox_fallback"}:
            raise ValueError("on_segmenter_error must be `raise` or `bbox_fallback`.")
        if not 0.0 <= self.max_bbox_fallback_rate <= 1.0:
            raise ValueError("max_bbox_fallback_rate must be between 0 and 1.")
        if not 0.0 < self.min_polygon_area_ratio <= 1.0:
            raise ValueError("min_polygon_area_ratio must be greater than 0 and at most 1.")


@dataclass(frozen=True)
class ShadowPolyBuildReport:
    images_seen: int
    images_written: int
    images_reused: int
    images_rebuilt: int
    images_skipped_no_targets: int
    objects_considered: int
    objects_segmented: int
    objects_from_sam: int
    objects_fell_back_to_bbox: int
    object_errors: int
    segmenter_preflight_completed: bool
    issues: tuple[ShadowPolyIssue, ...] = ()

    @property
    def bbox_fallback_rate(self) -> float:
        if self.objects_segmented == 0:
            return 0.0
        return self.objects_fell_back_to_bbox / self.objects_segmented


class ShadowPolyBuilder:
    def __init__(self, dataset: YoloDataset) -> None:
        self.dataset = dataset

    def generate(
        self,
        segmenter: BoxSegmenter,
        selected_classes: Sequence[int | str] | None = None,
        splits: Sequence[str] | None = None,
        overwrite: bool = False,
        config: ShadowPolyBuildConfig | None = None,
    ) -> ShadowPolyBuildReport:
        build_config = self._resolve_config(
            selected_classes=selected_classes,
            splits=splits,
            overwrite=overwrite,
            config=config,
        )
        class_ids = tuple(sorted(self.dataset.resolve_class_ids(build_config.selected_classes)))

        images_seen = 0
        images_written = 0
        images_reused = 0
        images_rebuilt = 0
        images_skipped_no_targets = 0
        objects_considered = 0
        objects_segmented = 0
        objects_from_sam = 0
        objects_fell_back_to_bbox = 0
        object_errors = 0
        segmenter_preflight_completed = False
        issues: list[ShadowPolyIssue] = []

        for sample in self.dataset.iter_samples(build_config.splits):
            images_seen += 1
            annotations = self.dataset.load_annotations(sample)
            selected_annotations = [annotation for annotation in annotations if annotation.class_id in class_ids]
            if not selected_annotations:
                images_skipped_no_targets += 1
                continue

            objects_considered += len(selected_annotations)
            existing_cache = sample.shadow_poly_path.exists()
            if build_config.cache_policy == "reuse_valid" and existing_cache:
                validation = self.dataset.validate_shadow_poly_cache(
                    sample,
                    expected_selected_class_ids=class_ids,
                    selection_match_mode="exact",
                )
                if validation.valid:
                    images_reused += 1
                    continue
                issues.extend(validation.issues)
                images_rebuilt += 1
            elif build_config.cache_policy == "overwrite" and existing_cache:
                images_rebuilt += 1

            image = self.dataset.load_image(sample)
            image_height, image_width = image.shape[:2]
            sample_path = sample.relative_image_path(self.dataset.root)
            sample_issues: list[ShadowPolyIssue] = []

            try:
                segmenter.prepare_image(image)
                segmenter_preflight_completed = True
            except Exception as exc:
                object_errors += len(selected_annotations)
                if build_config.on_segmenter_error == "raise":
                    raise RuntimeError(f"Segmenter failed while preparing image `{sample_path}`.") from exc
                prepare_message = f"Segmenter prepare_image failed: {type(exc).__name__}: {exc}"
                sample_issues.append(
                    ShadowPolyIssue(
                        code="segmenter_prepare_failed",
                        message=prepare_message,
                        severity="warning",
                        sample_path=sample_path,
                    )
                )
                polygons = tuple(
                    self._build_bbox_fallback_polygon(
                        annotation=annotation,
                        image_width=image_width,
                        image_height=image_height,
                        warning=prepare_message,
                    )
                    for annotation in selected_annotations
                )
                objects_segmented += len(polygons)
                objects_fell_back_to_bbox += len(polygons)
                self._save_sample_cache(
                    sample=sample,
                    segmenter=segmenter,
                    class_ids=class_ids,
                    image=image,
                    polygons=polygons,
                    issues=sample_issues,
                )
                images_written += 1
                continue

            polygons: list[ShadowPolygon] = []
            for annotation in selected_annotations:
                bbox_xyxy = annotation.bbox.to_xyxy(image_width, image_height)
                warnings_for_polygon: list[str] = []
                source = "sam"

                try:
                    segmenter_output = segmenter.segment_bbox(
                        image=image,
                        bbox_xyxy=bbox_xyxy,
                        annotation=annotation,
                        sample=sample,
                    )
                    polygon_pixels = self._segmenter_output_to_polygon(segmenter_output)
                except Exception as exc:
                    object_errors += 1
                    if build_config.on_segmenter_error == "raise":
                        raise RuntimeError(
                            f"Segmenter failed for `{sample_path}` object index {annotation.object_index}."
                        ) from exc
                    fallback_message = f"Segmenter error: {type(exc).__name__}: {exc}"
                    warnings_for_polygon.append(fallback_message)
                    sample_issues.append(
                        ShadowPolyIssue(
                            code="segmenter_object_failed",
                            message=fallback_message,
                            severity="warning",
                            sample_path=sample_path,
                            object_index=annotation.object_index,
                        )
                    )
                    polygon_pixels = bbox_to_polygon(bbox_xyxy)
                    source = "bbox_fallback"
                else:
                    polygon_area_ratio = self._polygon_area_ratio(polygon_pixels, bbox_xyxy)
                    if len(polygon_pixels) < 3:
                        warnings_for_polygon.append("Segmentation returned fewer than three polygon vertices.")
                        source = "bbox_fallback"
                    elif polygon_area_ratio < build_config.min_polygon_area_ratio:
                        warnings_for_polygon.append(
                            "Segmentation polygon area ratio was below the minimum threshold and fell back to bbox."
                        )
                        source = "bbox_fallback"

                    if source == "bbox_fallback":
                        sample_issues.append(
                            ShadowPolyIssue(
                                code="bbox_fallback",
                                message=warnings_for_polygon[-1],
                                severity="warning",
                                sample_path=sample_path,
                                object_index=annotation.object_index,
                            )
                        )
                        polygon_pixels = bbox_to_polygon(bbox_xyxy)

                polygon_area_ratio = self._polygon_area_ratio(polygon_pixels, bbox_xyxy)
                polygon_normalized = polygon_to_normalized(polygon_pixels, image_width, image_height)
                polygons.append(
                    ShadowPolygon(
                        object_index=annotation.object_index,
                        class_id=annotation.class_id,
                        class_name=annotation.class_name,
                        bbox=annotation.bbox,
                        polygon=polygon_normalized,
                        source=source,
                        warnings=tuple(warnings_for_polygon),
                        vertex_count=len(polygon_normalized),
                        area_ratio=float(polygon_area_ratio),
                    )
                )
                objects_segmented += 1
                if source == "bbox_fallback":
                    objects_fell_back_to_bbox += 1
                else:
                    objects_from_sam += 1

            self._save_sample_cache(
                sample=sample,
                segmenter=segmenter,
                class_ids=class_ids,
                image=image,
                polygons=tuple(polygons),
                issues=sample_issues,
            )
            issues.extend(sample_issues)
            images_written += 1

        report = ShadowPolyBuildReport(
            images_seen=images_seen,
            images_written=images_written,
            images_reused=images_reused,
            images_rebuilt=images_rebuilt,
            images_skipped_no_targets=images_skipped_no_targets,
            objects_considered=objects_considered,
            objects_segmented=objects_segmented,
            objects_from_sam=objects_from_sam,
            objects_fell_back_to_bbox=objects_fell_back_to_bbox,
            object_errors=object_errors,
            segmenter_preflight_completed=segmenter_preflight_completed,
            issues=tuple(issues),
        )
        if report.bbox_fallback_rate > build_config.max_bbox_fallback_rate:
            raise RuntimeError(
                "BBox fallback rate exceeded the configured threshold: "
                f"{report.bbox_fallback_rate:.3f} > {build_config.max_bbox_fallback_rate:.3f}"
            )
        return report

    def _resolve_config(
        self,
        *,
        selected_classes: Sequence[int | str] | None,
        splits: Sequence[str] | None,
        overwrite: bool,
        config: ShadowPolyBuildConfig | None,
    ) -> ShadowPolyBuildConfig:
        if config is None:
            config = ShadowPolyBuildConfig(selected_classes=selected_classes, splits=splits)
        if overwrite:
            warnings.warn(
                "`overwrite` is deprecated; set ShadowPolyBuildConfig(cache_policy='overwrite') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            config = ShadowPolyBuildConfig(
                selected_classes=selected_classes if selected_classes is not None else config.selected_classes,
                splits=splits if splits is not None else config.splits,
                cache_policy="overwrite",
                on_segmenter_error=config.on_segmenter_error,
                max_bbox_fallback_rate=config.max_bbox_fallback_rate,
                min_polygon_area_ratio=config.min_polygon_area_ratio,
            )
            return config
        if selected_classes is None and splits is None:
            return config
        return ShadowPolyBuildConfig(
            selected_classes=selected_classes if selected_classes is not None else config.selected_classes,
            splits=splits if splits is not None else config.splits,
            cache_policy=config.cache_policy,
            on_segmenter_error=config.on_segmenter_error,
            max_bbox_fallback_rate=config.max_bbox_fallback_rate,
            min_polygon_area_ratio=config.min_polygon_area_ratio,
        )

    def _save_sample_cache(
        self,
        *,
        sample: YoloSample,
        segmenter: BoxSegmenter,
        class_ids: Sequence[int],
        image: np.ndarray,
        polygons: tuple[ShadowPolygon, ...],
        issues: Sequence[ShadowPolyIssue],
    ) -> None:
        image_height, image_width = image.shape[:2]
        generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        meta = build_cache_meta(
            generated_at=generated_at,
            builder_version=LIBRARY_VERSION,
            segmenter_name=type(segmenter).__name__,
            selected_class_ids=class_ids,
            sample=sample,
            root=self.dataset.root,
            image_width=image_width,
            image_height=image_height,
            image_sha256=_sha256_file(sample.image_path),
            label_sha256=_sha256_file(sample.label_path) if sample.label_path.exists() else None,
        )
        cache = ShadowPolyCache(meta=meta, polygons=polygons, issues=tuple(issues))
        self.dataset.save_shadow_polys(sample, cache)

    @staticmethod
    def _build_bbox_fallback_polygon(
        *,
        annotation: YoloAnnotation,
        image_width: int,
        image_height: int,
        warning: str,
    ) -> ShadowPolygon:
        bbox_xyxy = annotation.bbox.to_xyxy(image_width, image_height)
        polygon_normalized = polygon_to_normalized(bbox_to_polygon(bbox_xyxy), image_width, image_height)
        return ShadowPolygon(
            object_index=annotation.object_index,
            class_id=annotation.class_id,
            class_name=annotation.class_name,
            bbox=annotation.bbox,
            polygon=polygon_normalized,
            source="bbox_fallback",
            warnings=(warning,),
            vertex_count=len(polygon_normalized),
            area_ratio=1.0,
        )

    @staticmethod
    def _segmenter_output_to_polygon(segmenter_output: np.ndarray | Sequence[tuple[float, float]]) -> np.ndarray:
        if isinstance(segmenter_output, np.ndarray):
            if segmenter_output.ndim == 2:
                return mask_to_polygon(segmenter_output)
            if segmenter_output.ndim == 3 and segmenter_output.shape[-1] == 2:
                return segmenter_output.astype(np.float32)
            return np.empty((0, 2), dtype=np.float32)

        polygon = np.asarray(list(segmenter_output), dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float32)
        return polygon

    @staticmethod
    def _polygon_area_ratio(polygon_pixels: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:
        if len(polygon_pixels) < 3:
            return 0.0
        x0, y0, x1, y1 = bbox_xyxy
        bbox_area = max((x1 - x0) * (y1 - y0), 1.0)
        return float(polygon_area_pixels(polygon_pixels) / bbox_area)
