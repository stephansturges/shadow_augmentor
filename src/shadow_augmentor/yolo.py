from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from PIL import Image

from shadow_augmentor.constants import CACHE_SCHEMA_VERSION
from shadow_augmentor.models import (
    CacheValidationResult,
    ClassSelectionSummary,
    DatasetSplitValidationReport,
    DatasetValidationReport,
    ShadowPolyCache,
    ShadowPolyCacheMeta,
    ShadowPolyIssue,
    ShadowPolygon,
    YoloAnnotation,
    YoloBBox,
    YoloSample,
)

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_yaml_path(config_path: Path, configured_root: str | None) -> Path:
    if configured_root is None:
        return config_path.parent.resolve()
    candidate = Path(configured_root).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {idx: str(value) for idx, value in enumerate(raw_names)}
    raise TypeError("YOLO dataset `names` must be a list or mapping.")


def _resolve_split_dir(root: Path, config_path: Path, split_value: object) -> Path:
    if isinstance(split_value, (list, tuple)):
        raise TypeError("This implementation expects each split path to be a single directory.")
    if not isinstance(split_value, str):
        raise TypeError("Split path must be a string.")
    candidate = Path(split_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    return (config_path.parent / candidate).resolve()


def _infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        image_idx = parts.index("images")
        parts[image_idx] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels" / image_dir.name


def _iter_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        image_path
        for image_path in image_dir.rglob("*")
        if image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _iter_label_files(label_dir: Path) -> list[Path]:
    if not label_dir.exists():
        return []
    return sorted(label_dir.rglob("*.txt"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


@dataclass
class YoloDataset:
    root: Path
    config_path: Path
    class_names: dict[int, str]
    split_dirs: dict[str, Path]
    shadow_poly_root: Path = field(default_factory=Path)
    _samples_cache: dict[str, list[YoloSample]] = field(default_factory=dict, init=False, repr=False)
    _shadow_poly_cache_cache: dict[Path, ShadowPolyCache | None] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "YoloDataset":
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        root = _resolve_yaml_path(config_path, payload.get("path"))
        class_names = _normalize_names(payload["names"])
        split_dirs: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            if split in payload:
                split_dirs[split] = _resolve_split_dir(root, config_path, payload[split])

        return cls(
            root=root,
            config_path=config_path,
            class_names=class_names,
            split_dirs=split_dirs,
            shadow_poly_root=root / "shadow_polys",
        )

    def resolve_class_ids(self, selected_classes: Sequence[int | str] | None) -> set[int]:
        if not selected_classes:
            return set(self.class_names)

        resolved: set[int] = set()
        reverse_names = {name: class_id for class_id, name in self.class_names.items()}
        for class_ref in selected_classes:
            if isinstance(class_ref, int):
                if class_ref not in self.class_names:
                    raise KeyError(f"Unknown class id: {class_ref}")
                resolved.add(class_ref)
                continue
            if class_ref not in reverse_names:
                raise KeyError(f"Unknown class name: {class_ref}")
            resolved.add(reverse_names[class_ref])
        return resolved

    def summarize_class_selection(
        self,
        selected_classes: Sequence[int | str] | None,
        splits: Iterable[str] | None = None,
    ) -> ClassSelectionSummary:
        resolved_class_ids = tuple(sorted(self.resolve_class_ids(selected_classes)))
        object_count_by_class = {class_id: 0 for class_id in resolved_class_ids}
        images_seen = 0
        images_with_selected_classes = 0

        for sample in self.iter_samples(splits):
            images_seen += 1
            annotations = self.load_annotations(sample)
            matched = [annotation for annotation in annotations if annotation.class_id in object_count_by_class]
            if matched:
                images_with_selected_classes += 1
            for annotation in matched:
                object_count_by_class[annotation.class_id] += 1

        return ClassSelectionSummary(
            resolved_class_ids=resolved_class_ids,
            resolved_class_names=tuple(self.class_names[class_id] for class_id in resolved_class_ids),
            images_seen=images_seen,
            images_with_selected_classes=images_with_selected_classes,
            object_count_by_class=object_count_by_class,
        )

    def validate_dataset(self, splits: Iterable[str] | None = None) -> DatasetValidationReport:
        split_names = tuple(splits) if splits is not None else tuple(self.split_dirs)
        split_reports: list[DatasetSplitValidationReport] = []
        issues: list[ShadowPolyIssue] = []

        for split in split_names:
            if split not in self.split_dirs:
                issues.append(
                    ShadowPolyIssue(
                        code="unknown_split",
                        message=f"Split `{split}` is not defined in {self.config_path}.",
                        severity="error",
                    )
                )
                continue

            image_dir = self.split_dirs[split]
            label_dir = _infer_label_dir(image_dir)
            split_issues: list[ShadowPolyIssue] = []
            image_files = _iter_image_files(image_dir)
            label_files = _iter_label_files(label_dir)

            if not image_dir.exists():
                split_issues.append(
                    ShadowPolyIssue(
                        code="missing_image_dir",
                        message=f"Image directory does not exist for split `{split}`: {image_dir}",
                        severity="error",
                    )
                )
            if not label_dir.exists():
                split_issues.append(
                    ShadowPolyIssue(
                        code="missing_label_dir",
                        message=f"Label directory does not exist for split `{split}`: {label_dir}",
                        severity="warning",
                    )
                )

            image_keys = {image_path.relative_to(image_dir).with_suffix(".txt") for image_path in image_files}
            label_keys = {label_path.relative_to(label_dir) for label_path in label_files} if label_dir.exists() else set()

            missing_labels = tuple(str(path) for path in sorted(image_keys - label_keys))
            orphan_labels = tuple(str(path) for path in sorted(label_keys - image_keys))

            if not image_files:
                split_issues.append(
                    ShadowPolyIssue(
                        code="empty_split",
                        message=f"No images were found for split `{split}`.",
                        severity="warning",
                    )
                )
            if missing_labels:
                split_issues.append(
                    ShadowPolyIssue(
                        code="missing_labels",
                        message=f"Split `{split}` has {len(missing_labels)} images without matching label files.",
                        severity="warning",
                    )
                )
            if orphan_labels:
                split_issues.append(
                    ShadowPolyIssue(
                        code="orphan_labels",
                        message=f"Split `{split}` has {len(orphan_labels)} label files without matching images.",
                        severity="warning",
                    )
                )

            split_reports.append(
                DatasetSplitValidationReport(
                    split=split,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    image_count=len(image_files),
                    label_count=len(label_files),
                    missing_label_paths=missing_labels,
                    orphan_label_paths=orphan_labels,
                    issues=tuple(split_issues),
                )
            )

        return DatasetValidationReport(
            root=self.root,
            config_path=self.config_path,
            split_reports=tuple(split_reports),
            issues=tuple(issues),
        )

    def get_samples(self, split: str) -> list[YoloSample]:
        if split not in self.split_dirs:
            raise KeyError(f"Split `{split}` is not defined in {self.config_path}.")
        if split in self._samples_cache:
            return self._samples_cache[split]

        image_dir = self.split_dirs[split]
        label_dir = _infer_label_dir(image_dir)
        samples: list[YoloSample] = []
        for image_path in _iter_image_files(image_dir):
            rel_path = image_path.relative_to(image_dir).with_suffix(".txt")
            label_path = label_dir / rel_path
            shadow_poly_path = self.shadow_poly_root / split / rel_path.with_suffix(".json")
            samples.append(
                YoloSample(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    shadow_poly_path=shadow_poly_path,
                )
            )

        self._samples_cache[split] = samples
        return samples

    def iter_samples(self, splits: Iterable[str] | None = None) -> Iterable[YoloSample]:
        split_names = tuple(splits) if splits is not None else tuple(self.split_dirs)
        for split in split_names:
            yield from self.get_samples(split)

    def load_image(self, sample: YoloSample) -> np.ndarray:
        with Image.open(sample.image_path) as image:
            return np.asarray(image.convert("RGB"))

    def load_annotations(self, sample: YoloSample) -> list[YoloAnnotation]:
        if not sample.label_path.exists():
            return []

        annotations: list[YoloAnnotation] = []
        with sample.label_path.open("r", encoding="utf-8") as handle:
            for object_index, raw_line in enumerate(handle):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 5:
                    raise ValueError(f"Invalid YOLO annotation in {sample.label_path}: {stripped}")

                class_id = int(float(parts[0]))
                bbox = YoloBBox(
                    class_id=class_id,
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                )
                annotations.append(
                    YoloAnnotation(
                        object_index=object_index,
                        bbox=bbox,
                        class_name=self.class_names.get(class_id, str(class_id)),
                    )
                )
        return annotations

    def save_shadow_polys(
        self,
        sample: YoloSample,
        polygons_or_cache: Sequence[ShadowPolygon] | ShadowPolyCache,
        *,
        meta: ShadowPolyCacheMeta | None = None,
        issues: Sequence[ShadowPolyIssue] | None = None,
    ) -> None:
        sample.shadow_poly_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(polygons_or_cache, ShadowPolyCache):
            cache = polygons_or_cache
        else:
            if meta is None:
                raise ValueError("`meta` is required when saving raw polygons.")
            cache = ShadowPolyCache(meta=meta, polygons=tuple(polygons_or_cache), issues=tuple(issues or ()))

        with sample.shadow_poly_path.open("w", encoding="utf-8") as handle:
            json.dump(cache.to_dict(), handle, indent=2)
        self._shadow_poly_cache_cache[sample.shadow_poly_path] = cache

    def load_shadow_poly_cache(self, sample: YoloSample) -> ShadowPolyCache | None:
        if sample.shadow_poly_path in self._shadow_poly_cache_cache:
            return self._shadow_poly_cache_cache[sample.shadow_poly_path]
        if not sample.shadow_poly_path.exists():
            self._shadow_poly_cache_cache[sample.shadow_poly_path] = None
            return None
        with sample.shadow_poly_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid shadow poly cache payload in {sample.shadow_poly_path}")
        cache = ShadowPolyCache.from_dict(payload)
        self._shadow_poly_cache_cache[sample.shadow_poly_path] = cache
        return cache

    def validate_shadow_poly_cache(
        self,
        sample: YoloSample,
        expected_selected_class_ids: Sequence[int] | None = None,
        *,
        selection_match_mode: str = "exact",
    ) -> CacheValidationResult:
        if selection_match_mode not in {"exact", "subset"}:
            raise ValueError("selection_match_mode must be `exact` or `subset`.")

        cache = self.load_shadow_poly_cache(sample)
        if cache is None:
            return CacheValidationResult(
                cache=None,
                issues=(
                    ShadowPolyIssue(
                        code="missing_shadow_poly_cache",
                        message=f"Missing shadow polygon cache: {sample.shadow_poly_path}",
                        severity="error",
                        sample_path=sample.relative_image_path(self.root),
                    ),
                ),
                exists=False,
            )

        issues = list(cache.issues)
        meta = cache.meta
        sample_path = sample.relative_image_path(self.root)

        if meta.schema_version != CACHE_SCHEMA_VERSION:
            issues.append(
                ShadowPolyIssue(
                    code="cache_schema_mismatch",
                    message=(
                        f"Cache schema version {meta.schema_version} does not match "
                        f"required version {CACHE_SCHEMA_VERSION}."
                    ),
                    severity="error",
                    sample_path=sample_path,
                )
            )

        if meta.image_path and meta.image_path != sample.relative_image_path(self.root):
            issues.append(
                ShadowPolyIssue(
                    code="cache_image_path_mismatch",
                    message="Cache image path does not match the sample image path.",
                    severity="error",
                    sample_path=sample_path,
                )
            )
        if meta.label_path and meta.label_path != sample.relative_label_path(self.root):
            issues.append(
                ShadowPolyIssue(
                    code="cache_label_path_mismatch",
                    message="Cache label path does not match the sample label path.",
                    severity="error",
                    sample_path=sample_path,
                )
            )
        if meta.split and meta.split != sample.split:
            issues.append(
                ShadowPolyIssue(
                    code="cache_split_mismatch",
                    message="Cache split does not match the sample split.",
                    severity="error",
                    sample_path=sample_path,
                )
            )

        if expected_selected_class_ids is not None:
            expected_class_ids = tuple(sorted(set(int(item) for item in expected_selected_class_ids)))
            actual_class_ids = tuple(sorted(set(meta.selected_class_ids)))
            matches = (
                actual_class_ids == expected_class_ids
                if selection_match_mode == "exact"
                else set(expected_class_ids).issubset(actual_class_ids)
            )
            if not matches:
                issues.append(
                    ShadowPolyIssue(
                        code="selected_class_mismatch",
                        message=(
                            f"Cache selected classes {actual_class_ids} do not satisfy "
                            f"expected classes {expected_class_ids}."
                        ),
                        severity="error",
                        sample_path=sample_path,
                    )
                )

        if meta.image_sha256 is None:
            issues.append(
                ShadowPolyIssue(
                    code="missing_image_hash",
                    message="Cache metadata does not include an image fingerprint.",
                    severity="error",
                    sample_path=sample_path,
                )
            )
        else:
            current_image_hash = _sha256_file(sample.image_path)
            if current_image_hash != meta.image_sha256:
                issues.append(
                    ShadowPolyIssue(
                        code="stale_image_hash",
                        message="Cached image fingerprint does not match the current image contents.",
                        severity="error",
                        sample_path=sample_path,
                    )
                )

        current_label_hash = _sha256_file(sample.label_path) if sample.label_path.exists() else None
        if meta.label_sha256 != current_label_hash:
            issues.append(
                ShadowPolyIssue(
                    code="stale_label_hash",
                    message="Cached label fingerprint does not match the current label contents.",
                    severity="error",
                    sample_path=sample_path,
                )
            )

        if meta.image_width <= 0 or meta.image_height <= 0:
            issues.append(
                ShadowPolyIssue(
                    code="missing_image_dimensions",
                    message="Cache metadata does not include valid image dimensions.",
                    severity="error",
                    sample_path=sample_path,
                )
            )
        else:
            image_width, image_height = _image_size(sample.image_path)
            if (image_width, image_height) != (meta.image_width, meta.image_height):
                issues.append(
                    ShadowPolyIssue(
                        code="image_dimension_mismatch",
                        message="Cache image dimensions do not match the current image size.",
                        severity="error",
                        sample_path=sample_path,
                    )
                )

        if not cache.polygons:
            issues.append(
                ShadowPolyIssue(
                    code="empty_shadow_poly_cache",
                    message="Shadow polygon cache does not contain any polygons.",
                    severity="warning",
                    sample_path=sample_path,
                )
            )

        for polygon in cache.polygons:
            if polygon.class_id not in meta.selected_class_ids:
                issues.append(
                    ShadowPolyIssue(
                        code="polygon_class_not_in_cache_selection",
                        message=(
                            f"Polygon class id {polygon.class_id} is not present in the cache "
                            f"selected classes {meta.selected_class_ids}."
                        ),
                        severity="error",
                        sample_path=sample_path,
                        object_index=polygon.object_index,
                    )
                )
            if len(polygon.polygon) < 3:
                issues.append(
                    ShadowPolyIssue(
                        code="degenerate_polygon",
                        message="Polygon contains fewer than three vertices.",
                        severity="error",
                        sample_path=sample_path,
                        object_index=polygon.object_index,
                    )
                )
            if any(point[0] < 0.0 or point[0] > 1.0 or point[1] < 0.0 or point[1] > 1.0 for point in polygon.polygon):
                issues.append(
                    ShadowPolyIssue(
                        code="polygon_out_of_bounds",
                        message="Polygon contains normalized coordinates outside the [0, 1] range.",
                        severity="error",
                        sample_path=sample_path,
                        object_index=polygon.object_index,
                    )
                )

        return CacheValidationResult(cache=cache, issues=tuple(issues), exists=True)

    def load_shadow_polys(self, sample: YoloSample) -> list[ShadowPolygon]:
        cache = self.load_shadow_poly_cache(sample)
        if cache is None:
            return []
        return list(cache.polygons)
