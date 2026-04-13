from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from shadow_augmentor.constants import CACHE_SCHEMA_VERSION

Point = tuple[float, float]


def _polygon_area(polygon: Iterable[Point]) -> float:
    points = list(polygon)
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += (x0 * y1) - (x1 * y0)
    return abs(area) * 0.5


@dataclass(frozen=True)
class ShadowPolyIssue:
    code: str
    message: str
    severity: str = "error"
    sample_path: str | None = None
    object_index: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning"}:
            raise ValueError("severity must be `error` or `warning`.")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.sample_path is not None:
            payload["sample_path"] = self.sample_path
        if self.object_index is not None:
            payload["object_index"] = self.object_index
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ShadowPolyIssue":
        return cls(
            code=str(payload["code"]),
            message=str(payload["message"]),
            severity=str(payload.get("severity", "error")),
            sample_path=str(payload["sample_path"]) if "sample_path" in payload else None,
            object_index=int(payload["object_index"]) if "object_index" in payload else None,
        )


@dataclass(frozen=True)
class YoloBBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        half_width = self.width * image_width * 0.5
        half_height = self.height * image_height * 0.5
        x_center = self.x_center * image_width
        y_center = self.y_center * image_height
        return (
            x_center - half_width,
            y_center - half_height,
            x_center + half_width,
            y_center + half_height,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class YoloAnnotation:
    object_index: int
    bbox: YoloBBox
    class_name: str

    @property
    def class_id(self) -> int:
        return self.bbox.class_id


@dataclass(frozen=True)
class ShadowPolygon:
    object_index: int
    class_id: int
    class_name: str
    bbox: YoloBBox
    polygon: tuple[Point, ...]
    source: str = "sam"
    warnings: tuple[str, ...] = ()
    vertex_count: int = 0
    area_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in {"sam", "bbox_fallback"}:
            raise ValueError("source must be `sam` or `bbox_fallback`.")
        if self.vertex_count == 0:
            object.__setattr__(self, "vertex_count", len(self.polygon))
        if self.area_ratio == 0.0 and self.polygon and self.bbox.area > 0.0:
            ratio = _polygon_area(self.polygon) / self.bbox.area
            object.__setattr__(self, "area_ratio", float(ratio))

    def to_dict(self) -> dict[str, object]:
        return {
            "object_index": self.object_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "polygon": [[x, y] for x, y in self.polygon],
            "source": self.source,
            "warnings": list(self.warnings),
            "vertex_count": self.vertex_count,
            "area_ratio": self.area_ratio,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ShadowPolygon":
        bbox_payload = payload["bbox"]
        if not isinstance(bbox_payload, Mapping):
            raise TypeError("Expected `bbox` to be a mapping.")
        bbox = YoloBBox(
            class_id=int(bbox_payload["class_id"]),
            x_center=float(bbox_payload["x_center"]),
            y_center=float(bbox_payload["y_center"]),
            width=float(bbox_payload["width"]),
            height=float(bbox_payload["height"]),
        )
        polygon = tuple((float(x), float(y)) for x, y in payload["polygon"])
        source = str(payload.get("source", "sam"))
        warnings = tuple(str(item) for item in payload.get("warnings", []))
        vertex_count = int(payload.get("vertex_count", len(polygon)))
        area_ratio = float(payload.get("area_ratio", 0.0))
        return cls(
            object_index=int(payload["object_index"]),
            class_id=int(payload["class_id"]),
            class_name=str(payload["class_name"]),
            bbox=bbox,
            polygon=polygon,
            source=source,
            warnings=warnings,
            vertex_count=vertex_count,
            area_ratio=area_ratio,
        )


@dataclass(frozen=True)
class ShadowPolyCacheMeta:
    schema_version: int
    generated_at: str
    builder_version: str
    segmenter_name: str
    selected_class_ids: tuple[int, ...]
    image_path: str
    label_path: str
    split: str
    image_width: int
    image_height: int
    image_sha256: str | None = None
    label_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "builder_version": self.builder_version,
            "segmenter_name": self.segmenter_name,
            "selected_class_ids": list(self.selected_class_ids),
            "image_path": self.image_path,
            "label_path": self.label_path,
            "split": self.split,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_sha256": self.image_sha256,
            "label_sha256": self.label_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ShadowPolyCacheMeta":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            generated_at=str(payload.get("generated_at", "")),
            builder_version=str(payload.get("builder_version", "")),
            segmenter_name=str(payload.get("segmenter_name", "unknown")),
            selected_class_ids=tuple(int(item) for item in payload.get("selected_class_ids", [])),
            image_path=str(payload.get("image_path", "")),
            label_path=str(payload.get("label_path", "")),
            split=str(payload.get("split", "")),
            image_width=int(payload.get("image_width", 0)),
            image_height=int(payload.get("image_height", 0)),
            image_sha256=str(payload["image_sha256"]) if payload.get("image_sha256") is not None else None,
            label_sha256=str(payload["label_sha256"]) if payload.get("label_sha256") is not None else None,
        )


@dataclass(frozen=True)
class ShadowPolyCache:
    meta: ShadowPolyCacheMeta
    polygons: tuple[ShadowPolygon, ...]
    issues: tuple[ShadowPolyIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "meta": self.meta.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "polygons": [polygon.to_dict() for polygon in self.polygons],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ShadowPolyCache":
        if "meta" in payload:
            meta_payload = payload["meta"]
            if not isinstance(meta_payload, Mapping):
                raise TypeError("Expected `meta` to be a mapping.")
            meta = ShadowPolyCacheMeta.from_dict(meta_payload)
            issues = tuple(
                ShadowPolyIssue.from_dict(item)
                for item in payload.get("issues", [])
                if isinstance(item, Mapping)
            )
            polygons = tuple(
                ShadowPolygon.from_dict(item)
                for item in payload.get("polygons", [])
                if isinstance(item, Mapping)
            )
            return cls(meta=meta, polygons=polygons, issues=issues)

        polygons = tuple(
            ShadowPolygon.from_dict(item)
            for item in payload.get("polygons", [])
            if isinstance(item, Mapping)
        )
        inferred_class_ids = tuple(sorted({polygon.class_id for polygon in polygons}))
        meta = ShadowPolyCacheMeta(
            schema_version=1,
            generated_at="",
            builder_version="",
            segmenter_name="unknown",
            selected_class_ids=inferred_class_ids,
            image_path=str(payload.get("image_path", "")),
            label_path="",
            split=str(payload.get("split", "")),
            image_width=0,
            image_height=0,
            image_sha256=None,
            label_sha256=None,
        )
        issues = (
            ShadowPolyIssue(
                code="legacy_cache_schema",
                message="Cache file predates the versioned cache schema.",
                severity="error",
                sample_path=meta.image_path or None,
            ),
        )
        return cls(meta=meta, polygons=polygons, issues=issues)


@dataclass(frozen=True)
class CacheValidationResult:
    cache: ShadowPolyCache | None
    issues: tuple[ShadowPolyIssue, ...]
    exists: bool

    @property
    def valid(self) -> bool:
        return self.exists and not any(issue.severity == "error" for issue in self.issues)

    @property
    def stale(self) -> bool:
        stale_codes = {"stale_image_hash", "stale_label_hash", "cache_schema_mismatch"}
        return any(issue.code in stale_codes for issue in self.issues)

    @property
    def rebuild_required(self) -> bool:
        return not self.valid


@dataclass(frozen=True)
class DatasetSplitValidationReport:
    split: str
    image_dir: Path
    label_dir: Path
    image_count: int
    label_count: int
    missing_label_paths: tuple[str, ...]
    orphan_label_paths: tuple[str, ...]
    issues: tuple[ShadowPolyIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class DatasetValidationReport:
    root: Path
    config_path: Path
    split_reports: tuple[DatasetSplitValidationReport, ...]
    issues: tuple[ShadowPolyIssue, ...] = ()

    @property
    def valid(self) -> bool:
        all_issues = self.issues + tuple(
            issue
            for split_report in self.split_reports
            for issue in split_report.issues
        )
        return not any(issue.severity == "error" for issue in all_issues)


@dataclass(frozen=True)
class ClassSelectionSummary:
    resolved_class_ids: tuple[int, ...]
    resolved_class_names: tuple[str, ...]
    images_seen: int
    images_with_selected_classes: int
    object_count_by_class: dict[int, int] = field(default_factory=dict)

    @property
    def selected_object_count(self) -> int:
        return int(sum(self.object_count_by_class.values()))


@dataclass(frozen=True)
class YoloSample:
    split: str
    image_path: Path
    label_path: Path
    shadow_poly_path: Path

    def relative_image_path(self, root: Path) -> str:
        if self.image_path.is_relative_to(root):
            return str(self.image_path.relative_to(root))
        return str(self.image_path)

    def relative_label_path(self, root: Path) -> str:
        if self.label_path.is_relative_to(root):
            return str(self.label_path.relative_to(root))
        return str(self.label_path)


def build_cache_meta(
    *,
    generated_at: str,
    builder_version: str,
    segmenter_name: str,
    selected_class_ids: Iterable[int],
    sample: YoloSample,
    root: Path,
    image_width: int,
    image_height: int,
    image_sha256: str | None,
    label_sha256: str | None,
) -> ShadowPolyCacheMeta:
    return ShadowPolyCacheMeta(
        schema_version=CACHE_SCHEMA_VERSION,
        generated_at=generated_at,
        builder_version=builder_version,
        segmenter_name=segmenter_name,
        selected_class_ids=tuple(sorted(int(item) for item in selected_class_ids)),
        image_path=sample.relative_image_path(root),
        label_path=sample.relative_label_path(root),
        split=sample.split,
        image_width=image_width,
        image_height=image_height,
        image_sha256=image_sha256,
        label_sha256=label_sha256,
    )
