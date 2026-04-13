from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np

from shadow_augmentor.geometry import (
    build_shadow_shape,
    choose_edge,
    polygon_to_pixels,
    rasterize_polygon,
)
from shadow_augmentor.models import ShadowPolygon, YoloAnnotation, YoloSample

if TYPE_CHECKING:
    from shadow_augmentor.augment import ShadowAugmentor


@dataclass(frozen=True)
class ShadowDebugAttempt:
    object_index: int
    class_id: int
    class_name: str
    accepted: bool
    reason: str
    edge_start: tuple[float, float] | None = None
    edge_end: tuple[float, float] | None = None
    direction: tuple[float, float] | None = None
    shadow_length: float | None = None
    darkness: float | None = None
    coverage_ratio: float | None = None
    overlap_ratio: float | None = None
    shadow_shape: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_index": self.object_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "accepted": self.accepted,
            "reason": self.reason,
            "edge_start": list(self.edge_start) if self.edge_start is not None else None,
            "edge_end": list(self.edge_end) if self.edge_end is not None else None,
            "direction": list(self.direction) if self.direction is not None else None,
            "shadow_length": self.shadow_length,
            "darkness": self.darkness,
            "coverage_ratio": self.coverage_ratio,
            "overlap_ratio": self.overlap_ratio,
            "shadow_shape": [list(point) for point in self.shadow_shape],
        }


@dataclass(frozen=True)
class ShadowDebugResult:
    augmented_image: np.ndarray
    shadow_mask: np.ndarray
    attempts: tuple[ShadowDebugAttempt, ...]

    def summary(self) -> dict[str, Any]:
        accepted_count = sum(1 for attempt in self.attempts if attempt.accepted)
        return {
            "accepted_count": accepted_count,
            "attempt_count": len(self.attempts),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def simulate_shadow_debug(
    augmenter: "ShadowAugmentor",
    image: np.ndarray,
    shadow_polys: Sequence[ShadowPolygon],
    *,
    rng: np.random.Generator | None = None,
    annotations: Sequence[YoloAnnotation] | None = None,
) -> ShadowDebugResult:
    rng = rng or np.random.default_rng()
    attempts: list[ShadowDebugAttempt] = []

    if augmenter.config.probability <= 0.0 or not shadow_polys:
        return ShadowDebugResult(
            augmented_image=image.copy(),
            shadow_mask=np.zeros(image.shape[:2], dtype=np.float32),
            attempts=tuple(attempts),
        )
    if float(rng.random()) > augmenter.config.probability:
        attempts.append(
            ShadowDebugAttempt(
                object_index=-1,
                class_id=-1,
                class_name="",
                accepted=False,
                reason="probability_skip",
            )
        )
        return ShadowDebugResult(
            augmented_image=image.copy(),
            shadow_mask=np.zeros(image.shape[:2], dtype=np.float32),
            attempts=tuple(attempts),
        )

    selected_class_ids = augmenter._resolve_selected_class_ids()
    eligible_polygons = [
        polygon
        for polygon in shadow_polys
        if selected_class_ids is None or polygon.class_id in selected_class_ids
    ]
    if not eligible_polygons:
        attempts.append(
            ShadowDebugAttempt(
                object_index=-1,
                class_id=-1,
                class_name="",
                accepted=False,
                reason="no_eligible_polygons",
            )
        )
        return ShadowDebugResult(
            augmented_image=image.copy(),
            shadow_mask=np.zeros(image.shape[:2], dtype=np.float32),
            attempts=tuple(attempts),
        )

    image_height, image_width = image.shape[:2]
    shadow_layers = np.zeros((image_height, image_width), dtype=np.float32)
    accepted_shadow_coverage = np.zeros((image_height, image_width), dtype=np.float32)
    shadow_count = augmenter._sample_int(rng, augmenter.config.shadow_count)
    target_polygons = augmenter._sample_target_polygons(rng, eligible_polygons, shadow_count)

    for polygon in target_polygons:
        accepted_mask: np.ndarray | None = None
        accepted_edge_start: np.ndarray | None = None
        accepted_edge_end: np.ndarray | None = None
        accepted_direction: np.ndarray | None = None
        accepted_length: float | None = None
        for _attempt in range(augmenter.config.max_shadow_attempts):
            polygon_pixels = polygon_to_pixels(polygon.polygon, image_width, image_height)
            if len(polygon_pixels) < 3:
                attempts.append(
                    ShadowDebugAttempt(
                        object_index=polygon.object_index,
                        class_id=polygon.class_id,
                        class_name=polygon.class_name,
                        accepted=False,
                        reason="degenerate_polygon",
                    )
                )
                continue

            edge_start, edge_end, outward_normal = choose_edge(
                polygon_pixels,
                mode=augmenter.config.side_mode,
                rng=rng,
            )
            direction = augmenter._choose_direction(rng, outward_normal)
            shadow_length = augmenter._sample_shadow_length(polygon_pixels, rng)
            shadow_shape_array = build_shadow_shape(
                edge_start=edge_start,
                edge_end=edge_end,
                direction=direction,
                scale_pixels=shadow_length,
                attachment_span_ratio=augmenter._sample_float(rng, augmenter.config.attachment_span_ratio),
                tip_width_ratio=augmenter._sample_float(rng, augmenter.config.tip_width_ratio),
                roundness_factor=augmenter._sample_float(rng, augmenter.config.roundness_factor),
                flare_factor=augmenter._sample_float(rng, augmenter.config.flare_factor),
                skew_factor=augmenter._sample_signed_float(rng, augmenter.config.skew_factor),
                bend_factor=augmenter._sample_float(rng, augmenter.config.bend_factor),
                jitter_factor=augmenter._sample_signed_float(rng, augmenter.config.jitter_factor),
            )
            mask = rasterize_polygon(shadow_shape_array, image.shape[:2])
            if not np.any(mask):
                attempts.append(
                    ShadowDebugAttempt(
                        object_index=polygon.object_index,
                        class_id=polygon.class_id,
                        class_name=polygon.class_name,
                        accepted=False,
                        reason="empty_shadow_mask",
                        edge_start=_to_point(edge_start),
                        edge_end=_to_point(edge_end),
                        direction=_to_point(direction),
                        shadow_length=float(shadow_length),
                        shadow_shape=tuple(_to_point(point) for point in shadow_shape_array),
                    )
                )
                continue

            coverage_ratio = augmenter._coverage_ratio(mask, accepted_shadow_coverage)
            overlap_ratio = augmenter._overlap_ratio(mask, polygon, annotations, image_width, image_height)
            attempt_payload = dict(
                object_index=polygon.object_index,
                class_id=polygon.class_id,
                class_name=polygon.class_name,
                edge_start=_to_point(edge_start),
                edge_end=_to_point(edge_end),
                direction=_to_point(direction),
                shadow_length=float(shadow_length),
                coverage_ratio=float(coverage_ratio),
                overlap_ratio=float(overlap_ratio),
                shadow_shape=tuple(_to_point(point) for point in shadow_shape_array),
            )

            if coverage_ratio > augmenter.config.max_shadow_coverage_ratio:
                attempts.append(
                    ShadowDebugAttempt(
                        accepted=False,
                        reason="coverage_limit",
                        **attempt_payload,
                    )
                )
                continue
            if overlap_ratio > augmenter.config.max_overlap_with_other_objects_ratio:
                attempts.append(
                    ShadowDebugAttempt(
                        accepted=False,
                        reason="overlap_limit",
                        **attempt_payload,
                    )
                )
                continue

            darkness = augmenter._sample_float(rng, augmenter.config.darkness)
            attempts.append(
                ShadowDebugAttempt(
                    accepted=True,
                    reason="accepted",
                    darkness=float(darkness),
                    **attempt_payload,
                )
            )
            accepted_mask = mask
            accepted_edge_start = edge_start
            accepted_edge_end = edge_end
            accepted_direction = direction
            accepted_length = shadow_length
            shadow_alpha = augmenter._build_shadow_alpha(
                mask,
                edge_start=edge_start,
                edge_end=edge_end,
                direction=direction,
                shadow_length=shadow_length,
                rng=rng,
            )
            shadow_layers = np.clip(shadow_layers + (shadow_alpha * darkness), 0.0, 1.0)
            accepted_shadow_coverage = np.maximum(accepted_shadow_coverage, mask)
            break

        if (
            accepted_mask is None
            or accepted_edge_start is None
            or accepted_edge_end is None
            or accepted_direction is None
            or accepted_length is None
        ):
            continue

    if not np.any(shadow_layers):
        return ShadowDebugResult(
            augmented_image=image.copy(),
            shadow_mask=shadow_layers,
            attempts=tuple(attempts),
        )

    augmented_image = _blend_shadow(image, shadow_layers, augmenter.config.blend_mode)
    return ShadowDebugResult(
        augmented_image=augmented_image,
        shadow_mask=shadow_layers,
        attempts=tuple(attempts),
    )


def render_shadow_debug_overlay(
    image: np.ndarray,
    shadow_polys: Sequence[ShadowPolygon],
    debug_result: ShadowDebugResult,
) -> np.ndarray:
    original_panel = image.copy()
    geometry_panel = image.copy()
    final_panel = debug_result.augmented_image.copy()
    image_height, image_width = image.shape[:2]

    for polygon in shadow_polys:
        polygon_pixels = polygon_to_pixels(polygon.polygon, image_width, image_height)
        if len(polygon_pixels) < 3:
            continue
        color = (0, 200, 255) if polygon.source == "sam" else (0, 120, 255)
        cv2.polylines(original_panel, [np.round(polygon_pixels).astype(np.int32)], True, color, 2)
        cv2.polylines(geometry_panel, [np.round(polygon_pixels).astype(np.int32)], True, color, 2)
        anchor = np.round(polygon_pixels[0]).astype(np.int32)
        label = f"{polygon.class_name}:{polygon.source}"
        cv2.putText(original_panel, label, tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    accepted_overlay = geometry_panel.copy()
    for attempt in debug_result.attempts:
        if not attempt.accepted or not attempt.shadow_shape:
            continue
        shadow_shape = np.round(np.asarray(attempt.shadow_shape, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(accepted_overlay, [shadow_shape], (35, 35, 35))
        cv2.polylines(geometry_panel, [shadow_shape], True, (20, 20, 220), 2)
        if attempt.edge_start is not None and attempt.edge_end is not None:
            edge_start = tuple(int(round(value)) for value in attempt.edge_start)
            edge_end = tuple(int(round(value)) for value in attempt.edge_end)
            cv2.line(geometry_panel, edge_start, edge_end, (40, 220, 40), 3)

    geometry_panel = cv2.addWeighted(accepted_overlay, 0.28, geometry_panel, 0.72, 0.0)
    mask_panel = np.clip(debug_result.shadow_mask * 255.0, 0.0, 255.0).astype(np.uint8)
    mask_panel = cv2.cvtColor(mask_panel, cv2.COLOR_GRAY2RGB)
    cv2.putText(original_panel, "Source + Polygons", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(geometry_panel, "Geometry + Shadow Shape", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(final_panel, "Augmented Output", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mask_panel, "Shadow Mask", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    top_row = np.concatenate([original_panel, geometry_panel], axis=1)
    bottom_row = np.concatenate([final_panel, mask_panel], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


def write_debug_overlay_bundle(
    *,
    image: np.ndarray,
    shadow_polys: Sequence[ShadowPolygon],
    debug_result: ShadowDebugResult,
    output_path: Path,
    sample: YoloSample | None = None,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_shadow_debug_overlay(image, shadow_polys, debug_result)
    image_path = output_path.with_suffix(".png")
    json_path = output_path.with_suffix(".json")
    cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    payload = debug_result.summary()
    if sample is not None:
        payload["sample"] = {
            "split": sample.split,
            "image_path": str(sample.image_path),
            "label_path": str(sample.label_path),
            "shadow_poly_path": str(sample.shadow_poly_path),
        }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return image_path, json_path


def _to_point(point: np.ndarray | Sequence[float]) -> tuple[float, float]:
    point_array = np.asarray(point, dtype=np.float32).reshape(-1)
    return float(point_array[0]), float(point_array[1])


def _blend_shadow(image: np.ndarray, shadow_layers: np.ndarray, blend_mode: str) -> np.ndarray:
    working = image.astype(np.float32)
    alpha = shadow_layers[..., None]
    if blend_mode == "multiply":
        working = working * (1.0 - alpha)
    else:
        working = np.clip(working - (255.0 * alpha), 0.0, 255.0)
    return working.astype(np.uint8)
