from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from shadow_augmentor.geometry import (
    blur_mask,
    build_shadow_shape,
    choose_edge,
    polygon_bbox,
    polygon_to_pixels,
    rasterize_polygon,
    unit_vector_from_degrees,
)
from shadow_augmentor.models import ShadowPolygon, YoloAnnotation


def _validate_float_range(name: str, value_range: tuple[float, float], *, minimum: float | None = None, maximum: float | None = None) -> None:
    low, high = value_range
    if low > high:
        raise ValueError(f"{name} range must be ordered low <= high.")
    if minimum is not None and low < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}.")
    if maximum is not None and high > maximum:
        raise ValueError(f"{name} upper bound must be at most {maximum}.")


def _validate_int_range(name: str, value_range: tuple[int, int], *, minimum: int = 0) -> None:
    low, high = value_range
    if low > high:
        raise ValueError(f"{name} range must be ordered low <= high.")
    if low < minimum:
        raise ValueError(f"{name} lower bound must be at least {minimum}.")


@dataclass
class ShadowAugmentConfig:
    selected_classes: Sequence[int | str] | None = None
    probability: float = 0.5
    side_mode: str = "down"
    shadow_count: tuple[int, int] = (1, 1)
    scale: tuple[float, float] = (1.0, 1.0)
    darkness: tuple[float, float] = (0.35, 0.6)
    direction_degrees: tuple[float, float] | None = None
    blur_ratio: float = 0.02
    blend_mode: str = "multiply"
    attachment_span_ratio: tuple[float, float] = (0.45, 0.95)
    tip_width_ratio: tuple[float, float] = (0.15, 0.7)
    roundness_factor: tuple[float, float] = (0.18, 0.5)
    flare_factor: tuple[float, float] = (0.05, 0.38)
    skew_factor: tuple[float, float] = (0.0, 0.24)
    bend_factor: tuple[float, float] = (0.08, 0.3)
    jitter_factor: tuple[float, float] = (0.0, 0.12)
    density_decay: tuple[float, float] = (0.0, 0.55)
    density_noise: tuple[float, float] = (0.0, 0.32)
    max_shadow_coverage_ratio: float = 0.25
    max_overlap_with_other_objects_ratio: float = 0.05
    max_shadow_attempts: int = 8

    def __post_init__(self) -> None:
        if self.side_mode not in {"random", "down"}:
            raise ValueError("side_mode must be `random` or `down`.")
        if self.blend_mode not in {"multiply", "darken"}:
            raise ValueError("blend_mode must be `multiply` or `darken`.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        _validate_int_range("shadow_count", self.shadow_count, minimum=0)
        _validate_float_range("scale", self.scale, minimum=0.0)
        _validate_float_range("darkness", self.darkness, minimum=0.0, maximum=1.0)
        _validate_float_range("attachment_span_ratio", self.attachment_span_ratio, minimum=0.05, maximum=1.0)
        _validate_float_range("tip_width_ratio", self.tip_width_ratio, minimum=0.05, maximum=1.5)
        _validate_float_range("roundness_factor", self.roundness_factor, minimum=0.0, maximum=1.0)
        _validate_float_range("flare_factor", self.flare_factor, minimum=0.0, maximum=1.5)
        _validate_float_range("skew_factor", self.skew_factor, minimum=0.0, maximum=1.0)
        _validate_float_range("bend_factor", self.bend_factor, minimum=0.0)
        _validate_float_range("jitter_factor", self.jitter_factor, minimum=0.0)
        _validate_float_range("density_decay", self.density_decay, minimum=0.0, maximum=1.0)
        _validate_float_range("density_noise", self.density_noise, minimum=0.0, maximum=1.0)
        if self.direction_degrees is not None:
            _validate_float_range("direction_degrees", self.direction_degrees)
        if not 0.0 <= self.blur_ratio <= 1.0:
            raise ValueError("blur_ratio must be between 0 and 1.")
        if not 0.0 < self.max_shadow_coverage_ratio <= 1.0:
            raise ValueError("max_shadow_coverage_ratio must be greater than 0 and at most 1.")
        if not 0.0 <= self.max_overlap_with_other_objects_ratio <= 1.0:
            raise ValueError("max_overlap_with_other_objects_ratio must be between 0 and 1.")
        if self.max_shadow_attempts < 1:
            raise ValueError("max_shadow_attempts must be at least 1.")


class ShadowAugmentor:
    def __init__(self, config: ShadowAugmentConfig) -> None:
        self.config = config
        self._selected_class_ids: set[int] | None = None

    def set_selected_class_ids(self, class_ids: Sequence[int] | None) -> None:
        self._selected_class_ids = None if class_ids is None else set(class_ids)

    def __call__(
        self,
        image: np.ndarray,
        shadow_polys: Sequence[ShadowPolygon],
        rng: np.random.Generator | None = None,
        *,
        annotations: Sequence[YoloAnnotation] | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.config.probability <= 0.0 or not shadow_polys:
            return image
        if float(rng.random()) > self.config.probability:
            return image

        selected_class_ids = self._resolve_selected_class_ids()
        eligible_polygons = [
            polygon
            for polygon in shadow_polys
            if selected_class_ids is None or polygon.class_id in selected_class_ids
        ]
        if not eligible_polygons:
            return image

        image_height, image_width = image.shape[:2]
        shadow_layers = np.zeros((image_height, image_width), dtype=np.float32)
        accepted_shadow_coverage = np.zeros((image_height, image_width), dtype=np.float32)
        shadow_count = self._sample_int(rng, self.config.shadow_count)
        target_polygons = self._sample_target_polygons(rng, eligible_polygons, shadow_count)

        for polygon in target_polygons:
            accepted_mask: np.ndarray | None = None
            accepted_polygon: ShadowPolygon | None = None
            accepted_edge_start: np.ndarray | None = None
            accepted_edge_end: np.ndarray | None = None
            accepted_direction: np.ndarray | None = None
            accepted_length: float | None = None
            for _attempt in range(self.config.max_shadow_attempts):
                polygon_pixels = polygon_to_pixels(polygon.polygon, image_width, image_height)
                if len(polygon_pixels) < 3:
                    continue

                edge_start, edge_end, outward_normal = choose_edge(
                    polygon_pixels,
                    mode=self.config.side_mode,
                    rng=rng,
                )
                direction = self._choose_direction(rng, outward_normal)
                shadow_length = self._sample_shadow_length(polygon_pixels, rng)
                shadow_shape = build_shadow_shape(
                    edge_start=edge_start,
                    edge_end=edge_end,
                    direction=direction,
                    scale_pixels=shadow_length,
                    attachment_span_ratio=self._sample_float(rng, self.config.attachment_span_ratio),
                    tip_width_ratio=self._sample_float(rng, self.config.tip_width_ratio),
                    roundness_factor=self._sample_float(rng, self.config.roundness_factor),
                    flare_factor=self._sample_float(rng, self.config.flare_factor),
                    skew_factor=self._sample_signed_float(rng, self.config.skew_factor),
                    bend_factor=self._sample_float(rng, self.config.bend_factor),
                    jitter_factor=self._sample_signed_float(rng, self.config.jitter_factor),
                )
                mask = rasterize_polygon(shadow_shape, image.shape[:2])
                if not np.any(mask):
                    continue
                if not self._coverage_is_allowed(mask, accepted_shadow_coverage):
                    continue
                if not self._overlap_is_allowed(mask, polygon, annotations, image_width, image_height):
                    continue

                accepted_mask = mask
                accepted_polygon = polygon
                accepted_edge_start = edge_start
                accepted_edge_end = edge_end
                accepted_direction = direction
                accepted_length = shadow_length
                break

            if (
                accepted_mask is None
                or accepted_polygon is None
                or accepted_edge_start is None
                or accepted_edge_end is None
                or accepted_direction is None
                or accepted_length is None
            ):
                continue

            shadow_strength = self._sample_float(rng, self.config.darkness)
            shadow_alpha = self._build_shadow_alpha(
                accepted_mask,
                edge_start=accepted_edge_start,
                edge_end=accepted_edge_end,
                direction=accepted_direction,
                shadow_length=accepted_length,
                rng=rng,
            )
            shadow_layers = np.clip(shadow_layers + (shadow_alpha * shadow_strength), 0.0, 1.0)
            accepted_shadow_coverage = np.maximum(accepted_shadow_coverage, accepted_mask)

        if not np.any(shadow_layers):
            return image

        working = image.astype(np.float32)
        alpha = shadow_layers[..., None]
        if self.config.blend_mode == "multiply":
            working = working * (1.0 - alpha)
        else:
            working = np.clip(working - (255.0 * alpha), 0.0, 255.0)
        return working.astype(np.uint8)

    def _resolve_selected_class_ids(self) -> set[int] | None:
        if self._selected_class_ids is not None:
            return self._selected_class_ids
        if self.config.selected_classes is None:
            return None
        if any(not isinstance(class_id, int) for class_id in self.config.selected_classes):
            raise ValueError(
                "selected_classes contains names. Resolve them against a YoloDataset first or "
                "use ShadowAugmentedYoloDataset."
            )
        return {int(class_id) for class_id in self.config.selected_classes}

    def _choose_direction(self, rng: np.random.Generator, outward_normal: np.ndarray) -> np.ndarray:
        if self.config.direction_degrees is None:
            return outward_normal
        sampled_angle = self._sample_float(rng, self.config.direction_degrees)
        direction = unit_vector_from_degrees(sampled_angle)
        if float(np.dot(direction, outward_normal)) < 0.0:
            return outward_normal
        return direction

    def _sample_shadow_length(self, polygon_pixels: np.ndarray, rng: np.random.Generator) -> float:
        x0, y0, x1, y1 = polygon_bbox(polygon_pixels)
        base_size = max(x1 - x0, y1 - y0, 1.0)
        return base_size * self._sample_float(rng, self.config.scale)

    def _sample_target_polygons(
        self,
        rng: np.random.Generator,
        eligible_polygons: Sequence[ShadowPolygon],
        shadow_count: int,
    ) -> list[ShadowPolygon]:
        if shadow_count <= 0 or not eligible_polygons:
            return []
        if shadow_count >= len(eligible_polygons):
            return list(eligible_polygons)

        sampled_indices = rng.choice(len(eligible_polygons), size=shadow_count, replace=False)
        return [eligible_polygons[int(index)] for index in np.asarray(sampled_indices).tolist()]

    def _build_shadow_alpha(
        self,
        mask: np.ndarray,
        *,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        direction: np.ndarray,
        shadow_length: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        blurred_mask = blur_mask(mask, self.config.blur_ratio)
        if not np.any(mask):
            return blurred_mask

        mask_uint8 = (mask > 0).astype(np.uint8)
        points = np.argwhere(mask_uint8 > 0)
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        crop_mask = mask_uint8[y0:y1, x0:x1]
        density_crop = self._build_density_crop(
            crop_mask,
            edge_start=edge_start,
            edge_end=edge_end,
            direction=direction,
            shadow_length=shadow_length,
            rng=rng,
            offset=(x0, y0),
        )

        density = np.zeros_like(mask, dtype=np.float32)
        density[y0:y1, x0:x1] = density_crop
        return np.clip(blurred_mask * density, 0.0, 1.0)

    def _build_density_crop(
        self,
        crop_mask: np.ndarray,
        *,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        direction: np.ndarray,
        shadow_length: float,
        rng: np.random.Generator,
        offset: tuple[int, int],
    ) -> np.ndarray:
        crop_height, crop_width = crop_mask.shape
        direction_unit = direction / max(float(np.linalg.norm(direction)), 1e-6)
        center = (edge_start + edge_end) * 0.5

        grid_y, grid_x = np.mgrid[0:crop_height, 0:crop_width].astype(np.float32)
        grid_x += float(offset[0])
        grid_y += float(offset[1])
        rel_x = grid_x - float(center[0])
        rel_y = grid_y - float(center[1])
        longitudinal = (rel_x * float(direction_unit[0])) + (rel_y * float(direction_unit[1]))
        longitudinal = np.clip(longitudinal / max(float(shadow_length), 1.0), 0.0, 1.0)

        decay_strength = self._sample_float(rng, self.config.density_decay)
        decay_power = float(rng.uniform(0.55, 1.85))
        density = 1.0 - (decay_strength * np.power(longitudinal, decay_power))

        distance = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
        max_distance = float(np.max(distance))
        if max_distance > 0.0:
            distance /= max_distance
            center_bias_power = float(rng.uniform(0.65, 1.45))
            density *= 0.65 + (0.35 * np.power(distance, center_bias_power))

        noise_strength = self._sample_float(rng, self.config.density_noise)
        if noise_strength > 0.0:
            density *= self._sample_noise_map(rng, crop_mask.shape, noise_strength)

        density *= crop_mask.astype(np.float32)
        return np.clip(density, 0.0, 1.0)

    def _coverage_is_allowed(self, mask: np.ndarray, accepted_shadow_coverage: np.ndarray) -> bool:
        coverage_ratio = self._coverage_ratio(mask, accepted_shadow_coverage)
        return coverage_ratio <= self.config.max_shadow_coverage_ratio

    def _overlap_is_allowed(
        self,
        mask: np.ndarray,
        polygon: ShadowPolygon,
        annotations: Sequence[YoloAnnotation] | None,
        image_width: int,
        image_height: int,
    ) -> bool:
        overlap_ratio = self._overlap_ratio(mask, polygon, annotations, image_width, image_height)
        return overlap_ratio <= self.config.max_overlap_with_other_objects_ratio

    def _coverage_ratio(self, mask: np.ndarray, accepted_shadow_coverage: np.ndarray) -> float:
        proposed_coverage = np.maximum(accepted_shadow_coverage, mask)
        return float(np.count_nonzero(proposed_coverage > 0.0) / proposed_coverage.size)

    def _overlap_ratio(
        self,
        mask: np.ndarray,
        polygon: ShadowPolygon,
        annotations: Sequence[YoloAnnotation] | None,
        image_width: int,
        image_height: int,
    ) -> float:
        if not annotations:
            return 0.0

        obstruction_mask = np.zeros_like(mask)
        for annotation in annotations:
            if annotation.object_index == polygon.object_index:
                continue
            x0, y0, x1, y1 = annotation.bbox.to_xyxy(image_width, image_height)
            x0_i = max(int(np.floor(x0)), 0)
            y0_i = max(int(np.floor(y0)), 0)
            x1_i = min(int(np.ceil(x1)), image_width)
            y1_i = min(int(np.ceil(y1)), image_height)
            if x1_i <= x0_i or y1_i <= y0_i:
                continue
            obstruction_mask[y0_i:y1_i, x0_i:x1_i] = 1.0

        shadow_area = float(np.count_nonzero(mask > 0.0))
        if shadow_area <= 0.0:
            return 1.0
        overlap_area = float(np.count_nonzero((mask > 0.0) & (obstruction_mask > 0.0)))
        return overlap_area / shadow_area

    @staticmethod
    def _sample_float(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return float(low)
        return float(rng.uniform(low, high))

    @staticmethod
    def _sample_int(rng: np.random.Generator, value_range: tuple[int, int]) -> int:
        low, high = value_range
        if low == high:
            return int(low)
        return int(rng.integers(low, high + 1))

    @staticmethod
    def _sample_signed_float(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
        magnitude = ShadowAugmentor._sample_float(rng, value_range)
        if magnitude == 0.0:
            return 0.0
        return float(magnitude if float(rng.random()) >= 0.5 else -magnitude)

    @staticmethod
    def _sample_noise_map(rng: np.random.Generator, image_shape: tuple[int, int], strength: float) -> np.ndarray:
        height, width = image_shape
        coarse_height = max(2, min(24, int(round(height / 12.0))))
        coarse_width = max(2, min(24, int(round(width / 12.0))))
        coarse = rng.uniform(1.0 - strength, 1.0, size=(coarse_height, coarse_width)).astype(np.float32)
        upsampled = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        kernel_size = max(int(round(max(height, width) * 0.08)), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        softened = cv2.GaussianBlur(upsampled, (kernel_size, kernel_size), 0)
        return np.clip(softened, 0.0, 1.0)
