from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from shadow_augmentor.models import Point


def polygon_to_pixels(
    polygon: Iterable[Point],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = np.asarray(list(polygon), dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 2)
    points[:, 0] *= image_width
    points[:, 1] *= image_height
    return points


def polygon_to_normalized(
    polygon: Iterable[Point],
    image_width: int,
    image_height: int,
) -> tuple[Point, ...]:
    points = np.asarray(list(polygon), dtype=np.float32)
    if points.size == 0:
        return ()
    points[:, 0] /= image_width
    points[:, 1] /= image_height
    return tuple((float(x), float(y)) for x, y in points)


def bbox_to_polygon(xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    return np.asarray(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        dtype=np.float32,
    )


def polygon_bbox(polygon: np.ndarray) -> tuple[float, float, float, float]:
    x0 = float(np.min(polygon[:, 0]))
    y0 = float(np.min(polygon[:, 1]))
    x1 = float(np.max(polygon[:, 0]))
    y1 = float(np.max(polygon[:, 1]))
    return x0, y0, x1, y1


def polygon_area_pixels(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    return float(abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1))) * 0.5)


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    return np.mean(polygon, axis=0).astype(np.float32)


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(perimeter * epsilon_ratio, 1.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return points


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    steps: int = 20,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    omt = 1.0 - t
    points = (
        (omt**3) * p0
        + 3.0 * (omt**2) * t * p1
        + 3.0 * omt * (t**2) * p2
        + (t**3) * p3
    )
    return points.astype(np.float32)


def unit_vector_from_degrees(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.asarray((math.cos(radians), math.sin(radians)), dtype=np.float32)


def build_shadow_shape(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    direction: np.ndarray,
    scale_pixels: float,
    attachment_span_ratio: float,
    tip_width_ratio: float,
    roundness_factor: float,
    flare_factor: float,
    skew_factor: float,
    bend_factor: float,
    jitter_factor: float,
    curve_steps: int = 14,
) -> np.ndarray:
    edge = edge_end - edge_start
    edge_length = max(float(np.linalg.norm(edge)), 1.0)
    tangent = edge / edge_length
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    midpoint = (edge_start + edge_end) * 0.5

    attachment_span = edge_length * float(np.clip(attachment_span_ratio, 0.05, 1.0))
    tip_width = attachment_span * float(np.clip(tip_width_ratio, 0.05, 1.5))
    roundness = float(np.clip(roundness_factor, 0.0, 1.0))
    flare = attachment_span * float(np.clip(flare_factor, 0.0, 1.5))
    skew = attachment_span * float(np.clip(skew_factor, -1.0, 1.0))

    half_attachment = attachment_span * 0.5
    near_start = midpoint - tangent * half_attachment
    near_end = midpoint + tangent * half_attachment

    tip_center = midpoint + direction * scale_pixels + tangent * ((attachment_span * 0.2 * jitter_factor) + skew)
    half_tip = tip_width * 0.5
    tip_start = tip_center + tangent * half_tip
    tip_end = tip_center - tangent * half_tip

    side_sway = tangent * (attachment_span * 0.35 * jitter_factor)
    tip_bulge = direction * (scale_pixels * (0.16 + roundness * 0.55 + bend_factor))
    right_pull = tangent * (attachment_span * (0.12 + roundness * 0.25) + flare * 0.45 + max(skew, 0.0))
    left_pull = -tangent * (attachment_span * (0.12 + roundness * 0.25) + flare * 0.45 + max(-skew, 0.0))
    side_depth = direction * (scale_pixels * (0.28 + roundness * 0.2))
    tip_return = direction * (scale_pixels * 0.18)

    right_curve = cubic_bezier(
        near_end,
        near_end + side_depth + side_sway,
        tip_start - tip_return + right_pull + side_sway * 0.5,
        tip_start,
        steps=curve_steps,
    )
    tip_curve = cubic_bezier(
        tip_start,
        tip_start + tip_bulge + side_sway * 0.3,
        tip_end + tip_bulge - side_sway * 0.3,
        tip_end,
        steps=curve_steps,
    )
    left_curve = cubic_bezier(
        tip_end,
        tip_end - tip_return + left_pull - side_sway * 0.5,
        near_start + side_depth - side_sway,
        near_start,
        steps=curve_steps,
    )

    shadow_shape = np.vstack(
        [
            near_start,
            near_end,
            right_curve[1:],
            tip_curve[1:],
            left_curve[1:],
            near_start,
        ]
    ).astype(np.float32)
    return shadow_shape


def choose_edge(
    polygon: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = polygon_centroid(polygon)
    candidate_edges: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for idx in range(len(polygon)):
        start = polygon[idx]
        end = polygon[(idx + 1) % len(polygon)]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 1.0:
            continue
        normal = np.asarray((edge[1], -edge[0]), dtype=np.float32) / edge_length
        midpoint = (start + end) * 0.5
        if float(np.dot(normal, midpoint - centroid)) < 0.0:
            normal *= -1.0

        if mode == "down":
            score = max(float(normal[1]), 0.0) * 2.0 + float(midpoint[1]) * 0.01 + edge_length * 0.001
        else:
            score = edge_length
        candidate_edges.append((score, start, end, normal))

    if not candidate_edges:
        raise ValueError("Polygon does not contain a usable edge.")

    if mode == "random":
        weights = np.asarray([max(score, 1e-3) for score, *_ in candidate_edges], dtype=np.float32)
        weights /= np.sum(weights)
        selected_idx = int(rng.choice(len(candidate_edges), p=weights))
        _, start, end, normal = candidate_edges[selected_idx]
        return start, end, normal

    _, start, end, normal = max(candidate_edges, key=lambda item: item[0])
    return start, end, normal


def rasterize_polygon(
    polygon: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)
    if len(polygon) < 3:
        return mask
    polygon_int = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [polygon_int], color=1.0)
    return mask


def blur_mask(mask: np.ndarray, blur_ratio: float) -> np.ndarray:
    if blur_ratio <= 0.0:
        return mask
    image_span = max(mask.shape)
    kernel_size = max(int(round(image_span * blur_ratio)), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
