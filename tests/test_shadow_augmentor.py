from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from shadow_augmentor import (
    BBoxRectangleSegmenter,
    CACHE_SCHEMA_VERSION,
    ShadowAugmentConfig,
    ShadowAugmentedYoloDataset,
    ShadowAugmentor,
    ShadowPolyBuildConfig,
    ShadowPolyBuilder,
    ShadowPolygon,
    YoloAnnotation,
    YoloBBox,
    YoloDataset,
)
from shadow_augmentor.cli import main as cli_main
from shadow_augmentor.debug import simulate_shadow_debug, write_debug_overlay_bundle
from shadow_augmentor.geometry import build_shadow_shape


class EmptyMaskSegmenter:
    def prepare_image(self, image: np.ndarray) -> None:
        _ = image

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        annotation: YoloAnnotation | None = None,
        sample: object | None = None,
    ) -> np.ndarray:
        _ = image, bbox_xyxy, annotation, sample
        return np.zeros((64, 64), dtype=np.uint8)


def _write_dataset(root: Path, *, missing_label_image: bool = False, orphan_label: bool = False) -> Path:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)

    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[16:48, 16:48] = 220
    Image.fromarray(image).save(root / "images" / "train" / "sample.jpg")

    (root / "labels" / "train" / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )

    if missing_label_image:
        Image.fromarray(image).save(root / "images" / "train" / "unlabeled.jpg")
    if orphan_label:
        (root / "labels" / "train" / "orphan.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nnames:\n  0: car\n  1: person\n",
        encoding="utf-8",
    )
    return root / "data.yaml"


def _build_default_cache(config_path: Path) -> tuple[YoloDataset, object]:
    dataset = YoloDataset.from_yaml(config_path)
    builder = ShadowPolyBuilder(dataset)
    report = builder.generate(
        segmenter=BBoxRectangleSegmenter(),
        selected_classes=["car"],
        splits=["train"],
    )
    return dataset, report


def test_validate_dataset_reports_missing_labels_and_orphans(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path, missing_label_image=True, orphan_label=True)
    dataset = YoloDataset.from_yaml(config_path)

    report = dataset.validate_dataset(["train"])

    assert report.valid
    split_report = report.split_reports[0]
    assert split_report.image_count == 2
    assert split_report.label_count == 2
    assert "unlabeled.txt" in split_report.missing_label_paths
    assert "orphan.txt" in split_report.orphan_label_paths


def test_builder_writes_versioned_cache_and_reuses_valid_cache(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path)
    dataset, first_report = _build_default_cache(config_path)

    sample = dataset.get_samples("train")[0]
    cache = dataset.load_shadow_poly_cache(sample)
    assert cache is not None
    assert first_report.images_written == 1
    assert first_report.images_reused == 0
    assert cache.meta.schema_version == CACHE_SCHEMA_VERSION
    assert cache.meta.image_sha256
    assert cache.meta.label_sha256
    assert cache.meta.selected_class_ids == (0,)
    assert cache.polygons[0].source == "sam"
    assert cache.polygons[0].area_ratio > 0.9

    second_report = ShadowPolyBuilder(dataset).generate(
        segmenter=BBoxRectangleSegmenter(),
        selected_classes=["car"],
        splits=["train"],
    )
    assert second_report.images_written == 0
    assert second_report.images_reused == 1


def test_builder_rebuilds_stale_cache_after_label_change(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path)
    dataset, _ = _build_default_cache(config_path)
    sample = dataset.get_samples("train")[0]

    sample.label_path.write_text(
        "0 0.55 0.55 0.4 0.4\n1 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )
    validation = dataset.validate_shadow_poly_cache(sample, expected_selected_class_ids=(0,))
    assert not validation.valid
    assert any(issue.code == "stale_label_hash" for issue in validation.issues)

    report = ShadowPolyBuilder(dataset).generate(
        segmenter=BBoxRectangleSegmenter(),
        selected_classes=["car"],
        splits=["train"],
    )
    assert report.images_rebuilt == 1
    assert report.images_written == 1
    refreshed_validation = dataset.validate_shadow_poly_cache(sample, expected_selected_class_ids=(0,))
    assert refreshed_validation.valid


def test_builder_records_bbox_fallback_provenance(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path)
    dataset = YoloDataset.from_yaml(config_path)
    builder = ShadowPolyBuilder(dataset)

    report = builder.generate(
        segmenter=EmptyMaskSegmenter(),
        selected_classes=["car"],
        splits=["train"],
        config=ShadowPolyBuildConfig(
            selected_classes=["car"],
            splits=["train"],
            on_segmenter_error="bbox_fallback",
            max_bbox_fallback_rate=1.0,
        ),
    )

    sample = dataset.get_samples("train")[0]
    cache = dataset.load_shadow_poly_cache(sample)
    assert cache is not None
    assert report.objects_fell_back_to_bbox == 1
    assert cache.polygons[0].source == "bbox_fallback"
    assert cache.polygons[0].warnings


def test_builder_raises_when_bbox_fallback_rate_exceeds_threshold(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path)
    dataset = YoloDataset.from_yaml(config_path)
    builder = ShadowPolyBuilder(dataset)

    with pytest.raises(RuntimeError, match="BBox fallback rate exceeded"):
        builder.generate(
            segmenter=EmptyMaskSegmenter(),
            selected_classes=["car"],
            splits=["train"],
            config=ShadowPolyBuildConfig(
                selected_classes=["car"],
                splits=["train"],
                on_segmenter_error="bbox_fallback",
                max_bbox_fallback_rate=0.0,
            ),
        )


def test_training_dataset_errors_on_stale_cache_and_warn_mode_skips_augmentation(tmp_path: Path) -> None:
    config_path = _write_dataset(tmp_path)
    dataset, _ = _build_default_cache(config_path)
    sample = dataset.get_samples("train")[0]
    original_image = dataset.load_image(sample)

    sample.label_path.write_text(
        "0 0.52 0.52 0.48 0.48\n1 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=["car"],
            probability=1.0,
            side_mode="down",
            shadow_count=(1, 1),
            scale=(1.0, 1.0),
            darkness=(0.5, 0.5),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
        )
    )

    error_dataset = ShadowAugmentedYoloDataset(
        dataset=dataset,
        split="train",
        augmenter=augmenter,
        cache_validation="error",
    )
    with pytest.raises(RuntimeError, match="Invalid shadow polygon cache"):
        _ = error_dataset[0]

    warn_dataset = ShadowAugmentedYoloDataset(
        dataset=dataset,
        split="train",
        augmenter=augmenter,
        cache_validation="warn",
    )
    with pytest.warns(RuntimeWarning, match="Invalid shadow polygon cache"):
        warn_sample = warn_dataset[0]
    assert warn_sample["shadow_polys"] == ()
    assert np.array_equal(warn_sample["image"], original_image)


def test_shadow_augmentor_rejects_overlapping_shadows_and_is_deterministic() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    shadow_poly = ShadowPolygon(
        object_index=0,
        class_id=0,
        class_name="car",
        bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
        polygon=((0.35, 0.2), (0.65, 0.2), (0.65, 0.5), (0.35, 0.5)),
        source="sam",
        area_ratio=1.0,
    )
    annotations = [
        YoloAnnotation(
            object_index=0,
            bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
            class_name="car",
        ),
        YoloAnnotation(
            object_index=1,
            bbox=YoloBBox(class_id=1, x_center=0.5, y_center=0.7, width=0.35, height=0.2),
            class_name="person",
        ),
    ]

    overlap_guard = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=[0],
            probability=1.0,
            side_mode="down",
            shadow_count=(1, 1),
            scale=(1.0, 1.0),
            darkness=(0.5, 0.5),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
            max_overlap_with_other_objects_ratio=0.0,
            max_shadow_attempts=1,
        )
    )
    rejected = overlap_guard(image.copy(), [shadow_poly], rng=np.random.default_rng(7), annotations=annotations)
    assert np.array_equal(rejected, image)

    deterministic = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=[0],
            probability=1.0,
            side_mode="down",
            shadow_count=(1, 1),
            scale=(1.0, 1.0),
            darkness=(0.5, 0.5),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
            max_overlap_with_other_objects_ratio=1.0,
        )
    )
    first = deterministic(image.copy(), [shadow_poly], rng=np.random.default_rng(11), annotations=annotations[:1])
    second = deterministic(image.copy(), [shadow_poly], rng=np.random.default_rng(11), annotations=annotations[:1])
    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


def test_build_shadow_shape_generates_curved_outline() -> None:
    shape = build_shadow_shape(
        edge_start=np.asarray((20.0, 20.0), dtype=np.float32),
        edge_end=np.asarray((44.0, 20.0), dtype=np.float32),
        direction=np.asarray((0.0, 1.0), dtype=np.float32),
        scale_pixels=24.0,
        attachment_span_ratio=0.7,
        tip_width_ratio=0.35,
        roundness_factor=0.45,
        flare_factor=0.25,
        skew_factor=0.12,
        bend_factor=0.2,
        jitter_factor=0.1,
    )

    assert shape.shape[0] > 20
    assert not np.allclose(shape[0], shape[1])
    assert float(np.max(shape[:, 1])) > 40.0
    unique_x = np.unique(np.round(shape[:, 0], decimals=1))
    assert unique_x.size > 10


def test_shadow_augmentor_allows_fully_black_extreme_shadow() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    shadow_poly = ShadowPolygon(
        object_index=0,
        class_id=0,
        class_name="car",
        bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
        polygon=((0.35, 0.2), (0.65, 0.2), (0.65, 0.5), (0.35, 0.5)),
        source="sam",
        area_ratio=1.0,
    )
    annotation = YoloAnnotation(
        object_index=0,
        bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
        class_name="car",
    )

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=[0],
            probability=1.0,
            side_mode="down",
            shadow_count=(1, 1),
            scale=(1.0, 1.0),
            darkness=(1.0, 1.0),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
            max_overlap_with_other_objects_ratio=1.0,
        )
    )
    result = augmenter(image.copy(), [shadow_poly], rng=np.random.default_rng(5), annotations=[annotation])
    assert int(result.min()) == 0


def test_shadow_debug_limits_to_one_shadow_per_object() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    shadow_poly = ShadowPolygon(
        object_index=0,
        class_id=0,
        class_name="car",
        bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
        polygon=((0.35, 0.2), (0.65, 0.2), (0.65, 0.5), (0.35, 0.5)),
        source="sam",
        area_ratio=1.0,
    )
    annotation = YoloAnnotation(
        object_index=0,
        bbox=YoloBBox(class_id=0, x_center=0.5, y_center=0.35, width=0.3, height=0.3),
        class_name="car",
    )

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=[0],
            probability=1.0,
            side_mode="down",
            shadow_count=(3, 3),
            scale=(1.0, 1.0),
            darkness=(0.7, 0.7),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
            max_shadow_coverage_ratio=1.0,
            max_overlap_with_other_objects_ratio=1.0,
            max_shadow_attempts=3,
        )
    )
    debug_result = simulate_shadow_debug(
        augmenter,
        image,
        [shadow_poly],
        rng=np.random.default_rng(9),
        annotations=[annotation],
    )
    assert debug_result.summary()["accepted_count"] == 1


def test_shadow_debug_matches_runtime_under_retry_heavy_settings() -> None:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    shadow_polys = [
        ShadowPolygon(
            object_index=0,
            class_id=0,
            class_name="car",
            bbox=YoloBBox(class_id=0, x_center=0.35, y_center=0.25, width=0.25, height=0.25),
            polygon=((0.2, 0.1), (0.5, 0.1), (0.5, 0.4), (0.2, 0.4)),
            source="sam",
            area_ratio=1.0,
        ),
        ShadowPolygon(
            object_index=1,
            class_id=0,
            class_name="car",
            bbox=YoloBBox(class_id=0, x_center=0.65, y_center=0.25, width=0.25, height=0.25),
            polygon=((0.5, 0.1), (0.8, 0.1), (0.8, 0.4), (0.5, 0.4)),
            source="sam",
            area_ratio=1.0,
        ),
    ]
    annotations = [
        YoloAnnotation(
            object_index=0,
            bbox=YoloBBox(class_id=0, x_center=0.35, y_center=0.25, width=0.25, height=0.25),
            class_name="car",
        ),
        YoloAnnotation(
            object_index=1,
            bbox=YoloBBox(class_id=0, x_center=0.65, y_center=0.25, width=0.25, height=0.25),
            class_name="car",
        ),
        YoloAnnotation(
            object_index=2,
            bbox=YoloBBox(class_id=1, x_center=0.5, y_center=0.6, width=0.6, height=0.25),
            class_name="person",
        ),
    ]

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=[0],
            probability=1.0,
            side_mode="random",
            shadow_count=(2, 2),
            scale=(0.8, 1.6),
            darkness=(0.35, 0.9),
            direction_degrees=(60.0, 120.0),
            blur_ratio=0.03,
            attachment_span_ratio=(0.4, 1.0),
            tip_width_ratio=(0.1, 1.2),
            roundness_factor=(0.0, 0.8),
            flare_factor=(0.0, 0.7),
            skew_factor=(0.0, 0.4),
            bend_factor=(0.0, 0.5),
            jitter_factor=(0.0, 0.2),
            density_decay=(0.0, 0.8),
            density_noise=(0.0, 0.5),
            max_shadow_coverage_ratio=0.08,
            max_overlap_with_other_objects_ratio=0.0,
            max_shadow_attempts=5,
        )
    )

    runtime_image = augmenter(
        image.copy(),
        shadow_polys,
        rng=np.random.default_rng(1),
        annotations=annotations,
    )
    debug_result = simulate_shadow_debug(
        augmenter,
        image.copy(),
        shadow_polys,
        rng=np.random.default_rng(1),
        annotations=annotations,
    )

    assert np.array_equal(debug_result.augmented_image, runtime_image)
    assert debug_result.summary()["accepted_count"] == 2


def test_debug_overlay_bundle_and_cli_workflows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_dataset(tmp_path)
    dataset, _ = _build_default_cache(config_path)
    sample = dataset.get_samples("train")[0]
    image = dataset.load_image(sample)
    annotations = dataset.load_annotations(sample)
    cache = dataset.load_shadow_poly_cache(sample)
    assert cache is not None

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=["car"],
            probability=1.0,
            side_mode="down",
            shadow_count=(1, 1),
            scale=(1.0, 1.0),
            darkness=(0.5, 0.5),
            direction_degrees=(90.0, 90.0),
            blur_ratio=0.0,
            attachment_span_ratio=(1.0, 1.0),
            tip_width_ratio=(1.0, 1.0),
            roundness_factor=(0.0, 0.0),
            flare_factor=(0.0, 0.0),
            skew_factor=(0.0, 0.0),
            bend_factor=(0.0, 0.0),
            jitter_factor=(0.0, 0.0),
            density_decay=(0.0, 0.0),
            density_noise=(0.0, 0.0),
        )
    )
    augmenter.set_selected_class_ids((0,))
    debug_result = simulate_shadow_debug(
        augmenter,
        image,
        cache.polygons,
        rng=np.random.default_rng(13),
        annotations=annotations,
    )
    image_path, json_path = write_debug_overlay_bundle(
        image=image,
        shadow_polys=cache.polygons,
        debug_result=debug_result,
        output_path=tmp_path / "debug" / "sample",
        sample=sample,
    )
    assert image_path.exists()
    assert json_path.exists()
    assert debug_result.attempts

    assert cli_main(["validate-dataset", str(config_path)]) == 0
    validate_output = capsys.readouterr().out
    assert "dataset valid: True" in validate_output

    assert cli_main(["audit-cache", str(config_path), "--classes", "car"]) == 0
    audit_output = capsys.readouterr().out
    assert "audited=1" in audit_output

    render_dir = tmp_path / "cli-render"
    assert (
        cli_main(
            [
                "render-overlays",
                str(config_path),
                "--output-dir",
                str(render_dir),
                "--classes",
                "car",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    render_output = capsys.readouterr().out
    assert "rendered=1" in render_output
    assert (render_dir / "sample.png").exists()
    assert (render_dir / "sample.json").exists()

    assert cli_main(["adapter-preview", str(config_path), "--json"]) == 0
    adapter_output = capsys.readouterr().out
    assert '"torchvision"' in adapter_output
    assert '"albumentations"' in adapter_output
