# shadow-augmentor

`shadow-augmentor` is a Python library for adding class-aware shadow augmentation to YOLO detection datasets.

It is built around a two-stage workflow:

1. Build time: read a YOLO dataset, select target classes, and cache object polygons for those classes into `shadow_polys/`.
2. Training time: load the cached polygons and project synthetic shadows from the object outline.

The expensive part, segmentation, is intentionally separated from the fast part, augmentation. That keeps training-time data loading lightweight and reproducible.

## What The Repo Contains

- `src/shadow_augmentor/`: library code
- `examples/basic_usage.py`: runnable smoke-test usage example
- `tests/test_shadow_augmentor.py`: unit and workflow tests

Core modules:

- `yolo.py`: YOLO dataset indexing, validation, annotation loading, cache IO
- `builder.py`: polygon generation and versioned cache writing
- `augment.py`: shadow sampling and image blending
- `debug.py`: overlay rendering and debug simulation
- `cli.py`: debug and audit commands
- `training.py`: DataLoader-friendly dataset wrapper
- `segmenters.py`: box-segmentation adapter interface

## What It Does

Given a standard YOLO dataset, the library can:

- load `data.yaml`, images, and bounding boxes
- resolve class names or class ids
- generate polygons for selected classes with a pluggable SAM-style box segmenter
- save those polygons as versioned `shadow_polys` label files
- apply synthetic shadows at training time only when relevant objects are present
- expose debug tooling to validate datasets, audit caches, and render visual overlays

## Dataset Expectations

The expected layout is a standard YOLO-style directory:

```text
dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Generated polygon caches are written under:

```text
dataset/
  shadow_polys/
    train/
      sample_001.json
```

Each cache file stores:

- schema version
- image and label relative paths
- image dimensions
- image and label SHA-256 fingerprints
- builder version
- segmenter name
- selected class ids
- per-object polygons
- provenance for each polygon: `sam` or `bbox_fallback`
- cache issues collected during build

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## User Journey

The intended user path is:

1. Validate the YOLO dataset layout.
2. Decide which classes should receive synthetic shadows.
3. Build `shadow_polys` for those classes.
4. Validate or audit the caches.
5. Wrap the dataset with `ShadowAugmentedYoloDataset`.
6. Train with the augmenter enabled.
7. Use the debug overlays when the results do not look right.

The important design point is that SAM is only needed during polygon generation, not during model training.

## Quick Start

This is the fastest end-to-end smoke test. It uses `BBoxRectangleSegmenter`, which turns the bbox into a rectangle mask. That is not a realistic segmentation model, but it exercises the whole pipeline without a heavy dependency.

```python
from shadow_augmentor import (
    BBoxRectangleSegmenter,
    ShadowAugmentConfig,
    ShadowAugmentedYoloDataset,
    ShadowAugmentor,
    ShadowPolyBuildConfig,
    ShadowPolyBuilder,
    YoloDataset,
)

dataset = YoloDataset.from_yaml("dataset/data.yaml")

validation = dataset.validate_dataset()
if not validation.valid:
    raise RuntimeError("Dataset validation failed.")

summary = dataset.summarize_class_selection(["car", "truck"], splits=["train"])
print(summary.images_with_selected_classes, summary.object_count_by_class)

builder = ShadowPolyBuilder(dataset)
build_report = builder.generate(
    segmenter=BBoxRectangleSegmenter(),
    config=ShadowPolyBuildConfig(
        selected_classes=["car", "truck"],
        splits=["train"],
        cache_policy="reuse_valid",
        on_segmenter_error="bbox_fallback",
        max_bbox_fallback_rate=1.0,
        min_polygon_area_ratio=0.05,
    ),
)
print(build_report.images_written, build_report.objects_segmented)

augmenter = ShadowAugmentor(
    ShadowAugmentConfig(
        selected_classes=["car", "truck"],
        probability=0.75,
        side_mode="down",
        shadow_count=(1, 1),
        scale=(0.8, 1.4),
        darkness=(0.35, 0.65),
        direction_degrees=(75.0, 110.0),
        blur_ratio=0.03,
    )
)

train_dataset = ShadowAugmentedYoloDataset(
    dataset=dataset,
    split="train",
    augmenter=augmenter,
    cache_validation="error",
)

sample = train_dataset[0]
print(sample["image"].shape, sample["boxes"].shape, len(sample["shadow_polys"]))
```

## Real Segmentation Integration

The library does not bundle SAM directly. Instead, it expects a box segmenter with this shape:

```python
class MySegmenter:
    def prepare_image(self, image):
        ...

    def segment_bbox(self, image, bbox_xyxy, annotation=None, sample=None):
        # Return either:
        # 1. a 2D binary mask
        # 2. a sequence of pixel-space (x, y) points
        ...
```

If your predictor looks like classic SAM with `set_image(...)` and `predict(...)`, you can use the included adapter:

```python
from shadow_augmentor import SAMBoxPredictorAdapter, ShadowPolyBuilder, ShadowPolyBuildConfig

segmenter = SAMBoxPredictorAdapter(your_predictor)

ShadowPolyBuilder(dataset).generate(
    segmenter=segmenter,
    config=ShadowPolyBuildConfig(
        selected_classes=["car", "truck"],
        splits=["train", "val"],
        cache_policy="reuse_valid",
        on_segmenter_error="raise",
        max_bbox_fallback_rate=0.2,
        min_polygon_area_ratio=0.05,
    ),
)
```

Important: SAM runtimes vary. `SAMBoxPredictorAdapter` is a convenience adapter, not a guarantee that every SAM3 runtime will work without a custom wrapper.

## Training-Time Augmentation

`ShadowAugmentedYoloDataset` loads:

- the image
- YOLO bounding boxes
- cached `shadow_polys`
- the augmented image, if an augmenter is attached

The returned sample is a plain dictionary containing:

- `image`
- `class_ids`
- `boxes`
- `shadow_polys`
- `path`
- `sample`

### Default Behavior

By default, shadow generation is biased toward the lower edge of the object:

- `side_mode="down"`
- one object can receive at most one shadow in a single augmentation pass
- multiple different objects can each receive one shadow

### Main Augmentation Controls

- `probability`: whether the image is augmented at all
- `shadow_count`: how many object shadows to attempt in the image
- `scale`: shadow length relative to the polygon bbox
- `darkness`: shadow strength; `1.0` can produce fully black regions
- `direction_degrees`: image-space direction in degrees; `90` points down
- `blur_ratio`: softness relative to image size
- `blend_mode`: `multiply` or `darken`

Shape controls:

- `attachment_span_ratio`
- `tip_width_ratio`
- `roundness_factor`
- `flare_factor`
- `skew_factor`
- `bend_factor`
- `jitter_factor`

Density controls:

- `density_decay`
- `density_noise`

Safety controls:

- `max_shadow_coverage_ratio`
- `max_overlap_with_other_objects_ratio`
- `max_shadow_attempts`

## Debug And Audit Tooling

The repo includes a debug CLI:

```bash
shadow-augmentor-debug validate-dataset dataset/data.yaml
shadow-augmentor-debug audit-cache dataset/data.yaml --classes car truck
shadow-augmentor-debug render-overlays dataset/data.yaml --classes car truck --output-dir debug_overlays
shadow-augmentor-debug adapter-preview dataset/data.yaml --json
```

These commands help answer different questions:

- `validate-dataset`: is the YOLO layout usable
- `audit-cache`: are the existing `shadow_polys` valid for the selected classes
- `render-overlays`: do the polygons and shadow geometry make visual sense
- `adapter-preview`: what shape does the training sample take for common frameworks

Programmatic debug helpers are also exposed:

- `simulate_shadow_debug(...)`
- `render_shadow_debug_overlay(...)`
- `write_debug_overlay_bundle(...)`
- `to_torchvision_detection_sample(...)`
- `to_albumentations_sample(...)`

## Cache Validation Rules

The training wrapper defaults to fail-closed behavior when caches are stale or malformed.

`YoloDataset.validate_shadow_poly_cache(...)` checks:

- schema version
- selected class ids
- image path, label path, and split
- image hash
- label hash
- image dimensions
- polygon validity and bounds

`ShadowAugmentedYoloDataset` supports:

- `cache_validation="error"`: raise when a needed cache is invalid
- `cache_validation="warn"`: warn and skip augmentation
- `cache_validation="ignore"`: load whatever exists

For training, `error` is the safest mode.

## Framework Adapters

The repo includes lightweight adapters for inspection and interop:

- `to_torchvision_detection_sample(...)`
- `to_albumentations_sample(...)`
- `summarize_framework_views(...)`

These do not add framework dependencies. They only reshape the sample into familiar formats.

## Common Failure Paths

The main failure modes along the user journey are:

1. Dataset layout issues
   Missing split directories, missing labels, or orphan labels will surface in `validate_dataset()`.

2. Segmenter integration mismatch
   Your SAM runtime may not match the assumptions in `SAMBoxPredictorAdapter`. If `prepare_image` or `predict` fail, build-time segmentation will fail or fall back depending on config.

3. Weak segmentation quality
   If the mask is too small or degenerate, the builder falls back to bbox polygons. This is recorded as `bbox_fallback`.

4. Stale caches
   If an image or label file changes after polygon generation, cache validation will reject the stored `shadow_polys`.

5. Over-constrained augmentation
   Aggressive overlap or coverage guards can cause the augmenter to skip shadows entirely on crowded scenes.

6. Direct augmentor usage with class names
   `ShadowAugmentConfig(selected_classes=["car"])` is fine when used through `ShadowAugmentedYoloDataset`, which resolves names for you. If you call `ShadowAugmentor(...)` directly, resolve names to ids first or call `set_selected_class_ids(...)`.

7. Debug/runtime mismatch
   The debug simulator is intended to mirror runtime augmentation. If you change augmentation logic, keep the debug path in sync and verify it with tests.

## Current Status

What is covered by tests today:

- dataset validation
- cache schema and fingerprint checks
- stale-cache rebuild behavior
- bbox-fallback provenance
- runtime overlap safety
- deterministic augmentation
- debug overlay and CLI flows
- debug/runtime parity under retry-heavy sampling

What is intentionally not bundled:

- a fixed SAM3 dependency
- a framework-specific training integration beyond the current Python API
- a full training loop

## Minimal Dev Workflow

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Run the example:

```bash
PYTHONPATH=src python examples/basic_usage.py
```

If you want a visual inspection pass, render overlays:

```bash
shadow-augmentor-debug render-overlays dataset/data.yaml --classes car truck --output-dir debug_overlays
```
