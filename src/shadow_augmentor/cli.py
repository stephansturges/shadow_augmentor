from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from shadow_augmentor.adapters import summarize_framework_views
from shadow_augmentor.augment import ShadowAugmentConfig, ShadowAugmentor
from shadow_augmentor.debug import simulate_shadow_debug, write_debug_overlay_bundle
from shadow_augmentor.training import ShadowAugmentedYoloDataset
from shadow_augmentor.yolo import YoloDataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shadow-augmentor-debug", description="Debug tooling for shadow_augmentor datasets and caches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-dataset", help="Validate YOLO dataset layout.")
    validate_parser.add_argument("data_yaml", type=Path)
    validate_parser.add_argument("--splits", nargs="*", default=None)
    validate_parser.add_argument("--json", action="store_true", dest="as_json")
    validate_parser.set_defaults(func=_cmd_validate_dataset)

    audit_parser = subparsers.add_parser("audit-cache", help="Validate versioned shadow polygon caches.")
    audit_parser.add_argument("data_yaml", type=Path)
    audit_parser.add_argument("--splits", nargs="*", default=None)
    audit_parser.add_argument("--classes", nargs="*", default=None)
    audit_parser.add_argument("--selection-match", choices=("exact", "subset"), default="subset")
    audit_parser.add_argument("--json", action="store_true", dest="as_json")
    audit_parser.set_defaults(func=_cmd_audit_cache)

    render_parser = subparsers.add_parser("render-overlays", help="Render per-sample polygon and shadow debug overlays.")
    render_parser.add_argument("data_yaml", type=Path)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--split", default="train")
    render_parser.add_argument("--classes", nargs="*", default=None)
    render_parser.add_argument("--limit", type=int, default=10)
    render_parser.add_argument("--seed", type=int, default=7)
    render_parser.add_argument("--side-mode", choices=("random", "down"), default="down")
    render_parser.add_argument("--shadow-count-min", type=int, default=1)
    render_parser.add_argument("--shadow-count-max", type=int, default=1)
    render_parser.add_argument("--scale-min", type=float, default=0.8)
    render_parser.add_argument("--scale-max", type=float, default=1.4)
    render_parser.add_argument("--darkness-min", type=float, default=0.35)
    render_parser.add_argument("--darkness-max", type=float, default=0.65)
    render_parser.add_argument("--direction-min", type=float, default=70.0)
    render_parser.add_argument("--direction-max", type=float, default=110.0)
    render_parser.add_argument("--blur-ratio", type=float, default=0.03)
    render_parser.add_argument("--blend-mode", choices=("multiply", "darken"), default="multiply")
    render_parser.add_argument("--max-shadow-coverage-ratio", type=float, default=0.25)
    render_parser.add_argument("--max-overlap-with-other-objects-ratio", type=float, default=0.05)
    render_parser.set_defaults(func=_cmd_render_overlays)

    adapter_parser = subparsers.add_parser("adapter-preview", help="Preview framework adapter shapes for a dataset sample.")
    adapter_parser.add_argument("data_yaml", type=Path)
    adapter_parser.add_argument("--split", default="train")
    adapter_parser.add_argument("--index", type=int, default=0)
    adapter_parser.add_argument("--json", action="store_true", dest="as_json")
    adapter_parser.set_defaults(func=_cmd_adapter_preview)

    return parser


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    dataset = YoloDataset.from_yaml(args.data_yaml)
    report = dataset.validate_dataset(args.splits)
    payload = {
        "valid": report.valid,
        "root": str(report.root),
        "config_path": str(report.config_path),
        "issues": [issue.to_dict() for issue in report.issues],
        "splits": [
            {
                "split": split_report.split,
                "image_dir": str(split_report.image_dir),
                "label_dir": str(split_report.label_dir),
                "image_count": split_report.image_count,
                "label_count": split_report.label_count,
                "missing_label_paths": list(split_report.missing_label_paths),
                "orphan_label_paths": list(split_report.orphan_label_paths),
                "issues": [issue.to_dict() for issue in split_report.issues],
            }
            for split_report in report.split_reports
        ],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"dataset valid: {report.valid}")
        for split_report in report.split_reports:
            print(
                f"{split_report.split}: images={split_report.image_count} labels={split_report.label_count} "
                f"missing_labels={len(split_report.missing_label_paths)} orphan_labels={len(split_report.orphan_label_paths)}"
            )
            for issue in split_report.issues:
                print(f"  {issue.severity}: {issue.code} - {issue.message}")
    return 0 if report.valid else 1


def _cmd_audit_cache(args: argparse.Namespace) -> int:
    dataset = YoloDataset.from_yaml(args.data_yaml)
    selected_classes = _parse_class_refs(args.classes)
    selected_class_ids = tuple(sorted(dataset.resolve_class_ids(selected_classes)))

    audited_samples = 0
    valid_samples = 0
    invalid_samples = 0
    missing_samples = 0
    sample_payloads: list[dict[str, Any]] = []

    for sample in dataset.iter_samples(args.splits):
        annotations = dataset.load_annotations(sample)
        if not any(annotation.class_id in selected_class_ids for annotation in annotations):
            continue
        audited_samples += 1
        validation = dataset.validate_shadow_poly_cache(
            sample,
            expected_selected_class_ids=selected_class_ids,
            selection_match_mode=args.selection_match,
        )
        if validation.valid:
            valid_samples += 1
        else:
            invalid_samples += 1
            if not validation.exists:
                missing_samples += 1
        sample_payloads.append(
            {
                "image_path": sample.relative_image_path(dataset.root),
                "valid": validation.valid,
                "exists": validation.exists,
                "issues": [issue.to_dict() for issue in validation.issues],
            }
        )

    payload = {
        "selected_class_ids": list(selected_class_ids),
        "audited_samples": audited_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "missing_samples": missing_samples,
        "samples": sample_payloads,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"audited={audited_samples} valid={valid_samples} invalid={invalid_samples} missing={missing_samples}"
        )
        for sample_payload in sample_payloads:
            if sample_payload["valid"]:
                continue
            print(f"{sample_payload['image_path']}: invalid")
            for issue in sample_payload["issues"]:
                print(f"  {issue['severity']}: {issue['code']} - {issue['message']}")
    return 0 if invalid_samples == 0 else 1


def _cmd_render_overlays(args: argparse.Namespace) -> int:
    dataset = YoloDataset.from_yaml(args.data_yaml)
    selected_classes = _parse_class_refs(args.classes)
    selected_class_ids = tuple(sorted(dataset.resolve_class_ids(selected_classes)))
    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=selected_class_ids,
            probability=1.0,
            side_mode=args.side_mode,
            shadow_count=(args.shadow_count_min, args.shadow_count_max),
            scale=(args.scale_min, args.scale_max),
            darkness=(args.darkness_min, args.darkness_max),
            direction_degrees=(args.direction_min, args.direction_max),
            blur_ratio=args.blur_ratio,
            blend_mode=args.blend_mode,
            max_shadow_coverage_ratio=args.max_shadow_coverage_ratio,
            max_overlap_with_other_objects_ratio=args.max_overlap_with_other_objects_ratio,
        )
    )
    augmenter.set_selected_class_ids(selected_class_ids)

    rendered = 0
    skipped_invalid = 0
    for sample in dataset.get_samples(args.split):
        annotations = dataset.load_annotations(sample)
        if not any(annotation.class_id in selected_class_ids for annotation in annotations):
            continue
        validation = dataset.validate_shadow_poly_cache(
            sample,
            expected_selected_class_ids=selected_class_ids,
            selection_match_mode="subset",
        )
        if not validation.valid or validation.cache is None:
            skipped_invalid += 1
            continue

        image = dataset.load_image(sample)
        rng = np.random.default_rng(args.seed + rendered)
        debug_result = simulate_shadow_debug(
            augmenter,
            image,
            validation.cache.polygons,
            rng=rng,
            annotations=annotations,
        )
        relative_stem = sample.image_path.relative_to(dataset.split_dirs[args.split]).with_suffix("")
        output_stem = args.output_dir / relative_stem
        write_debug_overlay_bundle(
            image=image,
            shadow_polys=validation.cache.polygons,
            debug_result=debug_result,
            output_path=output_stem,
            sample=sample,
        )
        rendered += 1
        if rendered >= args.limit:
            break

    print(f"rendered={rendered} skipped_invalid={skipped_invalid} output_dir={args.output_dir}")
    return 0 if rendered > 0 else 1


def _cmd_adapter_preview(args: argparse.Namespace) -> int:
    dataset = YoloDataset.from_yaml(args.data_yaml)
    debug_dataset = ShadowAugmentedYoloDataset(
        dataset=dataset,
        split=args.split,
        augmenter=None,
        cache_validation="ignore",
    )
    sample = debug_dataset[args.index]
    payload = summarize_framework_views(sample)
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"path: {payload['path']}")
        print(f"torchvision image_shape={payload['torchvision']['image_shape']} boxes_shape={payload['torchvision']['boxes_shape']}")
        print(f"albumentations image_shape={payload['albumentations']['image_shape']} bbox_count={payload['albumentations']['bbox_count']}")
    return 0


def _parse_class_refs(values: Sequence[str] | None) -> list[int | str] | None:
    if not values:
        return None
    parsed: list[int | str] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
