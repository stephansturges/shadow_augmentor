from shadow_augmentor import (
    SAMBoxPredictorAdapter,
    ShadowAugmentConfig,
    ShadowAugmentedYoloDataset,
    ShadowAugmentor,
    ShadowPolyBuildConfig,
    ShadowPolyBuilder,
    YoloDataset,
)


def main() -> None:
    dataset = YoloDataset.from_yaml("dataset/data.yaml")
    report = dataset.validate_dataset()
    if not report.valid:
        raise RuntimeError("Dataset validation failed.")

    # Replace this with your own SAM-style predictor instance.
    sam3_predictor = ...
    segmenter = SAMBoxPredictorAdapter(sam3_predictor)

    ShadowPolyBuilder(dataset).generate(
        segmenter=segmenter,
        config=ShadowPolyBuildConfig(
            selected_classes=["car", "truck"],
            splits=["train"],
            cache_policy="reuse_valid",
            on_segmenter_error="raise",
            max_bbox_fallback_rate=0.2,
            min_polygon_area_ratio=0.05,
        ),
    )

    augmenter = ShadowAugmentor(
        ShadowAugmentConfig(
            selected_classes=["car", "truck"],
            probability=0.75,
            side_mode="down",
            shadow_count=(1, 2),
            scale=(0.8, 1.5),
            darkness=(0.35, 0.65),
            direction_degrees=(75.0, 110.0),
        )
    )

    train_dataset = ShadowAugmentedYoloDataset(
        dataset=dataset,
        split="train",
        augmenter=augmenter,
        cache_validation="error",
    )
    sample = train_dataset[0]
    print(sample["image"].shape, len(sample["shadow_polys"]))


if __name__ == "__main__":
    main()
