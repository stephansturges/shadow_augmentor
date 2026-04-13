from shadow_augmentor.adapters import (
    summarize_framework_views,
    to_albumentations_sample,
    to_torchvision_detection_sample,
)
from shadow_augmentor.constants import CACHE_SCHEMA_VERSION, LIBRARY_VERSION
from shadow_augmentor.augment import ShadowAugmentConfig, ShadowAugmentor
from shadow_augmentor.builder import ShadowPolyBuildConfig, ShadowPolyBuildReport, ShadowPolyBuilder
from shadow_augmentor.debug import (
    ShadowDebugAttempt,
    ShadowDebugResult,
    render_shadow_debug_overlay,
    simulate_shadow_debug,
    write_debug_overlay_bundle,
)
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
from shadow_augmentor.segmenters import BBoxRectangleSegmenter, SAMBoxPredictorAdapter
from shadow_augmentor.training import ShadowAugmentedYoloDataset, TrainingSample
from shadow_augmentor.yolo import YoloDataset

__version__ = LIBRARY_VERSION

__all__ = [
    "__version__",
    "BBoxRectangleSegmenter",
    "CACHE_SCHEMA_VERSION",
    "CacheValidationResult",
    "ClassSelectionSummary",
    "DatasetSplitValidationReport",
    "DatasetValidationReport",
    "LIBRARY_VERSION",
    "SAMBoxPredictorAdapter",
    "ShadowDebugAttempt",
    "ShadowDebugResult",
    "ShadowAugmentConfig",
    "ShadowAugmentedYoloDataset",
    "ShadowAugmentor",
    "ShadowPolyBuildConfig",
    "ShadowPolyCache",
    "ShadowPolyCacheMeta",
    "ShadowPolyIssue",
    "ShadowPolygon",
    "ShadowPolyBuildReport",
    "ShadowPolyBuilder",
    "TrainingSample",
    "YoloAnnotation",
    "YoloBBox",
    "YoloDataset",
    "YoloSample",
    "render_shadow_debug_overlay",
    "simulate_shadow_debug",
    "summarize_framework_views",
    "to_albumentations_sample",
    "to_torchvision_detection_sample",
    "write_debug_overlay_bundle",
]
