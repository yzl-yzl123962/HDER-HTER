"""Public HTER model aliases.

This module exposes paper-facing names while keeping the original training
entrypoint compatible with the experiment logs and reproduction configs.
"""

from .Gate_fusion import Gate_fusion, build_model
from .GlobalFusionClassifier import GlobalFusionClassifier
from .LocalFusionClassifier import LocalFusionClassifier


HTERModel = Gate_fusion
SemanticEvidencePath = GlobalFusionClassifier
BehavioralEvidencePath = LocalFusionClassifier


__all__ = [
    "HTERModel",
    "SemanticEvidencePath",
    "BehavioralEvidencePath",
    "build_model",
]
