"""Identidade imutável da experiência V3 época 3. Sem PyTorch/Qt."""
from __future__ import annotations

from pathlib import Path

TRIAL_POINTER = Path("reports/aoi_quality_cnn_v3/production_v3_trial.json")
POINTER_SCHEMA = "visionx.aoi_quality_cnn_v3_production_ng_veto_trial.v1"
SNAPSHOT_RELATIVE_PATH = (
    "reports/aoi_quality_cnn_v3/candidates/"
    "20261104T192753_047124Z_6b733a/"
    "epoch_diagnostics/epoch_03_experimental_only.pt"
)
SNAPSHOT_SHA256 = (
    "fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce"
)
SOURCE_MANIFEST_SHA256 = (
    "c3f8f017d2ae17cbe876a6342056fbdfe041690bd4062ddeeba195982e64e843"
)
INITIAL_V3_SHA256 = (
    "781d5d4d6cb80461a990e9a325560b75688f24f4647846ef12023e3898e15e1c"
)
LIGHTS = ("SIDE", "TOP", "MID")

__all__ = [
    "TRIAL_POINTER", "POINTER_SCHEMA", "SNAPSHOT_RELATIVE_PATH",
    "SNAPSHOT_SHA256", "SOURCE_MANIFEST_SHA256", "INITIAL_V3_SHA256",
    "LIGHTS",
]
