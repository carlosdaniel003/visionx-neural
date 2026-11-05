"""Fonte V3 época 5 explicitamente autorizada pelo operador (05/11/2026).

Não altera pesos nem falsifica o checkpoint experimental como homologado.
O opt-in fica em um ponteiro SEPARADO, reversível, e prevalece sobre o
antigo ensaio de época 3 em todos os modos do AOI/VisionX.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.core.neural.aoi_v3_trial_contract import LIGHTS

EPOCH = 5
EPOCH5_POINTER = Path("reports/aoi_quality_cnn_v3/production_v3_epoch5.json")
EPOCH5_POINTER_SCHEMA = "visionx.aoi_v3_epoch5_operator_override.v1"
EPOCH5_VOTE_SCHEMA = "visionx.aoi_v3_epoch5_runtime_vote.v1"
REPORT_PATH = Path(
    "reports/aoi_quality_cnn_v3/candidates/"
    "20261105T171228_989626Z_44072c/training_report.json"
)
REPORT_SHA256 = "47d5d7c3bfe6863f887cf747f70e623ab69e299626ecee2bdf107e1593b92c32"
SNAPSHOT_PATH = Path(
    "reports/aoi_quality_cnn_v3/candidates/"
    "20261105T171228_989626Z_44072c/"
    "epoch_diagnostics/epoch_05_experimental_only.pt"
)
SNAPSHOT_SHA256 = "eff1d3bf5fd05e2763792e22f634ebd40f432ebab4712055c9441f2fff61fc6d"
MANIFEST_SHA256 = "32bc0f6abb871a6570e590638cdc86ff57fdf2105e456b5866b43d9f37eb70cc"
PREVIOUS_EPOCH3_SHA256 = "fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce"


def epoch5_pointer_payload() -> dict:
    return {
        "schema": EPOCH5_POINTER_SCHEMA,
        "enabled": True,
        "operator_authorized": True,
        "deployment_mode": "v3_primary_epoch5_operator_override",
        "checkpoint_relative_path": SNAPSHOT_PATH.as_posix(),
        "checkpoint_sha256": SNAPSHOT_SHA256,
        "source_report_sha256": REPORT_SHA256,
        "epoch": EPOCH,
        "validation_OK_false_NG": 2,
        "validation_NG_false_OK": 0,
        "human_ng_training_replay_not_independent": True,
        "blind_holdout": False,
        "automatic_ok_from_v3_authorized": False,
        "v2_runtime_reenabled": False,
    }


def epoch5_pointer_enabled(root: Path) -> bool:
    pointer = Path(root).resolve() / EPOCH5_POINTER
    if not pointer.exists() and not pointer.is_symlink():
        return False
    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("Ponteiro V3 época 5 inválido")
    if json.loads(pointer.read_text(encoding="utf-8")) != epoch5_pointer_payload():
        raise ValueError("Ponteiro V3 época 5 diverge da autorização exata")
    return True


def _local_file(root: Path, relative: Path) -> Path:
    base = Path(root).resolve()
    p = base / relative
    if p.is_symlink() or not p.is_file() or p.resolve() != p.absolute():
        raise ValueError("Artefato V3 época 5 ausente ou simbólico: " + str(relative))
    return p


def validate_epoch5_report(root: Path) -> dict:
    file = _local_file(root, REPORT_PATH)
    if sha256(file.read_bytes()).hexdigest() != REPORT_SHA256:
        raise ValueError("Relatório V3 época 5 com SHA-256 divergente")
    report = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Relatório V3 época 5 inválido")
    history = report.get("epoch_history")
    snapshots = report.get("epoch_diagnostic_checkpoints")
    if (
        report.get("schema") != "visionx.aoi_quality_cnn_v3_training_report.v1"
        or report.get("candidate_only") is not True
        or report.get("production_approved") is not False
        or report.get("automatic_xp_authorized") is not False
        or report.get("training", {}).get("selected_epoch_using_validation_only") != 0
        or report.get("training", {}).get("v3_initialized_from_verified_epoch3") is not True
        or report.get("operator_ng_escape_review_verified") != 25
        or report.get("provenance", {}).get("manifest_sha256") != MANIFEST_SHA256
        or report.get("provenance", {}).get("v3_ng_escape_sources_in_dataset") != 25
        or report.get("provenance", {}).get("v3_ng_escape_unique_pairs_in_dataset") != 24
        or report.get("initial_v3_candidate", {}).get("checkpoint_sha256") != PREVIOUS_EPOCH3_SHA256
        or not isinstance(history, list) or len(history) != 7
        or not isinstance(snapshots, list) or len(snapshots) != 6
    ):
        raise ValueError("Relatório V3 não corresponde ao treino das 25 fontes NG")
    selected = history[EPOCH]
    snapshot = snapshots[EPOCH-1]
    ng = selected.get("operator_ng_escape_training_replay", {})
    human = selected.get("operator_training_replay", {}).get("metrics", {})
    validation = selected.get("validation", {})
    if (
        selected.get("epoch") != EPOCH
        or selected.get("safe_gate_passed") is not False
        or selected.get("selected") is not False
        or validation.get("NG_false_OK") != 0
        or validation.get("OK_false_NG") != 2
        or ng.get("NG_total") != 24 or ng.get("NG_correct") != 24
        or human.get("correct") != 14
        or human.get("NG_correct") != 9 or human.get("OK_correct") != 5
        or snapshot.get("epoch") != EPOCH
        or snapshot.get("path") != SNAPSHOT_PATH.as_posix()
        or snapshot.get("sha256") != SNAPSHOT_SHA256
        or snapshot.get("safe_gate_selected") is not False
        or snapshot.get("test_evaluated_for_this_epoch") is not False
        or snapshot.get("production_approved") is not False
        or snapshot.get("automatic_xp_authorized") is not False
    ):
        raise ValueError("Métricas da época 5 ou identidade de snapshot divergentes")
    return report


def validate_epoch5_snapshot(root: Path) -> tuple[dict, Path]:
    validate_epoch5_report(root)
    path = _local_file(root, SNAPSHOT_PATH)
    if sha256(path.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("Checkpoint V3 época 5 com SHA-256 divergente")
    import torch
    from src.core.neural.aoi_quality_cnn_v3 import MODEL_SCHEMA_V3
    from src.services.aoi_v3_epoch_diagnostics import EPOCH_SCHEMA
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    v = checkpoint.get("validation", {}) if isinstance(checkpoint, dict) else {}
    h = checkpoint.get("operator_training_metrics", {}) if isinstance(checkpoint, dict) else {}
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != EPOCH_SCHEMA
        or checkpoint.get("architecture_schema") != MODEL_SCHEMA_V3
        or checkpoint.get("epoch") != EPOCH
        or checkpoint.get("source_manifest_sha256") != MANIFEST_SHA256
        or checkpoint.get("previous_candidate_sha256") != PREVIOUS_EPOCH3_SHA256
        or checkpoint.get("image_size") != 96
        or checkpoint.get("focus_fraction") != .70
        or tuple(checkpoint.get("lights", ())) != LIGHTS
        or checkpoint.get("diagnostics_only") is not True
        or checkpoint.get("candidate_only") is not True
        or checkpoint.get("production_approved") is not False
        or checkpoint.get("automatic_xp_authorized") is not False
        or checkpoint.get("selected_by_safe_gate") is not False
        or checkpoint.get("test_evaluated_for_this_epoch") is not False
        or v.get("NG_false_OK") != 0 or v.get("OK_false_NG") != 2
        or h.get("correct") != 14 or h.get("NG_correct") != 9 or h.get("OK_correct") != 5
        or not isinstance(checkpoint.get("model_weights_not_production_state_dict"), dict)
    ):
        raise ValueError("Payload do checkpoint V3 época 5 não corresponde às evidências")
    return checkpoint, path


__all__ = [
    "EPOCH", "EPOCH5_POINTER", "EPOCH5_POINTER_SCHEMA", "EPOCH5_VOTE_SCHEMA",
    "REPORT_PATH", "REPORT_SHA256", "SNAPSHOT_PATH", "SNAPSHOT_SHA256",
    "epoch5_pointer_payload", "epoch5_pointer_enabled",
    "validate_epoch5_report", "validate_epoch5_snapshot",
]
