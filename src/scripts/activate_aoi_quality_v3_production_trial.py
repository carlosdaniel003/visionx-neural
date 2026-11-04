"""Ativa/desativa localmente a V3 época 3 como veto NG reversível.

Executar com ODIN/XP fechados. NÃO substitui live_active.json,
não renomeia checkpoints e NÃO autoriza PRESS_0 pela CNN V3.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from src.core.neural.aoi_quality_cnn_v3_production_trial import (
    INITIAL_V3_SHA256, POINTER_SCHEMA, SNAPSHOT_RELATIVE_PATH,
    SNAPSHOT_SHA256, SOURCE_MANIFEST_SHA256, TRIAL_POINTER,
    trial_pointer_payload, validate_trial_snapshot,
)

REPORT_RELATIVE_PATH = (
    "reports/aoi_quality_cnn_v3/candidates/"
    "20261104T192753_047124Z_6b733a/training_report.json"
)


def verify_epoch3_report(root: Path) -> dict:
    root = Path(root).resolve()
    path = root / REPORT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("Relatório original da época 3 ausente")
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report.get("epoch_history") if isinstance(report, dict) else None
    snapshots = report.get("epoch_diagnostic_checkpoints") if isinstance(
        report, dict) else None
    if (
        report.get("schema") != "visionx.aoi_quality_cnn_v3_training_report.v1"
        or report.get("candidate_only") is not True
        or report.get("production_approved") is not False
        or report.get("automatic_xp_authorized") is not False
        or report.get("selected_epoch") != 0
        or report.get("operator_image_review_verified") != 14
        or report.get("aoi_category_used_for_training_selection_evaluation")
           is not False
        or report.get("provenance", {}).get("manifest_sha256")
           != SOURCE_MANIFEST_SHA256
        or report.get("initial_v3_candidate", {}).get("candidate_sha256")
           != INITIAL_V3_SHA256
        or not isinstance(history, list) or len(history) != 7
        or not isinstance(snapshots, list) or len(snapshots) != 6
    ):
        raise ValueError("Treino V3 não corresponde ao relatório revisado")
    previous, third = history[0], history[3]
    if (
        previous.get("epoch") != 0
        or previous.get("validation", {}).get("NG_false_OK") != 3
        or previous.get("validation", {}).get("OK_false_NG") != 0
        or previous.get("operator_training_replay", {}).get(
            "metrics", {}).get("correct") != 0
        or third.get("epoch") != 3
        or third.get("safe_gate_passed") is not True
        or third.get("validation", {}).get("NG_false_OK") != 3
        or third.get("validation", {}).get("OK_false_NG") != 0
        or third.get("operator_training_replay", {}).get(
            "metrics", {}).get("NG_correct") != 8
        or third.get("operator_training_replay", {}).get(
            "metrics", {}).get("OK_correct") != 5
        or third.get("operator_training_replay", {}).get(
            "metrics", {}).get("correct") != 13
        or snapshots[2].get("epoch") != 3
        or snapshots[2].get("path") != SNAPSHOT_RELATIVE_PATH
        or snapshots[2].get("sha256") != SNAPSHOT_SHA256
        or snapshots[2].get("production_approved") is not False
        or snapshots[2].get("automatic_xp_authorized") is not False
    ):
        raise ValueError("Evidência da época 3 ou seu SHA divergiram")
    return report


def activate(root: Path) -> dict:
    root = Path(root).resolve()
    verify_epoch3_report(root)
    validate_trial_snapshot(root)
    pointer = root / TRIAL_POINTER
    payload = trial_pointer_payload()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    if pointer.exists() or pointer.is_symlink():
        if pointer.is_symlink() or json.loads(
            pointer.read_text(encoding="utf-8")
        ) != payload:
            raise ValueError("Ponteiro V3 já existe e diverge; não sobrescrever")
        return payload
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp",
            prefix=".v3_trial_", dir=pointer.parent, delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, pointer)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return payload


def deactivate(root: Path) -> bool:
    path = Path(root).resolve() / TRIAL_POINTER
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or json.loads(
        path.read_text(encoding="utf-8")
    ) != trial_pointer_payload():
        raise ValueError("Ponteiro V3 inválido; intervenção manual necessária")
    path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--activate", action="store_true")
    action.add_argument("--deactivate", action="store_true")
    args = parser.parse_args()
    if args.activate:
        payload = activate(args.root)
        print(json.dumps({
            "status": "ENABLED_NG_VETO_ONLY",
            "checkpoint_epoch": 3,
            "sha256": SNAPSHOT_SHA256,
            "automatic_ok_from_v3_authorized": False,
            "pointer": str(Path(args.root).resolve() / TRIAL_POINTER),
            "source_production_v2_unchanged": True,
        }, indent=2))
    else:
        removed = deactivate(args.root)
        print(json.dumps({
            "status": "DISABLED",
            "pointer_removed": removed,
            "v2_production_unchanged": True,
        }, indent=2))


if __name__ == "__main__":
    main()
