"""Controlled, reversible activation of the SHA-pinned CNN V2 97.70% candidate.

IMPORTANT: the model FAILED historic OK regression (98/117). This activates
a HUMAN-REVIEW-ONLY trial, not production-approved autonomous decisions.
It does not overwrite or mutate either checkpoint; uses existing live pointer
loader. The machine's ODIN application must be closed for install/rollback.
No XP, KNN, label, mask, threshold or training changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from src.core.neural.faltando_checkpoint_contract import (
    POINTER_SCHEMA, TRIAL_97P70_CHECKPOINT, TRIAL_97P70_MODE,
    TRIAL_97P70_SHA256,
)
from src.core.neural.faltando_live import (
    FaltandoCNNLive, PINNED_CHECKPOINT_SHA256,
    PINNED_RELATIVE_CHECKPOINT,
)
from src.scripts.faltando_cnn_v2_blend_candidates import (
    _inside, _verified_candidate,
)

ORIGINAL_REPORT = (
    "reports/faltando_neural/archive_experiments/"
    "20261016T125315_996834Z_c3e70a/training_report.json"
)
BENCHMARK_REPORT = (
    "reports/aoi_universal_quality/benchmarks/"
    "20261016T181313_610125Z_e527c1/head_to_head_report.json"
)
POINTER = Path("reports/neural_online/live_active.json")
AUDITS = Path("reports/neural_online/manual_trial_audits")
ACK = "CONFIRMO_19_FALSOS_NG_E_REVISAO_MANUAL"
POINTER_BACKUP = Path("reports/neural_online/manual_trial_audits")
SCHEMA = "visionx.faltando_97p70_controlled_trial.v1"


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Arquivo ausente ou symlink: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def _read_benchmark(root: Path) -> dict:
    path = _inside(
        root, root / BENCHMARK_REPORT,
        root/"reports/aoi_universal_quality/benchmarks",
    )
    if path.name != "head_to_head_report.json":
        raise ValueError("Benchmark ilegítimo")
    data = json.loads(path.read_text(encoding="utf-8"))
    old = data.get("original_97p70_on_same_test", {})
    original = data.get("first_experiment_original_holdout", {})
    if (
        data.get("schema") != "visionx.aoi_head_to_head_97p70_readonly.v1"
        or data.get("first_97p70_candidate_sha256") != TRIAL_97P70_SHA256
        or data.get("comparison_only") is not True
        or data.get("trained") is not False
        or data.get("promoted") is not False
        or data.get("test_split_exactly_matched_original") is not True
        or old.get("total") != 106
        or old.get("correct") != 93
        or old.get("NG_total") != 63
        or old.get("NG_correct") != 63
        or old.get("NG_false_OK") != 0
        or old.get("OK_false_NG") != 13
        or original.get("correct") != 85 or original.get("total") != 87
        or original.get("historical_correct") != 98
        or original.get("historical_total") != 117
        or original.get("historical_false_ng_count") != 19
    ):
        raise ValueError("Benchmark 106/87/histórico não corresponde ao modelo fixado")
    return data


def _pointer_path(root: Path) -> Path:
    return root / POINTER


def _audit(root: Path, action: str, details: dict) -> Path:
    directory = root / AUDITS
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6] + "_" + action + ".json"
    )
    path.write_text(
        json.dumps({
            "schema": SCHEMA, "event": action,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with tmp.open("x", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        if path.exists() or path.is_symlink():
            raise ValueError("Ponteiro mudou durante a instalação")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _validate_original_station(root: Path) -> None:
    base = root / PINNED_RELATIVE_CHECKPOINT
    if _sha(base) != PINNED_CHECKPOINT_SHA256:
        raise ValueError("Modelo original instalado foi alterado/está ausente")


def _read_trial_pointer(root: Path) -> dict:
    path = _pointer_path(root)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Ponteiro de teste não encontrado ou symlink")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema") != POINTER_SCHEMA
        or data.get("category") != "FALTANDO"
        or data.get("checkpoint_relative_path") != TRIAL_97P70_CHECKPOINT
        or data.get("checkpoint_sha256") != TRIAL_97P70_SHA256
        or data.get("deployment_mode") != TRIAL_97P70_MODE
        or data.get("experimental") is not True
        or data.get("production_approved") is not False
        or data.get("manual_review_required") is not True
        or data.get("automatic_ok_authorized") is not False
        or data.get("historical_ok_regression_acknowledged") is not True
    ):
        raise ValueError("Ponteiro é de outra implantação: abortar sem sobrescrever")
    if _sha(root / TRIAL_97P70_CHECKPOINT) != TRIAL_97P70_SHA256:
        raise ValueError("Checkpoint ativo diferente do hash pinado")
    return data


def _validate_historical_light_regression(hold: dict, hist: dict) -> dict:
    """Check 117 individual *lighting views*, not 117 event records.

    _assess stores one case per event. A single event may hold SIDE, TOP
    and MID, so len(cases) can be much smaller than hist["images"].
    Each per_light.correct is the existing evaluation of that view, with
    its original event quality label. Do not invent per-light labels.
    """
    cases = hist.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Casos históricos ausentes ou inválidos")
    inspected = correct = ok_false_ng = ng_false_ok = 0
    for event in cases:
        if not isinstance(event, dict) or event.get("label") not in ("OK", "NG"):
            raise ValueError("Rótulo histórico do evento inválido")
        per_light = event.get("per_light")
        if not isinstance(per_light, dict) or not per_light:
            raise ValueError("Evento histórico sem observações por iluminação")
        if not set(per_light).issubset({"SIDE", "TOP", "MID"}):
            raise ValueError("Iluminação histórica desconhecida")
        for view in per_light.values():
            if not isinstance(view, dict) or type(view.get("correct")) is not bool:
                raise ValueError("Resultado histórico por luz inválido")
            inspected += 1
            correct += int(view["correct"])
            if not view["correct"]:
                if event["label"] == "OK":
                    ok_false_ng += 1
                else:
                    ng_false_ok += 1
    if (
        hold.get("correct_images") != 85 or hold.get("images") != 87
        or hold.get("ng_false_ok") != 0
        or hist.get("images") != 117
        or hist.get("correct_images") != 98
        or inspected != 117 or correct != 98
        or ok_false_ng != 19 or ng_false_ok != 0
    ):
        raise ValueError(
            "Gates históricos do candidato não correspondem às 117 luzes: "
            f"avaliadas={inspected}, corretas={correct}, "
            f"OK→NG={ok_false_ng}, NG→OK={ng_false_ok}"
        )
    return {
        "historical_events": len(cases),
        "historical_lights": inspected,
        "historical_correct_lights": correct,
        "historical_ok_false_ng_lights": ok_false_ng,
        "historical_ng_false_ok_lights": ng_false_ok,
    }


def _install(root: Path) -> dict:
    pointer = _pointer_path(root)
    if pointer.exists() or pointer.is_symlink():
        raise ValueError("Ponteiro online já existe: não sobrescrever nenhuma implantação")
    _validate_original_station(root)
    previous, checkpoint = _verified_candidate(root, root/ORIGINAL_REPORT)
    if previous["candidate_sha256"] != TRIAL_97P70_SHA256:
        raise ValueError("O checkpoint original não é o de 97,70%")
    hold = previous.get("new_holdout", {})
    hist = previous.get("new_historical", {})
    historical_regression = _validate_historical_light_regression(hold, hist)
    _read_benchmark(root)
    source = root/previous["candidate_relative_path"]
    target = root/TRIAL_97P70_CHECKPOINT
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("O destino do checkpoint é symlink")
    if target.exists():
        if _sha(target) != TRIAL_97P70_SHA256:
            raise ValueError("Já há outro checkpoint no destino do ensaio")
    else:
        tmp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        try:
            with source.open("rb") as inp, tmp.open("xb") as out:
                shutil.copyfileobj(inp, out)
                out.flush()
                os.fsync(out.fileno())
            if _sha(tmp) != TRIAL_97P70_SHA256:
                raise ValueError("Cópia do checkpoint não corresponde ao hash")
            if target.exists() or target.is_symlink():
                raise ValueError("Destino ocupado durante cópia")
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()
    if _sha(target) != TRIAL_97P70_SHA256:
        raise ValueError("SHA-256 do checkpoint copiado incorreto")
    # Execute EXACT production loader before publishing the active pointer.
    engine = FaltandoCNNLive(
        checkpoint_path=target,
        expected_sha256=TRIAL_97P70_SHA256,
        online_root=root,
    )
    warmup = engine.warmup()
    if warmup.get("state") != "ready":
        raise ValueError("Falha no warmup do checkpoint: " + str(warmup))
    row = {
        "schema": POINTER_SCHEMA,
        "category": "FALTANDO",
        "checkpoint_relative_path": TRIAL_97P70_CHECKPOINT,
        "checkpoint_sha256": TRIAL_97P70_SHA256,
        "experimental": True, "production_approved": False,
        "deployment_mode": TRIAL_97P70_MODE,
        "manual_review_required": True,
        "automatic_ok_authorized": False,
        "historical_ok_regression_acknowledged": True,
        "manual_trial_source": "OPERATOR_REQUESTED_CNN_97P70_REPLACEMENT",
        "original_historical_correct": 98,
        "original_historical_total": 117,
        "original_historical_false_ng": 19,
    }
    _atomic_json(pointer, row)
    audit = _audit(root, "installed", {
        "active_sha256": TRIAL_97P70_SHA256,
        "original_sha256_unchanged": PINNED_CHECKPOINT_SHA256,
        "warmup": warmup,
        "manual_review_only": True,
        "automatic_ok_and_ng_disabled": True,
        "old_checkpoint_retained": True,
        "historical_regression_checked_per_light": historical_regression,
    })
    print("CNN 97,70% ATIVA EM ENSAIO ASSISTIDO (requer reiniciar ODIN)")
    print("SHA:", TRIAL_97P70_SHA256)
    print("REVISÃO HUMANA OBRIGATÓRIA — SEM 0/1 AUTOMÁTICO")
    print("Rollback: python -X utf8 -m src.scripts.deploy_faltando_97p70_trial rollback")
    print("Auditoria:", audit)
    return {"status": "trial_active", "audit": str(audit)}


def _rollback(root: Path) -> dict:
    _read_trial_pointer(root)
    _validate_original_station(root)
    current = _pointer_path(root)
    dest = (root / AUDITS / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6] + "_rolled_back_live_active.json"
    ))
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(current, dest)
    audit = _audit(root, "rolled_back", {
        "restored_checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "trial_checkpoint_preserved_for_forensics": True,
        "pointer_archived_at": dest.relative_to(root).as_posix(),
        "requires_application_restart": True,
    })
    print("ROLLBACK CONCLUÍDO: checkpoint original será usado no próximo início.")
    print("REINICIE O ODIN (nenhuma mudança a quente permitida).")
    print("Auditoria:", audit)
    return {"status": "baseline_restored_after_restart", "audit": str(audit)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Ativação assistida ou rollback da CNN 97,70%, sem auto-OK"
    )
    parser.add_argument("action", choices=("install", "status", "rollback"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--ack", default="",
        help=("Obrigatório no install: " + ACK),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action == "install":
        if args.ack != ACK:
            raise ValueError(
                "Recusa ativação sem confirmação explícita da regressão: "
                + ACK
            )
        _install(root)
    elif args.action == "rollback":
        _rollback(root)
    else:
        pointer = _pointer_path(root)
        if pointer.exists() or pointer.is_symlink():
            _read_trial_pointer(root)
            print("STATUS: ensaio CNN 97,70% ativo, SEM AUTO-OK/AUTO-NG")
        else:
            _validate_original_station(root)
            print("STATUS: checkpoint original ativo (sem ponteiro de ensaio)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
