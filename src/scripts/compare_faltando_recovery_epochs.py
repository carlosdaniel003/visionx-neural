"""Compara 5 e 7 épocas da CNN FALTANDO V2, sem ativar pesos.

Cada rodada usa uma NOVA instância da rede, a MESMA base pinada verificada
e sementes reinicializadas por train_online(). Não há warm start 5 -> 7.
Não altera KNN, ponteiro ativo, histórico nem agente XP.

Uso somente com ODIN encerrado, em modo manutenção:
 python -m src.scripts.compare_faltando_recovery_epochs ^
   --root C:\visionx-neural-main ^
   --correction-id 20261013T175115Z_ebf986ef
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from src.services.faltando_cnn_recovery_preflight import (
    validate_recovery_inputs,
)

EPOCHS = (5, 7)
LIGHTS = ("SIDE", "TOP", "MID")
SUMMARY_SCHEMA = "visionx.faltando_recovery_comparison.v1"


def _hash(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Arquivo ausente ou inseguro: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _relative(root: Path, value: str, expected_folder: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Relatório sem caminho de candidato")
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
        raise ValueError("Caminho de candidato inseguro")
    path = (root / relative).resolve()
    if path.parent != expected_folder.resolve():
        raise ValueError("Candidato fora da pasta experimental")
    return path


def summarize_report(
    root: Path, correction_id: str, report: dict, *,
    epochs: int, base_sha256: str, required_events: set[str],
) -> dict:
    """Validação independente do relatório da rodada e suas 9 luzes NG."""
    if (report.get("schema") != "visionx.specialist_online_training_report.v1"
            or report.get("candidate_only") is not True
            or report.get("promoted") is not False
            or report.get("train_epochs") != epochs
            or report.get("recovery_correction_id") != correction_id
            or report.get("base_checkpoint_sha256") != base_sha256):
        raise ValueError(f"Relatório da rodada {epochs} incompatível")
    forced = report.get("forced_events_per_epoch") or []
    if len(forced) != 3 or set(forced) != required_events:
        raise ValueError(f"Rodada {epochs} não reforçou os três eventos corrigidos")
    online = report.get("online_candidate") or {}
    baseline = report.get("online_baseline") or {}
    historic = report.get("historical_candidate") or {}
    historic_old = report.get("historical_baseline") or {}
    cases = online.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Relatório do candidato sem resultados individuais completos")
    selected = {x["event_id_or_group_hint"]: x for x in cases
                if x.get("event_id_or_group_hint") in required_events}
    if len(selected) != len(required_events):
        raise ValueError("Resultados não cobrem os três eventos corrigidos")
    light_correct = {mode: 0 for mode in LIGHTS}
    individual = {}
    for event_id in sorted(required_events):
        row = selected[event_id]
        if row.get("label") != "NG":
            raise ValueError("Rótulo NG corrigido ausente no resultado")
        scores = row.get("per_light")
        if not isinstance(scores, dict) or set(scores) != set(LIGHTS):
            raise ValueError("Evento sem avaliação individual SIDE/TOP/MID")
        individual[event_id] = {}
        for mode in LIGHTS:
            data = scores[mode]
            score = data.get("score")
            if (not isinstance(score, (float, int)) or isinstance(score, bool)
                    or not 0 <= score <= 1 or
                    data.get("correct") is not (score >= 0.5)):
                raise ValueError("Score por iluminação inválido/incoerente")
            if data["correct"]:
                light_correct[mode] += 1
            individual[event_id][mode] = {
                "ng_score": round(float(score), 6),
                "correct": bool(data["correct"]),
            }
    if (len(forced) * 3 != 9
            or online.get("correct_images") is None
            or baseline.get("correct_images") is None
            or historic.get("correct_images") is None
            or historic.get("images") != historic_old.get("images")):
        raise ValueError("Métricas de regressão ausentes")
    destination = _relative(
        root, report.get("candidate"), root / "reports/neural_online/recovery_candidates"
    )
    actual_sha = _hash(destination)
    if actual_sha != report.get("candidate_sha256"):
        raise ValueError("Checkpoint candidato não corresponde ao SHA-256 informado")
    report_path = root / "reports/neural_online/training_reports" / (
        destination.stem + ".json"
    )
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    if saved != report:
        raise ValueError("Relatório gravado em disco difere da avaliação obtida")
    return {
        "epochs": epochs,
        "training_events": report.get("training_event_count"),
        "historical": {
            "correct_images": historic["correct_images"],
            "total_images": historic["images"],
            "correct_events": historic.get("correct_events"),
            "total_events": historic.get("events"),
            "ng_false_ok": historic.get("ng_false_ok"),
        },
        "online_all_events": {
            "correct_images": online["correct_images"],
            "total_images": online.get("images"),
            "correct_events": online.get("correct_events"),
            "total_events": online.get("events"),
            "ng_false_ok": online.get("ng_false_ok"),
        },
        "correction_only": {
            "images_correct": sum(light_correct.values()),
            "images_total": 9,
            "correct_by_light": light_correct,
            "scores_by_event": individual,
        },
        "gates": report.get("gates"),
        "passed_internal_gates": report.get("passed_internal_gates"),
        "promoted": False,
        "report_path": str(report_path.relative_to(root)),
        "candidate_path": str(destination.relative_to(root)),
        "candidate_sha256": actual_sha,
    }


def run_comparison(
    root: Path | str, correction_id: str, *, trainer=None,
) -> dict:
    """Treinamento sequencial no modo offline; nunca promove resultados."""
    root = Path(root).resolve()
    verified = validate_recovery_inputs(root, correction_id)
    from src.core.neural.faltando_live import (
        PINNED_CHECKPOINT_SHA256, PINNED_RELATIVE_CHECKPOINT,
    )
    base_path = root / PINNED_RELATIVE_CHECKPOINT
    original_sha = _hash(base_path)
    if original_sha != PINNED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint original não confere com SHA-256 fixado")
    active = root / "reports/neural_online/live_active.json"
    if active.exists():
        raise ValueError("Ponteiro de pesos online presente; interrompido para auditoria")
    if trainer is None:
        from src.scripts.train_faltando_cnn_v2_online import train_online
        trainer = train_online
    events = verified["events"]
    event_id_set = {item.stem for item in events}
    results = {}
    for epochs in EPOCHS:
        # Checar integridade novamente ANTES da nova execução e não dar
        # continuidade a dados parcialmente alterados entre as rodadas.
        validate_recovery_inputs(root, correction_id)
        if active.exists() or _hash(base_path) != original_sha:
            raise ValueError("Modelo ativo ou checkpoint original mudou entre execuções")
        print(
            f"\n=== CNN FALTANDO: INÍCIO DA RODADA INDEPENDENTE {epochs} ÉPOCAS ===",
            flush=True,
        )
        # train_online faz torch.manual_seed(42), np.random.seed(42),
        # random.seed(42) e carrega estado do checkpoint original de novo.
        report = trainer(
            root, events[0], epochs=epochs, candidate_only=True,
            recovery_correction_id=correction_id,
        )
        if active.exists() or _hash(base_path) != original_sha:
            raise ValueError("Tentativa de alteração do checkpoint base ou do ponteiro")
        results[str(epochs)] = summarize_report(
            root, correction_id, report, epochs=epochs,
            base_sha256=original_sha, required_events=event_id_set,
        )
        print(
            f"RODADA {epochs} ÉPOCAS: "
            f"{results[str(epochs)]['correction_only']['images_correct']}/9 NG; "
            "MID "
            f"{results[str(epochs)]['correction_only']['correct_by_light']['MID']}/3; "
            f"gate interno={results[str(epochs)]['passed_internal_gates']}; "
            "promovido=False",
            flush=True,
        )
        gc.collect()

    # Sem campeão automático: mesmo 9/9 do treino não prova generalização.
    result = {
        "schema": SUMMARY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "correction_id": correction_id,
        "epochs_compared": list(EPOCHS),
        "independent_from_original_checkpoint": True,
        "original_checkpoint_sha256": original_sha,
        "three_verified_event_ids": sorted(event_id_set),
        "candidate_only": True,
        "active_model_changed": False,
        "production_approved": False,
        "experiments": results,
        "note": (
            "Replay nos mesmos exemplos de treino não mede generalização; "
            "não alterar checkpoint ativo sem teste independente e autorização."
        ),
    }
    out = root / "reports" / "neural_online" / "recovery_comparisons"
    name = "comparacao_5_7_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    ) + "_" + uuid4().hex[:6]
    destination = out / (name + ".json")
    _atomic_json(destination, result)
    print(f"\nCOMPARACAO FINAL: {destination}", flush=True)
    print("PESOS ATIVOS INALTERADOS; NENHUM CANDIDATO PROMOVIDO.", flush=True)
    return {"report_path": str(destination), **result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Duas rodadas independentes de 5 e 7 épocas sem promoção"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--correction-id", required=True)
    args = parser.parse_args(argv)
    run_comparison(args.root, args.correction_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
