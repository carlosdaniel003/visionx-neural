"""Experimento CNN FALTANDO V2: pesos de perda por luz, SEM promover pesos.

Executar apenas com ODIN em manutenção:
 python -m src.scripts.compare_faltando_recovery_light_weights ^
     --root C:\visionx-neural-main ^
     --correction-id 20261013T175115Z_ebf986ef

Três rodadas independentes: peso auxiliar 0.15 (controle), 0.50, 1.00;
todas 7 épocas, seed 42, checkpoint original íntegro e os mesmos eventos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from src.scripts.compare_faltando_recovery_epochs import (
    _atomic_json, _hash, summarize_report,
)
from src.services.faltando_cnn_recovery_preflight import validate_recovery_inputs

SCHEMA = "visionx.faltando_recovery_light_weight_comparison.v1"
WEIGHTS = (0.15, 0.5, 1.0)
EPOCHS = 7


def _key(weight: float) -> str:
    return f"{weight:.2f}"


def compare_light_weights(
    root: Path | str, correction_id: str, *, trainer=None,
) -> dict:
    """Compara três pesos sem qualquer seleção ou promoção de checkpoint."""
    root = Path(root).resolve()
    verified = validate_recovery_inputs(root, correction_id)
    from src.core.neural.faltando_live import (
        PINNED_CHECKPOINT_SHA256, PINNED_RELATIVE_CHECKPOINT,
    )
    original_path = root / PINNED_RELATIVE_CHECKPOINT
    original_sha = _hash(original_path)
    if original_sha != PINNED_CHECKPOINT_SHA256:
        raise ValueError("O checkpoint original não corresponde ao SHA-256 fixado")
    active = root / "reports/neural_online/live_active.json"
    if active.exists():
        raise ValueError(
            "Há checkpoint online ativo: bloquear comparação até auditoria"
        )
    if trainer is None:
        from src.scripts.train_faltando_cnn_v2_online import train_online
        trainer = train_online

    event_files = verified["events"]
    forced_ids = {p.stem for p in event_files}
    if len(forced_ids) != 3:
        raise ValueError("A seleção de eventos corrigidos não é única")
    experiments = {}
    for weight in WEIGHTS:
        # Preflight e SHA repetidos para não continuar se dados mudaram.
        refreshed = validate_recovery_inputs(root, correction_id)
        if {p.stem for p in refreshed["events"]} != forced_ids:
            raise ValueError("Conjunto supervisionado mudou entre rodadas")
        if active.exists() or _hash(original_path) != original_sha:
            raise ValueError("Ponteiro ativo ou checkpoint base modificado")
        print(
            f"\n=== EXPERIMENTO: PERDA POR ILUMINAÇÃO {_key(weight)}, "
            f"{EPOCHS} ÉPOCAS, BASE ORIGINAL ===",
            flush=True,
        )
        # Toda chamada recria o modelo e reinicia as sementes. Nunca
        # reutilizar checkpoint gerado na rodada imediatamente anterior.
        report = trainer(
            root, event_files[0], epochs=EPOCHS, candidate_only=True,
            recovery_correction_id=correction_id, light_loss_weight=weight,
        )
        if active.exists() or _hash(original_path) != original_sha:
            raise ValueError(
                "O modelo ativo/base mudou durante a comparação; resultado inválido"
            )
        if (report.get("light_loss_weight") != weight
                or report.get("candidate_only") is not True
                or report.get("promoted") is not False):
            raise ValueError("O treinamento não respeitou peso e modo experimental")
        summary = summarize_report(
            root, correction_id, report, epochs=EPOCHS,
            base_sha256=original_sha, required_events=forced_ids,
        )
        # Comparar sem supor que 9 acertos treinados = desempenho inédito.
        summary["light_loss_weight"] = weight
        experiments[_key(weight)] = summary
        by_light = summary["correction_only"]["correct_by_light"]
        print(
            f"PESO {_key(weight)}: acertos "
            f"{summary['correction_only']['images_correct']}/9; "
            f"SIDE={by_light['SIDE']}/3, TOP={by_light['TOP']}/3, "
            f"MID={by_light['MID']}/3; "
            f"gate={summary['passed_internal_gates']}; PROMOVIDO=FALSE",
            flush=True,
        )
        gc.collect()

    if set(experiments) != {_key(w) for w in WEIGHTS}:
        raise ValueError("A comparação não completou todas as três rodadas")
    output = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "correction_id": correction_id,
        "epochs_per_experiment": EPOCHS,
        "auxiliary_loss_weights": list(WEIGHTS),
        "independent_from_original_checkpoint": True,
        "original_checkpoint_sha256": original_sha,
        "verified_event_ids": sorted(forced_ids),
        "candidate_only": True,
        "active_model_changed": False,
        "production_approved": False,
        "automatic_selection": False,
        "experiments": experiments,
        "note": (
            "Resultados de replay incluem imagens usadas no treinamento. "
            "Não escolher ou promover pesos para Produção por esta comparação. "
            "Exigir holdout por peça e inspeção em casos inéditos."
        ),
    }
    out = root / "reports/neural_online/recovery_comparisons"
    filename = (
        "comparacao_pesos_iluminacao_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6] + ".json"
    )
    destination = out / filename
    _atomic_json(destination, output)
    print(f"\nCOMPARAÇÃO CONCLUÍDA: {destination}", flush=True)
    print("Nenhum candidato foi promovido. Modelo ativo preservado.", flush=True)
    return {"report_path": str(destination), **output}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Comparar pesos de perda por iluminação, somente candidatos"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--correction-id", type=str, required=True)
    options = parser.parse_args(argv)
    compare_light_weights(options.root, options.correction_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
