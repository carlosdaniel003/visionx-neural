"""Auditoria da CNN V3 por época — treino humano separado da validação.

Somente registros de treino explicitamente confirmados podem entrar aqui.
Esta avaliação NÃO é holdout, não produz rótulos para a CNN e não habilita XP.
"""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path

import torch

from src.core.neural.aoi_quality_cnn_v3 import MODEL_SCHEMA_V3
from src.services.aoi_v3_binary_quality_metrics import binary_quality_metrics

EPOCH_SCHEMA = "visionx.aoi_quality_cnn_v3_epoch_diagnostic_only.v1"


def operator_epoch_replay(records: list[dict], predictions: dict) -> dict:
    """Confere os 14 julgamentos por época, sem contar treino como teste."""
    if (
        len(records) != 14
        or any(row.get("label_provenance") != "HUMAN_V3_IMAGE_SHA256"
               for row in records)
        or sum(row.get("label") == "NG" for row in records) != 9
        or sum(row.get("label") == "OK" for row in records) != 5
    ):
        raise ValueError("Replay CNN V3 exige 9 NG e 5 OK humanos no treino")
    metrics = binary_quality_metrics(records, predictions)
    cases = predictions["cases"]
    errors = []
    confidence = 0.
    log_loss = 0.
    for row, item in zip(records, cases):
        score = item["ng_score"]
        if (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0. <= score <= 1.
        ):
            raise ValueError("Score de supervisão humana inválido")
        truth_probability = score if row["label"] == "NG" else 1.-score
        confidence += truth_probability
        log_loss -= math.log(max(1.e-6, truth_probability))
        if not item["correct"]:
            errors.append({
                "source": row["sources"][0],
                "expected": row["label"],
                "predicted": item["predicted"],
                "ng_score": score,
            })
    return {
        "metrics": metrics,
        "cases": cases,
        "incorrect_sources": errors,
        "mean_true_label_score": round(confidence / len(cases), 8),
        "mean_true_label_log_loss": round(log_loss / len(cases), 8),
        "in_sample_not_independent_test": True,
        "must_not_be_reported_as_production_accuracy": True,
    }


def epoch_review_comparison(
    baseline_validation: dict, validation: dict,
    baseline_replay: dict, replay: dict,
) -> dict:
    """Separar progresso nos 14 humanos de regressões nos folds externos."""
    ng_delta = validation["NG_false_OK"] - baseline_validation["NG_false_OK"]
    ok_delta = validation["OK_false_NG"] - baseline_validation["OK_false_NG"]
    human_delta = replay["metrics"]["correct"] - baseline_replay["metrics"]["correct"]
    baseline_cases = {
        item["event_id_or_group_hint"]: item
        for item in baseline_replay["cases"]
    }
    new_cases = {
        item["event_id_or_group_hint"]: item for item in replay["cases"]
    }
    if set(baseline_cases) != set(new_cases) or len(new_cases) != 14:
        raise ValueError("Replay por época não corresponde às 14 imagens")
    became_correct = sorted(
        key for key, item in new_cases.items()
        if item["correct"] and not baseline_cases[key]["correct"]
    )
    became_wrong = sorted(
        key for key, item in new_cases.items()
        if not item["correct"] and baseline_cases[key]["correct"]
    )
    return {
        "human_correct_delta": human_delta,
        "human_mean_true_score_delta": round(
            replay["mean_true_label_score"] -
            baseline_replay["mean_true_label_score"], 8,
        ),
        "human_log_loss_delta": round(
            replay["mean_true_label_log_loss"] -
            baseline_replay["mean_true_label_log_loss"], 8,
        ),
        "human_became_correct": became_correct,
        "human_became_wrong": became_wrong,
        "NG_false_OK_change_on_validation": ng_delta,
        "OK_false_NG_change_on_validation": ok_delta,
        "validation_non_regression_gate_passed": ng_delta <= 0 and ok_delta <= 0,
        "validation_errors_worse": [
            label for label, delta in (
                ("NG_TO_OK", ng_delta), ("OK_TO_NG", ok_delta)
            ) if delta > 0
        ],
        "human_training_examples_only": True,
    }


def best_exploratory_epoch(history: list[dict]) -> int | None:
    """Seleciona diagnóstico POR TREINO HUMANO+VALIDAÇÃO, nunca pelo teste.

    Não participa da seleção do checkpoint conservador. Se nenhuma época
    cruzou o limiar de classificação em mais amostras humanas, retorna None.
    """
    baseline = history[0]["operator_training_replay"]["metrics"]["correct"]
    candidates = [item for item in history[1:]
                  if item["operator_training_replay"]["metrics"]["correct"] > baseline]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        item["operator_training_replay"]["metrics"]["correct"],
        -len(item["comparison_with_epoch_zero"]["human_became_wrong"]),
        item["validation"]["balanced_accuracy"],
        -item["validation"]["NG_false_OK"],
        -item["validation"]["OK_false_NG"],
        -item["epoch"],
    ))["epoch"]


def persist_epoch_diagnostics(
    folder: Path, snapshots: list[tuple[int, dict]],
    history: list[dict], *, manifest_sha256: str,
    previous_candidate_sha256: str | None, image_size: int,
    safe_selected_epoch: int,
) -> list[dict]:
    """Retém cada época treinada com schema incompatível com Produção.

    Somente dados/weights de TREINO e VALIDAÇÃO são serializados; o
    teste separado nunca é executado para essas variantes experimentais.
    """
    if (
        len(snapshots) != len(history)-1
        or {number for number, _ in snapshots} !=
           {row["epoch"] for row in history[1:]}
        or any(row["epoch"] != idx for idx, row in enumerate(history))
    ):
        raise ValueError("Snapshots não correspondem às épocas avaliadas")
    directory = folder / "epoch_diagnostics"
    directory.mkdir(exist_ok=False)
    records = []
    for number, state in snapshots:
        if number < 1:
            raise ValueError("Época diagnóstica inválida")
        assessment = history[number]
        filename = directory / f"epoch_{number:02d}_experimental_only.pt"
        torch.save({
            "schema": EPOCH_SCHEMA,
            "architecture_schema": MODEL_SCHEMA_V3,
            "model_weights_not_production_state_dict": state,
            "epoch": number,
            "source_manifest_sha256": manifest_sha256,
            "previous_candidate_sha256": previous_candidate_sha256,
            "image_size": image_size,
            "focus_fraction": .70,
            "lights": ("SIDE", "TOP", "MID"),
            "validation": assessment["validation"],
            "operator_training_metrics":
                assessment["operator_training_replay"]["metrics"],
            "in_sample_not_independent_test": True,
            "test_evaluated_for_this_epoch": False,
            "selected_by_safe_gate": number == safe_selected_epoch,
            "diagnostics_only": True,
            "candidate_only": True,
            "production_approved": False,
            "automatic_xp_authorized": False,
        }, filename)
        records.append({
            "epoch": number,
            "path": filename.relative_to(folder.parent.parent.parent.parent).as_posix(),
            "sha256": sha256(filename.read_bytes()).hexdigest(),
            "operator_14_correct": assessment["operator_training_replay"][
                "metrics"]["correct"],
            "human_correct_delta": assessment["comparison_with_epoch_zero"][
                "human_correct_delta"],
            "validation_NG_false_OK": assessment["validation"]["NG_false_OK"],
            "validation_OK_false_NG": assessment["validation"]["OK_false_NG"],
            "safe_gate_selected": number == safe_selected_epoch,
            "test_evaluated_for_this_epoch": False,
            "production_approved": False,
            "automatic_xp_authorized": False,
        })
    return records


__all__ = [
    "EPOCH_SCHEMA", "operator_epoch_replay", "epoch_review_comparison",
    "best_exploratory_epoch", "persist_epoch_diagnostics",
]
