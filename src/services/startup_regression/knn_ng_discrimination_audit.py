"""Auditoria NG histórica read-only: vizinhos desbalanceados, duplicatas, revisão.

Reutiliza EXATAMENTE compare_anomaly_signatures e o voto inverso da
KNN existente no cenário baseline. As alternativas são experiências,
NÃO mudanças do KNN operacional, nem seleção calibrada de limiares.

Cada consulta exclui o próprio registro e event_id compartilhado. Para
testes sem duplicação também exclui todas as cópias da assinatura da
consulta e conta cada hash de vizinho no máximo uma vez. Categoria e
iluminação são obrigatoriamente idênticas (memória estrita).

Os rótulos são utilizados como supervisão de auditoria, não como atalho
de votação. A avaliação por registros NÃO mede os 212 PNGs nem prova
generalização fora da base; grupos visuais idênticos podem vazar entre
capturas diferentes sem event_id declarado.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.anomaly_signature import (
    compare_anomaly_signatures, valid_anomaly_signature,
)
from src.core.experts.knn_expert import KNNExpert
from .legacy_knn_signature_audit import load_signature_records

SCHEMA = "visionx.knn_ng_discrimination_audit.v1"
MODES = (
    "BASELINE_TOP5",
    "SEM_DUPLICATAS_TOP5",
    "BALANCEADO_3_POR_CLASSE",
    "BALANCEADO_SEM_DUPLICATAS",
    "BALANCEADO_COM_REVISAO",
)
EPSILON = 1e-9


def _signature_hash(signature: dict) -> str:
    vector = np.asarray(signature["vector"], dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _validate(rows: list[dict]) -> tuple[list[dict], dict]:
    eligible = []
    rejection = Counter()
    seen = set()
    for raw in rows:
        if raw.get("status") != "ELEGIVEL":
            rejection[str(raw.get("status") or "INVALIDO")] += 1
            continue
        if not valid_anomaly_signature(raw.get("signature")):
            raise ValueError("Registro ELEGIVEL com assinatura inválida")
        if raw.get("label") not in ("OK", "NG"):
            raise ValueError("Registro ELEGIVEL sem rótulo OK/NG")
        if not raw.get("category") or raw.get("lighting_mode") not in (
            "SIDE", "TOP", "MID"
        ):
            raise ValueError("Registro ELEGIVEL com categoria/luz inválida")
        path = str(raw.get("path") or "")
        if not path or path in seen:
            raise ValueError("Caminho vazio/duplicado na auditoria")
        seen.add(path)
        row = dict(raw)
        row["signature_hash"] = _signature_hash(row["signature"])
        eligible.append(row)
    return eligible, dict(sorted(rejection.items()))


def _weighted_vote(neighbors: list[tuple]) -> float:
    """A mesma fórmula de votação por distância da KNN de produção."""
    return float(KNNExpert._weighted_vote(
        [(float(d), label) for d, label, _p, _h in neighbors]
    ))


def _balanced_vote(neighbors: list[tuple], n_per_label: int):
    by_label = {
        label: [r for r in neighbors if r[1] == label][:n_per_label]
        for label in ("OK", "NG")
    }
    if not all(by_label.values()):
        return None, list(by_label["OK"] + by_label["NG"])
    # Evidência média por classe, NÃO soma bruta: evita 891 OK contra
    # apenas 61 NG somarem mais votos apenas por quantidade de cópias.
    mean_weight = {
        label: float(np.mean([
            1.0 / max(float(r[0]), 0.0001) for r in selected
        ]))
        for label, selected in by_label.items()
    }
    denom = mean_weight["OK"] + mean_weight["NG"]
    score = mean_weight["NG"] / denom if denom else 0.5
    return float(np.clip(score, 0., 1.)), (
        by_label["OK"] + by_label["NG"]
    )


def _without_duplicates(current: dict, pool: list[tuple]):
    """Exclusão de todas cópias da consulta e colapso dos hashes vizinhos."""
    selected = []
    by_hash = {}
    query_hash = current["signature_hash"]
    skipped_query_copies = 0
    skipped_neighbor_copies = 0
    collision = False
    for row in pool:
        _distance, label, _path, sig_hash = row
        if sig_hash == query_hash:
            skipped_query_copies += 1
            if label != current["label"]:
                collision = True
            continue
        if sig_hash not in by_hash:
            by_hash[sig_hash] = label
            selected.append(row)
        else:
            skipped_neighbor_copies += 1
            if by_hash[sig_hash] != label:
                collision = True
    return selected, {
        "excluded_query_signature_copies": skipped_query_copies,
        "collapsed_neighbor_signature_copies": skipped_neighbor_copies,
        "signature_conflict": collision,
    }


def _evaluate_mode(
    current: dict, pool: list[tuple], mode: str,
    *, k: int, per_label: int,
    min_similarity: float, review_margin: float,
) -> dict:
    duplicate_info = {
        "excluded_query_signature_copies": 0,
        "collapsed_neighbor_signature_copies": 0,
        "signature_conflict": False,
    }
    working = pool
    if mode in (
        "SEM_DUPLICATAS_TOP5",
        "BALANCEADO_SEM_DUPLICATAS",
        "BALANCEADO_COM_REVISAO",
    ):
        working, duplicate_info = _without_duplicates(current, pool)

    if mode in ("BASELINE_TOP5", "SEM_DUPLICATAS_TOP5"):
        selected = working[:k]
        score = _weighted_vote(selected) if selected else None
    else:
        score, selected = _balanced_vote(working, per_label)

    best = 1.0 - working[0][0] if working else None
    status = None
    prediction = None
    if duplicate_info["signature_conflict"]:
        status = "REVISAR_CONFLITO_ASSINATURA"
    elif not working:
        status = "REVISAR_SEM_VIZINHOS"
    elif score is None:
        status = "REVISAR_CLASSE_AUSENTE"
    elif any(
        abs(r[0] - working[0][0]) <= EPSILON
        and r[1] != working[0][1]
        for r in working
    ):
        status = "REVISAR_EMPATE_DE_DISTANCIA"
    elif abs(score - 0.5) <= EPSILON:
        status = "REVISAR_EMPATE_DE_VOTACAO"
    elif mode == "BALANCEADO_COM_REVISAO" and (
        best is None or best < min_similarity
    ):
        status = "REVISAR_SIMILARIDADE_BAIXA"
    elif mode == "BALANCEADO_COM_REVISAO" and abs(score - .5) < review_margin:
        status = "REVISAR_MARGEM_INSUFICIENTE"
    else:
        prediction = "NG" if score > .5 else "OK"
        status = "AUTO_" + prediction

    return {
        "status": status,
        "prediction": prediction,
        "vote_ng": round(score, 8) if score is not None else None,
        "best_similarity": round(best, 8) if best is not None else None,
        "neighbors_used": len(selected),
        **duplicate_info,
        "neighbors": [
            {
                "path": path, "label": label,
                "similarity": round(1.0 - distance, 8),
            }
            for distance, label, path, _h in selected
        ],
    }


def _metrics(cases: list[dict], mode: str) -> dict:
    tally = Counter()
    by_category = defaultdict(Counter)
    for row in cases:
        expected = row["expected_human_label"]
        result = row["modes"][mode]
        pred = result["prediction"]
        category = row["category"]
        tally["total"] += 1
        tally["expected_" + expected] += 1
        if pred is None:
            key = "review_" + expected
        elif pred == expected:
            key = "correct_" + expected
        elif expected == "NG":
            key = "missed_NG_as_OK"
        else:
            key = "false_NG_on_OK"
        tally[key] += 1
        by_category[category][key] += 1
        by_category[category]["expected_" + expected] += 1
    denominator = tally["expected_NG"]
    ok_denominator = tally["expected_OK"]
    auto_total = sum(tally[key] for key in (
        "correct_OK", "correct_NG", "missed_NG_as_OK", "false_NG_on_OK"
    ))
    correct = tally["correct_OK"] + tally["correct_NG"]
    metrics = {
        key: tally[key] for key in (
            "total", "expected_OK", "expected_NG",
            "correct_OK", "correct_NG",
            "missed_NG_as_OK", "false_NG_on_OK",
            "review_OK", "review_NG",
        )
    }
    metrics.update({
        "automatic_decisions": auto_total,
        "correct_automatic": correct,
        "automatic_accuracy_on_decided_only": (
            round(correct / auto_total, 6) if auto_total else None
        ),
        "ng_detection_recall_over_all_NG": (
            round(tally["correct_NG"] / denominator, 6)
            if denominator else None
        ),
        "ng_unsafe_release_rate": (
            round(tally["missed_NG_as_OK"] / denominator, 6)
            if denominator else None
        ),
        "ng_auto_or_review_not_released_as_OK": (
            round((tally["correct_NG"] + tally["review_NG"]) / denominator, 6)
            if denominator else None
        ),
        "ok_false_ng_rate": (
            round(tally["false_NG_on_OK"] / ok_denominator, 6)
            if ok_denominator else None
        ),
        "review_fraction_total": (
            round((tally["review_OK"] + tally["review_NG"]) / tally["total"], 6)
            if tally["total"] else None
        ),
        "by_category": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(by_category.items())
        },
    })
    return metrics


def audit_knn_ng_discrimination(
    root: Path, *, records: list[dict] | None = None,
    k: int = 5, per_label: int = 3,
    min_similarity: float = .80,
    review_margin: float = .10,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    if not 1 <= k <= 50 or not 1 <= per_label <= 25:
        raise ValueError("k/per_label fora do intervalo permitido")
    if not (0 <= min_similarity <= 1 and 0 <= review_margin < .5):
        raise ValueError("Limiares fora de [0,1]; margem inferior a 0.5")
    root = Path(root).expanduser().resolve()
    raw = records if records is not None else load_signature_records(root)
    entries, rejected = _validate(raw)
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry["category"], entry["lighting_mode"])].append(entry)

    # Cache para reusar as mesmas comparações nos cinco métodos.
    cache = {}
    cases = []
    for group_name, group in sorted(groups.items()):
        for current in group:
            candidates = []
            for other in group:
                if other["path"] == current["path"]:
                    continue
                if (current.get("event_id")
                        and current["event_id"] == other.get("event_id")):
                    continue
                pair = tuple(sorted((
                    current["path"], other["path"]
                )))
                distance = cache.get(pair)
                if distance is None:
                    sim, _detail = compare_anomaly_signatures(
                        current["signature"], other["signature"]
                    )
                    if not np.isfinite(sim):
                        raise ValueError("Similaridade KNN inválida")
                    distance = float(1.0 - sim)
                    cache[pair] = distance
                candidates.append((
                    distance, other["label"], other["path"],
                    other["signature_hash"],
                ))
            candidates.sort(key=lambda row: (row[0], row[2]))
            modes = {
                mode: _evaluate_mode(
                    current, candidates, mode, k=k,
                    per_label=per_label,
                    min_similarity=min_similarity,
                    review_margin=review_margin,
                )
                for mode in MODES
            }
            cases.append({
                "path": current["path"],
                "schema": current.get("schema"),
                "category": group_name[0],
                "lighting_mode": group_name[1],
                "event_id_declared": bool(current.get("event_id")),
                "expected_human_label": current["label"],
                "other_records_in_same_scope": len(candidates),
                "original_duplicate_neighbors": sum(
                    item[3] == current["signature_hash"] for item in candidates
                ),
                "modes": modes,
            })
            if progress is not None:
                progress(len(cases), len(entries), current["path"])

    comparisons = {mode: _metrics(cases, mode) for mode in MODES}
    duplicate_groups = Counter(
        (item["category"], item["lighting_mode"], item["signature_hash"])
        for item in entries
    )
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_EXPERIMENTAL_NG_DISCRIMINATION",
        "evaluated_records": len(cases),
        "original_json_records": len(raw),
        "rejected_records": rejected,
        "declared_top_k_baseline": k,
        "balanced_neighbors_per_class": per_label,
        "review_similarity_min_experimental": min_similarity,
        "review_margin_experimental": review_margin,
        "modes": list(MODES),
        "duplicate_signature_groups": sum(
            count > 1 for count in duplicate_groups.values()
        ),
        "records_inside_duplicate_groups": sum(
            count for count in duplicate_groups.values() if count > 1
        ),
        "pair_similarity_comparisons": len(cache),
        "comparisons": comparisons,
        "cases": cases,
        "dataset_modified": False,
        "cnn_modified": False,
        "knn_runtime_modified": False,
        "startup_gate_enabled": False,
        "archive_212_accuracy_measured": False,
        "new_images_generalization_measured": False,
        "thresholds_calibrated_on_independent_validation": False,
        "limitations": (
            "Amostras podem compartilhar placa/inspeção sem event_id; "
            "deduplicar assinaturas não substitui partição por lote/evento. "
            "Os limiares exploratórios de revisão NÃO foram calibrados. "
            "Não considerar revisão um acerto NG automático. "
            "Apenas simulação leave-one-record-out; "
            "não usar este relatório para liberar a produção."
        ),
    }


__all__ = [
    "SCHEMA", "MODES", "audit_knn_ng_discrimination",
]
