"""Avaliação KNN seletiva com partição por DIA e teste final intocado.

Somente leitura. Divisão determinística por dia (proxy de sessão), não por
JSON aleatório. Não altera o motor nem considera o rótulo de consulta
durante predição. A seleção usa APENAS calibração; teste é avaliado uma
vez com a política congelada. Vínculos de evento/hash entre partições
são excluídos da avaliação, com contagem explícita.

IMPORTANTE: Partição por dia não demonstra independência de placa/lote.
Ausência de NG, datas ou exemplos de ambas as classes bloqueia aprovação.
Mesmo zero NG perdidos em um teste pequeno não libera produção.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from src.core.anomaly_signature import compare_anomaly_signatures
from src.core.experts.knn_expert import KNNExpert
from .knn_ng_discrimination_audit import _validate, _balanced_vote
from .legacy_knn_signature_audit import load_signature_records

SCHEMA = "visionx.knn_selective_session_holdout.v1"
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
PARTS = ("memory_train", "calibration", "heldout_test")
# Grades declaradas ANTES de avaliar; o teste jamais escolhe parâmetros.
OK_VOTE_CEILINGS = (0.0, .10, .20, .25, .30, .35, .40)
SIMILARITY_FLOORS = (.80, .85, .90, .95)
MIN_CALIBRATION_OK_AUTOMATIC_FRACTION = .20


def _day(item: dict) -> str | None:
    # Usar a data somente como indicador de sessão; nunca como rótulo.
    if item.get("session_day"):
        s = str(item["session_day"])
        if re.fullmatch(r"20\d{2}-[01]\d-[0-3]\d", s):
            return s
        return None
    match = DATE_RE.search(str(item.get("path") or ""))
    return "-".join(match.groups()) if match else None


def _split_dates(entries: list[dict]) -> dict:
    groups = defaultdict(list)
    ungrouped = []
    for item in entries:
        day = _day(item)
        if day is None:
            ungrouped.append(item["path"])
        else:
            groups[day].append(item)

    assigned = {part: [] for part in PARTS}
    total_ng = sum(x["label"] == "NG" for x in entries)
    if len(groups) < 3 or total_ng < 3:
        return {
            "status": "INSUFFICIENT_SESSION_DAYS_OR_NG",
            "by_part": assigned, "days_by_part": {k: [] for k in PARTS},
            "ungrouped_paths": ungrouped,
            "days_total": len(groups),
        }

    # A alocação só observa contagens conhecidas por DIA, nunca os votos
    # ou acertos dos classificadores. Preserva o dia inteiro numa partição.
    ng_days = sorted(
        (d for d in groups if any(r["label"] == "NG" for r in groups[d])),
        key=lambda d: (-sum(r["label"] == "NG" for r in groups[d]), d),
    )
    if len(ng_days) < 3:
        return {
            "status": "NG_PRESENT_IN_FEWER_THAN_THREE_DAYS",
            "by_part": assigned, "days_by_part": {k: [] for k in PARTS},
            "ungrouped_paths": ungrouped,
            "days_total": len(groups),
        }
    day_assignment = {}
    # Garantir NG em todas as três partições antes de distribuir os demais.
    for idx, day in enumerate(ng_days[:3]):
        day_assignment[day] = PARTS[idx]
    ng_count = {p: sum(
        r["label"] == "NG" for d in day_assignment if day_assignment[d] == p
        for r in groups[d]
    ) for p in PARTS}
    size = {p: sum(
        len(groups[d]) for d in day_assignment if day_assignment[d] == p
    ) for p in PARTS}
    for day in ng_days[3:]:
        part = min(PARTS, key=lambda p: (ng_count[p], size[p], PARTS.index(p)))
        day_assignment[day] = part
        ng_count[part] += sum(r["label"] == "NG" for r in groups[day])
        size[part] += len(groups[day])
    for day in sorted((d for d in groups if d not in day_assignment),
                      key=lambda d: (-len(groups[d]), d)):
        part = min(PARTS, key=lambda p: (size[p], PARTS.index(p)))
        day_assignment[day] = part
        size[part] += len(groups[day])

    for day, records in groups.items():
        assigned[day_assignment[day]].extend(records)
    return {
        "status": "GROUPED_BY_DAY_NON_CHRONOLOGICAL",
        "by_part": assigned,
        "days_by_part": {
            p: sorted(d for d in day_assignment if day_assignment[d] == p)
            for p in PARTS
        },
        "ungrouped_paths": ungrouped, "days_total": len(groups),
    }


def _cross_partition_exclusions(by_part: dict) -> tuple[dict, list]:
    """Exclui MESMO hash, mesmo event_id em partições distintas, sem inferir label."""
    hash_locations, event_locations = defaultdict(set), defaultdict(set)
    for part in PARTS:
        for row in by_part[part]:
            scope = (row["category"], row["lighting_mode"], row["signature_hash"])
            hash_locations[scope].add(part)
            if row.get("event_id"):
                event_locations[row["event_id"]].add(part)
    bad_hashes = {k for k, loc in hash_locations.items() if len(loc) > 1}
    bad_events = {k for k, loc in event_locations.items() if len(loc) > 1}
    excluded = []
    cleaned = {p: [] for p in PARTS}
    for part in PARTS:
        for row in by_part[part]:
            signature_key = (
                row["category"], row["lighting_mode"], row["signature_hash"]
            )
            if signature_key in bad_hashes or (
                row.get("event_id") and row["event_id"] in bad_events
            ):
                excluded.append({
                    "path": row["path"], "part": part,
                    "reason": (
                        "CROSS_PARTITION_SIGNATURE"
                        if signature_key in bad_hashes else "CROSS_PARTITION_EVENT"
                    ),
                    "expected_human_label": row["label"],
                })
            else:
                cleaned[part].append(row)
    return cleaned, excluded


def _predictions(
    queries: list[dict], memory: list[dict], progress=None,
):
    grouped = defaultdict(list)
    for candidate in memory:
        grouped[(candidate["category"], candidate["lighting_mode"])].append(candidate)
    results = []
    for i, query in enumerate(queries, 1):
        candidates = []
        for other in grouped[(query["category"], query["lighting_mode"])]:
            if other["path"] == query["path"]:
                continue
            if query.get("event_id") and query["event_id"] == other.get("event_id"):
                continue
            sim, _ = compare_anomaly_signatures(
                query["signature"], other["signature"]
            )
            if not np.isfinite(sim):
                raise ValueError("Similaridade KNN não finita")
            candidates.append((
                1.0 - float(sim), other["label"],
                other["path"], other["signature_hash"],
            ))
        candidates.sort(key=lambda x: (x[0], x[2]))
        # Vizinhos idênticos na memória são apenas UMA evidência por hash.
        dedup = []
        known = {}
        conflict = False
        query_copy = False
        for item in candidates:
            _dist, label, _path, digest = item
            if digest == query["signature_hash"]:
                query_copy = True
                continue
            if digest in known:
                if known[digest] != label:
                    conflict = True
                continue
            known[digest] = label
            dedup.append(item)
        score, selected = _balanced_vote(dedup, 3)
        # Com apenas uma classe, _balanced_vote devolve score=None;
        # a decisão segura é revisão, nunca exceção nem AUTO_OK.
        base_five = candidates[:5]
        baseline_score = (
            float(KNNExpert._weighted_vote(
                [(distance, label) for distance, label, _p, _h in base_five]
            )) if base_five else None
        )
        results.append({
            "path": query["path"],
            "category": query["category"],
            "lighting_mode": query["lighting_mode"],
            "expected_human_label": query["label"],
            "candidate_pool": len(candidates),
            "balanced_neighbors": len(selected),
            "score_ng": score,
            "best_similarity": (
                1.0 - dedup[0][0] if dedup else None
            ),
            "baseline_vote_ng": baseline_score,
            "baseline_best_similarity": (
                1.0 - candidates[0][0] if candidates else None
            ),
            "has_conflicting_neighbor_signatures": conflict,
            "has_exact_query_copy": query_copy,
            "class_support": {
                "OK": sum(item[1] == "OK" for item in selected),
                "NG": sum(item[1] == "NG" for item in selected),
            },
        })
        if progress:
            progress(i, len(queries), query["path"])
    return results


def _baseline_decision(row: dict) -> dict:
    if row["has_exact_query_copy"] or row["has_conflicting_neighbor_signatures"]:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "CONFLITO_OU_COPIA_EXATA"}
    score = row["baseline_vote_ng"]
    if score is None:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "SEM_VIZINHOS"}
    return {
        "decision": "NG" if score > .5 else "OK",
        "reason": "BASELINE_KNN_TOP5_SEM_ABSTENCAO_POR_MARGEM",
    }


def _decision(row: dict, policy: dict | None) -> dict:
    if policy is None:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "POLITICA_NAO_CALIBRADA"}
    if row["has_exact_query_copy"] or row["has_conflicting_neighbor_signatures"]:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "CONFLITO_OU_COPIA_EXATA"}
    score, best = row["score_ng"], row["best_similarity"]
    if score is None or best is None:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "FALTA_CLASSE_OU_VIZINHO"}
    if best < policy["min_similarity"]:
        return {"decision": "REVISAO_OBRIGATORIA", "reason": "SEMELHANCA_INSUFICIENTE"}
    if score > .5:
        return {"decision": "NG", "reason": "VOTO_NG_CONCORDANTE"}
    if score <= policy["ok_vote_ceiling"]:
        return {"decision": "OK", "reason": "LIBERACAO_SELETIVA_OK"}
    return {"decision": "REVISAO_OBRIGATORIA", "reason": "MARGEM_OK_INSUFICIENTE"}


def _measure(rows: list[dict], policy: dict | None, *, baseline=False) -> dict:
    counts = Counter()
    by_category = defaultdict(Counter)
    decisions = []
    for row in rows:
        ans = _baseline_decision(row) if baseline else _decision(row, policy)
        label = row["expected_human_label"]
        decision = ans["decision"]
        counts[f"actual_{label}"] += 1
        counts[f"{label}_as_{decision}"] += 1
        by_category[row["category"]][f"{label}_as_{decision}"] += 1
        decisions.append({
            "path": row["path"], "category": row["category"],
            "lighting_mode": row["lighting_mode"],
            "expected_human_label": label,
            "decision": decision, "reason": ans["reason"],
            "vote_ng": round(row["score_ng"], 8) if row["score_ng"] is not None else None,
            "best_similarity": (
                round(row["best_similarity"], 8)
                if row["best_similarity"] is not None else None
            ),
        })
    return {
        "evaluated": len(rows),
        "actual_NG": counts["actual_NG"],
        "actual_OK": counts["actual_OK"],
        "NG_released_as_OK": counts["NG_as_OK"],
        "NG_auto_NG": counts["NG_as_NG"],
        "NG_review": counts["NG_as_REVISAO_OBRIGATORIA"],
        "OK_auto_OK": counts["OK_as_OK"],
        "OK_false_NG": counts["OK_as_NG"],
        "OK_review": counts["OK_as_REVISAO_OBRIGATORIA"],
        "total_reviews": (
            counts["NG_as_REVISAO_OBRIGATORIA"] +
            counts["OK_as_REVISAO_OBRIGATORIA"]
        ),
        "by_category": {
            k: dict(sorted(v.items())) for k, v in sorted(by_category.items())
        },
        "cases": decisions,
    }


def _calibrate(rows: list[dict]):
    results = []
    for minimum in SIMILARITY_FLOORS:
        for ceiling in OK_VOTE_CEILINGS:
            policy = {
                "ok_vote_ceiling": ceiling, "min_similarity": minimum,
            }
            stats = _measure(rows, policy)
            ok_fraction = (
                stats["OK_auto_OK"] / stats["actual_OK"]
                if stats["actual_OK"] else 0.0
            )
            admitted = (
                stats["actual_NG"] > 0 and stats["actual_OK"] > 0
                and stats["NG_released_as_OK"] == 0
                and ok_fraction >= MIN_CALIBRATION_OK_AUTOMATIC_FRACTION
            )
            results.append({
                **policy, "admitted_on_calibration": admitted,
                "NG_released_as_OK": stats["NG_released_as_OK"],
                "NG_review": stats["NG_review"],
                "NG_auto_NG": stats["NG_auto_NG"],
                "OK_auto_OK": stats["OK_auto_OK"],
                "OK_review": stats["OK_review"],
                "OK_auto_fraction": round(ok_fraction, 6),
            })
    suitable = [r for r in results if r["admitted_on_calibration"]]
    chosen = min(
        suitable,
        key=lambda r: (
            -r["OK_auto_OK"], r["OK_review"],
            r["ok_vote_ceiling"], -r["min_similarity"],
        ),
        default=None
    )
    policy = (
        {
            "ok_vote_ceiling": chosen["ok_vote_ceiling"],
            "min_similarity": chosen["min_similarity"],
        } if chosen else None
    )
    return policy, results


def selective_knn_holdout(
    root: Path, *, records: list[dict] | None = None,
    progress=None,
) -> dict:
    root = Path(root).expanduser().resolve()
    raw = records if records is not None else load_signature_records(root)
    entries, rejected = _validate(raw)
    grouping = _split_dates(entries)
    cleaned, excluded = _cross_partition_exclusions(grouping["by_part"])
    train, cal, test = (cleaned[p] for p in PARTS)

    # Nunca chamar uma partição de independente quando houver <3 dias NG,
    # NG ausente ou amostras idênticas atravessando a partição.
    enough = (
        grouping["status"] == "GROUPED_BY_DAY_NON_CHRONOLOGICAL"
        and all(any(r["label"] == "NG" for r in cleaned[p]) for p in PARTS)
        and all(any(r["label"] == "OK" for r in cleaned[p]) for p in PARTS)
    )
    cal_rows = _predictions(cal, train, progress=progress) if enough else []
    policy, search = _calibrate(cal_rows) if enough else (None, [])
    # Com parâmetros travados (ou política sempre-revisar) avalia TESTE
    # apenas uma vez. Sem calibração útil, o teste não qualifica o modelo.
    test_rows = _predictions(test, train, progress=progress) if enough else []
    cal_metrics = _measure(cal_rows, policy) if enough else None
    test_metrics = _measure(test_rows, policy) if enough else None

    actual_ng = test_metrics["actual_NG"] if test_metrics else 0
    errors = test_metrics["NG_released_as_OK"] if test_metrics else 0
    zero_error_upper_bound_95 = (
        round(1 - pow(.05, 1 / actual_ng), 6)
        if actual_ng and errors == 0 else None
    )
    if not enough:
        status = "BLOCKED_INSUFFICIENT_INDEPENDENT_CLASSES"
    elif policy is None:
        status = "BLOCKED_NO_USEFUL_CALIBRATED_POLICY"
    elif errors:
        status = "BLOCKED_NG_RELEASED_AS_OK_ON_HOLDOUT"
    else:
        status = "EXPLORATORY_NO_NG_MISSED_NOT_PRODUCTION_VALIDATED"

    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_GROUPED_HOLDOUT",
        "status": status,
        "partition_basis": "WHOLE_DAY_FROM_FILENAME_NOT_MACHINE_SESSION_ID",
        "partition_non_chronological": True,
        "groups": grouping["days_by_part"],
        "groups_total": grouping["days_total"],
        "ungrouped_paths": grouping["ungrouped_paths"],
        "excluded_cross_partition": excluded,
        "eligible_records": len(entries),
        "rejected_records": rejected,
        "after_leakage_exclusion": {p: len(cleaned[p]) for p in PARTS},
        "class_counts_after_exclusion": {
            p: dict(Counter(row["label"] for row in cleaned[p]))
            for p in PARTS
        },
        "selection_only_on_calibration": True,
        "calibration_grid": search,
        "calibrated_policy": policy,
        "calibration": cal_metrics,
        "heldout_test": test_metrics,
        "heldout_baseline_top5": (
            _measure(test_rows, None, baseline=True) if enough else None
        ),
        "illustrative_zero_error_bound_requires_independence": True,
        "heldout_zero_ng_miss_upper_bound_95": zero_error_upper_bound_95,
        "startup_gate_enabled": False,
        "dataset_modified": False,
        "cnn_modified": False,
        "knn_runtime_modified": False,
        "production_approved": False,
        "test_of_new_independent_physical_boards": False,
        "note": (
            "Sessões agrupadas pelo dia inferido do nome JSON; sem identificação "
            "física de placa/lote, independência visual não garantida; "
            "limite de risco 95% é apenas ilustrativo sob independência não "
            "demonstrada. Teste "
            "nunca seleciona parâmetros. Revisão não é acerto NG automático; "
            "nenhuma taxa de erro zero amostral libera produção."
        ),
    }


__all__ = ["SCHEMA", "selective_knn_holdout"]
