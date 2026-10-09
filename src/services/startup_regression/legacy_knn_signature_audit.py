"""Diagnóstico leave-one-record-out de memória de assinaturas KNN histórica.

Usa o comparador de produção compare_anomaly_signatures e seu voto
por distância, com contexto obrigatório categoria + iluminação. Não
reconstrói imagens ou confunde a assinatura com hash visual. O rótulo
guardado só é usado DEPOIS da predição para avaliar a concordância.

Autoexclusão, exclusão por event_id declarado, conflitos idênticos e
possíveis duplicações são registrados. Acurácia histórica NÃO é
cobertura de screenshots nem qualidade medida em imagens inéditas.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from src.core.anomaly_signature import (
    compare_anomaly_signatures, valid_anomaly_signature,
)
from src.core.experts.knn_expert import KNNExpert
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.startup_regression.archive_reconciler import (
    DATA_FOLDERS, HUMAN_SOURCES,
)

SCHEMA = "visionx.legacy_knn_signature_leave_one_out.v1"


def _record(root: Path, path: Path, folder_label: str):
    row = {
        "path": path.relative_to(root).as_posix(),
        "schema": None, "category": None, "lighting_mode": None,
        "label": None, "status": "REJEITADO",
        "event_id": None, "signature": None, "signature_hash": None,
        "error": None,
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON não é um objeto")
        row["schema"] = str(data.get("schema", "") or "SEM_SCHEMA")
        label = str(data.get("label", "")).strip().upper()
        decision = data.get("decision")
        decision = decision if isinstance(decision, dict) else {}
        source = str(decision.get("source", "")).strip().lower()
        info = data.get("aoi_info")
        info = info if isinstance(info, dict) else {}
        category = canonical_memory_category(info.get("category"))
        light = canonical_memory_lighting(info.get("lighting_mode"))
        row.update({
            "label": label, "category": category,
            "lighting_mode": light,
        })
        if (label not in {"OK", "NG"} or label != folder_label
                or str(decision.get("operator_label", "")).upper() != label
                or not (
                    source in HUMAN_SOURCES
                    or source.startswith("operator_")
                )):
            row["status"] = "SEM_ROTULO_HUMANO_VERIFICADO"
            return row
        if not category:
            row["status"] = "CATEGORIA_AUSENTE"
            return row
        signature = KNNExpert._extract_anomaly_memory(data)
        if not valid_anomaly_signature(signature):
            row["status"] = "ASSINATURA_INVALIDA"
            return row
        vector = np.asarray(signature["vector"], dtype=np.float32)
        row["signature"] = signature
        row["signature_hash"] = hashlib.sha256(
            vector.tobytes()
        ).hexdigest()
        multilight = data.get("multilight")
        if isinstance(multilight, dict) and multilight.get("event_id"):
            row["event_id"] = str(multilight["event_id"])
        row["status"] = "ELEGIVEL"
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        row["status"] = "JSON_INVALIDO"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def load_signature_records(root: Path):
    root = Path(root).expanduser().resolve()
    rows = []
    for dirname, folder_label in DATA_FOLDERS:
        base = root / "public" / "dataset" / dirname
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*.json")):
            if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                continue
            rows.append(_record(root, path, folder_label))
    return rows


def audit_signature_knn(
    root: Path, *, records: list | None = None,
    top_k: int = 5, progress: Callable | None = None,
) -> dict:
    """KNN por assinatura, SEM consultar a própria observação.

    Lógica de voto e distância da produção; isolação por categoria/luz
    da strict_category_memory. Não instancia KNN (não baixa modelo).
    """
    if not 1 <= int(top_k) <= 50:
        raise ValueError("top_k deve estar entre 1 e 50")
    root = Path(root).expanduser().resolve()
    raw = records if records is not None else load_signature_records(root)
    groups = defaultdict(list)
    for row in raw:
        if row.get("status") == "ELEGIVEL":
            if not valid_anomaly_signature(row.get("signature")):
                raise ValueError("Registro marcado ELEGIVEL sem assinatura válida")
            groups[(row["category"], row["lighting_mode"])].append(row)

    cases = []
    candidates_count = sum(map(len, groups.values()))
    done = 0
    for (category, light), entries in sorted(groups.items()):
        for current in entries:
            done += 1
            candidate_pool = [
                other for other in entries
                if other["path"] != current["path"]
                and not (
                    current.get("event_id")
                    and current.get("event_id") == other.get("event_id")
                )
            ]
            distances = []
            for other in candidate_pool:
                similarity, _ = compare_anomaly_signatures(
                    current["signature"], other["signature"]
                )
                if not np.isfinite(similarity):
                    raise ValueError("Similaridade KNN não finita")
                distances.append(
                    (1.0 - float(similarity), other["label"], other["path"],
                     other.get("signature_hash"))
                )
            distances.sort(key=lambda item: (item[0], item[2]))
            nearest = distances[:int(top_k)]
            same_fingerprint = [
                other for other in candidate_pool
                if other.get("signature_hash") == current.get("signature_hash")
            ]
            row = {
                "path": current["path"], "schema": current.get("schema"),
                "category": category, "lighting_mode": light,
                "expected_human_label": current["label"],
                "predicted_label": None, "status": "SEM_VIZINHOS",
                "candidate_pool": len(candidate_pool),
                "neighbors_used": len(nearest),
                "best_similarity": None, "vote_ng": None,
                "same_signature_other_records": len(same_fingerprint),
                "same_signature_conflict": len({
                    other["label"] for other in same_fingerprint
                } | {current["label"]}) > 1,
                "neighbors": [
                    {"path": p, "label": label, "similarity": round(1-dist, 8)}
                    for dist, label, p, _ in nearest
                ],
            }
            if nearest:
                score = KNNExpert._weighted_vote(
                    [(dist, label) for dist, label, _, _ in nearest]
                )
                row["vote_ng"] = round(score, 8)
                row["best_similarity"] = round(1-nearest[0][0], 8)
                closest = [
                    label for dist, label, _, _ in distances
                    if abs(dist-nearest[0][0]) <= 1e-9
                ]
                if len(set(closest)) > 1 or row["same_signature_conflict"]:
                    row["status"] = "CONFLITO_DE_ASSINATURA"
                elif abs(score - 0.5) < 1e-9:
                    row["status"] = "EMPATE_DE_VOTACAO"
                else:
                    verdict = "NG" if score > .5 else "OK"
                    row["predicted_label"] = verdict
                    row["status"] = (
                        "CONCORDA" if verdict == current["label"]
                        else "DIVERGE"
                    )
            cases.append(row)
            if progress is not None:
                progress(done, candidates_count, current["path"])

    rejected = Counter(
        row["status"] for row in raw if row.get("status") != "ELEGIVEL"
    )
    counts = Counter(row["status"] for row in cases)
    by_category = {}
    for category in sorted({row["category"] for row in cases}):
        selected = [r for r in cases if r["category"] == category]
        by_category[category] = dict(sorted(Counter(
            r["status"] for r in selected
        ).items()))
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_LEAVE_ONE_RECORD_OUT",
        "top_k": top_k,
        "json_records_total": len(raw),
        "eligible_signatures": len(cases),
        "rejected_records": dict(sorted(rejected.items())),
        "status_counts": dict(sorted(counts.items())),
        "by_category": by_category,
        "duplicate_signature_records": sum(
            bool(r["same_signature_other_records"]) for r in cases
        ),
        "same_signature_conflict_records": sum(
            r["same_signature_conflict"] for r in cases
        ),
        "cases": cases,
        "dataset_modified": False,
        "knn_runtime_modified": False,
        "startup_gate_enabled": False,
        "archive_212_accuracy_measured": False,
        "test_of_new_images": False,
        "note": (
            "Reprodução KNN histórica por assinatura em leave-one-record-out. "
            "Não é a memória exata de produção, não valida os 212 PNGs, "
            "e não mede generalização em novos defeitos."
        ),
    }


__all__ = ["SCHEMA", "load_signature_records", "audit_signature_knn"]
