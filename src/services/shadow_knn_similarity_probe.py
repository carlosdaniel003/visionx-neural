"""Sonda de KNN por similaridade em Modo Sombra: NÃO atua no veredito.

Lê snapshot persistente numa thread de diagnóstico de capacidade um; não
bloqueia inspeções, não usa event_id por proximidade de horário e não envia XP.
A proposta por evento é gravada em reports/shadow_similarity/latest.json.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

from src.core.anomaly_signature import (
    build_anomaly_signature, compare_anomaly_signatures, valid_anomaly_signature,
)
from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.knn_similarity_shadow_candidate import select_shadow_candidate, _day
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knn-shadow-sim")
_PENDING = threading.Lock()
_BUSY = False
LIGHTS = ("SIDE", "TOP", "MID")


def evaluate_shadow_snapshot(
    samples: dict, category: str, records: list[dict], *,
    event_id: str = "",
) -> dict:
    """Calcula sugestões sobre memória persistida, com rótulos desconhecidos.

    Não altera objetos de memória nem recebe rótulo de avaliação da peça.
    """
    cat = canonical_memory_category(category)
    out: dict[str, Any] = {
        "schema": "visionx.knn_live_shadow_snapshot.v1",
        "mode": "SHADOW_DIAGNOSTIC_ONLY",
        "category": cat, "event_id": str(event_id)[:180],
        "decision_authorized": False, "xp_command": None,
        "record_count": len(records),
        "lights": {}, "overall": "REVISAO",
    }
    if not cat or cat in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
        out["reason"] = "INVALID_OCR_CATEGORY"
        return out
    for light in LIGHTS:
        sample = samples.get(light)
        if not isinstance(sample, dict) or sample.get("lighting_mode") != light:
            continue
        ref, test = sample.get("sample_image"), sample.get("test_image")
        if (getattr(ref, "size", 0) == 0 or getattr(test, "size", 0) == 0):
            continue
        signature = build_anomaly_signature(
            ref, test, {}, {"category": cat, "lighting_mode": light}
        )
        if not valid_anomaly_signature(signature):
            out["lights"][light] = {
                "status": "REVISAO", "reason": "INVALID_SIGNATURE",
            }
            continue
        ranked = []
        scope = {"OK": set(), "NG": set()}
        for rec in records:
            if not isinstance(rec, dict) or rec.get("label") not in scope:
                continue
            ident = rec.get("identity") or []
            if (len(ident) != 4 or canonical_memory_category(ident[0]) != cat
                    or canonical_memory_lighting(ident[1]) != light):
                continue
            stored = rec.get("anomaly_signature")
            if not valid_anomaly_signature(stored):
                continue
            paths = [src.get("source_path", "") for src in rec.get("sources", ())
                     if isinstance(src, dict)]
            source = next((x for x in paths if _day(x)), "")
            if not source:
                continue
            scope[rec["label"]].add(_day(source))
            sim, _detail = compare_anomaly_signatures(signature, stored)
            ranked.append({
                "label": rec["label"], "similarity": sim,
                "source_path": source,
            })
        result = select_shadow_candidate(
            ranked, category=cat, lighting=light,
            scope_days={key: len(days) for key, days in scope.items()},
        )
        out["lights"][light] = result
    suggestions = [entry["status"] for entry in out["lights"].values()]
    if suggestions and all(x == suggestions[0] for x in suggestions):
        out["overall"] = suggestions[0]
    else:
        out["overall"] = "REVISAO"
    out["reason"] = (
        "CONSENSUS_AVAILABLE_LIGHTS" if out["overall"] != "REVISAO"
        else "UNAVAILABLE_OR_DISAGREEMENT"
    )
    # Todos os dados deste JSON são APENAS hipóteses não calibradas.
    return out


def _worker(samples: dict, category: str, event_id: str, root: Path):
    global _BUSY
    try:
        records = PersistentArchiveKNN(root).records()
        report = evaluate_shadow_snapshot(
            samples, category, records, event_id=event_id
        )
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        dest = root / "reports" / "shadow_similarity"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "latest.json"
        tmp = dest / "latest.json.tmp"
        tmp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        # Apenas diagnóstico: qualquer falha jamais afeta a captura.
        pass
    finally:
        with _PENDING:
            _BUSY = False


def enqueue_shadow_candidate(
    samples: dict | None, category: str, event_id: str, root: Path,
) -> bool:
    """Fila limitada: um diagnóstico ativo, sem backlog de peças antigas."""
    global _BUSY
    if not isinstance(samples, dict) or not samples:
        return False
    with _PENDING:
        if _BUSY:
            return False
        copied = {}
        for light in LIGHTS:
            sample = samples.get(light)
            if not isinstance(sample, dict):
                continue
            ref, test = sample.get("sample_image"), sample.get("test_image")
            if not callable(getattr(ref, "copy", None)):
                continue
            if not callable(getattr(test, "copy", None)):
                continue
            copied[light] = {
                "lighting_mode": light, "sample_image": ref.copy(),
                "test_image": test.copy(),
            }
        if not copied:
            return False
        _BUSY = True
    try:
        _EXECUTOR.submit(
            _worker, copied, str(category), str(event_id), Path(root).resolve()
        )
        return True
    except RuntimeError:
        with _PENDING:
            _BUSY = False
        return False


__all__ = ["enqueue_shadow_candidate", "evaluate_shadow_snapshot"]
