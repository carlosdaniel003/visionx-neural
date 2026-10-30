"""Primeiro rótulo humano -> memória visual durável ANTES da próxima inspeção.

CNN FALTANDO v2 tem duas velocidades complementares:
1. experiência exata imediata (mesmo par/luz/categoria -> mesmo rótulo);
2. backpropagation de pesos no worker online já existente, com gate.
A experiência é lookup de caso visto, NÃO atualização instantânea de pesos.
Nenhuma predição da IA vira gabarito por esta rotina.
"""
from __future__ import annotations

import time

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_operator_record,
)
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category

HUMAN_SOURCES = frozenset({
    "button", "xp_keyboard", "keyboard", "manual", "operator",
    "physical_keyboard", "human",
})


def immediate_confirmed_memory(task: dict, *, store=None) -> dict:
    """O retorno indica COMMIT verificado, não sucesso do treino CNN."""
    source = str(task.get("source", "") or "").strip().lower()
    label = str(task.get("label", "") or "").strip().upper()
    if source not in HUMAN_SOURCES or label not in {"OK", "NG"}:
        return {"committed": False, "reason": "UNVERIFIED_OR_AUTO_LABEL",
                "new": 0}
    info = task.get("aoi_info") or {}
    if not isinstance(info, dict):
        return {"committed": False, "reason": "INVALID_INFO", "new": 0}
    category = canonical_memory_category(info.get("category", ""))
    # O snapshot corrente pode ser tri-light, parcial, ou apenas SIDE.
    samples = (task.get("multilight_samples")
               or task.get("shadow_partial_samples")
               or [{
                   "lighting_mode": info.get("lighting_mode", "SIDE"),
                   "sample_image": task.get("sample_image"),
                   "test_image": task.get("ng_image"),
               }])
    if not isinstance(samples, list) or not samples:
        return {"committed": False, "reason": "EMPTY_SAMPLES", "new": 0}
    # Um único instante para todas as iluminações do mesmo clique.
    # A ordem é a dos julgamentos HUMANOS, nunca de inferências/XP.
    confirmed_at_ns = time.time_ns()
    proposed = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Amostra de confirmação inválida")
        reference = sample.get("sample_image")
        test = sample.get("test_image")
        lighting = str(sample.get("lighting_mode") or
                       info.get("lighting_mode") or "SIDE")
        observed_info = {**info, "lighting_mode": lighting}
        signature = build_anomaly_signature(
            reference, test, {}, observed_info
        )
        proposed.append(make_operator_record(
            label=label, source=source,
            event_id=str(task.get("event_id", "") or ""),
            category=category, lighting=lighting,
            reference=reference, test=test,
            anomaly_signature=signature, aoi_info=observed_info,
            confirmed_at_ns=confirmed_at_ns,
        ))
    memory = store if store is not None else PersistentArchiveKNN()
    outcome = memory.add(proposed)
    return {
        "committed": True,
        "reason": "HUMAN_LABEL_PERSISTED",
        "new": outcome["new"],
        "total": outcome["total"],
        "cnn_v2_experience": uses_faltando_v2(category),
        "source": source,
        "labels_conflict": outcome["conflicting_identities"],
        "latest_human_at_ns": confirmed_at_ns,
    }


__all__ = ["immediate_confirmed_memory", "HUMAN_SOURCES"]
