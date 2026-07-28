"""Integra memória de anomalia e especialistas exclusivos por categoria."""

from __future__ import annotations

import re

import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.missing_component_expert import MissingComponentExpert


ADHESIVE_CATEGORIES = frozenset({"MUCH ADHESIVE", "MUITO ADESIVO"})
MISSING_CATEGORIES = frozenset({"MISSING", "FALTANDO"})
STANDARD_ROUTES = ("silk", "ssim", "semantic", "knn")
ADHESIVE_ROUTES = ("shift",) + STANDARD_ROUTES
MISSING_ROUTES = ("missing",) + STANDARD_ROUTES

CATEGORY_KEY_ALIASES = {
    "INVERTIDO": "INVERTIDO",
    "INVERTED": "INVERTIDO",
    "REVERSE": "INVERTIDO",
    "UPSIDEDOWN": "INVERTIDO",
    "FALTANDO": "FALTANDO",
    "MISSING": "FALTANDO",
    "MUSING": "FALTANDO",
    "MISSMG": "FALTANDO",
    "MUITOADESIVO": "MUITOADESIVO",
    "MUCHADHESIVE": "MUITOADESIVO",
    "EXCESSADHESIVE": "MUITOADESIVO",
    "ADESIVOEMEXCESSO": "MUITOADESIVO",
}


def normalize_category_name(category: str) -> str:
    return " ".join(str(category or "").strip().upper().split())


def canonical_category_key(category: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", normalize_category_name(category))
    return CATEGORY_KEY_ALIASES.get(compact, compact)


def is_adhesive_category(category: str) -> bool:
    return normalize_category_name(category) in ADHESIVE_CATEGORIES


def is_missing_category(category: str) -> bool:
    return normalize_category_name(category) in MISSING_CATEGORIES


def routes_for_category(category: str) -> tuple[str, ...]:
    if is_adhesive_category(category):
        return ADHESIVE_ROUTES
    if is_missing_category(category):
        return MISSING_ROUTES
    return STANDARD_ROUTES


def _normalize_knn_memory_categories(knn_expert) -> None:
    """Converte em memória as categorias antigas para as novas chaves."""
    for attribute in ("signatures_ok", "signatures_ng"):
        for record in getattr(knn_expert, attribute, []) or []:
            if isinstance(record, dict):
                record["category"] = canonical_category_key(
                    record.get("category", "")
                )


def _focus_box(aoi_epicenters, analysis: dict, detail: dict):
    if aoi_epicenters:
        candidate = aoi_epicenters[0]
        if candidate and len(candidate) >= 4:
            return tuple(int(value) for value in candidate[:4])

    for candidate in (
        detail.get("semantic_focus_box"),
        detail.get("adhesive_roi_box"),
        detail.get("missing_roi_box"),
        detail.get("roi_box"),
        analysis.get("bounding_box"),
    ):
        if candidate and len(candidate) >= 4:
            return tuple(int(value) for value in candidate[:4])
    return None


def _append_engine(
    orchestrator,
    engines: list[dict],
    engine_id: str,
    label: str,
    active: bool,
    triggered: bool,
    raw_score: float,
    effective_score: float,
    threshold: float,
    summary: str,
) -> dict:
    entry = orchestrator._engine_entry(
        engine_id,
        label,
        active,
        triggered,
        raw_score,
        effective_score,
        threshold,
        summary,
    )
    engines.append(entry)
    return entry


def _dynamic_fusion(
    orchestrator,
    detail: dict,
    category: str,
    missing_result: dict | None,
    knn: dict | None,
):
    """Replica as regras existentes, criando somente motores aplicáveis."""
    engines: list[dict] = []
    physical_reasons: list[str] = []

    if is_adhesive_category(category) and (
        "shift_active" in detail or "adhesive_score" in detail
    ):
        active = bool(detail.get("shift_active", False))
        raw = float(detail.get("adhesive_score", 0.0))
        threshold = float(detail.get("adhesive_tolerance", 0.32))
        triggered = bool(
            active
            and detail.get(
                "adhesive_is_defect",
                raw > threshold,
            )
        )
        effective = max(0.80, min(1.0, raw)) if triggered else 0.0
        summary = str(detail.get("adhesive_reason", ""))
        _append_engine(
            orchestrator,
            engines,
            "adhesive",
            "Fluxo de adesivo",
            active,
            triggered,
            raw,
            effective,
            threshold,
            summary,
        )
        if triggered and summary:
            physical_reasons.append(summary)

    if is_missing_category(category) and missing_result:
        active = bool(missing_result.get("missing_active", False))
        raw = float(missing_result.get("missing_score", 0.0))
        threshold = float(missing_result.get("missing_tolerance", 0.42))
        triggered = bool(
            active
            and missing_result.get(
                "missing_is_defect",
                raw > threshold,
            )
        )
        effective = max(0.88, min(1.0, raw)) if triggered else 0.0
        summary = str(missing_result.get("missing_reason", ""))
        _append_engine(
            orchestrator,
            engines,
            "missing",
            "Presença do componente",
            active,
            triggered,
            raw,
            effective,
            threshold,
            summary,
        )
        if triggered and summary:
            physical_reasons.append(summary)

    if "silk_error_pct" in detail:
        raw = float(detail.get("silk_error_pct", 0.0))
        threshold = 0.08
        triggered = raw > threshold
        effective = 0.85 if triggered else 0.0
        summary = (
            f"Divergência estrutural {raw:.0%}"
            if raw > 0
            else "Estrutura coincidente"
        )
        _append_engine(
            orchestrator,
            engines,
            "structural",
            "Comparador estrutural",
            True,
            triggered,
            raw,
            effective,
            threshold,
            summary,
        )
        if triggered:
            physical_reasons.append(summary)

    if "semantic_loss" in detail:
        raw = float(detail.get("semantic_loss", 0.0))
        threshold = 0.45
        triggered = raw > threshold
        effective = min(1.0, max(0.85, raw * 1.5)) if triggered else 0.0
        summary = str(
            detail.get(
                "semantic_reason",
                f"Evidência semântica {raw:.0%}",
            )
        )
        _append_engine(
            orchestrator,
            engines,
            "semantic",
            "Debug semântico",
            True,
            triggered,
            raw,
            effective,
            threshold,
            summary,
        )
        if triggered and summary:
            physical_reasons.append(summary)

    if "local_score" in detail or "ssim" in detail:
        local_score = float(detail.get("local_score", 0.0))
        context_score = float(detail.get("ctx_score", 0.0))
        raw = local_score * 0.65 + context_score * 0.35
        threshold = float(detail.get("decision_threshold", 0.45))
        triggered = raw > threshold
        summary = (
            f"SSIM {float(detail.get('ssim', 1.0)):.2f}; pixels alterados "
            f"{float(detail.get('pct_changed', 0.0)):.0%}"
        )
        _append_engine(
            orchestrator,
            engines,
            "texture",
            "Laboratório de textura",
            True,
            triggered,
            raw,
            raw,
            threshold,
            summary,
        )
        if triggered:
            physical_reasons.append(summary)

    physical_candidates = [engine for engine in engines if engine["active"]]
    physical_dominant = (
        max(physical_candidates, key=lambda item: item["effective_score"])
        if physical_candidates
        else None
    )
    physical_score = (
        float(physical_dominant["effective_score"])
        if physical_dominant
        else 0.0
    )
    if physical_dominant:
        physical_dominant["selected"] = True

    physical_defect = any(engine["triggered"] for engine in engines)

    has_memory = bool(knn and knn.get("has_memory", False))
    memory_vote = float(knn.get("vote_defect", 0.5)) if knn else 0.5
    memory_similarity = float(knn.get("best_similarity", 0.0)) if knn else 0.0
    neighbors = int(knn.get("n_neighbors", 0)) if knn else 0

    physical_weight = 1.0
    memory_weight = 0.0
    fusion_rule = "physical_only"
    memory_role = "SEM MEMÓRIA"
    final_score = physical_score

    if has_memory:
        if physical_defect:
            if memory_similarity >= 0.85:
                physical_weight, memory_weight = 0.0, 1.0
                fusion_rule = "memory_veto"
                memory_role = "VETO DA MEMÓRIA"
            elif memory_vote < 0.30 and memory_similarity >= 0.75:
                physical_weight, memory_weight = 0.20, 0.80
                fusion_rule = "memory_priority"
                memory_role = "MEMÓRIA PRIORITÁRIA"
            else:
                physical_weight, memory_weight = 0.70, 0.30
                fusion_rule = "weighted_physical"
                memory_role = "FUSÃO 70/30"
        else:
            if memory_similarity >= 0.85:
                physical_weight, memory_weight = 0.0, 1.0
                fusion_rule = "memory_override"
                memory_role = "MEMÓRIA DECISIVA"
            elif memory_similarity >= 0.75:
                physical_weight, memory_weight = 0.50, 0.50
                fusion_rule = "balanced_fusion"
                memory_role = "FUSÃO EQUILIBRADA"
            else:
                physical_weight, memory_weight = 0.70, 0.30
                fusion_rule = "weighted_low_similarity"
                memory_role = "MEMÓRIA AUXILIAR"
        final_score = physical_score * physical_weight + memory_vote * memory_weight

    final_score = float(np.clip(final_score, 0.0, 1.0))
    is_defect = bool(final_score > orchestrator.DECISION_CUTOFF)
    distance_max = (
        1.0 - orchestrator.DECISION_CUTOFF
        if is_defect
        else orchestrator.DECISION_CUTOFF
    )
    current_distance = (
        final_score - orchestrator.DECISION_CUTOFF
        if is_defect
        else orchestrator.DECISION_CUTOFF - final_score
    )
    confidence = float(
        max(
            0.50,
            min(
                0.99,
                0.50
                + 0.49 * (current_distance / max(distance_max, 1e-6)),
            ),
        )
    )

    for engine in engines:
        if engine["selected"]:
            engine["final_influence"] = float(
                engine["effective_score"] * physical_weight
            )

    memory_entry = orchestrator._engine_entry(
        "knn",
        "Memória local KNN",
        has_memory,
        has_memory and memory_vote > orchestrator.DECISION_CUTOFF,
        memory_vote,
        memory_vote,
        orchestrator.DECISION_CUTOFF,
        (
            f"{memory_role}; similaridade {memory_similarity:.0%}; "
            f"{neighbors} vizinho(s)"
            if has_memory
            else "Dataset sem anomalias compatíveis"
        ),
    )
    memory_entry["selected"] = bool(memory_weight > 0.0)
    memory_entry["final_influence"] = float(memory_vote * memory_weight)
    engines.append(memory_entry)

    if memory_weight >= physical_weight and memory_weight > 0:
        dominant_engine = "knn"
    elif physical_dominant:
        dominant_engine = physical_dominant["id"]
    else:
        dominant_engine = "none"

    if not physical_reasons:
        physical_reasons.append("Sem anomalias físicas significativas")
    reason = " | ".join(item for item in physical_reasons if item)
    if has_memory:
        reason += (
            f" || KNN {memory_vote:.0%} NG; similaridade "
            f"{memory_similarity:.0%}; {memory_role}"
        )

    decision_trace = {
        "schema": orchestrator.DECISION_SCHEMA,
        "cutoff": orchestrator.DECISION_CUTOFF,
        "final_score": final_score,
        "confidence": confidence,
        "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
        "physical_score": physical_score,
        "physical_defect": bool(physical_defect),
        "dominant_engine": dominant_engine,
        "fusion_rule": fusion_rule,
        "weights": {
            "physical": float(physical_weight),
            "knn": float(memory_weight),
        },
        "memory": {
            "has_memory": has_memory,
            "vote_defect": float(memory_vote),
            "best_similarity": float(memory_similarity),
            "n_neighbors": neighbors,
            "role": memory_role,
            "best_match_label": str(
                (knn or {}).get("best_match_label", "")
            ),
            "memory_mode": str((knn or {}).get("memory_mode", "none")),
            "memory_scope": str((knn or {}).get("memory_scope", "none")),
        },
        "engines": engines,
    }
    return final_score, is_defect, confidence, reason, decision_trace


def install_anomaly_memory_integration(orchestrator_cls) -> None:
    """Executa motores por categoria e consulta o KNN após a anomalia existir."""
    if getattr(orchestrator_cls, "_anomaly_memory_installed", False):
        return

    original_inspect = orchestrator_cls.inspect

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        category = str((aoi_info or {}).get("category", "Unknown"))
        had_route = category in self.routing_table
        original_route = list(self.routing_table.get(category, []))
        self.routing_table[category] = [
            engine
            for engine in routes_for_category(category)
            if engine not in {"knn", "missing"}
        ]

        try:
            analysis = original_inspect(
                self,
                full_gab,
                full_test,
                raw_anomalies,
                aoi_info,
                global_box_info,
                aoi_epicenters,
            )
        finally:
            if had_route:
                self.routing_table[category] = original_route
            else:
                self.routing_table.pop(category, None)

        detail = analysis.setdefault("detail", {})
        active_engines = analysis.setdefault("active_engines", [])

        missing_result = None
        if is_missing_category(category):
            if "missing" not in self.experts:
                self.experts["missing"] = MissingComponentExpert()
            missing_result = self.experts["missing"].analyze(
                full_gab,
                full_test,
                global_box_info,
                aoi_info,
                aoi_epicenters,
            )
            detail.update(missing_result)
            if missing_result.get("missing_active", False):
                if "missing_expert.py" not in active_engines:
                    active_engines.append("missing_expert.py")
                if missing_result.get("missing_bounding_box"):
                    analysis["bounding_box"] = missing_result["missing_bounding_box"]
                    analysis.setdefault("all_boxes", {})["missing"] = missing_result[
                        "missing_bounding_box"
                    ]

        focus = _focus_box(aoi_epicenters, analysis, detail)
        signature_detail = dict(detail)
        if missing_result and missing_result.get("component_missing_mask") is not None:
            signature_detail["missing_mask"] = missing_result[
                "component_missing_mask"
            ]
            signature_detail["missing_pct"] = max(
                float(signature_detail.get("missing_pct", 0.0)),
                float(missing_result.get("missing_coverage", 0.0)),
            )
        signature = build_anomaly_signature(
            full_gab,
            full_test,
            signature_detail,
            aoi_info,
            focus,
        )

        knn_expert = self.experts["knn"]
        _normalize_knn_memory_categories(knn_expert)
        knn_result = knn_expert.analyze(
            full_gab,
            full_test,
            None,
            None,
            aoi_info,
            anomaly_signature=signature,
        )

        (
            final_score,
            is_defect,
            confidence,
            reason,
            decision_trace,
        ) = _dynamic_fusion(
            self,
            detail,
            category,
            missing_result,
            knn_result,
        )

        analysis["is_defect"] = is_defect
        analysis["confidence"] = confidence
        analysis["verdict"] = "DEFEITO REAL" if is_defect else "FALHA FALSA"
        analysis["reason"] = reason

        detail.update(knn_result)
        detail.update(
            {
                "anomaly_signature": signature,
                "query_anomaly_signature": signature,
                "final_score": final_score,
                "physical_score": decision_trace["physical_score"],
                "decision_cutoff": decision_trace["cutoff"],
                "dominant_engine": decision_trace["dominant_engine"],
                "fusion_rule": decision_trace["fusion_rule"],
                "decision_trace": decision_trace,
            }
        )

        if "knn_expert.py" not in active_engines:
            active_engines.append("knn_expert.py")
        return analysis

    orchestrator_cls.inspect = inspect
    orchestrator_cls._anomaly_memory_installed = True


__all__ = [
    "canonical_category_key",
    "install_anomaly_memory_integration",
    "is_adhesive_category",
    "is_missing_category",
    "routes_for_category",
]
