"""Integra a assinatura de anomalia ao MoE sem duplicar o orquestrador."""

from __future__ import annotations

import re

from src.core.anomaly_signature import build_anomaly_signature


ADHESIVE_CATEGORIES = frozenset({"MUCH ADHESIVE", "MUITO ADESIVO"})
STANDARD_ROUTES = ("silk", "ssim", "semantic", "knn")
ADHESIVE_ROUTES = ("shift",) + STANDARD_ROUTES

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
    """O fluxo de adesivo depende exclusivamente da categoria AOI."""
    return normalize_category_name(category) in ADHESIVE_CATEGORIES


def routes_for_category(category: str) -> tuple[str, ...]:
    return ADHESIVE_ROUTES if is_adhesive_category(category) else STANDARD_ROUTES


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
        detail.get("roi_box"),
        analysis.get("bounding_box"),
    ):
        if candidate and len(candidate) >= 4:
            return tuple(int(value) for value in candidate[:4])
    return None


def _physical_inputs(detail: dict, category: str):
    shift = None
    if is_adhesive_category(category) and (
        "shift_active" in detail or "adhesive_score" in detail
    ):
        adhesive_active = bool(detail.get("shift_active", False))
        adhesive_score = float(detail.get("adhesive_score", 0.0))
        adhesive_tolerance = float(detail.get("adhesive_tolerance", 0.32))
        adhesive_defect = bool(
            detail.get(
                "adhesive_is_defect",
                adhesive_active and adhesive_score > adhesive_tolerance,
            )
        )
        shift = {
            "shift_active": adhesive_active,
            "adhesive_score": adhesive_score,
            "adhesive_tolerance": adhesive_tolerance,
            "adhesive_is_defect": adhesive_defect,
            "is_defect": adhesive_defect,
            "adhesive_reason": detail.get("adhesive_reason", ""),
            "reason": detail.get("adhesive_reason", ""),
        }

    silk = None
    if "silk_error_pct" in detail:
        structural_error = float(detail.get("silk_error_pct", 0.0))
        structural_tolerance = 0.08
        silk = {
            "silk_error_pct": structural_error,
            "tolerance": structural_tolerance,
            "is_defect": structural_error > structural_tolerance,
            "reason": (
                f"Divergência estrutural {structural_error:.0%}"
                if structural_error > 0
                else ""
            ),
        }

    semantic = None
    if "semantic_loss" in detail:
        semantic_score = float(detail.get("semantic_loss", 0.0))
        semantic = {
            "semantic_loss": semantic_score,
            "is_defect": semantic_score > 0.45,
            "reason": detail.get(
                "semantic_reason",
                f"Evidência semântica {semantic_score:.0%}",
            ),
        }

    ssim = None
    if "local_score" in detail or "ssim" in detail:
        ssim = {
            "local_score": float(detail.get("local_score", 0.0)),
            "ctx_score": float(detail.get("ctx_score", 0.0)),
            "decision_threshold": float(
                detail.get("decision_threshold", 0.45)
            ),
            "ssim": float(detail.get("ssim", 1.0)),
            "pct_changed": float(detail.get("pct_changed", 0.0)),
        }
    return shift, silk, semantic, ssim


def install_anomaly_memory_integration(orchestrator_cls) -> None:
    """Faz o KNN consultar a anomalia após os motores físicos."""
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

        # O KNN é executado depois da construção da assinatura. O motor Shift
        # só entra na primeira etapa quando a categoria é realmente adesiva.
        self.routing_table[category] = [
            engine
            for engine in routes_for_category(category)
            if engine != "knn"
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
        focus = _focus_box(aoi_epicenters, analysis, detail)
        signature = build_anomaly_signature(
            full_gab,
            full_test,
            detail,
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

        shift, silk, semantic, ssim = _physical_inputs(detail, category)
        (
            final_score,
            is_defect,
            confidence,
            reason,
            decision_trace,
        ) = self._master_fusion_score(
            shift,
            silk,
            semantic,
            ssim,
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

        active_engines = analysis.setdefault("active_engines", [])
        if "knn_expert.py" not in active_engines:
            active_engines.append("knn_expert.py")
        return analysis

    orchestrator_cls.inspect = inspect
    orchestrator_cls._anomaly_memory_installed = True
