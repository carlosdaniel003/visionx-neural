"""Guarda transversal de ausência física para categorias não-FALTANDO.

A categoria da AOI continua sendo preservada para roteamento, memória e auditoria.
Esta guarda só impede que uma memória OK forte transforme em falha falsa um caso
em que o componente desapareceu fisicamente, mesmo que a AOI tenha nomeado a
anomalia como EMBORCADO, DESLOCADO ou INVERTIDO.
"""

from __future__ import annotations

from typing import Any

from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert


class PhysicalAbsenceGuard(ROIPatchExpectationExpert):
    """Confirma ausência física inequívoca fora da categoria FALTANDO."""

    CATEGORIES = frozenset({"EMBORCADO", "DESLOCADO", "INVERTIDO"})

    POLICY = "cross_category_physical_absence_guard_v1"

    MIN_SCORE = 0.82
    MIN_COVERAGE = 0.45
    MIN_RESIDUAL_MEAN = 0.38
    MIN_APPEARANCE_LOSS = 0.40
    MAX_DIRECT_SIMILARITY = 0.60
    MIN_EDGE_MISMATCH = 0.40
    MAX_NEARBY_SIMILARITY = 0.25

    MIN_STRUCTURAL_SUPPORT = 0.35
    MIN_SEMANTIC_SUPPORT = 0.60

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _physical_support(cls, detail: dict | None) -> dict:
        detail = detail if isinstance(detail, dict) else {}
        structural = cls._float(detail.get("silk_error_pct", 0.0))
        semantic = cls._float(detail.get("semantic_loss", 0.0))
        physical_score = cls._float(detail.get("physical_score", 0.0))

        supported = bool(
            structural >= cls.MIN_STRUCTURAL_SUPPORT
            and semantic >= cls.MIN_SEMANTIC_SUPPORT
            and physical_score >= 0.85
        )
        return {
            "supported": supported,
            "structural": structural,
            "semantic": semantic,
            "physical_score": physical_score,
        }

    @classmethod
    def hard_absence_evidence(
        cls,
        result: dict,
        physical_detail: dict | None,
    ) -> tuple[bool, str, dict]:
        support = cls._physical_support(physical_detail)

        if not bool(result.get("missing_active", False)):
            return False, "guarda inativa para a categoria", support
        if not bool(result.get("missing_is_defect", False)):
            return False, "ROI sem divergência suficiente", support

        classification = str(
            result.get("missing_classification", "")
        ).strip().upper()
        if classification == "DESLOCAMENTO PROVÁVEL":
            return False, "conteúdo compatível encontrado deslocado", support

        if not support["supported"]:
            return (
                False,
                "motores físicos independentes não confirmaram a ausência",
                support,
            )

        score = cls._float(result.get("missing_score", 0.0))
        coverage = cls._float(result.get("missing_changed_coverage", 0.0))
        residual = cls._float(result.get("missing_residual_mean", 0.0))
        appearance_loss = cls._float(
            result.get("missing_appearance_loss", 0.0)
        )
        direct_similarity = cls._float(
            result.get("missing_direct_similarity", 1.0),
            1.0,
        )
        edge_mismatch = cls._float(
            result.get("missing_edge_mismatch", 0.0)
        )
        nearby_similarity = cls._float(
            result.get("missing_best_similarity", 0.0)
        )

        confirmed = bool(
            score >= cls.MIN_SCORE
            and coverage >= cls.MIN_COVERAGE
            and residual >= cls.MIN_RESIDUAL_MEAN
            and appearance_loss >= cls.MIN_APPEARANCE_LOSS
            and direct_similarity <= cls.MAX_DIRECT_SIMILARITY
            and edge_mismatch >= cls.MIN_EDGE_MISMATCH
            and nearby_similarity < cls.MAX_NEARBY_SIMILARITY
        )

        if confirmed:
            return (
                True,
                "aparência do componente desapareceu sem correspondência "
                "próxima, confirmada pelos motores estrutural e semântico",
                support,
            )
        return (
            False,
            "diferença visual forte, mas abaixo do contrato transversal "
            "de ausência física",
            support,
        )

    def analyze(
        self,
        full_reference,
        full_test,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
        physical_detail: dict | None = None,
    ) -> dict:
        result = super().analyze(
            full_reference,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )

        category = str((aoi_info or {}).get("category", "") or "").strip().upper()
        hard, reason, support = self.hard_absence_evidence(
            result,
            physical_detail,
        )

        result.update(
            {
                "missing_cross_category_guard": True,
                "missing_guard_policy": self.POLICY,
                "missing_guard_source_category": category,
                "missing_guard_physical_support": support,
                "missing_hard_absence": bool(hard),
                "missing_hard_absence_reason": reason,
                "missing_hard_absence_thresholds": {
                    "score": self.MIN_SCORE,
                    "coverage": self.MIN_COVERAGE,
                    "residual_mean": self.MIN_RESIDUAL_MEAN,
                    "appearance_loss": self.MIN_APPEARANCE_LOSS,
                    "direct_similarity_max": self.MAX_DIRECT_SIMILARITY,
                    "edge_mismatch": self.MIN_EDGE_MISMATCH,
                    "nearby_similarity_max": self.MAX_NEARBY_SIMILARITY,
                    "structural_support": self.MIN_STRUCTURAL_SUPPORT,
                    "semantic_support": self.MIN_SEMANTIC_SUPPORT,
                },
            }
        )

        if hard:
            result["missing_classification"] = (
                "COMPONENTE FISICAMENTE AUSENTE — GUARDA TRANSVERSAL"
            )
            base = str(result.get("missing_reason", "") or "")
            suffix = (
                "AUSÊNCIA FÍSICA FORTE FORA DA CATEGORIA FALTANDO: "
                f"{reason}"
            )
            result["missing_reason"] = (
                f"{base} • {suffix}" if base else suffix
            )

        return result


__all__ = ["PhysicalAbsenceGuard"]
