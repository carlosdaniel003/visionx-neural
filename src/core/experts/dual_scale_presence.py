"""Guarda dual-scale de presença física.

A escala local (epicentro da AOI) continua sendo a fonte de precisão. Quando
essa ROI representa uma fração pequena do componente ou contradiz evidências
físicas globais, este módulo analisa uma segunda ROI contextual maior antes de
permitir que a memória KNN decida OK/NG.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DualScalePresenceAnalyzer:
    """Compara presença local + contexto sem alterar a categoria da AOI."""

    POLICY = "dual_scale_presence_v1"

    MAX_LOCAL_GLOBAL_AREA_RATIO = 0.25
    CONTEXT_EXPANSION = 2.50
    CONTEXT_GLOBAL_FRACTION = 0.55

    MIN_CONTEXT_SCORE = 0.72
    MIN_CONTEXT_COVERAGE = 0.30
    MIN_CONTEXT_RESIDUAL = 0.50
    MIN_CONTEXT_APPEARANCE_LOSS = 0.35
    MIN_CONTEXT_STRUCTURE_LOSS = 0.20
    MIN_CONTEXT_EDGE_MISMATCH = 0.30
    MAX_CONTEXT_NEARBY_SIMILARITY = 0.35

    MIN_STRUCTURAL_SUPPORT = 0.45
    MIN_SEMANTIC_SUPPORT = 0.45
    STRONG_SINGLE_SUPPORT = 0.65

    EXTREME_CONTEXT_SCORE = 0.85
    EXTREME_CONTEXT_COVERAGE = 0.45
    EXTREME_CONTEXT_RESIDUAL = 0.60
    EXTREME_CONTEXT_APPEARANCE_LOSS = 0.50
    EXTREME_CONTEXT_NEARBY_SIMILARITY = 0.25

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _local_global_ratio(
        cls,
        roi_box,
        image_shape,
        global_box_info: dict | None,
    ) -> float:
        image_height, image_width = image_shape[:2]
        _, _, roi_width, roi_height = roi_box
        local_area = max(1, int(roi_width) * int(roi_height))

        global_width = min(
            image_width,
            max(
                1,
                int(
                    round(
                        cls._float(
                            (global_box_info or {}).get("w"),
                            image_width,
                        )
                    )
                ),
            ),
        )
        global_height = min(
            image_height,
            max(
                1,
                int(
                    round(
                        cls._float(
                            (global_box_info or {}).get("h"),
                            image_height,
                        )
                    )
                ),
            ),
        )
        global_area = max(1, global_width * global_height)
        return float(np.clip(local_area / global_area, 0.0, 1.0))

    @classmethod
    def _context_box(
        cls,
        roi_box,
        image_shape,
        global_box_info: dict | None,
    ):
        image_height, image_width = image_shape[:2]
        x, y, roi_width, roi_height = [int(round(v)) for v in roi_box]

        global_width = min(
            image_width,
            max(
                roi_width,
                int(
                    round(
                        cls._float(
                            (global_box_info or {}).get("w"),
                            image_width,
                        )
                    )
                ),
            ),
        )
        global_height = min(
            image_height,
            max(
                roi_height,
                int(
                    round(
                        cls._float(
                            (global_box_info or {}).get("h"),
                            image_height,
                        )
                    )
                ),
            ),
        )

        target_width = min(
            image_width,
            max(
                int(round(roi_width * cls.CONTEXT_EXPANSION)),
                int(round(global_width * cls.CONTEXT_GLOBAL_FRACTION)),
            ),
        )
        target_height = min(
            image_height,
            max(
                int(round(roi_height * cls.CONTEXT_EXPANSION)),
                int(round(global_height * cls.CONTEXT_GLOBAL_FRACTION)),
            ),
        )

        center_x = float(x) + float(roi_width) / 2.0
        center_y = float(y) + float(roi_height) / 2.0
        x1 = int(round(center_x - target_width / 2.0))
        y1 = int(round(center_y - target_height / 2.0))
        x1 = max(0, min(image_width - target_width, x1))
        y1 = max(0, min(image_height - target_height, y1))
        return x1, y1, target_width, target_height

    @classmethod
    def _physical_support(cls, physical_detail: dict | None) -> dict:
        detail = physical_detail if isinstance(physical_detail, dict) else {}
        structural = cls._float(detail.get("silk_error_pct", 0.0))
        semantic = cls._float(detail.get("semantic_loss", 0.0))

        supported = bool(
            (
                structural >= cls.MIN_STRUCTURAL_SUPPORT
                and semantic >= cls.MIN_SEMANTIC_SUPPORT
            )
            or max(structural, semantic) >= cls.STRONG_SINGLE_SUPPORT
        )
        return {
            "supported": supported,
            "structural": structural,
            "semantic": semantic,
        }

    @classmethod
    def _context_hard_absence(
        cls,
        metrics: dict,
        physical_support: dict,
    ) -> tuple[bool, str]:
        score = cls._float(metrics.get("score"))
        coverage = cls._float(metrics.get("coverage"))
        residual = cls._float(metrics.get("residual_mean"))
        appearance = cls._float(metrics.get("appearance_loss"))
        structure = cls._float(metrics.get("structure_loss"))
        edge = cls._float(metrics.get("edge_mismatch"))
        nearby = cls._float(metrics.get("best_similarity"))

        contextual = bool(
            score >= cls.MIN_CONTEXT_SCORE
            and coverage >= cls.MIN_CONTEXT_COVERAGE
            and residual >= cls.MIN_CONTEXT_RESIDUAL
            and appearance >= cls.MIN_CONTEXT_APPEARANCE_LOSS
            and nearby < cls.MAX_CONTEXT_NEARBY_SIMILARITY
            and (
                structure >= cls.MIN_CONTEXT_STRUCTURE_LOSS
                or edge >= cls.MIN_CONTEXT_EDGE_MISMATCH
            )
        )
        extreme = bool(
            score >= cls.EXTREME_CONTEXT_SCORE
            and coverage >= cls.EXTREME_CONTEXT_COVERAGE
            and residual >= cls.EXTREME_CONTEXT_RESIDUAL
            and appearance >= cls.EXTREME_CONTEXT_APPEARANCE_LOSS
            and nearby < cls.EXTREME_CONTEXT_NEARBY_SIMILARITY
        )

        if contextual and bool(physical_support.get("supported", False)):
            return (
                True,
                "contexto maior confirma desaparecimento físico que a ROI "
                "local não representa",
            )
        if extreme:
            return (
                True,
                "contexto maior apresenta colapso visual extremo independente "
                "da leitura local",
            )
        return False, "contexto não confirmou ausência física forte"

    @classmethod
    def analyze(
        cls,
        expert,
        full_reference,
        full_test,
        local_result: dict,
        global_box_info: dict | None = None,
        physical_detail: dict | None = None,
    ) -> dict:
        output = {
            "missing_dual_scale_policy": cls.POLICY,
            "missing_dual_scale_active": False,
            "missing_dual_scale_triggered": False,
            "missing_scale_disagreement": False,
            "missing_local_global_area_ratio": 1.0,
            "missing_context_box": None,
            "missing_context_area_ratio": 0.0,
            "missing_context_score": 0.0,
            "missing_context_coverage": 0.0,
            "missing_context_residual_mean": 0.0,
            "missing_context_residual_p90": 0.0,
            "missing_context_structure_loss": 0.0,
            "missing_context_edge_mismatch": 0.0,
            "missing_context_direct_similarity": 1.0,
            "missing_context_appearance_loss": 0.0,
            "missing_context_best_similarity": 0.0,
            "missing_context_hard_absence": False,
            "missing_context_hard_reason": "",
            "missing_context_physical_support": cls._physical_support(
                physical_detail
            ),
        }

        if (
            full_reference is None
            or full_test is None
            or not isinstance(local_result, dict)
            or full_reference.size == 0
            or full_test.size == 0
        ):
            output["missing_context_hard_reason"] = "imagem/resultado local inválido"
            return output

        local_box = local_result.get("missing_roi_box")
        if not local_box or len(local_box) < 4:
            output["missing_context_hard_reason"] = "ROI local indisponível"
            return output

        full_reference, full_test = expert._safe_pair(
            full_reference,
            full_test,
        )
        local_ratio = cls._local_global_ratio(
            local_box,
            full_reference.shape,
            global_box_info,
        )
        output["missing_local_global_area_ratio"] = local_ratio

        support = output["missing_context_physical_support"]
        local_classification = str(
            local_result.get("missing_classification", "")
        ).strip().upper()
        if local_classification == "DESLOCAMENTO PROVÁVEL":
            output["missing_dual_scale_active"] = False
            output["missing_context_hard_reason"] = (
                "escala local encontrou conteúdo compatível deslocado; "
                "dual-scale não pode converter deslocamento em ausência"
            )
            return output

        local_is_defect = bool(local_result.get("missing_is_defect", False))
        scale_disagreement = bool(
            not local_is_defect and support.get("supported", False)
        )
        output["missing_scale_disagreement"] = scale_disagreement

        should_run = bool(
            local_ratio <= cls.MAX_LOCAL_GLOBAL_AREA_RATIO
            or scale_disagreement
        )
        output["missing_dual_scale_active"] = should_run
        if not should_run:
            output["missing_context_hard_reason"] = (
                "epicentro representa área suficiente e não há contradição "
                "entre escalas"
            )
            return output

        context_box = cls._context_box(
            local_box,
            full_reference.shape,
            global_box_info,
        )
        output["missing_context_box"] = context_box
        output["missing_dual_scale_triggered"] = True

        _, _, local_width, local_height = [int(round(v)) for v in local_box]
        _, _, context_width, context_height = context_box
        context_area = max(1, context_width * context_height)
        output["missing_context_area_ratio"] = float(
            np.clip(
                (local_width * local_height) / context_area,
                0.0,
                1.0,
            )
        )

        normalized_test = expert._normalize_illumination(
            full_reference,
            full_test,
            context_box,
        )
        reference = expert._crop(full_reference, context_box)
        test = expert._crop(normalized_test, context_box)
        if (
            reference.size == 0
            or test.size == 0
            or min(reference.shape[:2]) < 8
        ):
            output["missing_context_hard_reason"] = "ROI contextual inválida"
            return output

        (
            residual,
            anomaly_mask,
            structure_loss,
            _extra_structure,
            edge_mismatch,
            _texture,
            _edge_density,
        ) = expert._residual_and_mask(reference, test)

        coverage = float(np.mean(anomaly_mask > 0))
        residual_mean = (
            float(np.mean(residual[anomaly_mask > 0]))
            if coverage > 0
            else float(np.mean(residual))
        )
        residual_p90 = float(np.percentile(residual, 90))
        direct_similarity = expert._direct_similarity(
            residual,
            coverage,
            edge_mismatch,
        )
        score = expert._score(
            coverage,
            residual_mean,
            residual_p90,
            edge_mismatch,
        )
        distinctness = expert._patch_distinctness(reference)
        best_similarity, _, _, _ = expert._nearby_match(
            full_reference,
            normalized_test,
            context_box,
            distinctness,
        )

        metrics = {
            "score": score,
            "coverage": coverage,
            "residual_mean": residual_mean,
            "residual_p90": residual_p90,
            "structure_loss": float(structure_loss),
            "edge_mismatch": float(edge_mismatch),
            "direct_similarity": float(direct_similarity),
            "appearance_loss": float(1.0 - direct_similarity),
            "best_similarity": float(best_similarity),
        }
        hard, reason = cls._context_hard_absence(metrics, support)

        output.update(
            {
                "missing_context_score": metrics["score"],
                "missing_context_coverage": metrics["coverage"],
                "missing_context_residual_mean": metrics["residual_mean"],
                "missing_context_residual_p90": metrics["residual_p90"],
                "missing_context_structure_loss": metrics["structure_loss"],
                "missing_context_edge_mismatch": metrics["edge_mismatch"],
                "missing_context_direct_similarity": metrics[
                    "direct_similarity"
                ],
                "missing_context_appearance_loss": metrics[
                    "appearance_loss"
                ],
                "missing_context_best_similarity": metrics[
                    "best_similarity"
                ],
                "missing_context_hard_absence": bool(hard),
                "missing_context_hard_reason": reason,
            }
        )
        return output


__all__ = ["DualScalePresenceAnalyzer"]
