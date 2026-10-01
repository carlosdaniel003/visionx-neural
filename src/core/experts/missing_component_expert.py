"""Compatibilidade de importação para o motor FALTANDO orientado ao patch."""

import cv2
import numpy as np

from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert


class MissingComponentExpert(ROIPatchExpectationExpert):
    """Especialista FALTANDO com confirmação de ausência física forte.

    A memória KNN pode reconhecer padrões antigos, mas não deve anular uma
    ausência física comprovada pelo comparador gabarito × teste.
    """

    HARD_ABSENCE_MIN_SCORE = 0.85
    HARD_ABSENCE_MIN_COVERAGE = 0.30
    HARD_ABSENCE_MIN_RESIDUAL = 0.45
    HARD_ABSENCE_MIN_STRUCTURE_LOSS = 0.30
    HARD_ABSENCE_MIN_BACKGROUND_EXPOSURE = 0.28
    HARD_ABSENCE_MAX_NEARBY_SIMILARITY = 0.60

    # Terceiro padrão real de ausência: o componente desaparece, mas deixa
    # footprint/base escura ou material subjacente semelhante à sua região.
    # Nessa situação background_exposure e structure_loss podem ficar baixos
    # apesar de a aparência original ter desaparecido quase por completo.
    HARD_FOOTPRINT_MIN_SCORE = 0.90
    HARD_FOOTPRINT_MIN_COVERAGE = 0.45
    HARD_FOOTPRINT_MIN_RESIDUAL = 0.60
    HARD_FOOTPRINT_MIN_APPEARANCE_LOSS = 0.55
    HARD_FOOTPRINT_MAX_NEARBY_SIMILARITY = 0.35

    @staticmethod
    def _palette_residual(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
        """Mede se o teste ainda pertence à paleta cromática do patch.

        Em patches homogêneos, como um trecho do corpo preto do componente,
        pequenas mudanças de brilho não devem transformar toda a ROI em defeito.
        """
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        test_lab = cv2.cvtColor(test, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = reference_lab.reshape(-1, 3)
        center = np.median(pixels, axis=0)
        mad = np.median(np.abs(pixels - center), axis=0) * 1.4826
        scale = np.maximum(mad * 2.6, np.asarray([14.0, 9.0, 9.0], dtype=np.float32))
        normalized = (test_lab - center.reshape(1, 1, 3)) / scale.reshape(1, 1, 3)
        distance = np.sqrt(np.mean(normalized * normalized, axis=2))
        return np.clip((distance - 0.75) / 3.2, 0.0, 1.0).astype(np.float32)

    @classmethod
    def _hard_absence_evidence(cls, result: dict) -> tuple[bool, str]:
        """Confirma desaparecimento físico sem confundir deslocamento com falta."""
        if not bool(result.get("missing_active", False)):
            return False, "motor inativo"
        if not bool(result.get("missing_is_defect", False)):
            return False, "ROI sem divergência física suficiente"

        classification = str(
            result.get("missing_classification", "")
        ).strip().upper()
        if classification == "DESLOCAMENTO PROVÁVEL":
            return False, "conteúdo compatível encontrado deslocado"

        score = float(result.get("missing_score", 0.0) or 0.0)
        coverage = float(result.get("missing_changed_coverage", 0.0) or 0.0)
        residual = float(result.get("missing_residual_mean", 0.0) or 0.0)
        structure_loss = float(result.get("missing_structure_loss", 0.0) or 0.0)
        appearance_loss = float(
            result.get("missing_appearance_loss", 0.0) or 0.0
        )
        direct_similarity = float(
            result.get("missing_direct_similarity", 1.0) or 0.0
        )
        background = float(
            result.get("missing_background_exposure", 0.0) or 0.0
        )
        nearby_similarity = float(
            result.get("missing_best_similarity", 0.0) or 0.0
        )

        background_absence = bool(
            score >= 0.72
            and coverage >= 0.25
            and background >= cls.HARD_ABSENCE_MIN_BACKGROUND_EXPOSURE
            and nearby_similarity < cls.HARD_ABSENCE_MAX_NEARBY_SIMILARITY
        )
        structural_collapse = bool(
            score >= cls.HARD_ABSENCE_MIN_SCORE
            and coverage >= cls.HARD_ABSENCE_MIN_COVERAGE
            and residual >= cls.HARD_ABSENCE_MIN_RESIDUAL
            and structure_loss >= cls.HARD_ABSENCE_MIN_STRUCTURE_LOSS
            and nearby_similarity < cls.HARD_ABSENCE_MAX_NEARBY_SIMILARITY
        )

        footprint_absence = bool(
            score >= cls.HARD_FOOTPRINT_MIN_SCORE
            and coverage >= cls.HARD_FOOTPRINT_MIN_COVERAGE
            and residual >= cls.HARD_FOOTPRINT_MIN_RESIDUAL
            and appearance_loss >= cls.HARD_FOOTPRINT_MIN_APPEARANCE_LOSS
            and direct_similarity <= (
                1.0 - cls.HARD_FOOTPRINT_MIN_APPEARANCE_LOSS
            )
            and nearby_similarity < cls.HARD_FOOTPRINT_MAX_NEARBY_SIMILARITY
        )

        if background_absence:
            return (
                True,
                "conteúdo do gabarito foi substituído pelo fundo da região",
            )
        if structural_collapse:
            return (
                True,
                "estrutura esperada colapsou sem correspondência próxima válida",
            )
        if footprint_absence:
            return (
                True,
                "aparência esperada desapareceu e restou apenas footprint/base sem correspondência válida",
            )
        return False, "divergência presente, mas sem prova forte de ausência"

    def analyze(
        self,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
    ) -> dict:
        result = super().analyze(
            full_reference,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )

        hard_absence, hard_reason = self._hard_absence_evidence(result)
        result["missing_hard_absence"] = bool(hard_absence)
        result["missing_hard_absence_reason"] = hard_reason
        result["missing_hard_absence_thresholds"] = {
            "score": self.HARD_ABSENCE_MIN_SCORE,
            "coverage": self.HARD_ABSENCE_MIN_COVERAGE,
            "residual_mean": self.HARD_ABSENCE_MIN_RESIDUAL,
            "structure_loss": self.HARD_ABSENCE_MIN_STRUCTURE_LOSS,
            "background_exposure": self.HARD_ABSENCE_MIN_BACKGROUND_EXPOSURE,
            "nearby_similarity_max": self.HARD_ABSENCE_MAX_NEARBY_SIMILARITY,
            "footprint_score": self.HARD_FOOTPRINT_MIN_SCORE,
            "footprint_coverage": self.HARD_FOOTPRINT_MIN_COVERAGE,
            "footprint_residual_mean": self.HARD_FOOTPRINT_MIN_RESIDUAL,
            "footprint_appearance_loss": self.HARD_FOOTPRINT_MIN_APPEARANCE_LOSS,
            "footprint_nearby_similarity_max": (
                self.HARD_FOOTPRINT_MAX_NEARBY_SIMILARITY
            ),
        }

        if hard_absence:
            result["missing_classification"] = "COMPONENTE FISICAMENTE AUSENTE"
            base_reason = str(result.get("missing_reason", "") or "")
            result["missing_reason"] = (
                f"{base_reason} • AUSÊNCIA FÍSICA FORTE: {hard_reason}"
                if base_reason
                else f"AUSÊNCIA FÍSICA FORTE: {hard_reason}"
            )

        return result

    @classmethod
    def _residual_and_mask(cls, reference: np.ndarray, test: np.ndarray):
        color_residual = cls._local_color_residual(reference, test)
        edge_anomaly, missing_edges, extra_edges, edge_mismatch = cls._edge_metrics(
            reference,
            test,
        )
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        texture = float(np.std(reference_gray) / 64.0)
        edge_density = float(np.mean(cls._auto_edges(reference) > 0))
        homogeneous = texture < 0.48 and edge_density < 0.10

        if homogeneous:
            palette_residual = cls._palette_residual(reference, test)
            color_residual = np.minimum(color_residual, palette_residual)

        edge_layer = (edge_anomaly > 0).astype(np.float32)
        residual = np.clip(0.84 * color_residual + 0.16 * edge_layer, 0.0, 1.0)
        threshold = 0.29 if homogeneous else 0.31
        mask = (residual >= threshold).astype(np.uint8) * 255
        mask = cls._clean_mask(mask)
        return (
            residual,
            mask,
            missing_edges,
            extra_edges,
            edge_mismatch,
            texture,
            edge_density,
        )


__all__ = ["MissingComponentExpert"]
