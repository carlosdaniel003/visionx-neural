"""Compatibilidade de importação para o motor FALTANDO orientado ao patch."""

import cv2
import numpy as np

from src.core.experts.dual_scale_presence import DualScalePresenceAnalyzer
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
    HARD_FOOTPRINT_MIN_APPEARANCE_LOSS = 0.50
    HARD_FOOTPRINT_MAX_NEARBY_SIMILARITY = 0.35

    # Testemunha geométrica de presença. Ela procura o corpo do componente em
    # baixa frequência, ignorando serigrafia, texto e mudança global de brilho.
    BODY_PRESENCE_MIN_COARSE_SIMILARITY = 0.70
    BODY_PRESENCE_MIN_SILHOUETTE_DICE = 0.78
    BODY_PRESENCE_MIN_AREA_RATIO = 0.62
    BODY_PRESENCE_MAX_AREA_RATIO = 1.55
    BODY_PRESENCE_MAX_CENTROID_SHIFT = 0.10

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

    @staticmethod
    def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_f = left.astype(np.float32).reshape(-1)
        right_f = right.astype(np.float32).reshape(-1)
        left_f -= float(np.mean(left_f))
        right_f -= float(np.mean(right_f))
        denominator = float(
            np.linalg.norm(left_f) * np.linalg.norm(right_f)
        )
        if denominator <= 1e-8:
            return 0.0
        return float(
            np.clip(
                np.dot(left_f, right_f) / denominator,
                -1.0,
                1.0,
            )
        )

    @classmethod
    def _coarse_body_similarity(
        cls,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> float:
        """Compara a massa/forma de baixa frequência, não a serigrafia."""
        if (
            not isinstance(reference, np.ndarray)
            or not isinstance(test, np.ndarray)
            or reference.size == 0
            or test.size == 0
        ):
            return 0.0

        if reference.shape[:2] != test.shape[:2]:
            test = cv2.resize(
                test,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        reference_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY,
        )
        test_gray = cv2.cvtColor(
            test,
            cv2.COLOR_BGR2GRAY,
        )

        minimum_side = max(1, min(reference_gray.shape[:2]))
        kernel = max(5, int(round(minimum_side * 0.09)))
        if kernel % 2 == 0:
            kernel += 1

        reference_blur = cv2.GaussianBlur(
            reference_gray,
            (kernel, kernel),
            0,
        )
        test_blur = cv2.GaussianBlur(
            test_gray,
            (kernel, kernel),
            0,
        )

        # Remove a borda externa da ROI/AOI para que linhas verdes e contornos
        # da interface não se tornem a "prova" de presença.
        trim = max(1, int(round(minimum_side * 0.04)))
        if (
            reference_blur.shape[0] > trim * 2 + 12
            and reference_blur.shape[1] > trim * 2 + 12
        ):
            reference_blur = reference_blur[trim:-trim, trim:-trim]
            test_blur = test_blur[trim:-trim, trim:-trim]

        reference_small = cv2.resize(
            reference_blur,
            (32, 32),
            interpolation=cv2.INTER_AREA,
        )
        test_small = cv2.resize(
            test_blur,
            (32, 32),
            interpolation=cv2.INTER_AREA,
        )
        return cls._normalized_correlation(
            reference_small,
            test_small,
        )

    @staticmethod
    def _central_silhouette(image: np.ndarray):
        """Extrai a maior massa central escura após forte suavização."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        minimum_side = max(1, min(height, width))
        kernel = max(5, int(round(minimum_side * 0.07)))
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )

        trim = max(1, int(round(minimum_side * 0.04)))
        mask[:trim, :] = 0
        mask[-trim:, :] = 0
        mask[:, :trim] = 0
        mask[:, -trim:] = 0

        morph = max(3, int(round(minimum_side * 0.03)))
        if morph % 2 == 0:
            morph += 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (morph, morph),
            ),
        )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            8,
        )
        image_area = float(max(1, height * width))
        image_center = np.asarray(
            [width / 2.0, height / 2.0],
            dtype=np.float32,
        )
        diagonal = float(max(np.hypot(width, height), 1.0))

        best_index = None
        best_score = -1e9
        for index in range(1, count):
            x, y, box_width, box_height, area = stats[index]
            area_ratio = float(area / image_area)
            if area_ratio < 0.03:
                continue
            centroid = centroids[index]
            center_distance = float(
                np.linalg.norm(centroid - image_center) / diagonal
            )
            score = area_ratio - 0.80 * center_distance
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            return None, None

        selected = (labels == best_index).astype(np.uint8)
        x, y, box_width, box_height, area = stats[best_index]
        centroid = centroids[best_index]
        info = {
            "area_ratio": float(area / image_area),
            "centroid_x": float(centroid[0] / max(width, 1)),
            "centroid_y": float(centroid[1] / max(height, 1)),
            "box_width_ratio": float(box_width / max(width, 1)),
            "box_height_ratio": float(box_height / max(height, 1)),
        }
        return selected, info

    @classmethod
    def _component_body_presence(
        cls,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> dict:
        """Testemunha independente para corpo físico preservado na mesma ROI."""
        if (
            not isinstance(reference, np.ndarray)
            or not isinstance(test, np.ndarray)
            or reference.size == 0
            or test.size == 0
        ):
            return {
                "missing_body_presence_active": False,
                "missing_component_body_present": False,
                "missing_body_presence_reason": "ROI indisponível",
            }

        if reference.shape[:2] != test.shape[:2]:
            test = cv2.resize(
                test,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        coarse_similarity = cls._coarse_body_similarity(
            reference,
            test,
        )
        reference_mask, reference_info = cls._central_silhouette(reference)
        test_mask, test_info = cls._central_silhouette(test)

        silhouette_dice = 0.0
        area_ratio = 0.0
        centroid_shift = 1.0
        if (
            reference_mask is not None
            and test_mask is not None
            and reference_info is not None
            and test_info is not None
        ):
            intersection = float(
                np.count_nonzero(
                    (reference_mask > 0) & (test_mask > 0)
                )
            )
            denominator = float(
                np.count_nonzero(reference_mask)
                + np.count_nonzero(test_mask)
            )
            silhouette_dice = (
                2.0 * intersection / denominator
                if denominator > 0
                else 0.0
            )
            reference_area = max(
                float(reference_info["area_ratio"]),
                1e-6,
            )
            area_ratio = float(
                test_info["area_ratio"] / reference_area
            )
            centroid_shift = float(
                np.hypot(
                    test_info["centroid_x"]
                    - reference_info["centroid_x"],
                    test_info["centroid_y"]
                    - reference_info["centroid_y"],
                )
            )

        body_present = bool(
            coarse_similarity
            >= cls.BODY_PRESENCE_MIN_COARSE_SIMILARITY
            and silhouette_dice
            >= cls.BODY_PRESENCE_MIN_SILHOUETTE_DICE
            and cls.BODY_PRESENCE_MIN_AREA_RATIO
            <= area_ratio
            <= cls.BODY_PRESENCE_MAX_AREA_RATIO
            and centroid_shift
            <= cls.BODY_PRESENCE_MAX_CENTROID_SHIFT
        )

        reason = (
            "corpo geométrico preservado apesar da divergência de aparência"
            if body_present
            else "sem testemunha geométrica suficiente de corpo preservado"
        )
        return {
            "missing_body_presence_active": True,
            "missing_component_body_present": body_present,
            "missing_body_coarse_similarity": float(coarse_similarity),
            "missing_body_silhouette_dice": float(silhouette_dice),
            "missing_body_area_ratio": float(area_ratio),
            "missing_body_centroid_shift": float(centroid_shift),
            "missing_body_presence_reason": reason,
        }

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
        physical_detail: dict | None = None,
    ) -> dict:
        result = super().analyze(
            full_reference,
            full_test,
            global_box_info,
            aoi_info,
            aoi_epicenters,
        )

        body_presence = {
            "missing_body_presence_active": False,
            "missing_component_body_present": False,
            "missing_body_presence_reason": "ROI indisponível",
        }
        roi_box = result.get("missing_roi_box")
        if (
            result.get("missing_active", False)
            and roi_box
            and len(roi_box) >= 4
        ):
            safe_reference, safe_test = self._safe_pair(
                full_reference,
                full_test,
            )
            reference_roi = self._crop(safe_reference, roi_box)
            test_roi = self._crop(safe_test, roi_box)
            body_presence = self._component_body_presence(
                reference_roi,
                test_roi,
            )
        result.update(body_presence)

        raw_classification = str(
            result.get("missing_classification", "")
        ).strip().upper()
        body_presence_veto = bool(
            result.get("missing_component_body_present", False)
            and result.get("missing_is_defect", False)
            and raw_classification
            in {
                "CONTEÚDO ESPERADO AUSENTE",
                "QUEBRA DA EXPECTATIVA VISUAL",
            }
        )
        result["missing_body_presence_veto"] = body_presence_veto

        if body_presence_veto:
            # O especialista FALTANDO responde presença física. Mudança de
            # brilho/serigrafia com silhueta preservada não é falta física.
            result["missing_is_defect"] = False
            result["missing_classification"] = (
                "COMPONENTE PRESENTE — APARÊNCIA DIVERGENTE"
            )
            result["missing_reason"] = (
                "CORPO DO COMPONENTE PRESENTE: geometria e ocupação "
                "preservadas; divergência interna tratada por outros motores"
            )

        hard_absence, hard_reason = self._hard_absence_evidence(result)

        if result.get("missing_component_body_present", False):
            result.update(
                {
                    "missing_dual_scale_policy": (
                        DualScalePresenceAnalyzer.POLICY
                    ),
                    "missing_dual_scale_active": False,
                    "missing_dual_scale_triggered": False,
                    "missing_scale_disagreement": False,
                    "missing_context_hard_absence": False,
                    "missing_context_hard_reason": (
                        "corpo do componente já confirmado na escala local"
                    ),
                }
            )
            hard_absence = False
            hard_reason = (
                "corpo do componente preservado; hard missing bloqueado"
            )
        elif not hard_absence:
            dual_scale = DualScalePresenceAnalyzer.analyze(
                self,
                full_reference,
                full_test,
                result,
                global_box_info=global_box_info,
                physical_detail=physical_detail,
            )
            result.update(dual_scale)
            if dual_scale.get("missing_context_hard_absence", False):
                hard_absence = True
                hard_reason = str(
                    dual_scale.get(
                        "missing_context_hard_reason",
                        "contexto maior confirmou ausência física",
                    )
                )
        else:
            result.update(
                {
                    "missing_dual_scale_policy": (
                        DualScalePresenceAnalyzer.POLICY
                    ),
                    "missing_dual_scale_active": False,
                    "missing_dual_scale_triggered": False,
                    "missing_scale_disagreement": False,
                    "missing_context_hard_absence": False,
                    "missing_context_hard_reason": (
                        "escala local já confirmou ausência física"
                    ),
                }
            )

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
            "body_presence_coarse_similarity": (
                self.BODY_PRESENCE_MIN_COARSE_SIMILARITY
            ),
            "body_presence_silhouette_dice": (
                self.BODY_PRESENCE_MIN_SILHOUETTE_DICE
            ),
            "body_presence_area_ratio_min": (
                self.BODY_PRESENCE_MIN_AREA_RATIO
            ),
            "body_presence_area_ratio_max": (
                self.BODY_PRESENCE_MAX_AREA_RATIO
            ),
            "body_presence_centroid_shift_max": (
                self.BODY_PRESENCE_MAX_CENTROID_SHIFT
            ),
            "dual_scale_policy": DualScalePresenceAnalyzer.POLICY,
            "dual_scale_local_global_ratio_max": (
                DualScalePresenceAnalyzer.MAX_LOCAL_GLOBAL_AREA_RATIO
            ),
            "dual_scale_context_score": (
                DualScalePresenceAnalyzer.MIN_CONTEXT_SCORE
            ),
            "dual_scale_context_coverage": (
                DualScalePresenceAnalyzer.MIN_CONTEXT_COVERAGE
            ),
        }

        if hard_absence:
            result["missing_classification"] = (
                "COMPONENTE FISICAMENTE AUSENTE — DUAL-SCALE"
                if result.get("missing_context_hard_absence", False)
                else "COMPONENTE FISICAMENTE AUSENTE"
            )
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
