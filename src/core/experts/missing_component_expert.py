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

    # Segunda rota: silhueta/ocupação quase idênticas podem provar presença
    # mesmo quando acabamento, brilho e serigrafia destroem a correlação tonal.
    BODY_GEOMETRY_MIN_SILHOUETTE_DICE = 0.84
    BODY_GEOMETRY_MIN_AREA_RATIO = 0.70
    BODY_GEOMETRY_MAX_AREA_RATIO = 1.35
    BODY_GEOMETRY_MAX_CENTROID_SHIFT = 0.07
    BODY_GEOMETRY_MIN_BOX_RATIO = 0.75
    BODY_GEOMETRY_MAX_BOX_RATIO = 1.33

    # Testemunha complementar do envelope global detectado pela AOI. Usa
    # perfis de baixa frequência em vez de depender da serigrafia central.
    GLOBAL_ENVELOPE_MIN_ROW_PROFILE = 0.84
    GLOBAL_ENVELOPE_MIN_COL_PROFILE = 0.84
    GLOBAL_ENVELOPE_MIN_COARSE_SIMILARITY = 0.66
    GLOBAL_ENVELOPE_MAX_BACKGROUND_EXPOSURE = 0.08

    # Segunda rota do envelope: compara a massa escura sem depender da posição
    # exata dentro da caixa. Ordenar os perfis remove translação e inversão de
    # orientação, mantendo a distribuição física do corpo.
    GLOBAL_ENVELOPE_MIN_DARK_REFERENCE_FRACTION = 0.18
    GLOBAL_ENVELOPE_MIN_DARK_TEST_FRACTION = 0.18
    GLOBAL_ENVELOPE_MIN_DARK_RETENTION = 0.55
    GLOBAL_ENVELOPE_MAX_DARK_RETENTION = 1.80
    GLOBAL_ENVELOPE_MIN_INVARIANT_ROW_PROFILE = 0.84
    GLOBAL_ENVELOPE_MIN_INVARIANT_COL_PROFILE = 0.84
    GLOBAL_ENVELOPE_INVARIANT_MAX_BACKGROUND_EXPOSURE = 0.15

    # Testemunha composta para ROIs locais pequenas. Ela não afrouxa a
    # testemunha rígida de corpo: só atua quando o envelope global já preserva
    # massa física invariável e a ocupação geométrica local continua coerente.
    INVARIANT_OCCUPANCY_MIN_SILHOUETTE_DICE = 0.70
    INVARIANT_OCCUPANCY_MIN_AREA_RATIO = 0.70
    INVARIANT_OCCUPANCY_MAX_AREA_RATIO = 1.35
    INVARIANT_OCCUPANCY_MAX_CENTROID_SHIFT = 0.10
    INVARIANT_OCCUPANCY_MIN_BOX_RATIO = 0.75
    INVARIANT_OCCUPANCY_MAX_BOX_RATIO = 1.33

    # Rota dedicada para FALTANDO com footprint/pads preservados. Ela cobre o
    # caso em que a silhueta local parece semelhante por geometria, mas o corpo
    # real desapareceu e contexto + motores independentes confirmam a perda.
    FOOTPRINT_ABSENCE_MAX_BODY_COARSE = 0.10
    FOOTPRINT_ABSENCE_MIN_LOCAL_RESIDUAL = 0.60
    FOOTPRINT_ABSENCE_MIN_LOCAL_STRUCTURE = 0.55
    FOOTPRINT_ABSENCE_MIN_LOCAL_EDGE_MISMATCH = 0.42
    FOOTPRINT_ABSENCE_MAX_LOCAL_NEARBY = 0.30
    FOOTPRINT_ABSENCE_MIN_CONTEXT_SCORE = 0.72
    FOOTPRINT_ABSENCE_MIN_CONTEXT_COVERAGE = 0.32
    FOOTPRINT_ABSENCE_MIN_CONTEXT_RESIDUAL = 0.60
    FOOTPRINT_ABSENCE_MIN_CONTEXT_APPEARANCE = 0.38
    FOOTPRINT_ABSENCE_MIN_CONTEXT_STRUCTURE = 0.35
    FOOTPRINT_ABSENCE_MAX_CONTEXT_DIRECT_SIMILARITY = 0.62
    FOOTPRINT_ABSENCE_MAX_CONTEXT_NEARBY = 0.50
    FOOTPRINT_ABSENCE_MIN_PHYSICAL_STRUCTURAL = 0.35
    FOOTPRINT_ABSENCE_MIN_PHYSICAL_SEMANTIC = 0.50

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

    @classmethod
    def _global_envelope_presence_support(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        global_box_info: dict | None,
    ) -> dict:
        """Testemunha conservadora da forma global do componente.

        Esta rota não declara OK. Ela apenas impede que uma ROI interna estreita
        tenha autoridade de "ausência física forte" quando o envelope AOI
        completo preserva a estrutura de baixa frequência do componente.
        """
        default = {
            "missing_global_envelope_active": False,
            "missing_global_envelope_support": False,
            "missing_global_envelope_box": None,
            "missing_global_envelope_row_profile": 0.0,
            "missing_global_envelope_col_profile": 0.0,
            "missing_global_envelope_coarse_similarity": 0.0,
            "missing_global_envelope_background_exposure": 0.0,
            "missing_global_envelope_dark_threshold": 0.0,
            "missing_global_envelope_reference_dark_fraction": 0.0,
            "missing_global_envelope_test_dark_fraction": 0.0,
            "missing_global_envelope_dark_retention": 0.0,
            "missing_global_envelope_invariant_row_profile": 0.0,
            "missing_global_envelope_invariant_col_profile": 0.0,
            "missing_global_envelope_invariant_support": False,
            "missing_global_envelope_reason": (
                "caixa global da AOI indisponível"
            ),
        }
        if not isinstance(global_box_info, dict):
            return default
        if not bool(global_box_info.get("detected", False)):
            return default

        try:
            box = (
                int(global_box_info.get("x", 0)),
                int(global_box_info.get("y", 0)),
                int(global_box_info.get("w", 0)),
                int(global_box_info.get("h", 0)),
            )
        except Exception:
            return default

        x, y, width, height = box
        if width <= 0 or height <= 0:
            return default

        safe_reference, safe_test = cls._safe_pair(
            full_reference,
            full_test,
        )
        reference_roi = cls._crop(safe_reference, box)
        test_roi = cls._crop(safe_test, box)
        if (
            not isinstance(reference_roi, np.ndarray)
            or not isinstance(test_roi, np.ndarray)
            or reference_roi.size == 0
            or test_roi.size == 0
        ):
            return default

        if reference_roi.shape[:2] != test_roi.shape[:2]:
            test_roi = cv2.resize(
                test_roi,
                (reference_roi.shape[1], reference_roi.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        reference_gray = cv2.cvtColor(
            reference_roi,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)
        test_gray = cv2.cvtColor(
            test_roi,
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32)

        minimum_side = max(1, min(reference_gray.shape[:2]))
        kernel = max(9, int(round(minimum_side * 0.18)))
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

        # Usa apenas os 70% centrais para remover a moldura verde e reduzir
        # influência das bordas da interface.
        roi_height, roi_width = reference_blur.shape[:2]
        x_margin = int(round(roi_width * 0.15))
        y_margin = int(round(roi_height * 0.15))
        x1 = min(max(0, x_margin), max(0, roi_width - 2))
        x2 = max(x1 + 1, roi_width - x_margin)
        y1 = min(max(0, y_margin), max(0, roi_height - 2))
        y2 = max(y1 + 1, roi_height - y_margin)

        reference_core = reference_blur[y1:y2, x1:x2]
        test_core = test_blur[y1:y2, x1:x2]

        row_similarity = cls._normalized_correlation(
            np.mean(reference_core, axis=1),
            np.mean(test_core, axis=1),
        )
        col_similarity = cls._normalized_correlation(
            np.mean(reference_core, axis=0),
            np.mean(test_core, axis=0),
        )
        coarse_similarity = cls._coarse_body_similarity(
            reference_roi,
            test_roi,
        )

        # Importante: fundo exposto é espacial. Não pode reutilizar a métrica
        # da ROI local em uma caixa global diferente.
        background = cls._background_replacement_signal(
            safe_reference,
            safe_test,
            reference_roi,
            test_roi,
            box,
        )

        # Presença invariante a deslocamento/orientação interna. Usa o núcleo
        # do envelope e mede massa escura relativa à distribuição do gabarito.
        dark_x_margin = int(round(width * 0.18))
        dark_y_margin = int(round(height * 0.12))
        dark_x1 = min(max(0, dark_x_margin), max(0, width - 2))
        dark_x2 = max(dark_x1 + 1, width - dark_x_margin)
        dark_y1 = min(max(0, dark_y_margin), max(0, height - 2))
        dark_y2 = max(dark_y1 + 1, height - dark_y_margin)

        reference_dark_core = reference_gray[
            dark_y1:dark_y2,
            dark_x1:dark_x2,
        ]
        test_dark_core = test_gray[
            dark_y1:dark_y2,
            dark_x1:dark_x2,
        ]

        dark_threshold = float(
            np.clip(
                np.percentile(reference_dark_core, 25) + 26.0,
                35.0,
                85.0,
            )
        )
        reference_dark_mask = (
            reference_dark_core <= dark_threshold
        ).astype(np.float32)
        test_dark_mask = (
            test_dark_core <= dark_threshold
        ).astype(np.float32)

        reference_dark_fraction = float(np.mean(reference_dark_mask))
        test_dark_fraction = float(np.mean(test_dark_mask))
        dark_retention = float(
            test_dark_fraction / max(reference_dark_fraction, 1e-6)
        )

        reference_row_mass = np.sort(
            np.mean(reference_dark_mask, axis=1)
        )
        test_row_mass = np.sort(
            np.mean(test_dark_mask, axis=1)
        )
        reference_col_mass = np.sort(
            np.mean(reference_dark_mask, axis=0)
        )
        test_col_mass = np.sort(
            np.mean(test_dark_mask, axis=0)
        )

        invariant_row_similarity = float(
            np.clip(
                1.0 - np.mean(
                    np.abs(reference_row_mass - test_row_mass)
                ),
                0.0,
                1.0,
            )
        )
        invariant_col_similarity = float(
            np.clip(
                1.0 - np.mean(
                    np.abs(reference_col_mass - test_col_mass)
                ),
                0.0,
                1.0,
            )
        )

        aligned_support = bool(
            row_similarity >= cls.GLOBAL_ENVELOPE_MIN_ROW_PROFILE
            and col_similarity >= cls.GLOBAL_ENVELOPE_MIN_COL_PROFILE
            and coarse_similarity
            >= cls.GLOBAL_ENVELOPE_MIN_COARSE_SIMILARITY
            and background
            <= cls.GLOBAL_ENVELOPE_MAX_BACKGROUND_EXPOSURE
        )
        invariant_support = bool(
            reference_dark_fraction
            >= cls.GLOBAL_ENVELOPE_MIN_DARK_REFERENCE_FRACTION
            and test_dark_fraction
            >= cls.GLOBAL_ENVELOPE_MIN_DARK_TEST_FRACTION
            and cls.GLOBAL_ENVELOPE_MIN_DARK_RETENTION
            <= dark_retention
            <= cls.GLOBAL_ENVELOPE_MAX_DARK_RETENTION
            and invariant_row_similarity
            >= cls.GLOBAL_ENVELOPE_MIN_INVARIANT_ROW_PROFILE
            and invariant_col_similarity
            >= cls.GLOBAL_ENVELOPE_MIN_INVARIANT_COL_PROFILE
            and background
            <= cls.GLOBAL_ENVELOPE_INVARIANT_MAX_BACKGROUND_EXPOSURE
        )
        # A rota invariável é testemunha auxiliar. Ela não pode, sozinha,
        # vetar hard missing porque footprints escuros reais também podem
        # preservar massa. A autoridade direta do envelope continua restrita
        # à rota alinhada; a rota invariável exige confirmação OK da memória.
        supported = bool(aligned_support)

        if aligned_support:
            reason = (
                "envelope global preserva perfis alinhados do componente"
            )
        elif invariant_support:
            reason = (
                "massa física invariável preservada; requer testemunha OK "
                "forte antes de contrariar hard missing"
            )
        else:
            reason = "envelope global sem suporte estrutural suficiente"
        return {
            "missing_global_envelope_active": True,
            "missing_global_envelope_support": supported,
            "missing_global_envelope_box": list(box),
            "missing_global_envelope_row_profile": float(row_similarity),
            "missing_global_envelope_col_profile": float(col_similarity),
            "missing_global_envelope_coarse_similarity": float(
                coarse_similarity
            ),
            "missing_global_envelope_background_exposure": float(background),
            "missing_global_envelope_dark_threshold": float(dark_threshold),
            "missing_global_envelope_reference_dark_fraction": float(
                reference_dark_fraction
            ),
            "missing_global_envelope_test_dark_fraction": float(
                test_dark_fraction
            ),
            "missing_global_envelope_dark_retention": float(dark_retention),
            "missing_global_envelope_invariant_row_profile": float(
                invariant_row_similarity
            ),
            "missing_global_envelope_invariant_col_profile": float(
                invariant_col_similarity
            ),
            "missing_global_envelope_invariant_support": bool(
                invariant_support
            ),
            "missing_global_envelope_reason": reason,
        }


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
        box_width_ratio = 0.0
        box_height_ratio = 0.0
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
            box_width_ratio = float(
                test_info["box_width_ratio"]
                / max(float(reference_info["box_width_ratio"]), 1e-6)
            )
            box_height_ratio = float(
                test_info["box_height_ratio"]
                / max(float(reference_info["box_height_ratio"]), 1e-6)
            )

        appearance_body_present = bool(
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
        geometry_body_present = bool(
            silhouette_dice >= cls.BODY_GEOMETRY_MIN_SILHOUETTE_DICE
            and cls.BODY_GEOMETRY_MIN_AREA_RATIO
            <= area_ratio
            <= cls.BODY_GEOMETRY_MAX_AREA_RATIO
            and centroid_shift
            <= cls.BODY_GEOMETRY_MAX_CENTROID_SHIFT
            and cls.BODY_GEOMETRY_MIN_BOX_RATIO
            <= box_width_ratio
            <= cls.BODY_GEOMETRY_MAX_BOX_RATIO
            and cls.BODY_GEOMETRY_MIN_BOX_RATIO
            <= box_height_ratio
            <= cls.BODY_GEOMETRY_MAX_BOX_RATIO
        )
        body_present = bool(
            appearance_body_present or geometry_body_present
        )

        if geometry_body_present and not appearance_body_present:
            policy = "geometry_only"
            reason = (
                "silhueta, área e centro preservados apesar da mudança "
                "forte de brilho/serigrafia"
            )
        elif appearance_body_present:
            policy = "coarse_and_geometry"
            reason = (
                "corpo geométrico preservado apesar da divergência de aparência"
            )
        else:
            policy = "none"
            reason = "sem testemunha geométrica suficiente de corpo preservado"
        return {
            "missing_body_presence_active": True,
            "missing_component_body_present": body_present,
            "missing_body_coarse_similarity": float(coarse_similarity),
            "missing_body_silhouette_dice": float(silhouette_dice),
            "missing_body_area_ratio": float(area_ratio),
            "missing_body_centroid_shift": float(centroid_shift),
            "missing_body_box_width_ratio": float(box_width_ratio),
            "missing_body_box_height_ratio": float(box_height_ratio),
            "missing_body_presence_policy": policy,
            "missing_body_presence_reason": reason,
        }

    @classmethod
    def _component_body_presence_witness(
        cls,
        full_reference: np.ndarray,
        full_test: np.ndarray,
        local_box,
        aoi_epicenters,
    ) -> dict:
        """Busca presença no patch local e também no envelope do componente.

        A ROI interna pode concentrar serigrafia/texto e parecer completamente
        diferente mesmo quando o corpo físico continua presente. O epicentro
        final da AOI fornece uma segunda escala geométrica independente.
        """
        default = {
            "missing_body_presence_active": False,
            "missing_component_body_present": False,
            "missing_body_presence_source": "none",
            "missing_body_presence_box": None,
            "missing_body_presence_reason": "ROI indisponível",
        }

        safe_reference, safe_test = cls._safe_pair(
            full_reference,
            full_test,
        )
        candidates = []
        seen = set()

        def add_candidate(source: str, box) -> None:
            if not box or len(box) < 4:
                return
            candidate = tuple(int(value) for value in box[:4])
            x, y, width, height = candidate
            if width <= 0 or height <= 0:
                return
            if candidate in seen:
                return
            seen.add(candidate)
            candidates.append((source, candidate))

        add_candidate("missing_roi", local_box)
        for index, box in enumerate(aoi_epicenters or []):
            add_candidate(
                "aoi_epicenter" if index == 0 else f"aoi_epicenter_{index + 1}",
                box,
            )

        if not candidates:
            return default

        best = None
        best_strength = -1.0
        for source, box in candidates:
            reference_roi = cls._crop(safe_reference, box)
            test_roi = cls._crop(safe_test, box)
            witness = cls._component_body_presence(
                reference_roi,
                test_roi,
            )
            witness["missing_body_presence_source"] = source
            witness["missing_body_presence_box"] = list(box)

            coarse = float(
                witness.get("missing_body_coarse_similarity", 0.0) or 0.0
            )
            dice = float(
                witness.get("missing_body_silhouette_dice", 0.0) or 0.0
            )
            area_ratio = float(
                witness.get("missing_body_area_ratio", 0.0) or 0.0
            )
            centroid_shift = float(
                witness.get("missing_body_centroid_shift", 1.0) or 1.0
            )
            area_fit = (
                1.0
                if cls.BODY_PRESENCE_MIN_AREA_RATIO
                <= area_ratio
                <= cls.BODY_PRESENCE_MAX_AREA_RATIO
                else 0.0
            )
            centroid_fit = max(
                0.0,
                1.0
                - centroid_shift
                / max(cls.BODY_PRESENCE_MAX_CENTROID_SHIFT, 1e-6),
            )
            strength = (
                max(0.0, coarse) * 0.35
                + max(0.0, dice) * 0.45
                + area_fit * 0.10
                + centroid_fit * 0.10
            )

            if bool(witness.get("missing_component_body_present", False)):
                # Uma testemunha positiva na escala do componente é suficiente
                # para impedir que diferença de aparência vire "falta física".
                return witness

            if strength > best_strength:
                best_strength = strength
                best = witness

        return best or default


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

    @classmethod
    def _dedicated_footprint_absence_evidence(
        cls,
        result: dict,
        physical_detail: dict | None,
    ) -> tuple[bool, str]:
        """Confirma ausência quando footprint preservado engana a geometria.

        Esta rota existe somente dentro do especialista FALTANDO. Ela não é
        usada pela guarda transversal de DESLOCADO/EMBORCADO/INVERTIDO.
        """
        if not isinstance(result, dict):
            return False, "resultado missing indisponível"
        if not bool(result.get("missing_active", False)):
            return False, "motor missing inativo"
        if not bool(result.get("missing_is_defect", False)):
            return False, "motor missing não detectou divergência"

        classification = str(
            result.get("missing_classification", "")
        ).strip().upper()
        if classification == "DESLOCAMENTO PROVÁVEL":
            return False, "deslocamento provável não pode virar ausência"

        if bool(result.get("missing_global_envelope_support", False)):
            return False, "envelope global alinhado ainda confirma presença"

        body_present = bool(
            result.get("missing_component_body_present", False)
        )
        body_policy = str(
            result.get("missing_body_presence_policy", "")
        ).strip().lower()
        body_coarse = float(
            result.get("missing_body_coarse_similarity", 0.0) or 0.0
        )

        # Presença com suporte de aparência continua soberana. A exceção é
        # apenas geometry_only com correlação coarse praticamente nula:
        # footprint/pads podem conservar forma mesmo sem o componente.
        geometry_contradicted = bool(
            body_present
            and body_policy == "geometry_only"
            and body_coarse <= cls.FOOTPRINT_ABSENCE_MAX_BODY_COARSE
        )
        if body_present and not geometry_contradicted:
            return False, "testemunha de presença física não foi contradita"

        local_residual = float(
            result.get("missing_residual_mean", 0.0) or 0.0
        )
        local_structure = float(
            result.get("missing_structure_loss", 0.0) or 0.0
        )
        local_edge = float(
            result.get("missing_edge_mismatch", 0.0) or 0.0
        )
        local_nearby = float(
            result.get("missing_best_similarity", 1.0) or 0.0
        )

        context_triggered = bool(
            result.get("missing_dual_scale_triggered", False)
        )
        context_score = float(
            result.get("missing_context_score", 0.0) or 0.0
        )
        context_coverage = float(
            result.get("missing_context_coverage", 0.0) or 0.0
        )
        context_residual = float(
            result.get("missing_context_residual_mean", 0.0) or 0.0
        )
        context_appearance = float(
            result.get("missing_context_appearance_loss", 0.0) or 0.0
        )
        context_structure = float(
            result.get("missing_context_structure_loss", 0.0) or 0.0
        )
        context_direct = float(
            result.get("missing_context_direct_similarity", 1.0) or 0.0
        )
        context_nearby = float(
            result.get("missing_context_best_similarity", 1.0) or 0.0
        )

        detail = physical_detail if isinstance(physical_detail, dict) else {}
        physical_structural = float(
            detail.get("silk_error_pct", 0.0) or 0.0
        )
        physical_semantic = float(
            detail.get("semantic_loss", 0.0) or 0.0
        )

        local_support = bool(
            local_residual >= cls.FOOTPRINT_ABSENCE_MIN_LOCAL_RESIDUAL
            and local_structure >= cls.FOOTPRINT_ABSENCE_MIN_LOCAL_STRUCTURE
            and local_edge >= cls.FOOTPRINT_ABSENCE_MIN_LOCAL_EDGE_MISMATCH
            and local_nearby < cls.FOOTPRINT_ABSENCE_MAX_LOCAL_NEARBY
        )
        context_support = bool(
            context_triggered
            and context_score >= cls.FOOTPRINT_ABSENCE_MIN_CONTEXT_SCORE
            and context_coverage >= cls.FOOTPRINT_ABSENCE_MIN_CONTEXT_COVERAGE
            and context_residual >= cls.FOOTPRINT_ABSENCE_MIN_CONTEXT_RESIDUAL
            and context_appearance
            >= cls.FOOTPRINT_ABSENCE_MIN_CONTEXT_APPEARANCE
            and context_structure
            >= cls.FOOTPRINT_ABSENCE_MIN_CONTEXT_STRUCTURE
            and context_direct
            <= cls.FOOTPRINT_ABSENCE_MAX_CONTEXT_DIRECT_SIMILARITY
            and context_nearby < cls.FOOTPRINT_ABSENCE_MAX_CONTEXT_NEARBY
        )
        independent_support = bool(
            physical_structural
            >= cls.FOOTPRINT_ABSENCE_MIN_PHYSICAL_STRUCTURAL
            and physical_semantic
            >= cls.FOOTPRINT_ABSENCE_MIN_PHYSICAL_SEMANTIC
        )

        if not local_support:
            return False, "ROI local não confirmou perda física suficiente"
        if not context_support:
            return False, "contexto não confirmou perda do corpo esperado"
        if not independent_support:
            return False, "motores estrutural/semântico não corroboraram ausência"

        reason = (
            "footprint preservou geometria, mas correlação coarse do corpo "
            f"caiu para {body_coarse:.3f}; ROI local + contexto + motores "
            "estrutural/semântico confirmam desaparecimento físico"
        )
        return True, reason

    @classmethod
    def _invariant_occupancy_presence_support(
        cls,
        result: dict,
        image_shape,
        global_box_info: dict | None,
    ) -> dict:
        """Corrobora presença sem exigir aparência/alinhamento idênticos.

        Esta rota só existe para uma ROI local pequena em relação ao envelope
        global. A massa global invariável precisa estar preservada e a própria
        ROI deve continuar contendo uma ocupação geométrica coerente. Assim,
        uma faixa local de terminal/serigrafia não pode declarar ausência do
        componente inteiro apenas porque mudou de posição ou aparência.
        """
        output = {
            "missing_invariant_occupancy_support": False,
            "missing_invariant_occupancy_veto": False,
            "missing_invariant_occupancy_reason": (
                "evidência composta de presença indisponível"
            ),
            "missing_local_global_area_ratio": 1.0,
        }
        if not isinstance(result, dict):
            return output

        roi_box = result.get("missing_roi_box")
        if not roi_box or len(roi_box) < 4:
            output["missing_invariant_occupancy_reason"] = (
                "ROI local indisponível"
            )
            return output

        local_ratio = DualScalePresenceAnalyzer._local_global_ratio(
            roi_box,
            image_shape,
            global_box_info,
        )
        output["missing_local_global_area_ratio"] = float(local_ratio)

        if local_ratio > DualScalePresenceAnalyzer.MAX_LOCAL_GLOBAL_AREA_RATIO:
            output["missing_invariant_occupancy_reason"] = (
                "ROI local representa área suficiente do envelope global"
            )
            return output

        if not bool(
            result.get("missing_global_envelope_invariant_support", False)
        ):
            output["missing_invariant_occupancy_reason"] = (
                "envelope global não preservou massa física invariável"
            )
            return output

        silhouette = float(
            result.get("missing_body_silhouette_dice", 0.0) or 0.0
        )
        area_ratio = float(
            result.get("missing_body_area_ratio", 0.0) or 0.0
        )
        centroid_shift = float(
            result.get("missing_body_centroid_shift", 1.0) or 1.0
        )
        box_width_ratio = float(
            result.get("missing_body_box_width_ratio", 0.0) or 0.0
        )
        box_height_ratio = float(
            result.get("missing_body_box_height_ratio", 0.0) or 0.0
        )

        geometry_support = bool(
            silhouette >= cls.INVARIANT_OCCUPANCY_MIN_SILHOUETTE_DICE
            and cls.INVARIANT_OCCUPANCY_MIN_AREA_RATIO
            <= area_ratio
            <= cls.INVARIANT_OCCUPANCY_MAX_AREA_RATIO
            and centroid_shift
            <= cls.INVARIANT_OCCUPANCY_MAX_CENTROID_SHIFT
            and cls.INVARIANT_OCCUPANCY_MIN_BOX_RATIO
            <= box_width_ratio
            <= cls.INVARIANT_OCCUPANCY_MAX_BOX_RATIO
            and cls.INVARIANT_OCCUPANCY_MIN_BOX_RATIO
            <= box_height_ratio
            <= cls.INVARIANT_OCCUPANCY_MAX_BOX_RATIO
        )
        output["missing_invariant_occupancy_support"] = geometry_support

        if geometry_support:
            output["missing_invariant_occupancy_reason"] = (
                "ROI local pequena, massa global invariável e ocupação "
                "geométrica local preservadas"
            )
        else:
            output["missing_invariant_occupancy_reason"] = (
                "ocupação geométrica local não confirmou presença física"
            )
        return output

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
            "missing_body_presence_source": "none",
            "missing_body_presence_box": None,
            "missing_body_presence_reason": "ROI indisponível",
        }
        roi_box = result.get("missing_roi_box")
        if result.get("missing_active", False):
            body_presence = self._component_body_presence_witness(
                full_reference,
                full_test,
                roi_box,
                aoi_epicenters,
            )
        result.update(body_presence)

        global_envelope = self._global_envelope_presence_support(
            full_reference,
            full_test,
            global_box_info,
        )
        result.update(global_envelope)

        invariant_occupancy = self._invariant_occupancy_presence_support(
            result,
            full_reference.shape,
            global_box_info,
        )
        result.update(invariant_occupancy)

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
            source = str(
                result.get("missing_body_presence_source", "geometria")
                or "geometria"
            )
            result["missing_reason"] = (
                "CORPO DO COMPONENTE PRESENTE: geometria e ocupação "
                f"preservadas ({source}); divergência interna tratada por "
                "outros motores"
            )

        hard_absence, hard_reason = self._hard_absence_evidence(result)

        if (
            hard_absence
            and result.get("missing_global_envelope_support", False)
        ):
            hard_absence = False
            hard_reason = (
                "envelope global do componente preservado; ausência forte "
                "rebaixada para a fusão normal"
            )
            result["missing_global_envelope_veto"] = True
        else:
            result["missing_global_envelope_veto"] = False

        invariant_occupancy_veto = bool(
            hard_absence
            and result.get("missing_invariant_occupancy_support", False)
        )
        result["missing_invariant_occupancy_veto"] = (
            invariant_occupancy_veto
        )
        if invariant_occupancy_veto:
            hard_absence = False
            local_ratio = float(
                result.get("missing_local_global_area_ratio", 1.0) or 1.0
            )
            hard_reason = (
                "ROI local cobre apenas "
                f"{local_ratio:.1%} do envelope; massa global invariável e "
                "ocupação geométrica local preservadas; ausência forte "
                "rebaixada para a fusão normal"
            )

        if result.get("missing_body_presence_veto", False):
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
        elif result.get("missing_global_envelope_veto", False):
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
                        "envelope global preservado; decisão devolvida à "
                        "fusão física + memória"
                    ),
                }
            )
        elif result.get("missing_invariant_occupancy_veto", False):
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
                        "ROI local pequena contradita por massa global "
                        "invariável + ocupação geométrica preservada; decisão "
                        "devolvida à fusão física + memória"
                    ),
                }
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

            dedicated_footprint, footprint_reason = (
                self._dedicated_footprint_absence_evidence(
                    result,
                    physical_detail,
                )
            )
            result["missing_dedicated_footprint_absence"] = bool(
                dedicated_footprint
            )
            result["missing_dedicated_footprint_reason"] = footprint_reason
            if dedicated_footprint:
                hard_absence = True
                hard_reason = footprint_reason
        else:
            result["missing_dedicated_footprint_absence"] = False
            result["missing_dedicated_footprint_reason"] = (
                "hard missing já confirmado por rota anterior"
            )
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

        result.setdefault(
            "missing_dedicated_footprint_absence",
            False,
        )
        result.setdefault(
            "missing_dedicated_footprint_reason",
            "rota dedicada não avaliada",
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
            "body_geometry_silhouette_dice": (
                self.BODY_GEOMETRY_MIN_SILHOUETTE_DICE
            ),
            "body_geometry_area_ratio_min": (
                self.BODY_GEOMETRY_MIN_AREA_RATIO
            ),
            "body_geometry_area_ratio_max": (
                self.BODY_GEOMETRY_MAX_AREA_RATIO
            ),
            "body_geometry_centroid_shift_max": (
                self.BODY_GEOMETRY_MAX_CENTROID_SHIFT
            ),
            "body_geometry_box_ratio_min": (
                self.BODY_GEOMETRY_MIN_BOX_RATIO
            ),
            "body_geometry_box_ratio_max": (
                self.BODY_GEOMETRY_MAX_BOX_RATIO
            ),
            "global_envelope_row_profile": (
                self.GLOBAL_ENVELOPE_MIN_ROW_PROFILE
            ),
            "global_envelope_col_profile": (
                self.GLOBAL_ENVELOPE_MIN_COL_PROFILE
            ),
            "global_envelope_coarse_similarity": (
                self.GLOBAL_ENVELOPE_MIN_COARSE_SIMILARITY
            ),
            "global_envelope_background_exposure_max": (
                self.GLOBAL_ENVELOPE_MAX_BACKGROUND_EXPOSURE
            ),
            "global_envelope_dark_reference_fraction_min": (
                self.GLOBAL_ENVELOPE_MIN_DARK_REFERENCE_FRACTION
            ),
            "global_envelope_dark_test_fraction_min": (
                self.GLOBAL_ENVELOPE_MIN_DARK_TEST_FRACTION
            ),
            "global_envelope_dark_retention_min": (
                self.GLOBAL_ENVELOPE_MIN_DARK_RETENTION
            ),
            "global_envelope_dark_retention_max": (
                self.GLOBAL_ENVELOPE_MAX_DARK_RETENTION
            ),
            "global_envelope_invariant_row_profile_min": (
                self.GLOBAL_ENVELOPE_MIN_INVARIANT_ROW_PROFILE
            ),
            "global_envelope_invariant_col_profile_min": (
                self.GLOBAL_ENVELOPE_MIN_INVARIANT_COL_PROFILE
            ),
            "global_envelope_invariant_background_exposure_max": (
                self.GLOBAL_ENVELOPE_INVARIANT_MAX_BACKGROUND_EXPOSURE
            ),
            "invariant_occupancy_silhouette_dice_min": (
                self.INVARIANT_OCCUPANCY_MIN_SILHOUETTE_DICE
            ),
            "invariant_occupancy_area_ratio_min": (
                self.INVARIANT_OCCUPANCY_MIN_AREA_RATIO
            ),
            "invariant_occupancy_area_ratio_max": (
                self.INVARIANT_OCCUPANCY_MAX_AREA_RATIO
            ),
            "invariant_occupancy_centroid_shift_max": (
                self.INVARIANT_OCCUPANCY_MAX_CENTROID_SHIFT
            ),
            "invariant_occupancy_box_ratio_min": (
                self.INVARIANT_OCCUPANCY_MIN_BOX_RATIO
            ),
            "invariant_occupancy_box_ratio_max": (
                self.INVARIANT_OCCUPANCY_MAX_BOX_RATIO
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
            "footprint_body_coarse_max": (
                self.FOOTPRINT_ABSENCE_MAX_BODY_COARSE
            ),
            "footprint_local_residual_min": (
                self.FOOTPRINT_ABSENCE_MIN_LOCAL_RESIDUAL
            ),
            "footprint_local_structure_min": (
                self.FOOTPRINT_ABSENCE_MIN_LOCAL_STRUCTURE
            ),
            "footprint_local_edge_mismatch_min": (
                self.FOOTPRINT_ABSENCE_MIN_LOCAL_EDGE_MISMATCH
            ),
            "footprint_local_nearby_max": (
                self.FOOTPRINT_ABSENCE_MAX_LOCAL_NEARBY
            ),
            "footprint_context_score_min": (
                self.FOOTPRINT_ABSENCE_MIN_CONTEXT_SCORE
            ),
            "footprint_context_coverage_min": (
                self.FOOTPRINT_ABSENCE_MIN_CONTEXT_COVERAGE
            ),
            "footprint_context_residual_min": (
                self.FOOTPRINT_ABSENCE_MIN_CONTEXT_RESIDUAL
            ),
            "footprint_context_appearance_min": (
                self.FOOTPRINT_ABSENCE_MIN_CONTEXT_APPEARANCE
            ),
            "footprint_context_structure_min": (
                self.FOOTPRINT_ABSENCE_MIN_CONTEXT_STRUCTURE
            ),
            "footprint_context_direct_similarity_max": (
                self.FOOTPRINT_ABSENCE_MAX_CONTEXT_DIRECT_SIMILARITY
            ),
            "footprint_context_nearby_max": (
                self.FOOTPRINT_ABSENCE_MAX_CONTEXT_NEARBY
            ),
            "footprint_physical_structural_min": (
                self.FOOTPRINT_ABSENCE_MIN_PHYSICAL_STRUCTURAL
            ),
            "footprint_physical_semantic_min": (
                self.FOOTPRINT_ABSENCE_MIN_PHYSICAL_SEMANTIC
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
