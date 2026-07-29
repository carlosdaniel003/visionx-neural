# src/core/epicenter_extractor.py
"""Extração direta da menor caixa verde desenhada pela AOI."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.config.settings import settings


class EpicenterExtractor:
    """Usa a regra objetiva do layout da AOI.

    Existem duas caixas verdes:
    - a maior delimita o componente;
    - a menor delimita o defeito.

    O epicentro é sempre a menor caixa verde. Não há pontuação, classificação
    semântica da moldura ou tentativa de escolher pela posição na imagem.
    """

    MIN_SIDE_PX = 6
    DUPLICATE_IOU = 0.65

    @staticmethod
    def _valid_image(image: np.ndarray) -> bool:
        return isinstance(image, np.ndarray) and image.size > 0

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        """Máscara flexível dos verdes usados nas linhas da AOI.

        Os tons fornecidos ficam aproximadamente em H=55..66 no HSV do OpenCV.
        A faixa é ampliada para absorver transparência, blur e antialiasing.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)

        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        green_excess = green - np.maximum(red, blue)

        hsv_green = cv2.inRange(
            hsv,
            np.asarray((45, 25, 35), dtype=np.uint8),
            np.asarray((82, 255, 255), dtype=np.uint8),
        ) > 0
        dominant_green = (green_excess >= 4) & (green >= 40)

        prototypes = np.asarray(
            getattr(settings, "AOI_GREEN_RGB_SAMPLES", ()),
            dtype=np.int16,
        )
        if prototypes.size:
            delta = rgb[:, :, None, :] - prototypes[None, None, :, :]
            distance = np.sqrt(np.sum(delta.astype(np.float32) ** 2, axis=3))
            near_sample = np.min(distance, axis=2) <= 110.0
        else:
            near_sample = np.zeros(hsv_green.shape, dtype=bool)

        mask = (
            (hsv_green & dominant_green)
            | (near_sample & dominant_green)
        ).astype(np.uint8) * 255

        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

    @staticmethod
    def _iou(first, second) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - intersection
        return float(intersection / union) if union > 0 else 0.0

    @classmethod
    def _same_rectangle(cls, first, second) -> bool:
        if cls._iou(first, second) >= cls.DUPLICATE_IOU:
            return True
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        tolerance = max(3, int(round(min(aw, ah, bw, bh) * 0.18)))
        return (
            abs(ax - bx) <= tolerance
            and abs(ay - by) <= tolerance
            and abs(aw - bw) <= tolerance * 2
            and abs(ah - bh) <= tolerance * 2
        )

    @staticmethod
    def _contains(outer, inner, tolerance: int = 10) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (
            iw * ih < ow * oh
            and ix >= ox - tolerance
            and iy >= oy - tolerance
            and ix + iw <= ox + ow + tolerance
            and iy + ih <= oy + oh + tolerance
        )

    @classmethod
    def _normalize_box(cls, box, image_shape):
        if not box or len(box) < 4:
            return None
        image_height, image_width = image_shape[:2]
        try:
            x, y, width, height = (
                int(round(float(value))) for value in box[:4]
            )
        except (TypeError, ValueError):
            return None

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        width = x2 - x1
        height = y2 - y1
        if width < cls.MIN_SIDE_PX or height < cls.MIN_SIDE_PX:
            return None
        return int(x1), int(y1), int(width), int(height)

    @classmethod
    def _deduplicate_boxes(cls, boxes: list[tuple]) -> list[tuple]:
        unique: list[tuple] = []
        # Mantém a caixa maior de cada par duplicado. A espessura da linha verde
        # normalmente gera dois contornos quase iguais, interno e externo.
        for box in sorted(boxes, key=lambda value: value[2] * value[3], reverse=True):
            if any(cls._same_rectangle(box, current) for current in unique):
                continue
            unique.append(box)
        return unique

    @classmethod
    def _legacy_boxes(cls, old_epicenters, image_shape) -> list[tuple]:
        normalized = []
        for raw_box in old_epicenters or []:
            box = cls._normalize_box(raw_box, image_shape)
            if box is not None:
                normalized.append(box)
        return cls._deduplicate_boxes(normalized)

    @classmethod
    def _visual_boxes(cls, image: np.ndarray) -> list[tuple]:
        """Fallback: encontra contornos verdes retangulares no gabarito."""
        if not cls._valid_image(image):
            return []

        mask = cls._green_mask(image)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        image_height, image_width = image.shape[:2]
        image_area = max(1, image_height * image_width)
        boxes = []

        for contour in contours:
            box = cls._normalize_box(cv2.boundingRect(contour), image.shape)
            if box is None:
                continue
            x, y, width, height = box
            area = width * height
            if area >= image_area * 0.96:
                continue
            if width >= image_width * 0.99 or height >= image_height * 0.99:
                continue
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > 18.0:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = (
                cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if perimeter > 0
                else contour
            )
            # Linhas da AOI podem ser interrompidas por etiquetas. Aceita de 4
            # a 10 vértices, mas rejeita pontos e formas muito irregulares.
            if not (4 <= len(approximation) <= 10):
                continue
            boxes.append(box)

        return cls._deduplicate_boxes(boxes)

    @classmethod
    def _select_smallest(
        cls,
        old_epicenters,
        sample_crop: np.ndarray,
        global_box_info: dict,
    ):
        """Seleciona diretamente a menor caixa interna disponível."""
        # Caminho principal: detect_anomalies já removeu a caixa maior e nos
        # entregou apenas as caixas internas. Portanto basta usar a menor.
        legacy = cls._legacy_boxes(old_epicenters, sample_crop.shape)
        if legacy:
            return min(legacy, key=lambda box: box[2] * box[3]), "old_epicenters"

        # Fallback visual: detecta todas as caixas no gabarito, considera a maior
        # como caixa global e escolhe a menor que esteja dentro dela.
        visual = cls._visual_boxes(sample_crop)
        if len(visual) >= 2:
            outer = max(visual, key=lambda box: box[2] * box[3])
            internal = [
                box
                for box in visual
                if not cls._same_rectangle(box, outer)
                and cls._contains(outer, box)
            ]
            if internal:
                return min(internal, key=lambda box: box[2] * box[3]), "visual"

        # Se a caixa global veio do estágio anterior e só uma caixa verde menor
        # foi detectada visualmente, ela já é o epicentro.
        global_box = cls._normalize_box(
            (
                (global_box_info or {}).get("x", 0),
                (global_box_info or {}).get("y", 0),
                (global_box_info or {}).get("w", 0),
                (global_box_info or {}).get("h", 0),
            ),
            sample_crop.shape,
        )
        if global_box is not None and visual:
            internal = [
                box
                for box in visual
                if not cls._same_rectangle(box, global_box)
                and cls._contains(global_box, box)
            ]
            if internal:
                return min(internal, key=lambda box: box[2] * box[3]), "visual+global"

        return None, "none"

    @classmethod
    def _content_box(cls, box, image_shape):
        """Retira apenas 1 pixel da linha, sem mudar a ROI escolhida."""
        image_height, image_width = image_shape[:2]
        x, y, width, height = box
        inset = 1
        if width - 2 >= cls.MIN_SIDE_PX:
            x += inset
            width -= 2
        if height - 2 >= cls.MIN_SIDE_PX:
            y += inset
            height -= 2
        x = max(0, min(image_width - 1, x))
        y = max(0, min(image_height - 1, y))
        width = max(1, min(width, image_width - x))
        height = max(1, min(height, image_height - y))
        return int(x), int(y), int(width), int(height)

    @classmethod
    def extract_focus(
        cls,
        sample_crop: np.ndarray,
        ng_crop: np.ndarray,
        old_epicenters: list,
        global_box_info: dict,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Recorta a menor caixa verde nas mesmas coordenadas das duas imagens."""
        real_epicenters: list[tuple[int, int, int, int]] = []
        focus_gab = np.array([])
        focus_ng = np.array([])

        if not cls._valid_image(sample_crop) or not cls._valid_image(ng_crop):
            return real_epicenters, focus_gab, focus_ng

        if sample_crop.shape != ng_crop.shape:
            ng_crop = cv2.resize(
                ng_crop,
                (sample_crop.shape[1], sample_crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        try:
            selected, source = cls._select_smallest(
                old_epicenters,
                sample_crop,
                global_box_info,
            )
            if selected is not None:
                content_box = cls._content_box(selected, sample_crop.shape)
                real_epicenters.append(content_box)
                print(
                    "Menor retângulo verde selecionado: "
                    f"externo={selected} • conteúdo={content_box} • origem={source} • "
                    f"candidatos={old_epicenters}"
                )
            else:
                print(
                    "Menor retângulo verde não encontrado • "
                    f"candidatos recebidos={old_epicenters} • "
                    f"candidatos visuais={cls._visual_boxes(sample_crop)}"
                )
        except Exception as exc:
            print(f"Erro ao selecionar o menor retângulo verde: {exc}")

        if real_epicenters:
            x, y, width, height = real_epicenters[0]
            x2 = min(sample_crop.shape[1], x + width)
            y2 = min(sample_crop.shape[0], y + height)
            if x2 > x and y2 > y:
                focus_gab = sample_crop[y:y2, x:x2].copy()
                focus_ng = ng_crop[y:y2, x:x2].copy()

        return real_epicenters, focus_gab, focus_ng
