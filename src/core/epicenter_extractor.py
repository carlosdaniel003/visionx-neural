# src/core/epicenter_extractor.py
"""Extração simples da menor moldura verde desenhada pela AOI."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.config.settings import settings


class EpicenterExtractor:
    """Encontra os dois retângulos verdes e usa o menor como epicentro.

    Regra da AOI:
    - retângulo verde maior: componente/região global;
    - retângulo verde menor: ROI do defeito.

    A escolha é feita somente no gabarito. A imagem de teste recebe exatamente
    as mesmas coordenadas, independentemente da cor da marcação NG.
    """

    MIN_SIDE_PX = 7
    MAX_IMAGE_RATIO = 0.98
    DUPLICATE_IOU = 0.68
    MAX_INNER_AREA_RATIO = 0.78

    @staticmethod
    def _valid_image(image: np.ndarray) -> bool:
        return isinstance(image, np.ndarray) and image.size > 0

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        """Aceita os verdes reais da AOI e seus pixels antialiasados.

        Os exemplos fornecidos ficam aproximadamente entre H=55..66 no HSV do
        OpenCV. A faixa é ampliada porque a linha é misturada ao conteúdo da
        fotografia durante redimensionamento e antialiasing.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)

        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        green_excess = green - np.maximum(red, blue)

        hue_mask = cv2.inRange(
            hsv,
            np.asarray((45, 25, 35), dtype=np.uint8),
            np.asarray((82, 255, 255), dtype=np.uint8),
        ) > 0
        dominance = (green_excess >= 4) & (green >= 45)

        # Mantém também proximidade dos sete tons informados, mas não exige
        # correspondência exata: a moldura pode estar translúcida ou suavizada.
        prototypes = np.asarray(
            getattr(settings, "AOI_GREEN_RGB_SAMPLES", ()),
            dtype=np.int16,
        )
        if prototypes.size:
            delta = rgb[:, :, None, :] - prototypes[None, None, :, :]
            distance = np.sqrt(np.sum(delta.astype(np.float32) ** 2, axis=3))
            prototype_mask = np.min(distance, axis=2) <= 105.0
        else:
            prototype_mask = np.zeros(hue_mask.shape, dtype=bool)

        mask = ((hue_mask & dominance) | (prototype_mask & dominance)).astype(
            np.uint8
        ) * 255

        # Une interrupções de 1–2 pixels sem engrossar excessivamente a linha.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return mask

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
        tolerance = max(3, int(round(min(aw, ah, bw, bh) * 0.15)))
        return (
            abs(ax - bx) <= tolerance
            and abs(ay - by) <= tolerance
            and abs(aw - bw) <= tolerance * 2
            and abs(ah - bh) <= tolerance * 2
        )

    @staticmethod
    def _contains(outer, inner, tolerance: int = 8) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        if iw * ih >= ow * oh:
            return False
        return (
            ix >= ox - tolerance
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
        if x2 - x1 < cls.MIN_SIDE_PX or y2 - y1 < cls.MIN_SIDE_PX:
            return None
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _border_support(mask: np.ndarray, box) -> dict:
        """Mede somente se o contorno possui aparência de moldura fina."""
        x, y, width, height = box
        image_height, image_width = mask.shape[:2]
        band = max(1, min(4, int(round(min(width, height) * 0.08))))

        def density(x1, y1, x2, y2) -> float:
            patch = mask[
                max(0, y1) : min(image_height, y2),
                max(0, x1) : min(image_width, x2),
            ]
            return float(np.mean(patch > 0)) if patch.size else 0.0

        top = density(x, y - band, x + width, y + band + 1)
        bottom = density(
            x,
            y + height - 1 - band,
            x + width,
            y + height + band,
        )
        left = density(x - band, y, x + band + 1, y + height)
        right = density(
            x + width - 1 - band,
            y,
            x + width + band,
            y + height,
        )

        inset = max(2, min(6, int(round(min(width, height) * 0.16))))
        interior = density(
            x + inset,
            y + inset,
            x + width - inset,
            y + height - inset,
        )
        sides = (top, bottom, left, right)
        opposite = max(min(top, bottom), min(left, right))
        side_count = sum(value >= 0.10 for value in sides)
        frame_score = max(opposite, float(np.mean(sorted(sides)[-3:])))
        return {
            "sides": sides,
            "side_count": side_count,
            "opposite_support": opposite,
            "frame_score": frame_score,
            "interior_density": interior,
        }

    @classmethod
    def _visual_rectangles(cls, image: np.ndarray) -> list[dict]:
        """Retorna molduras verdes do gabarito sem classificá-las por função."""
        if not cls._valid_image(image):
            return []
        image_height, image_width = image.shape[:2]
        image_area = max(1, image_height * image_width)
        mask = cls._green_mask(image)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        for contour in contours:
            box = cls._normalize_box(cv2.boundingRect(contour), image.shape)
            if box is None:
                continue
            x, y, width, height = box
            area = width * height
            if area >= image_area * 0.96:
                continue
            if width >= image_width * cls.MAX_IMAGE_RATIO:
                continue
            if height >= image_height * cls.MAX_IMAGE_RATIO:
                continue
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > 16.0:
                continue

            metrics = cls._border_support(mask, box)
            # Aceita moldura parcialmente coberta por etiqueta da AOI. Basta um
            # par de lados opostos ou três lados detectáveis.
            looks_like_frame = (
                metrics["opposite_support"] >= 0.08
                or metrics["side_count"] >= 3
            )
            if not looks_like_frame:
                continue
            if metrics["interior_density"] > 0.72:
                continue
            candidates.append(
                {
                    "box": box,
                    "area": area,
                    "source": "visual",
                    **metrics,
                }
            )

        return cls._deduplicate(candidates)

    @classmethod
    def _legacy_rectangles(cls, boxes, image_shape) -> list[dict]:
        candidates = []
        for raw_box in boxes or []:
            box = cls._normalize_box(raw_box, image_shape)
            if box is None:
                continue
            candidates.append(
                {
                    "box": box,
                    "area": box[2] * box[3],
                    "source": "legacy",
                    "sides": (0.0, 0.0, 0.0, 0.0),
                    "side_count": 0,
                    "opposite_support": 0.0,
                    "frame_score": 0.0,
                    "interior_density": 0.0,
                }
            )
        return candidates

    @classmethod
    def _deduplicate(cls, candidates: list[dict]) -> list[dict]:
        unique: list[dict] = []
        for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
            duplicate = next(
                (
                    item
                    for item in unique
                    if cls._same_rectangle(item["box"], candidate["box"])
                ),
                None,
            )
            if duplicate is None:
                unique.append(candidate.copy())
                continue
            if candidate.get("source") == "visual":
                duplicate.update(candidate)
                duplicate["source"] = "visual+legacy"
            elif duplicate.get("source") == "visual":
                duplicate["source"] = "visual+legacy"
        return unique

    @classmethod
    def _select_smaller_rectangle(
        cls,
        visual: list[dict],
        legacy: list[dict],
        global_box_info: dict,
        image_shape,
    ):
        """Descarta o maior retângulo e retorna literalmente o menor interno."""
        candidates = cls._deduplicate(visual + legacy)
        if not candidates:
            return None

        # A maior moldura visual é a caixa do componente. global_box_info é
        # usado apenas quando não existe uma moldura visual grande.
        if visual:
            outer = max(visual, key=lambda item: item["area"])["box"]
        else:
            outer = cls._normalize_box(
                (
                    (global_box_info or {}).get("x", 0),
                    (global_box_info or {}).get("y", 0),
                    (global_box_info or {}).get("w", 0),
                    (global_box_info or {}).get("h", 0),
                ),
                image_shape,
            )
            if outer is None:
                return None

        outer_area = max(1, outer[2] * outer[3])
        internal = [
            item
            for item in candidates
            if not cls._same_rectangle(item["box"], outer)
            and cls._contains(outer, item["box"])
            and item["area"] <= outer_area * cls.MAX_INNER_AREA_RATIO
        ]
        if not internal:
            return None

        # Regra solicitada: dentre os retângulos internos, usar o menor.
        # Uma detecção visual ganha de uma caixa legada de área parecida.
        internal.sort(
            key=lambda item: (
                int(item["area"]),
                0 if "visual" in item.get("source", "") else 1,
            )
        )
        return internal[0]

    @classmethod
    def _content_box(cls, box, image_shape):
        """Remove somente a linha verde, preservando praticamente toda a ROI."""
        image_height, image_width = image_shape[:2]
        x, y, width, height = box
        inset = 1 if min(width, height) < 20 else 2
        if width - inset * 2 >= cls.MIN_SIDE_PX:
            x += inset
            width -= inset * 2
        if height - inset * 2 >= cls.MIN_SIDE_PX:
            y += inset
            height -= inset * 2
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
        """Recorta o menor retângulo verde do gabarito nas duas imagens."""
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
            visual = cls._visual_rectangles(sample_crop)
            legacy = cls._legacy_rectangles(old_epicenters, sample_crop.shape)
            selected = cls._select_smaller_rectangle(
                visual,
                legacy,
                global_box_info,
                sample_crop.shape,
            )
            if selected is not None:
                box = cls._content_box(selected["box"], sample_crop.shape)
                real_epicenters.append(box)
                x, y, width, height = box
                ordered = sorted(
                    [item["box"] for item in visual],
                    key=lambda value: value[2] * value[3],
                    reverse=True,
                )
                print(
                    "ROI AOI menor selecionada: "
                    f"X:{x} Y:{y} W:{width} H:{height} • "
                    f"retângulos verdes:{ordered} • origem:{selected.get('source')}"
                )
            else:
                print(
                    "ROI AOI menor não encontrada • "
                    f"visuais:{[item['box'] for item in visual]} • "
                    f"legadas:{[item['box'] for item in legacy]}"
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
