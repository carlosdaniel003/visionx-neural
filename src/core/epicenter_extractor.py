# src/core/epicenter_extractor.py
"""Extração da caixa verde menor desenhada pela AOI."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from src.config.settings import settings


class EpicenterExtractor:
    """Encontra a ROI verde do defeito usando o gabarito como autoridade.

    A caixa global também é verde, mas a caixa menor de referência sempre está
    no gabarito. No teste, a indicação da AOI pode ser azul, vermelha ou amarela;
    por isso candidatos do teste não participam da escolha da ROI.
    """

    MIN_SIDE_PX = 8
    MAX_IMAGE_RATIO = 0.96
    MIN_INNER_AREA_RATIO = 0.0012
    MAX_INNER_AREA_RATIO = 0.55
    DUPLICATE_IOU = 0.72
    MIN_FRAME_SCORE = 0.36

    @staticmethod
    def _valid_image(image: np.ndarray) -> bool:
        return isinstance(image, np.ndarray) and image.size > 0

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        """Máscara da cor de sobreposição, não da placa verde natural."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            np.asarray(settings.COLOR_GREEN_LOWER, dtype=np.uint8),
            np.asarray(settings.COLOR_GREEN_UPPER, dtype=np.uint8),
        )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        prototypes = np.asarray(
            settings.AOI_GREEN_RGB_SAMPLES,
            dtype=np.float32,
        )
        distance = np.sqrt(
            np.sum(
                (rgb[:, :, None, :] - prototypes[None, None, :, :]) ** 2,
                axis=3,
            )
        )
        prototype_mask = (
            np.min(distance, axis=2)
            <= float(settings.AOI_GREEN_MAX_RGB_DISTANCE)
        )

        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        green_excess = green - np.maximum(red, blue)
        dominance = green_excess >= float(settings.AOI_GREEN_MIN_EXCESS)

        # Verdes muito vivos podem sofrer interpolação e escapar alguns pontos
        # da distância RGB, mas ainda conservam saturação e dominância fortes.
        vivid = (
            (hsv[:, :, 1] >= 175)
            & (hsv[:, :, 2] >= 115)
            & (green_excess >= 22)
        )
        strict = (hsv_mask > 0) & dominance & (prototype_mask | vivid)
        mask = strict.astype(np.uint8) * 255

        # Fecha interrupções pequenas sem transformar manchas da placa em caixas.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
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
        tolerance = max(3, int(round(min(aw, ah, bw, bh) * 0.12)))
        return (
            abs(ax - bx) <= tolerance
            and abs(ay - by) <= tolerance
            and abs(aw - bw) <= tolerance * 2
            and abs(ah - bh) <= tolerance * 2
        )

    @staticmethod
    def _contains(outer, inner, tolerance: int = 6) -> bool:
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
        if width < cls.MIN_SIDE_PX or height < cls.MIN_SIDE_PX:
            return None
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        if x2 - x1 < cls.MIN_SIDE_PX or y2 - y1 < cls.MIN_SIDE_PX:
            return None
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _band_support(mask: np.ndarray, box, offset: int = 0) -> tuple:
        x, y, width, height = box
        image_height, image_width = mask.shape[:2]
        thickness = max(1, min(4, int(round(min(width, height) * 0.08))))

        def horizontal(row: int) -> float:
            y1 = max(0, row - thickness)
            y2 = min(image_height, row + thickness + 1)
            x1 = max(0, x + thickness)
            x2 = min(image_width, x + width - thickness)
            if y2 <= y1 or x2 <= x1:
                return 0.0
            return float(np.mean(mask[y1:y2, x1:x2] > 0))

        def vertical(column: int) -> float:
            x1 = max(0, column - thickness)
            x2 = min(image_width, column + thickness + 1)
            y1 = max(0, y + thickness)
            y2 = min(image_height, y + height - thickness)
            if y2 <= y1 or x2 <= x1:
                return 0.0
            return float(np.mean(mask[y1:y2, x1:x2] > 0))

        top = horizontal(y + offset)
        bottom = horizontal(y + height - 1 - offset)
        left = vertical(x + offset)
        right = vertical(x + width - 1 - offset)
        return top, bottom, left, right

    @classmethod
    def _frame_metrics(cls, mask: np.ndarray, box) -> dict:
        x, y, width, height = box
        image_height, image_width = mask.shape[:2]

        # O boundingRect pode cair na borda externa ou interna da linha. Testa
        # pequenos deslocamentos e conserva o melhor suporte de cada lado.
        all_supports = [cls._band_support(mask, box, offset) for offset in range(0, 4)]
        supports = tuple(max(values) for values in zip(*all_supports))
        ordered = sorted(supports, reverse=True)
        side_count = sum(value >= 0.24 for value in supports)
        three_side_score = float(np.mean(ordered[:3]))
        four_side_score = float(np.mean(supports))

        inset = max(2, min(6, int(round(min(width, height) * 0.14))))
        ix1 = min(image_width, max(0, x + inset))
        iy1 = min(image_height, max(0, y + inset))
        ix2 = min(image_width, max(ix1, x + width - inset))
        iy2 = min(image_height, max(iy1, y + height - inset))
        interior = mask[iy1:iy2, ix1:ix2]
        interior_density = (
            float(np.mean(interior > 0)) if interior.size else 1.0
        )
        hollow_score = float(np.clip(1.0 - interior_density / 0.38, 0.0, 1.0))

        corner_size = max(2, min(6, int(round(min(width, height) * 0.12))))
        corners = []
        for cx, cy in (
            (x, y),
            (x + width - 1, y),
            (x, y + height - 1),
            (x + width - 1, y + height - 1),
        ):
            x1 = max(0, cx - corner_size)
            x2 = min(image_width, cx + corner_size + 1)
            y1 = max(0, cy - corner_size)
            y2 = min(image_height, cy + corner_size + 1)
            patch = mask[y1:y2, x1:x2]
            corners.append(float(np.mean(patch > 0)) if patch.size else 0.0)
        corner_score = float(np.mean(sorted(corners, reverse=True)[:3]))

        frame_score = float(
            np.clip(
                0.42 * three_side_score
                + 0.22 * four_side_score
                + 0.20 * corner_score
                + 0.16 * hollow_score,
                0.0,
                1.0,
            )
        )
        return {
            "supports": supports,
            "side_count": side_count,
            "three_side_score": three_side_score,
            "four_side_score": four_side_score,
            "corner_score": corner_score,
            "interior_density": interior_density,
            "hollow_score": hollow_score,
            "frame_score": frame_score,
        }

    @classmethod
    def _contour_candidates(cls, image: np.ndarray) -> list[dict]:
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
            if area >= image_area * 0.92:
                continue
            if width >= image_width * cls.MAX_IMAGE_RATIO:
                continue
            if height >= image_height * cls.MAX_IMAGE_RATIO:
                continue
            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > 12.0:
                continue

            metrics = cls._frame_metrics(mask, box)
            perimeter = float(cv2.arcLength(contour, True))
            approximation = (
                cv2.approxPolyDP(contour, 0.035 * perimeter, True)
                if perimeter > 0
                else contour
            )
            quadrilateral = bool(
                4 <= len(approximation) <= 6
                and cv2.isContourConvex(approximation)
            )
            if metrics["side_count"] < 3 and not quadrilateral:
                continue
            if metrics["frame_score"] < 0.20:
                continue
            candidates.append(
                {
                    "box": box,
                    "area": area,
                    "sources": {"gabarito"},
                    "quadrilateral": quadrilateral,
                    **metrics,
                }
            )
        return candidates

    @classmethod
    def _legacy_candidates(
        cls,
        old_epicenters: list | None,
        image: np.ndarray,
    ) -> list[dict]:
        if not old_epicenters or not cls._valid_image(image):
            return []
        mask = cls._green_mask(image)
        candidates = []
        for raw_box in old_epicenters:
            box = cls._normalize_box(raw_box, image.shape)
            if box is None:
                continue
            metrics = cls._frame_metrics(mask, box)
            candidates.append(
                {
                    "box": box,
                    "area": box[2] * box[3],
                    "sources": {"legacy"},
                    "quadrilateral": False,
                    **metrics,
                }
            )
        return candidates

    @classmethod
    def _deduplicate(cls, candidates: list[dict]) -> list[dict]:
        ordered = sorted(candidates, key=lambda item: item["area"], reverse=True)
        unique: list[dict] = []
        for candidate in ordered:
            duplicate = next(
                (
                    existing
                    for existing in unique
                    if cls._same_rectangle(candidate["box"], existing["box"])
                ),
                None,
            )
            if duplicate is None:
                unique.append(
                    {
                        **candidate,
                        "sources": set(candidate.get("sources", set())),
                    }
                )
                continue
            duplicate["sources"].update(candidate.get("sources", set()))
            duplicate["frame_score"] = max(
                duplicate.get("frame_score", 0.0),
                candidate.get("frame_score", 0.0),
            )
            duplicate["side_count"] = max(
                duplicate.get("side_count", 0),
                candidate.get("side_count", 0),
            )
            duplicate["corner_score"] = max(
                duplicate.get("corner_score", 0.0),
                candidate.get("corner_score", 0.0),
            )
            duplicate["hollow_score"] = max(
                duplicate.get("hollow_score", 0.0),
                candidate.get("hollow_score", 0.0),
            )
            duplicate["quadrilateral"] = bool(
                duplicate.get("quadrilateral", False)
                or candidate.get("quadrilateral", False)
            )
        return unique

    @classmethod
    def _outer_box(cls, candidates: list[dict], global_box_info, image_shape):
        if isinstance(global_box_info, dict):
            box = cls._normalize_box(
                (
                    global_box_info.get("x", 0),
                    global_box_info.get("y", 0),
                    global_box_info.get("w", 0),
                    global_box_info.get("h", 0),
                ),
                image_shape,
            )
            if box is not None:
                return box

        reliable = [
            candidate
            for candidate in candidates
            if candidate.get("frame_score", 0.0) >= 0.28
            and candidate.get("side_count", 0) >= 3
        ]
        if reliable:
            return max(reliable, key=lambda item: item["area"])["box"]
        return 0, 0, image_shape[1], image_shape[0]

    @classmethod
    def _select_epicenter(
        cls,
        candidates: list[dict],
        global_box_info,
        image_shape,
    ):
        if not candidates:
            return None
        outer = cls._outer_box(candidates, global_box_info, image_shape)
        outer_area = max(1, outer[2] * outer[3])
        outer_center = (
            outer[0] + outer[2] / 2.0,
            outer[1] + outer[3] / 2.0,
        )

        valid = []
        for candidate in candidates:
            box = candidate["box"]
            if cls._same_rectangle(box, outer):
                continue
            if not cls._contains(outer, box, tolerance=8):
                continue
            area_ratio = candidate["area"] / outer_area
            if not (
                cls.MIN_INNER_AREA_RATIO
                <= area_ratio
                <= cls.MAX_INNER_AREA_RATIO
            ):
                continue
            if candidate.get("side_count", 0) < 3:
                continue
            if candidate.get("frame_score", 0.0) < cls.MIN_FRAME_SCORE:
                continue
            if candidate.get("interior_density", 1.0) > 0.46:
                continue

            x, y, width, height = box
            center_distance = math.hypot(
                x + width / 2.0 - outer_center[0],
                y + height / 2.0 - outer_center[1],
            ) / max(1.0, math.hypot(outer[2], outer[3]))
            source_bonus = (
                1.0
                if candidate.get("sources", set()) == {"gabarito", "legacy"}
                else 0.55 if "legacy" in candidate.get("sources", set()) else 0.0
            )
            confirmation = float(
                np.clip(
                    0.52 * candidate.get("frame_score", 0.0)
                    + 0.16 * candidate.get("corner_score", 0.0)
                    + 0.14 * candidate.get("hollow_score", 0.0)
                    + 0.14 * source_bonus
                    + 0.04 * (1.0 - center_distance),
                    0.0,
                    1.0,
                )
            )
            candidate["selection_score"] = confirmation
            candidate["area_ratio"] = area_ratio
            valid.append(candidate)

        if not valid:
            return None

        # Primeiro exige a melhor confirmação de moldura. Área só desempata;
        # reflexos pequenos ou regiões naturais da placa não vencem pela posição.
        return max(
            valid,
            key=lambda item: (
                float(item.get("selection_score", 0.0)),
                len(item.get("sources", set())),
                int(item.get("side_count", 0)),
                int(item.get("area", 0)),
            ),
        )

    @classmethod
    def _content_box(cls, box, image_shape):
        image_height, image_width = image_shape[:2]
        x, y, width, height = box
        minimum_side = min(width, height)
        inset = max(1, min(5, int(round(minimum_side * 0.08))))
        if width - 2 * inset < cls.MIN_SIDE_PX:
            inset = 0
        if height - 2 * inset < cls.MIN_SIDE_PX:
            inset = 0
        x += inset
        y += inset
        width -= inset * 2
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
        """Retorna a caixa interna e o mesmo recorte em gabarito e teste."""
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
            candidates = cls._contour_candidates(sample_crop)
            candidates.extend(cls._legacy_candidates(old_epicenters, sample_crop))
            candidates = cls._deduplicate(candidates)
            selected = cls._select_epicenter(
                candidates,
                global_box_info,
                sample_crop.shape,
            )
            if selected is not None:
                content_box = cls._content_box(selected["box"], sample_crop.shape)
                real_epicenters.append(content_box)
                x, y, width, height = content_box
                print(
                    "Epicentro AOI do gabarito selecionado: "
                    f"X:{x} Y:{y} W:{width} H:{height} • "
                    f"score:{selected.get('selection_score', 0.0):.2f} • "
                    f"lados:{selected.get('side_count', 0)} • "
                    f"fontes:{','.join(sorted(selected.get('sources', set())))}"
                )
        except Exception as exc:
            print(f"Erro na seleção da ROI verde menor: {exc}")

        # Se a detecção visual falhar, usa somente uma caixa antiga que ainda
        # apresente três lados verdes no gabarito. Nunca escolhe pela menor área.
        if not real_epicenters:
            legacy = cls._legacy_candidates(old_epicenters, sample_crop)
            reliable = [
                item
                for item in legacy
                if item.get("side_count", 0) >= 3
                and item.get("frame_score", 0.0) >= 0.24
            ]
            if reliable:
                selected = max(
                    reliable,
                    key=lambda item: (
                        item.get("frame_score", 0.0),
                        item.get("area", 0),
                    ),
                )
                real_epicenters.append(
                    cls._content_box(selected["box"], sample_crop.shape)
                )

        image_height, image_width = sample_crop.shape[:2]
        if real_epicenters:
            x, y, width, height = real_epicenters[0]
            x2 = min(image_width, x + width)
            y2 = min(image_height, y + height)
            if x2 > x and y2 > y:
                focus_gab = sample_crop[y:y2, x:x2].copy()
                focus_ng = ng_crop[y:y2, x:x2].copy()
                if focus_gab.shape != focus_ng.shape:
                    focus_ng = cv2.resize(
                        focus_ng,
                        (focus_gab.shape[1], focus_gab.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )

        return real_epicenters, focus_gab, focus_ng
