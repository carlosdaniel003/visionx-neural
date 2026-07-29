# src/core/epicenter_extractor.py
"""Extração hierárquica da ROI de epicentro desenhada pela AOI."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from src.config.settings import settings


class EpicenterExtractor:
    """Seleciona a caixa verde interna, não a moldura global do componente.

    A AOI normalmente desenha duas regiões:
    - uma caixa verde maior delimitando o componente;
    - uma caixa verde menor apontando o defeito.

    Contornos internos e externos da mesma linha verde são deduplicados. A
    seleção privilegia a caixa geometricamente mais aninhada e, em empate, a
    menor área válida. A proximidade do centro é apenas critério final.
    """

    MIN_SIDE_PX = 7
    MAX_IMAGE_RATIO = 0.94
    INNER_MAX_AREA_RATIO = 0.78
    DUPLICATE_IOU = 0.78

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.asarray(settings.COLOR_GREEN_LOWER, dtype=np.uint8)
        upper = np.asarray(settings.COLOR_GREEN_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Fecha pequenos intervalos da linha sem unir caixas separadas.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _hierarchy_depth(index: int, hierarchy: np.ndarray | None) -> int:
        if hierarchy is None or hierarchy.size == 0:
            return 0
        depth = 0
        parent = int(hierarchy[0][index][3])
        guard = 0
        while parent >= 0 and guard < len(hierarchy[0]):
            depth += 1
            parent = int(hierarchy[0][parent][3])
            guard += 1
        return depth

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
        tolerance = max(3, int(round(min(aw, ah, bw, bh) * 0.10)))
        return (
            abs(ax - bx) <= tolerance
            and abs(ay - by) <= tolerance
            and abs(aw - bw) <= tolerance * 2
            and abs(ah - bh) <= tolerance * 2
        )

    @staticmethod
    def _contains(outer, inner, tolerance: int = 4) -> bool:
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

    @staticmethod
    def _border_support(mask: np.ndarray, box) -> float:
        x, y, width, height = box
        image_height, image_width = mask.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        local = mask[y1:y2, x1:x2]
        thickness = max(1, min(4, int(round(min(width, height) * 0.08))))
        border = np.zeros(local.shape, dtype=np.uint8)
        border[:thickness, :] = 255
        border[-thickness:, :] = 255
        border[:, :thickness] = 255
        border[:, -thickness:] = 255
        selected = border > 0
        return float(np.mean(local[selected] > 0)) if np.any(selected) else 0.0

    @classmethod
    def _contour_candidates(cls, image: np.ndarray, source: str) -> list[dict]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return []

        image_height, image_width = image.shape[:2]
        image_area = max(1, image_height * image_width)
        mask = cls._green_mask(image)
        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[dict] = []
        for index, contour in enumerate(contours):
            x, y, width, height = cv2.boundingRect(contour)
            box_area = width * height
            if width < cls.MIN_SIDE_PX or height < cls.MIN_SIDE_PX:
                continue
            if width >= image_width * cls.MAX_IMAGE_RATIO:
                continue
            if height >= image_height * cls.MAX_IMAGE_RATIO:
                continue
            if box_area >= image_area * 0.90:
                continue

            aspect = max(width, height) / max(1.0, min(width, height))
            if aspect > 12.0:
                continue

            contour_area = abs(float(cv2.contourArea(contour)))
            rotated = cv2.minAreaRect(contour)
            rotated_area = max(1.0, float(rotated[1][0] * rotated[1][1]))
            shape_support = float(np.clip(contour_area / rotated_area, 0.0, 1.0))
            border_support = cls._border_support(mask, (x, y, width, height))

            # Linhas quebradas podem ter contourArea pequeno; a presença verde
            # no perímetro ainda as mantém como candidatas.
            quality = float(np.clip(0.62 * shape_support + 0.38 * border_support, 0.0, 1.0))
            if quality < 0.08 and border_support < 0.12:
                continue

            candidates.append(
                {
                    "box": (int(x), int(y), int(width), int(height)),
                    "area": int(box_area),
                    "hierarchy_depth": cls._hierarchy_depth(index, hierarchy),
                    "quality": quality,
                    "sources": {source},
                }
            )
        return candidates

    @classmethod
    def _legacy_candidates(
        cls,
        old_epicenters: list | None,
        image_shape,
    ) -> list[dict]:
        if not old_epicenters:
            return []
        image_height, image_width = image_shape[:2]
        candidates = []
        for candidate in old_epicenters:
            if not candidate or len(candidate) < 4:
                continue
            try:
                x, y, width, height = (
                    int(round(float(value))) for value in candidate[:4]
                )
            except (TypeError, ValueError):
                continue
            if width < cls.MIN_SIDE_PX or height < cls.MIN_SIDE_PX:
                continue
            if x + width <= 0 or y + height <= 0:
                continue
            if x >= image_width or y >= image_height:
                continue
            x = max(0, x)
            y = max(0, y)
            width = min(width, image_width - x)
            height = min(height, image_height - y)
            if width <= 0 or height <= 0:
                continue
            candidates.append(
                {
                    "box": (x, y, width, height),
                    "area": width * height,
                    "hierarchy_depth": 0,
                    "quality": 0.35,
                    "sources": {"legacy"},
                }
            )
        return candidates

    @classmethod
    def _deduplicate(cls, candidates: list[dict]) -> list[dict]:
        # Começa pelas caixas maiores para conservar o limite externo da linha
        # grossa, mas agrega a maior profundidade do contorno interno duplicado.
        ordered = sorted(candidates, key=lambda item: item["area"], reverse=True)
        unique: list[dict] = []
        for candidate in ordered:
            duplicate = None
            for existing in unique:
                if cls._same_rectangle(candidate["box"], existing["box"]):
                    duplicate = existing
                    break
            if duplicate is None:
                unique.append(
                    {
                        **candidate,
                        "sources": set(candidate.get("sources", set())),
                    }
                )
                continue

            duplicate["hierarchy_depth"] = max(
                int(duplicate.get("hierarchy_depth", 0)),
                int(candidate.get("hierarchy_depth", 0)),
            )
            duplicate["quality"] = max(
                float(duplicate.get("quality", 0.0)),
                float(candidate.get("quality", 0.0)),
            )
            duplicate["sources"].update(candidate.get("sources", set()))
        return unique

    @classmethod
    def _geometric_depth(cls, candidate: dict, candidates: list[dict]) -> int:
        box = candidate["box"]
        return sum(
            1
            for other in candidates
            if other is not candidate
            and other["area"] > candidate["area"] * 1.20
            and cls._contains(other["box"], box)
        )

    @classmethod
    def _select_epicenter(cls, candidates: list[dict], image_shape):
        if not candidates:
            return None

        image_height, image_width = image_shape[:2]
        center_x = image_width / 2.0
        center_y = image_height / 2.0

        for candidate in candidates:
            candidate["geometric_depth"] = cls._geometric_depth(
                candidate,
                candidates,
            )
            x, y, width, height = candidate["box"]
            candidate["center_distance"] = math.hypot(
                x + width / 2.0 - center_x,
                y + height / 2.0 - center_y,
            )

        possible_outers = [
            candidate
            for candidate in candidates
            if any(
                other is not candidate
                and other["area"] < candidate["area"] * cls.INNER_MAX_AREA_RATIO
                and cls._contains(candidate["box"], other["box"])
                for other in candidates
            )
        ]
        outer = (
            max(possible_outers, key=lambda item: item["area"])
            if possible_outers
            else max(candidates, key=lambda item: item["area"])
        )

        inner_candidates = [
            candidate
            for candidate in candidates
            if candidate is not outer
            and candidate["area"] < outer["area"] * cls.INNER_MAX_AREA_RATIO
            and cls._contains(outer["box"], candidate["box"])
        ]

        pool = inner_candidates or candidates
        selected = min(
            pool,
            key=lambda item: (
                -int(item.get("geometric_depth", 0)),
                -int(item.get("hierarchy_depth", 0)),
                int(item["area"]),
                -len(item.get("sources", set())),
                -float(item.get("quality", 0.0)),
                float(item.get("center_distance", 0.0)),
            ),
        )
        return selected

    @classmethod
    def extract_focus(
        cls,
        sample_crop: np.ndarray,
        ng_crop: np.ndarray,
        old_epicenters: list,
        global_box_info: dict,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Retorna epicentro selecionado e os recortes correspondentes."""
        real_epicenters: list[tuple[int, int, int, int]] = []
        focus_gab = np.array([])
        focus_ng = np.array([])

        if (
            not isinstance(sample_crop, np.ndarray)
            or not isinstance(ng_crop, np.ndarray)
            or sample_crop.size == 0
            or ng_crop.size == 0
        ):
            return real_epicenters, focus_gab, focus_ng

        if sample_crop.shape != ng_crop.shape:
            ng_crop = cv2.resize(
                ng_crop,
                (sample_crop.shape[1], sample_crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        image_height, image_width = sample_crop.shape[:2]

        try:
            candidates = []
            candidates.extend(cls._contour_candidates(sample_crop, "gabarito"))
            candidates.extend(cls._contour_candidates(ng_crop, "teste"))
            candidates.extend(
                cls._legacy_candidates(
                    old_epicenters,
                    sample_crop.shape,
                )
            )
            candidates = cls._deduplicate(candidates)
            selected = cls._select_epicenter(candidates, sample_crop.shape)
            if selected is not None:
                real_epicenters.append(selected["box"])
                x, y, width, height = selected["box"]
                print(
                    "Epicentro AOI interno selecionado: "
                    f"X:{x} Y:{y} W:{width} H:{height} • "
                    f"nível:{selected.get('geometric_depth', 0)} • "
                    f"fontes:{','.join(sorted(selected.get('sources', set())))}"
                )
        except Exception as exc:
            print(f"Erro na seleção hierárquica do epicentro: {exc}")

        # Fallback: prioriza a menor caixa antiga válida, nunca a maior moldura.
        if not real_epicenters and old_epicenters:
            legacy = cls._legacy_candidates(old_epicenters, sample_crop.shape)
            if legacy:
                legacy.sort(key=lambda item: item["area"])
                real_epicenters.append(legacy[0]["box"])

        # Último fallback: caixa global somente quando sua posição é conhecida.
        if not real_epicenters and isinstance(global_box_info, dict):
            try:
                x = int(global_box_info.get("x", 0))
                y = int(global_box_info.get("y", 0))
                width = int(global_box_info.get("w", image_width))
                height = int(global_box_info.get("h", image_height))
                if (
                    cls.MIN_SIDE_PX <= width < image_width * cls.MAX_IMAGE_RATIO
                    and cls.MIN_SIDE_PX <= height < image_height * cls.MAX_IMAGE_RATIO
                    and 0 <= x < image_width
                    and 0 <= y < image_height
                ):
                    real_epicenters.append((x, y, width, height))
            except (TypeError, ValueError):
                pass

        if real_epicenters:
            x, y, width, height = real_epicenters[0]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image_width, x1 + max(1, int(width)))
            y2 = min(image_height, y1 + max(1, int(height)))
            if x2 > x1 and y2 > y1:
                focus_gab = sample_crop[y1:y2, x1:x2].copy()
                focus_ng = ng_crop[y1:y2, x1:x2].copy()
                if focus_gab.shape != focus_ng.shape:
                    focus_ng = cv2.resize(
                        focus_ng,
                        (focus_gab.shape[1], focus_gab.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )

        return real_epicenters, focus_gab, focus_ng
