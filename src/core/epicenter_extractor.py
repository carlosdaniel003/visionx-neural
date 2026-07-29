# src/core/epicenter_extractor.py
"""Extração robusta da ROI de epicentro desenhada pela AOI."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from src.config.settings import settings


class EpicenterExtractor:
    """Seleciona a moldura verde menor que indica o defeito.

    A imagem pode conter reflexos verdes, pequenos textos e os contornos interno
    e externo de cada linha desenhada pela AOI. Por isso, "menor caixa" não é
    suficiente. O seletor procura uma moldura retangular fechada dentro da caixa
    global e privilegia confirmação no gabarito, no teste e no detector legado.
    """

    MIN_SIDE_PX = 10
    MAX_IMAGE_RATIO = 0.94
    INNER_MIN_AREA_RATIO = 0.0015
    INNER_MAX_AREA_RATIO = 0.60
    DUPLICATE_IOU = 0.78
    MIN_FRAME_SCORE = 0.30
    MAX_FILLED_GREEN_DENSITY = 0.60

    @staticmethod
    def _green_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.asarray(settings.COLOR_GREEN_LOWER, dtype=np.uint8)
        upper = np.asarray(settings.COLOR_GREEN_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
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
    def _contains(outer, inner, tolerance: int = 5) -> bool:
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
    def _side_supports(mask: np.ndarray, box) -> tuple[float, float, float, float, float]:
        x, y, width, height = box
        image_height, image_width = mask.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0, 0.0, 1.0

        local = mask[y1:y2, x1:x2]
        thickness = max(1, min(5, int(round(min(width, height) * 0.08))))
        top = float(np.mean(local[:thickness, :] > 0))
        bottom = float(np.mean(local[-thickness:, :] > 0))
        left = float(np.mean(local[:, :thickness] > 0))
        right = float(np.mean(local[:, -thickness:] > 0))
        density = float(np.mean(local > 0))
        return top, bottom, left, right, density

    @classmethod
    def _frame_metrics(
        cls,
        mask: np.ndarray,
        contour: np.ndarray,
        hierarchy: np.ndarray | None,
        index: int,
        box,
    ) -> dict:
        top, bottom, left, right, density = cls._side_supports(mask, box)
        side_values = sorted((top, bottom, left, right), reverse=True)
        # Três lados fortes bastam porque a etiqueta azul/vermelha pode cobrir
        # parte de uma das linhas da moldura.
        axis_frame = float(np.mean(side_values[:3]))

        contour_area = abs(float(cv2.contourArea(contour)))
        rotated = cv2.minAreaRect(contour)
        rotated_area = max(1.0, float(rotated[1][0] * rotated[1][1]))
        shape_support = float(np.clip(contour_area / rotated_area, 0.0, 1.0))

        perimeter = float(cv2.arcLength(contour, True))
        approximate = (
            cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if perimeter > 0
            else contour
        )
        polygon_frame = bool(
            len(approximate) == 4
            and cv2.isContourConvex(approximate)
            and shape_support >= 0.45
        )

        child = int(hierarchy[0][index][2]) if hierarchy is not None else -1
        has_frame_hole = child >= 0

        # Uma moldura é vazada. Blocos verdes preenchidos são normalmente
        # reflexos, textos ou elementos naturais da placa.
        density_penalty = float(
            np.clip(1.0 - max(0.0, density - 0.36) / 0.34, 0.0, 1.0)
        )
        frame_score = max(axis_frame, 0.70 if polygon_frame else 0.0)
        frame_score *= density_penalty

        quality = float(
            np.clip(
                0.50 * frame_score
                + 0.30 * shape_support
                + 0.20 * (1.0 if has_frame_hole else 0.0),
                0.0,
                1.0,
            )
        )
        return {
            "axis_frame_score": axis_frame,
            "frame_score": float(frame_score),
            "green_density": density,
            "shape_support": shape_support,
            "polygon_frame": polygon_frame,
            "has_frame_hole": has_frame_hole,
            "quality": quality,
        }

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
            if aspect > 10.0:
                continue

            metrics = cls._frame_metrics(
                mask,
                contour,
                hierarchy,
                index,
                (x, y, width, height),
            )
            if (
                not metrics["has_frame_hole"]
                and metrics["frame_score"] < 0.22
            ):
                continue
            if (
                metrics["green_density"] > cls.MAX_FILLED_GREEN_DENSITY
                and not metrics["has_frame_hole"]
            ):
                continue

            candidates.append(
                {
                    "box": (int(x), int(y), int(width), int(height)),
                    "area": int(box_area),
                    "hierarchy_depth": cls._hierarchy_depth(index, hierarchy),
                    "sources": {source},
                    **metrics,
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
                    "quality": 0.52,
                    "frame_score": 0.52,
                    "axis_frame_score": 0.0,
                    "green_density": 0.0,
                    "shape_support": 0.0,
                    "polygon_frame": False,
                    "has_frame_hole": False,
                    "sources": {"legacy"},
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

            duplicate["hierarchy_depth"] = max(
                int(duplicate.get("hierarchy_depth", 0)),
                int(candidate.get("hierarchy_depth", 0)),
            )
            for key in (
                "quality",
                "frame_score",
                "axis_frame_score",
                "shape_support",
            ):
                duplicate[key] = max(
                    float(duplicate.get(key, 0.0)),
                    float(candidate.get(key, 0.0)),
                )
            duplicate["green_density"] = min(
                float(duplicate.get("green_density", 1.0)),
                float(candidate.get("green_density", 1.0)),
            )
            duplicate["polygon_frame"] = bool(
                duplicate.get("polygon_frame", False)
                or candidate.get("polygon_frame", False)
            )
            duplicate["has_frame_hole"] = bool(
                duplicate.get("has_frame_hole", False)
                or candidate.get("has_frame_hole", False)
            )
            duplicate["sources"].update(candidate.get("sources", set()))
        return unique

    @classmethod
    def _select_outer(cls, candidates: list[dict]):
        reliable = [
            candidate
            for candidate in candidates
            if candidate.get("has_frame_hole", False)
            or candidate.get("frame_score", 0.0) >= cls.MIN_FRAME_SCORE
        ]
        return max(reliable or candidates, key=lambda item: item["area"])

    @classmethod
    def _select_epicenter(cls, candidates: list[dict], image_shape):
        if not candidates:
            return None

        image_height, image_width = image_shape[:2]
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        for candidate in candidates:
            x, y, width, height = candidate["box"]
            candidate["center_distance"] = math.hypot(
                x + width / 2.0 - center_x,
                y + height / 2.0 - center_y,
            )

        outer = cls._select_outer(candidates)
        _, _, outer_width, outer_height = outer["box"]
        inner_candidates: list[dict] = []

        for candidate in candidates:
            if candidate is outer:
                continue
            if not cls._contains(outer["box"], candidate["box"]):
                continue

            area_ratio = candidate["area"] / max(1.0, outer["area"])
            _, _, width, height = candidate["box"]
            minimum_relative_side = min(
                width / max(1.0, outer_width),
                height / max(1.0, outer_height),
            )
            reliable_frame = bool(
                candidate.get("has_frame_hole", False)
                or candidate.get("frame_score", 0.0) >= cls.MIN_FRAME_SCORE
                or "legacy" in candidate.get("sources", set())
            )
            if not reliable_frame:
                continue
            if not (
                cls.INNER_MIN_AREA_RATIO
                <= area_ratio
                <= cls.INNER_MAX_AREA_RATIO
            ):
                continue
            if minimum_relative_side < 0.025:
                continue

            candidate["area_ratio"] = float(area_ratio)
            inner_candidates.append(candidate)

        if not inner_candidates:
            return None

        # A ROI verdadeira costuma aparecer nos dois painéis e também no detector
        # legado. Entre molduras confiáveis, escolhemos a maior caixa interna:
        # reflexos e letras verdes são menores e não têm moldura completa.
        return max(
            inner_candidates,
            key=lambda item: (
                len(item.get("sources", set())),
                1 if "legacy" in item.get("sources", set()) else 0,
                1 if item.get("has_frame_hole", False) else 0,
                float(item.get("frame_score", 0.0)),
                int(item["area"]),
                -float(item.get("center_distance", 0.0)),
            ),
        )

    @classmethod
    def _content_box(cls, box, image_shape):
        """Remove a espessura da linha verde do recorte analisado."""
        image_height, image_width = image_shape[:2]
        x, y, width, height = box
        inset = max(1, min(3, int(round(min(width, height) * 0.04))))
        if width - 2 * inset < cls.MIN_SIDE_PX:
            inset = 0
        if height - 2 * inset < cls.MIN_SIDE_PX:
            inset = 0

        x = max(0, int(x + inset))
        y = max(0, int(y + inset))
        width = min(int(width - 2 * inset), image_width - x)
        height = min(int(height - 2 * inset), image_height - y)
        return x, y, max(1, width), max(1, height)

    @classmethod
    def extract_focus(
        cls,
        sample_crop: np.ndarray,
        ng_crop: np.ndarray,
        old_epicenters: list,
        global_box_info: dict,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Retorna o epicentro interno e os dois recortes correspondentes."""
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
                cls._legacy_candidates(old_epicenters, sample_crop.shape)
            )
            candidates = cls._deduplicate(candidates)
            selected = cls._select_epicenter(candidates, sample_crop.shape)
            if selected is not None:
                content_box = cls._content_box(
                    selected["box"],
                    sample_crop.shape,
                )
                real_epicenters.append(content_box)
                x, y, width, height = content_box
                print(
                    "Epicentro AOI interno selecionado: "
                    f"X:{x} Y:{y} W:{width} H:{height} • "
                    f"moldura:{selected.get('frame_score', 0.0):.2f} • "
                    f"fontes:{','.join(sorted(selected.get('sources', set())))}"
                )
        except Exception as exc:
            print(f"Erro na seleção robusta do epicentro: {exc}")

        # O detector legado já remove a caixa global. Se for necessário usá-lo,
        # a maior caixa interna é mais segura que pontos e reflexos pequenos.
        if not real_epicenters and old_epicenters:
            legacy = cls._legacy_candidates(old_epicenters, sample_crop.shape)
            if legacy:
                legacy.sort(key=lambda item: item["area"], reverse=True)
                real_epicenters.append(
                    cls._content_box(legacy[0]["box"], sample_crop.shape)
                )

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
