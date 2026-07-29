# src/core/epicenter_extractor.py
import cv2
import numpy as np
import math
from typing import Tuple


class EpicenterExtractor:
    """Extrai a ROI menor indicada pela AOI.

    Mantém o comportamento estável do Radar Euclidiano, mas prioriza as caixas
    internas já identificadas por ``inspection.py``. Isso impede que barras da
    interface, como ``0.OK``, sejam escolhidas apenas por estarem próximas do
    centro da imagem.
    """

    @staticmethod
    def _valid_box(box, img_w: int, img_h: int):
        if not box or len(box) < 4:
            return None
        try:
            x, y, w, h = (int(round(float(value))) for value in box[:4])
        except (TypeError, ValueError):
            return None
        if w <= 15 or h <= 15:
            return None
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return None
        if w >= img_w * 0.90 or h >= img_h * 0.90:
            return None
        return x, y, w, h

    @staticmethod
    def _same_box(first, second, tolerance: int = 9) -> bool:
        return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(first, second))

    @classmethod
    def _deduplicate(cls, boxes):
        unique = []
        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            if not any(cls._same_box(box, existing) for existing in unique):
                unique.append(box)
        return unique

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
    def _legacy_epicenter(cls, old_epicenters, img_w: int, img_h: int):
        """Usa primeiro as ROIs que ``inspection.py`` já filtrou da caixa global."""
        valid = []
        for candidate in old_epicenters or []:
            box = cls._valid_box(candidate, img_w, img_h)
            if box is not None:
                valid.append(box)
        valid = cls._deduplicate(valid)
        if not valid:
            return None

        # inspection.py já remove a maior caixa global. Restam a ROI verdadeira
        # e, ocasionalmente, pequenos contornos internos. O maior candidato
        # restante preserva o comportamento estável do commit 1849e338.
        return max(valid, key=lambda box: box[2] * box[3])

    @classmethod
    def _radar_epicenter(cls, sample_crop: np.ndarray):
        """Fallback visual: caixa menor contida na maior moldura verde."""
        img_h, img_w = sample_crop.shape[:2]
        hsv = cv2.cvtColor(sample_crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 150, 100])
        upper_green = np.array([75, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 15 or h <= 15:
                continue
            if w >= img_w * 0.98 or h >= img_h * 0.98:
                continue
            boxes.append((x, y, w, h))
        boxes = cls._deduplicate(boxes)
        if not boxes:
            return None

        # A maior moldura representa a área geral da peça. Barras da interface,
        # como 0.OK, ficam fora dela e deixam de competir com o epicentro.
        outer = max(boxes, key=lambda box: box[2] * box[3])
        inner = [
            box
            for box in boxes
            if box != outer
            and cls._contains(outer, box)
            and box[2] * box[3] < outer[2] * outer[3] * 0.85
        ]
        if inner:
            # Depois de remover a moldura global, escolhemos o maior retângulo
            # interno. Isso evita escolher letras, reflexos ou a borda dupla da
            # própria linha verde.
            return max(inner, key=lambda box: box[2] * box[3])

        # Compatibilidade com imagens antigas que não possuem moldura global.
        center_x, center_y = img_w / 2.0, img_h / 2.0
        valid = [
            box
            for box in boxes
            if box[2] < img_w * 0.85 and box[3] < img_h * 0.85
        ]
        if not valid:
            return None
        return min(
            valid,
            key=lambda box: math.hypot(
                center_x - (box[0] + box[2] / 2.0),
                center_y - (box[1] + box[3] / 2.0),
            ),
        )

    @classmethod
    def extract_focus(
        cls,
        sample_crop: np.ndarray,
        ng_crop: np.ndarray,
        old_epicenters: list,
        global_box_info: dict,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Retorna a ROI, o recorte do gabarito e o mesmo recorte do teste."""
        real_epicenters = []
        focus_gab = np.array([])
        focus_ng = np.array([])

        if (
            not isinstance(sample_crop, np.ndarray)
            or not isinstance(ng_crop, np.ndarray)
            or sample_crop.size == 0
            or ng_crop.size == 0
        ):
            return real_epicenters, focus_gab, focus_ng

        img_h, img_w = sample_crop.shape[:2]
        if ng_crop.shape[:2] != sample_crop.shape[:2]:
            ng_crop = cv2.resize(ng_crop, (img_w, img_h))

        selected = cls._legacy_epicenter(old_epicenters, img_w, img_h)
        source = "inspection"

        if selected is None:
            try:
                selected = cls._radar_epicenter(sample_crop)
                source = "radar"
            except Exception as error:
                print(f"Erro no Radar Euclidiano: {error}")

        if selected is None and global_box_info:
            fallback = (
                global_box_info.get("x", 0),
                global_box_info.get("y", 0),
                global_box_info.get("w", img_w),
                global_box_info.get("h", img_h),
            )
            selected = cls._valid_box(fallback, img_w, img_h)
            source = "global"

        if selected is None:
            return real_epicenters, focus_gab, focus_ng

        ex, ey, ew, eh = selected
        x1 = max(0, ex)
        y1 = max(0, ey)
        x2 = min(img_w, ex + ew)
        y2 = min(img_h, ey + eh)
        if x2 <= x1 or y2 <= y1:
            return real_epicenters, focus_gab, focus_ng

        selected = (x1, y1, x2 - x1, y2 - y1)
        real_epicenters.append(selected)
        focus_gab = sample_crop[y1:y2, x1:x2].copy()
        focus_ng = ng_crop[y1:y2, x1:x2].copy()
        if focus_gab.shape != focus_ng.shape:
            focus_ng = cv2.resize(
                focus_ng,
                (focus_gab.shape[1], focus_gab.shape[0]),
            )

        print(
            "Epicentro AOI selecionado: "
            f"X:{selected[0]} Y:{selected[1]} "
            f"W:{selected[2]} H:{selected[3]} • origem:{source}"
        )
        return real_epicenters, focus_gab, focus_ng
