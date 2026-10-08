"""Reconstrói caixas de foco verdes quando molduras da AOI se cruzam.

O radar de contornos continua sendo a primeira escolha. Esta recuperação só
é usada quando ele falha e exige duas molduras geometricamente independentes:
um quadro externo e outro mais estreito, com suas próprias laterais e topo.
Uma única moldura, mesmo grande, nunca cria um epicentro artificial.
"""

import cv2
import numpy as np


RADAR_GREEN_LOWER = np.array((50, 150, 100), dtype=np.uint8)
RADAR_GREEN_UPPER = np.array((75, 255, 255), dtype=np.uint8)


def _line_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cv2.boundingRect(contour) for contour in contours]


def _find_nested_frame(
    vertical_source: np.ndarray, horizontal_source: np.ndarray
) -> tuple[int, int, int, int] | None:
    height, width = vertical_source.shape[:2]
    vertical_mask = cv2.morphologyEx(
        vertical_source,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(25, int(height * 0.14)))
        ),
    )
    horizontal_mask = cv2.morphologyEx(
        horizontal_source,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(18, int(width * 0.10)), 1)
        ),
    )
    verticals = [
        (x, y, w, h)
        for x, y, w, h in _line_rectangles(vertical_mask)
        if h >= max(25, int(height * 0.30))
        and w <= max(6, int(width * 0.03))
    ]
    horizontals = [
        (x, y, w, h)
        for x, y, w, h in _line_rectangles(horizontal_mask)
        if w > 15 and h <= max(6, int(height * 0.04))
    ]
    if len(verticals) < 4 or len(horizontals) < 2:
        return None

    edge_tolerance = max(5, int(width * 0.025))
    top_tolerance = max(5, int(height * 0.035))
    frames: list[tuple[int, int, int, int]] = []

    for hx, hy, hw, hh in horizontals:
        horizontal_top = hy + hh / 2
        lefts = [
            line
            for line in verticals
            if abs(line[0] + line[2] / 2 - hx) <= edge_tolerance
            and abs(line[1] - horizontal_top) <= top_tolerance
        ]
        rights = [
            line
            for line in verticals
            if abs(line[0] + line[2] / 2 - (hx + hw - 1))
            <= edge_tolerance
            and abs(line[1] - horizontal_top) <= top_tolerance
        ]
        for lx, ly, lw, lh in lefts:
            for rx, ry, rw, rh in rights:
                if rx <= lx:
                    continue
                left = min(lx, hx)
                right = max(rx + rw, hx + hw)
                top = min(ly, ry)
                bottom = min(ly + lh, ry + rh)
                frame = (left, top, right - left, bottom - top)
                if frame[2] > 15 and frame[3] > 15 and frame not in frames:
                    frames.append(frame)

    # Há casos reais em que o topo do quadro menor fica ACIMA do quadro
    # global. A relação de enquadramento deve usar as laterais e a grande
    # sobreposição vertical, e não exigir inclusão estrita nos dois eixos.
    margin = max(5, int(width * 0.014))
    inner_frames = []
    for frame in frames:
        x, y, w, h = frame
        for outer in frames:
            ox, oy, ow, oh = outer
            vertical_overlap = max(
                0, min(y + h, oy + oh) - max(y, oy)
            )
            if (
                ox + margin <= x
                and x + w + margin <= ox + ow
                and vertical_overlap > min(h, oh) * 0.25
            ):
                inner_frames.append(frame)
                break

    if not inner_frames:
        return None
    return min(
        inner_frames,
        key=lambda frame: (
            frame[2] * frame[3],
            abs(frame[0] + frame[2] / 2 - width / 2),
        ),
    )


def recover_nested_frame_focus(
    sample_crop: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Recupera ROI cortada na borda inferior, sem usar imagem do TESTE.

    A segunda passagem fecha somente pequenas falhas de linhas horizontais
    ou verticais; ambas mantêm a exigência de duas molduras confirmadas.
    """
    height, width = sample_crop.shape[:2]
    hsv = cv2.cvtColor(sample_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RADAR_GREEN_LOWER, RADAR_GREEN_UPPER)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    focus = _find_nested_frame(mask, mask)
    if focus is not None:
        return focus

    vertical_repaired = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(7, int(height * 0.025)))
        ),
    )
    horizontal_repaired = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(7, int(width * 0.025)), 1)
        ),
    )
    return _find_nested_frame(vertical_repaired, horizontal_repaired)
