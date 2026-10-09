"""Overlay de features REAIS da CNN sobre a imagem de teste do epicentro.

Não detecta defeitos nem calcula atenção: recebe SOMENTE os mapas 2D
derivados de encoder/Grad-CAM já calculados em faltando_explainability.
Preserva os pixels do teste como base, realça intensidades altas e insere
uma miniatura do gabarito com borda amarela para comparação visual.

Executado apenas no subprocesso isolado da CNN, nunca na GUI Qt.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_EDGE = 640
MIN_PRESENTATION_EDGE = 300
KINDS = ("latent", "gradcam", "activation")


def _display_size(h: int, w: int) -> tuple[int, int]:
    if h < 1 or w < 1:
        raise ValueError("Recorte da AOI vazio")
    # Evita aumentar dimensões de entrada já grandes; amplia epicentros
    # pequenos apenas para tornar o gabarito legível no card.
    scale = min(MAX_EDGE / max(h, w), max(1.0, MIN_PRESENTATION_EDGE / max(h, w)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def compose_neural_overlay(
    test_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    activation_2d: np.ndarray,
    *,
    kind: str,
) -> np.ndarray:
    """Devolve TESTE + overlay da CNN + miniatura GABARITO.

    A imagem do teste fica intacta onde a intensidade CNN é zero. A opacidade
    cresce APENAS com ativações efetivas da rede, evitando pintar de azul
    regiões inteiras e aparentar falsamente um defeito detectado.
    """
    if kind not in KINDS:
        raise ValueError("Tipo de projeção neural não suportado")
    if (
        not isinstance(test_bgr, np.ndarray)
        or test_bgr.dtype != np.uint8 or test_bgr.ndim != 3
        or test_bgr.shape[2] != 3 or not min(test_bgr.shape[:2])
        or not isinstance(reference_bgr, np.ndarray)
        or reference_bgr.dtype != np.uint8 or reference_bgr.ndim != 3
        or reference_bgr.shape[2] != 3 or not min(reference_bgr.shape[:2])
        or not isinstance(activation_2d, np.ndarray)
        or activation_2d.ndim != 2
        or activation_2d.shape != test_bgr.shape[:2]
        or not np.isfinite(activation_2d).all()
    ):
        raise ValueError("Imagens e mapa de ativação incompatíveis")

    h, w = test_bgr.shape[:2]
    target_w, target_h = _display_size(h, w)
    dims = (target_w, target_h)
    method = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
    test = cv2.resize(test_bgr, dims, interpolation=method)
    ref = cv2.resize(reference_bgr, dims, interpolation=method)
    strength = cv2.resize(
        np.clip(activation_2d.astype(np.float32), 0.0, 1.0),
        dims, interpolation=cv2.INTER_LINEAR,
    )
    # The Grad-CAM can be exactly zero for its local class. In that
    # situation show the original, never a fabricated colorful heatmap.
    # Do not rescale strength after the activation normalization upstream.
    threshold = .15 if kind == "latent" else .22 if kind == "gradcam" else .35
    alpha_max = .60 if kind == "latent" else .56 if kind == "gradcam" else .38
    visual_response = np.maximum(strength-threshold, 0.0)/(1.0-threshold)
    alpha = (np.power(visual_response, 1.2)*alpha_max)[..., None]

    if kind == "latent":
        palette = cv2.COLORMAP_JET
    elif kind == "gradcam":
        palette = cv2.COLORMAP_HOT
    else:
        palette = cv2.COLORMAP_BONE
    colors = cv2.applyColorMap(
        np.uint8(np.round(strength*255)), palette,
    )
    blended = np.clip(
        test.astype(np.float32)*(1-alpha)
        + colors.astype(np.float32)*alpha, 0, 255,
    ).astype(np.uint8)

    # Small reference thumbnail in top-left. The thumbnail does NOT
    # participate in map statistics, inference or operator judgement.
    thumb_w = max(18, min(target_w//3, int(round(target_w*.29))))
    thumb_h = max(16, min(target_h//3, int(round(target_h*.32))))
    thumb_w = min(thumb_w, max(1, target_w-8))
    thumb_h = min(thumb_h, max(1, target_h-8))
    x, y = (7 if target_w >= 100 else 1), (7 if target_h >= 100 else 1)
    thumb_w = min(thumb_w, target_w-x)
    thumb_h = min(thumb_h, target_h-y)
    if thumb_w < 3 or thumb_h < 3:
        return np.ascontiguousarray(blended)
    inset = cv2.resize(ref, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    blended[y:y+thumb_h, x:x+thumb_w] = inset
    cv2.rectangle(
        blended, (x, y), (x+thumb_w-1, y+thumb_h-1),
        (0, 215, 255), 2,
    )
    # Tiny caption INSIDE the inset, not superimposed over the component.
    if thumb_w >= 45 and thumb_h >= 33:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = .38 if thumb_w >= 72 else .30
        width, height = cv2.getTextSize("GAB", font, scale, 1)[0]
        cv2.rectangle(
            blended, (x+1, y+1),
            (min(x+thumb_w-2, x+width+7), y+height+8),
            (12, 12, 12), -1,
        )
        cv2.putText(
            blended, "GAB", (x+4, y+height+4),
            font, scale, (0, 215, 255), 1, cv2.LINE_AA,
        )
    return np.ascontiguousarray(blended)
