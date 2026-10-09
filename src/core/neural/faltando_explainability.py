"""Visualização EXCLUSIVAMENTE de ativações reais da CNN FALTANDO v2.

As entradas de cada ROI são sondagens auxiliares (não a inferência de produção).
A CNN v2 concatena [gabarito, teste, abs(gabarito-teste)] em NOVE canais,
com encoder compartilhado entre escala completa e central. Portanto não há
features independentes de referência/teste e não existe decoder treinado.

Três mapas:
  1. diferença das ativações do encoder entre (ref,teste) e (ref,ref);
  2. Grad-CAM do logit local, classe prevista por essa própria sonda;
  3. projeção RMS das ativações espaciais do encoder (NÃO reconstrução RGB).

Somente pesos de checkpoint com SHA validado. Nunca muda score, KNN ou 0/1.
"""
from __future__ import annotations

from threading import Lock
import math

import cv2
import numpy as np

from src.core.neural.faltando_live import FaltandoCNNLive
from src.scripts.train_faltando_cnn_v2 import _letterbox_rgb, _focus_crop

LAYER_NAME = "encoder.4"  # último bloco convolucional, antes de AdaptiveAvgPool
VIEWS = ("DIF. LATENTE", "GRAD-CAM", "ATIVAÇÃO CNN")
REGIONS = (("major", "large_reference", "large"), ("minor", "small_reference", "small"))
LOCK = Lock()


def _valid(frame) -> bool:
    return (
        isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        and frame.ndim == 3 and frame.shape[-1] == 3
        and min(frame.shape[:2]) >= 12
    )


def _normalize(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    arr = np.maximum(arr, 0)
    hi = float(np.max(arr))
    if not math.isfinite(hi) or hi < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    # A escala é sempre relativa à maior resposta dentro DESTE mapa.
    return np.clip(arr / hi, 0, 1).astype(np.float32)


def _to_bgr(raw: np.ndarray, mode: str = "jet") -> np.ndarray:
    scaled = np.uint8(np.clip(raw, 0, 1)*255)
    if mode == "gray":
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def _unletterbox(spatial: np.ndarray, source_shape, target_size: int) -> np.ndarray:
    """Reverte o padding do _letterbox_rgb antes de exibir na ROI AOI."""
    h, w = source_shape[:2]
    factor = min(target_size/h, target_size/w)
    tw, th = max(1, round(w*factor)), max(1, round(h*factor))
    ox, oy = (target_size-tw)//2, (target_size-th)//2
    up = cv2.resize(spatial, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    core = up[oy:oy+th, ox:ox+tw]
    return cv2.resize(core, (w, h), interpolation=cv2.INTER_LINEAR)


def generate_explainability_triplet(
    model, image_reference_bgr, image_test_bgr, *, image_size: int,
    focus_fraction: float,
) -> dict:
    """Gradientes apenas para visualização; a rede é chamada em eval() CPU.

    Usar somente com modelo previamente autenticado/checkpoint validado.
    Este forward é uma SONDAGEM AOI, não a decisão real da peça completa.
    """
    import torch

    if not (_valid(image_reference_bgr) and _valid(image_test_bgr)):
        raise ValueError("Par ROI gabarito/teste ausente ou inválido")
    if not isinstance(image_size, int) or not 64 <= image_size <= 512:
        raise ValueError("Tamanho de entrada CNN fora do contrato")
    if not .5 <= float(focus_fraction) <= .90:
        raise ValueError("Fração central CNN fora do contrato")
    if model.training:
        raise ValueError("Modelo deve estar em eval()")
    if not hasattr(model, "encoder") or len(model.encoder) < 5:
        raise ValueError("Checkpoint não tem encoder convolucional v2")

    reference = image_reference_bgr
    test = image_test_bgr
    full_ref = _letterbox_rgb(reference, image_size)
    full_test = _letterbox_rgb(test, image_size)
    focus_ref = _letterbox_rgb(_focus_crop(reference, focus_fraction), image_size)
    focus_test = _letterbox_rgb(_focus_crop(test, focus_fraction), image_size)
    # A rede de produção usa B=1, L=3 e mascara 2 luzes em cada inferência.
    inputs = [torch.zeros((1, 3, 3, image_size, image_size))
              for _ in range(4)]
    for pos, tensor in enumerate((full_ref, full_test, focus_ref, focus_test)):
        inputs[pos][0, 0] = tensor
    mask = torch.tensor([[1., 0., 0.]], dtype=torch.float32)

    captures = []
    handle = model.encoder[4].register_forward_hook(
        lambda _module, _input, output: captures.append(output)
    )
    try:
        # Não usar inference_mode(): é necessário gradiente para Grad-CAM.
        with torch.enable_grad():
            logit, _per_light = model(*inputs, mask)
            if len(captures) != 2:
                raise ValueError("Encoder não expôs as duas escalas")
            full_map, focus_map = captures
            target_name = "NG" if float(logit.item()) >= 0 else "OK"
            target = logit if target_name == "NG" else -logit
            gradients = torch.autograd.grad(
                target.sum(), full_map, allow_unused=False,
                retain_graph=False, create_graph=False,
            )[0]
    finally:
        handle.remove()

    # O checkpoint não tem "ref_encoder"/"test_encoder" independentes.
    # A diferença latente compara a ativação do par com o mesmo gabarito
    # nos dois canais, usando OS MESMOS pesos do encoder treinado.
    with torch.no_grad():
        baseline = model.encoder[:5](
            model._combine(
                inputs[0], inputs[0],
            ).reshape(3, 9, image_size, image_size)
        )[0]

    feat = full_map.detach()[0]
    latent = (feat - baseline).abs().mean(dim=0).cpu().numpy()
    weight = gradients.detach()[0].mean(dim=(1, 2))
    cam = torch.relu(
        (weight[:, None, None]*feat).sum(dim=0)
    ).cpu().numpy()
    # RMS não é inversão/decoder: é projeção da energia de features.
    activation = torch.sqrt(torch.mean(feat.square(), dim=0)).cpu().numpy()
    norm_maps = tuple(
        _normalize(_unletterbox(array, test.shape, image_size))
        for array in (latent, cam, activation)
    )
    images = (
        _to_bgr(norm_maps[0]),
        _to_bgr(norm_maps[1]),
        _to_bgr(norm_maps[2], "gray"),
    )
    return {
        "images": images,
        "dimensions": (int(test.shape[1]), int(test.shape[0])),
        "layer": LAYER_NAME,
        "target_class": target_name,
        "probe_ng_score_uncalibrated": float(torch.sigmoid(logit.detach())[0]),
        "map_peaks": [float(x.max()) for x in norm_maps],
        "raw_feature_means": [
            round(float(np.asarray(array).mean()), 6)
            for array in (latent, cam, activation)
        ],
        "schema": "visionx.cnn_v2_neural_evidence.v1",
        "neural": True,
        "reconstruction_decoder": False,
        "probe_only": True,
    }


class FaltandoExplainability:
    """Reutiliza os pesos online ativos verificados, sem criar outro modelo."""

    def __init__(self, predictor=None):
        self.predictor = predictor if predictor is not None else FaltandoCNNLive()

    def explain_epicenters(self, payload: dict) -> dict:
        out: dict = {}
        with LOCK:  # hooks no modelo compartilhado nunca rodam concorrentes
            model = self.predictor._load()  # exige SHA e metadados verificados
            metadata = self.predictor._metadata
            for key, reference_key, test_key in REGIONS:
                ref = payload.get(reference_key)
                test = payload.get(test_key)
                if not _valid(ref) or not _valid(test):
                    out[key] = None
                    continue
                out[key] = generate_explainability_triplet(
                    model, ref, test,
                    image_size=metadata["image_size"],
                    focus_fraction=metadata["focus_fraction"],
                )
                out[key]["checkpoint_sha256"] = self.predictor.expected_sha256
        return out


_shared_predictor = FaltandoExplainability()


def explain_epicenters(payload: dict) -> dict:
    """Ponto de entrada para os workers Qt, com modelo carregado sob demanda."""
    return _shared_predictor.explain_epicenters(payload)
