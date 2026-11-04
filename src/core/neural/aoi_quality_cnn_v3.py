"""CNN V3 comparativa AOI — arquitetura NOVA, somente experimento offline.

Aprende a diferença espacial entre gabarito e teste, e não um classificador
de imagens isoladas. O mesmo encoder processa os dois lados; a representação
é simétrica para reduzir atalhos de iluminação e da posição do painel AOI.
Não é compatível com o checkpoint CNN V2 nem com seu carregador produtivo.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.core.neural.faltando_cnn_v2 import LIGHTS

MODEL_SCHEMA_V3 = "visionx.aoi_pair_quality_cnn.v3.offline"


def _block(inp: int, out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out),
        nn.SiLU(),
    )


class AOIQualityCNNV3(nn.Module):
    """CNN de pares, duas escalas e luzes independentes com pesos compartilhados.

    Um mapa de diferenças em features (não apenas a diferença RGB bruta)
    é combinado com correlação espacial. Essa operação não exige que duas
    imagens visualmente equivalentes sejam pixel-idênticas.
    """

    def __init__(self, dropout: float = .20) -> None:
        super().__init__()
        if not 0. <= dropout < 1.:
            raise ValueError("Dropout V3 inválido")
        self.encoder = nn.Sequential(
            _block(3, 24, 2),
            _block(24, 32, 2),
            _block(32, 48, 2),
            _block(48, 64, 2),
            _block(64, 96, 2),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        features = 96 * 3 * 3 * 2 * 2  # (|ref-test|, ref*test) × 2 escalas
        self.head = nn.Sequential(
            nn.Linear(features, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _pair_features(self, reference: Tensor, test: Tensor) -> Tensor:
        count = reference.shape[0] * len(LIGHTS)
        h, w = reference.shape[-2:]
        # Encoder compartilhado garante que a comparação tenha a mesma base.
        stacked = torch.cat((
            reference.reshape(count, 3, h, w),
            test.reshape(count, 3, h, w),
        ), dim=0)
        ref_feature, test_feature = self.encoder(stacked).chunk(2, dim=0)
        return torch.cat((
            torch.abs(ref_feature - test_feature),
            ref_feature * test_feature,
        ), dim=1).flatten(1)

    def forward(
        self, full_ref: Tensor, full_test: Tensor,
        focus_ref: Tensor, focus_test: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = (full_ref, full_test, focus_ref, focus_test)
        if (
            any(not isinstance(x, Tensor) or x.ndim != 5 for x in inputs)
            or any(x.shape != full_ref.shape for x in inputs)
            or full_ref.shape[1:3] != (3, 3)
            or min(full_ref.shape[-2:]) < 64
            or not isinstance(mask, Tensor)
            or mask.shape != (full_ref.shape[0], 3)
            or not bool(torch.all((mask == 0) | (mask == 1)).item())
            or not bool(torch.all(mask.sum(1) >= 1).item())
        ):
            raise ValueError("Par, luz, máscara ou escalas incompatíveis na CNN V3")
        full_features = self._pair_features(full_ref, full_test)
        focus_features = self._pair_features(focus_ref, focus_test)
        light_logits = self.head(
            torch.cat((full_features, focus_features), dim=1)
        ).reshape(-1, len(LIGHTS))
        event_logits = light_logits.masked_fill(
            mask <= 0, torch.finfo(light_logits.dtype).min
        ).max(dim=1).values
        return event_logits, light_logits


__all__ = ["AOIQualityCNNV3", "MODEL_SCHEMA_V3", "LIGHTS"]
