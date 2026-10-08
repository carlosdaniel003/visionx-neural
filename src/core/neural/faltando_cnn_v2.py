"""CNN FALTANDO v2 — par AOI comparativo em duas escalas e três luzes.

Modelo experimental isolado: nenhum impacto nos motores ativos do ODIN.
Entradas RGB: [B, 3 luzes, 3 canais, H, W]; máscara [B, 3].
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LIGHTS = ("SIDE", "TOP", "MID")
MODEL_SCHEMA_V2 = "visionx.faltando_comparative_cnn.v2"


def _conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                  stride=stride, bias=False),
        nn.GroupNorm(8, out_ch),
        nn.SiLU(),
    )


class FaltandoCNNV2(nn.Module):
    """Concatena gabarito, teste e diferença |ref-teste| antes da CNN.

    Duas passagens com encoder COMPARTILHADO:
      1) quadro físico integral, redimensionado sem esticar;
      2) região central mais detalhada (zoom 70% da imagem original).
    Os mapas preservam grade espacial 2x2 antes do classificador.
    Um evento pode ter somente SIDE ou SIDE/TOP/MID.

    O maior logit NG entre luzes disponíveis representa o evento; na
    produção real será necessário calibrar limiares e medir incerteza.
    """

    def __init__(self, dropout: float = .30):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(9, 24, 2),
            _conv(24, 32, 2),
            _conv(32, 48, 2),
            _conv(48, 64, 2),
            _conv(64, 96, 2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(96 * 2 * 2 * 2, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _combine(ref: Tensor, test: Tensor) -> Tensor:
        if ref.ndim != 5 or ref.shape[1:3] != (3, 3):
            raise ValueError("Gabarito deve ter forma [B, 3, 3, H, W]")
        if ref.shape != test.shape:
            raise ValueError("Gabarito e teste têm formas diferentes")
        if ref.shape[-1] < 64 or ref.shape[-2] < 64:
            raise ValueError("Tamanho mínimo de imagem: 64x64")
        return torch.cat((ref, test, torch.abs(ref - test)), dim=2)

    def forward(
        self, full_ref: Tensor, full_test: Tensor,
        focus_ref: Tensor, focus_test: Tensor, mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        full = self._combine(full_ref, full_test)
        focus = self._combine(focus_ref, focus_test)
        if full.shape != focus.shape or mask.shape != (full.shape[0], 3):
            raise ValueError("Duas escalas e máscara SIDE/TOP/MID incompatíveis")
        if not bool(torch.all(mask.sum(1) >= 1).item()):
            raise ValueError("Evento sem iluminação observada")

        n = full.shape[0] * 3
        feature_full = self.encoder(full.reshape(n, 9, *full.shape[-2:]))
        feature_focus = self.encoder(focus.reshape(n, 9, *focus.shape[-2:]))
        per_light = self.head(
            torch.cat((feature_full, feature_focus), dim=1)
        ).reshape(-1, 3)
        logits = per_light.masked_fill(mask <= 0, -1e6).max(dim=1).values
        return logits, per_light
