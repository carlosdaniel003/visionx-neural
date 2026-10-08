"""CNN comparativa FALTANDO (gabarito/teste), sem dependência de KNN.

O modelo opera com uma, duas ou três iluminações. Treinamento e checkpoint
são experimentais; este módulo NÃO conecta o resultado à Produção.
"""
from __future__ import annotations

import torch
from torch import nn

LIGHTS = ("SIDE", "TOP", "MID")
MODEL_SCHEMA = "visionx.faltando_comparative_cnn.v1"


def _layer(a: int, b: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(a, b, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, b), b),
        nn.SiLU(inplace=True),
    )


class FaltandoCNN(nn.Module):
    """Extrator compartilhado, comparação de pares e fusão multilight.

    x_ref/x_test: [B, 3, 3, H, W] em RGB [0..1].
    mask: [B, 3], 1 para iluminação disponível.
    Uma ausência observada numa luz não pode ser apagada pela média das outras:
    a fusão conservadora usa o MAIOR logit de NG nas luzes disponíveis.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _layer(3, 16, 2),
            _layer(16, 24, 2),
            _layer(24, 40, 2),
            _layer(40, 64, 2),
            _layer(64, 80, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(80 * 4, 96), nn.SiLU(),
            nn.Dropout(p=0.15),
            nn.Linear(96, 1),
        )

    def forward(self, x_ref: torch.Tensor, x_test: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_ref.ndim != 5 or x_ref.shape[1:3] != (3, 3):
            raise ValueError("Referência esperada [B, 3, 3, H, W]")
        if x_ref.shape != x_test.shape or mask.shape != (x_ref.shape[0], 3):
            raise ValueError("Teste/máscara multilight incompatíveis")
        if not torch.all(mask.sum(dim=1) > 0):
            raise ValueError("Cada evento requer ao menos uma iluminação")
        count = x_ref.shape[0] * 3
        a = self.encoder(x_ref.reshape(count, *x_ref.shape[2:]))
        b = self.encoder(x_test.reshape(count, *x_test.shape[2:]))
        evidence = torch.cat((a, b, torch.abs(a - b), a * b), dim=1)
        per_light = self.head(evidence).reshape(-1, 3)
        event_logits = per_light.masked_fill(mask <= 0, -1e6).max(dim=1).values
        return event_logits, per_light
