"""CNN especializada DESLOCADO — arquitetura comparativa v2, pesos próprios.

Nunca reutiliza pesos da CNN FALTANDO. Sem NG DESLOCADO real, só pode
aprender contraste com deslocamentos SINTÉTICOS como protótipo offline.
"""
from __future__ import annotations

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2

MODEL_SCHEMA_DESLOCADO = "visionx.deslocado_comparative_cnn.v1"


class DeslocadoCNN(FaltandoCNNV2):
    """Entrada dupla escala RGB(gabarito,teste,abs(diff)) × SIDE/TOP/MID."""
