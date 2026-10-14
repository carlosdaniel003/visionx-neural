"""Contrato leve do ponteiro do checkpoint FALTANDO v2.

Importar este módulo não inicializa PyTorch, datasets ou o treinamento
online. Usado em leitura operacional e na publicação dos pesos, para
evitar deriva entre as duas partes do contrato.
"""
POINTER_SCHEMA = "visionx.specialist_active_pointer.v1"

__all__ = ["POINTER_SCHEMA"]
