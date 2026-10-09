"""Adaptador de validação da MEMÓRIA KNN real, sem motor físico ou CNN.

A memória de produção só reconhece par exato humano validado via
VerifiedKNNMemory. A falta desse registro produz SEM COBERTURA, não OK.

Não lê rótulos de ok_archive/ng_archive para responder. Estes são usados
apenas externamente pelo validador para comparar o resultado recuperado.
"""
from __future__ import annotations

from pathlib import Path

from src.config.settings import BASE_DIR
from src.core.verified_memory_router import VerifiedKNNMemory
from src.core.experts.knn_expert import KNNExpert
from src.core.strict_category_memory import (
    canonical_memory_category,
    canonical_memory_lighting,
)

MODEL_KIND = "knn_verified_exact"


class KNNArchivePredictor:
    """Consulta a mesma fonte de memórias verificadas da operação normal."""

    def __init__(self, *, root: Path | None = None, knn=None, index=None):
        requested = Path(root or BASE_DIR).expanduser().resolve()
        if knn is None and requested != BASE_DIR.resolve():
            raise ValueError(
                "KNN usa o dataset configurado na aplicação. "
                "Raiz distinta exige KNN injetado; não usar memória de outra estação."
            )
        self.knn = knn if knn is not None else KNNExpert()
        self.index = index if index is not None else VerifiedKNNMemory()

    def inspect(self, reference, test, mode: str, aoi_info: dict) -> dict:
        info = dict(aoi_info or {})
        category = canonical_memory_category(info.get("category", ""))
        if not category or category in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
            raise ValueError("Categoria OCR ausente: KNN não pode buscar globalmente")
        if not str(info.get("board", "")).strip() or not str(
            info.get("parts", "")
        ).strip() or not str(info.get("value", "")).strip():
            raise ValueError(
                "Board/Parts/Value ausentes: não buscar KNN sem escopo verificado"
            )
        info["category"] = category
        info["lighting_mode"] = canonical_memory_lighting(mode)
        match = self.index.lookup(self.knn, reference, test, info)
        if not isinstance(match, dict):
            raise ValueError("Verificador KNN retornou resposta inválida")
        status = match.get("status")
        if status == "KNOWN":
            label = str(match.get("label", "")).upper()
            if label not in {"OK", "NG"}:
                raise ValueError("Rótulo da memória KNN inválido")
            return {
                "verdict": "DEFEITO REAL" if label == "NG" else "FALHA FALSA",
                "detail": {
                    "model_kind": MODEL_KIND,
                    "verified_exact_match": True,
                    "memory_status": status,
                    "memory_label": label,
                    "memory_source_json": match.get("source_json"),
                    "memory_matches": match.get("matches"),
                    "reason": match.get("reason", ""),
                },
            }
        if status in {"NEW", "UNAVAILABLE", "CONFLICT"}:
            return {
                "verdict": "REVISÃO OBRIGATÓRIA",
                "detail": {
                    "model_kind": MODEL_KIND,
                    "verified_exact_match": False,
                    "memory_status": status,
                    "reason": match.get("reason", ""),
                },
            }
        raise ValueError(f"Status de memória KNN inválido: {status!r}")


__all__ = ["KNNArchivePredictor", "MODEL_KIND"]
