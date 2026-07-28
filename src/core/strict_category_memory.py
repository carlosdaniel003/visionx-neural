"""Memória de anomalias estritamente isolada pela categoria da AOI.

Não há fallback por componente, busca global ou imagem legada. Uma inspeção de
INVERTIDO consulta somente JSONs INVERTIDO; FALTANDO e MUITO ADESIVO seguem a
mesma regra.
"""

from __future__ import annotations

import re

from src.core.anomaly_signature import valid_anomaly_signature


CATEGORY_ALIASES = {
    "INVERTIDO": "INVERTIDO",
    "INVERTED": "INVERTIDO",
    "REVERSE": "INVERTIDO",
    "UPSIDEDOWN": "INVERTIDO",
    "FALTANDO": "FALTANDO",
    "MISSING": "FALTANDO",
    "MUSING": "FALTANDO",
    "MISSMG": "FALTANDO",
    "MUITOADESIVO": "MUITOADESIVO",
    "MUCHADHESIVE": "MUITOADESIVO",
    "EXCESSADHESIVE": "MUITOADESIVO",
    "ADESIVOEMEXCESSO": "MUITOADESIVO",
}


def canonical_memory_category(value: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return CATEGORY_ALIASES.get(compact, compact)


def install_strict_category_memory(knn_expert_cls) -> None:
    if getattr(knn_expert_cls, "_strict_category_memory_installed", False):
        return

    def analyze(
        self,
        full_gab,
        full_test,
        crop_gab=None,
        crop_test=None,
        aoi_info=None,
        top_k=5,
        anomaly_signature=None,
    ):
        info = aoi_info if isinstance(aoi_info, dict) else {}
        target_category = canonical_memory_category(info.get("category", ""))

        if not target_category or not valid_anomaly_signature(anomaly_signature):
            result = self._empty_result(
                query_anomaly_signature=(
                    anomaly_signature
                    if valid_anomaly_signature(anomaly_signature)
                    else None
                )
            )
            result.update(
                {
                    "memory_scope": "categoria",
                    "memory_category": target_category,
                    "memory_candidate_count": 0,
                    "memory_filter_strict": True,
                    "memory_mode": "anomaly",
                    "memory_reason": (
                        "Categoria ausente"
                        if not target_category
                        else "Assinatura de anomalia inválida"
                    ),
                }
            )
            return result

        all_ok = [
            record
            for record in self.signatures_ok
            if record.get("mode") == "anomaly"
            and canonical_memory_category(record.get("category", ""))
            == target_category
        ]
        all_ng = [
            record
            for record in self.signatures_ng
            if record.get("mode") == "anomaly"
            and canonical_memory_category(record.get("category", ""))
            == target_category
        ]
        candidate_count = len(all_ok) + len(all_ng)

        if candidate_count == 0:
            result = self._empty_result(
                query_anomaly_signature=anomaly_signature
            )
            result.update(
                {
                    "memory_scope": "categoria",
                    "memory_category": target_category,
                    "memory_candidate_count": 0,
                    "memory_filter_strict": True,
                    "memory_mode": "anomaly",
                    "memory_reason": (
                        f"Nenhum JSON de anomalia da categoria {target_category}"
                    ),
                }
            )
            return result

        result = self._analyze_anomaly_memory(
            anomaly_signature,
            all_ok,
            all_ng,
            top_k,
            "categoria",
        )
        result.update(
            {
                "memory_category": target_category,
                "memory_candidate_count": candidate_count,
                "memory_filter_strict": True,
                "memory_reason": (
                    f"Consulta restrita a {candidate_count} JSON(s) de "
                    f"{target_category}"
                ),
            }
        )
        return result

    knn_expert_cls.analyze = analyze
    knn_expert_cls._strict_category_memory_installed = True


__all__ = [
    "canonical_memory_category",
    "install_strict_category_memory",
]
