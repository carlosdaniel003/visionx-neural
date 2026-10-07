"""Memória de anomalias estritamente isolada pela categoria da AOI.

Não há fallback por componente, busca global ou imagem legada. Cada inspeção
consulta somente JSONs da mesma categoria canônica.
"""

from __future__ import annotations

import re

from src.core.anomaly_signature import valid_anomaly_signature


CATEGORY_ALIASES = {
    "DESLOCADO": "DESLOCADO",
    "DDESLOCADO": "DESLOCADO",
    "DESLOCAMENTO": "DESLOCADO",
    "SHIFTED": "DESLOCADO",
    "MISALIGNED": "DESLOCADO",
    "OFFSET": "DESLOCADO",
    "EMBORCADO": "EMBORCADO",
    "EMBORCADA": "EMBORCADO",
    "TOMBSTONE": "EMBORCADO",
    "TOMBSTONED": "EMBORCADO",
    "STANDING": "EMBORCADO",
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


def canonical_memory_lighting(value: str) -> str:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in {"SIDE", "TOP", "MID"} else "SIDE"


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
        target_lighting = canonical_memory_lighting(
            info.get("lighting_mode", "")
        )

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
                    "memory_lighting": target_lighting,
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

        memory_lock = getattr(self, "_memory_lock", None)
        if memory_lock is not None:
            with memory_lock:
                signatures_ok = list(self.signatures_ok)
                signatures_ng = list(self.signatures_ng)
        else:
            signatures_ok = list(self.signatures_ok)
            signatures_ng = list(self.signatures_ng)

        all_ok = [
            record
            for record in signatures_ok
            if record.get("mode") == "anomaly"
            and canonical_memory_category(record.get("category", ""))
            == target_category
            and canonical_memory_lighting(
                record.get("lighting_mode", "")
            )
            == target_lighting
        ]
        all_ng = [
            record
            for record in signatures_ng
            if record.get("mode") == "anomaly"
            and canonical_memory_category(record.get("category", ""))
            == target_category
            and canonical_memory_lighting(
                record.get("lighting_mode", "")
            )
            == target_lighting
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
                        f"Nenhum JSON de anomalia da categoria {target_category} "
                        f"na iluminação {target_lighting}"
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
                "memory_lighting": target_lighting,
                "memory_candidate_count": candidate_count,
                "memory_filter_strict": True,
                "memory_reason": (
                    f"Consulta restrita a {candidate_count} JSON(s) de "
                    f"{target_category} em {target_lighting}"
                ),
            }
        )
        return result

    knn_expert_cls.analyze = analyze
    knn_expert_cls._strict_category_memory_installed = True


__all__ = [
    "canonical_memory_category",
    "canonical_memory_lighting",
    "install_strict_category_memory",
]
