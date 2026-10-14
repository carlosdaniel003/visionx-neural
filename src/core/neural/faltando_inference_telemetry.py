"""Cronometria monotônica e passiva da inferência CNN FALTANDO v2.

Não altera tensores, resultados, pesos, limiares ou política de produção.
A medição pertence à chamada atual e nunca integra a chave de cache.
"""
from __future__ import annotations

import time


PERFORMANCE_SCHEMA = "visionx.faltando_cnn_v2_performance.v1"


def elapsed_ms(started_at: float) -> float:
    """Tempo decorrido em milissegundos, com resolução do perf_counter."""
    return round(max(0.0, (time.perf_counter() - started_at) * 1000.0), 2)


class InferencePhaseProfile:
    """Coleciona durações de uma única chamada, sem estado compartilhado."""

    def __init__(self, light: str):
        self._started = time.perf_counter()
        self.light = str(light)
        self.stages: dict[str, float] = {}
        self.checkpoint: dict[str, float] = {}
        self.cache_hit = False
        self.checkpoint_loaded = False
        self.torch_threads: int | None = None
        self.torch_interop_threads: int | None = None

    def record(self, name: str, started_at: float) -> None:
        self.stages[str(name)] = elapsed_ms(started_at)

    def record_checkpoint(self, timings: dict | None) -> None:
        if isinstance(timings, dict):
            self.checkpoint = {
                str(key): round(float(value), 2)
                for key, value in timings.items()
                if isinstance(value, (int, float))
            }

    def snapshot(self, *, status: str = "INFERENCE_OK") -> dict:
        return {
            "schema": PERFORMANCE_SCHEMA,
            "lighting_mode": self.light,
            "status": str(status),
            "cache_hit": bool(self.cache_hit),
            "checkpoint_loaded": bool(self.checkpoint_loaded),
            "total_ms": elapsed_ms(self._started),
            "stages_ms": dict(self.stages),
            "checkpoint_load_ms": dict(self.checkpoint),
            "torch_num_threads": self.torch_threads,
            "torch_num_interop_threads": self.torch_interop_threads,
            "note": (
                "Os tempos sao do PC ODIN. checkpoint_load_ms detalha "
                "a etapa model_acquire_ms; nao somar as duas. "
                "cache_hit=True significa reutilizacao exata do resultado, "
                "nao execucao da CNN."
            ),
        }


def format_inference_profile(data: dict | None) -> list[str]:
    """Bloco textual para o Debug Técnico; não depende de Qt."""
    if not isinstance(data, dict) or not data:
        return []
    lines = [
        "PERFIL INTERNO CNN FALTANDO V2 (ms)",
        "Cache de inferencia da sessao: {}".format(data.get("cache_hit", "-")),
        "Checkpoint carregado nesta chamada: {}".format(
            data.get("checkpoint_loaded", "-")
        ),
        "Estado: {}".format(data.get("status", "-")),
        "Total desta chamada: {} ms".format(data.get("total_ms", "-")),
        "Threads PyTorch CPU: {} | interop: {}".format(
            data.get("torch_num_threads", "-"),
            data.get("torch_num_interop_threads", "-"),
        ),
    ]
    names = {
        "live_pointer_ms": "Consulta de checkpoint ativo",
        "checkpoint_stat_ms": "Verificacao do arquivo checkpoint",
        "fingerprints_ms": "Hashes das imagens para o cache",
        "cache_lookup_ms": "Consulta do cache",
        "model_acquire_ms": "Obtencao/carregamento do modelo",
        "input_prepare_ms": "Preparacao e normalizacao dos tensores",
        "torch_forward_ms": "Execucao PyTorch (forward + score)",
        "cache_store_ms": "Gravacao do resultado no cache",
        "checkpoint_hash_ms": "Leitura e SHA-256 do checkpoint",
        "checkpoint_deserialize_ms": "Desserializacao do checkpoint PyTorch",
        "model_construct_ms": "Construcao/carregamento dos pesos no modelo",
        "checkpoint_total_ms": "Carregamento total do checkpoint",
    }
    for key in (
        "live_pointer_ms", "checkpoint_stat_ms", "fingerprints_ms",
        "cache_lookup_ms", "model_acquire_ms", "input_prepare_ms",
        "torch_forward_ms", "cache_store_ms",
    ):
        if key in data.get("stages_ms", {}):
            lines.append("{}: {} ms".format(names[key], data["stages_ms"][key]))
    checkpoint = data.get("checkpoint_load_ms", {})
    if isinstance(checkpoint, dict) and checkpoint:
        lines.append("Detalhe do carregamento do checkpoint:")
        for key in (
            "checkpoint_hash_ms", "checkpoint_deserialize_ms",
            "model_construct_ms", "checkpoint_total_ms",
        ):
            if key in checkpoint:
                lines.append("  {}: {} ms".format(names[key], checkpoint[key]))
    lines.append(
        "Observacao: obter modelo inclui o carregamento; nao somar essas etapas."
    )
    return lines


__all__ = [
    "PERFORMANCE_SCHEMA", "InferencePhaseProfile", "elapsed_ms",
    "format_inference_profile",
]
