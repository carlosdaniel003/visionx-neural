"""Preparação antecipada, não bloqueante, dos índices e pesos de inspeção.

Os workers não interagem com objetos Qt nem enviam comandos ao XP.
Toda decisão continua feita exclusivamente pelo orquestrador original.
"""
from __future__ import annotations

import logging
from threading import Thread


LOG = logging.getLogger(__name__)


def _run_warmup(name: str, job) -> None:
    try:
        job()
    except Exception:
        # Warmup é oportunista. A rota síncrona da inspeção preserva
        # validação de integridade e fallback de revisão obrigatória.
        LOG.exception("Pré-aquecimento %s não concluído", name)


def start_inspection_warmup(orchestrator) -> tuple[Thread, ...]:
    """Inicia um worker KNN e um CNN após a UI estar criada."""
    # Reservar a instância do índice no thread da interface evita duas
    # instâncias na primeira corrida entre a primeira SIDE e o warmup.
    get_index = getattr(orchestrator, "_get_verified_memory_index", None)
    if callable(get_index):
        try:
            get_index()
        except Exception:
            LOG.exception("Não foi possível preparar referência do índice KNN")

    jobs = (
        ("KNN", getattr(orchestrator, "prepare_verified_memory_index", None)),
        ("CNN", getattr(orchestrator, "prepare_faltando_cnn", None)),
    )
    workers = []
    for name, job in jobs:
        if not callable(job):
            continue
        worker = Thread(
            target=_run_warmup,
            args=(name, job),
            name="ODINWarmup" + name,
            daemon=True,
        )
        worker.start()
        workers.append(worker)
    return tuple(workers)


__all__ = ["start_inspection_warmup"]
