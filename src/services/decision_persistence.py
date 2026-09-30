"""Persistência serial em segundo plano para decisões humanas do VisionX.

A decisão produtiva não deve esperar gravação em disco nem recarga da memória
KNN. Esta fila executa essas tarefas fora da thread da interface e preserva a
ordem das decisões.
"""

from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Any

from src.services.dataset_manager import DatasetManager


class DecisionPersistenceQueue:
    """Fila serial daemon para salvar aprendizado e atualizar a memória."""

    def __init__(self, orchestrator, dataset_manager=DatasetManager):
        self.orchestrator = orchestrator
        self.dataset_manager = dataset_manager
        self._queue: Queue[dict[str, Any] | None] = Queue()
        self._worker = Thread(
            target=self._run,
            name="VisionXDecisionPersistence",
            daemon=True,
        )
        self._worker.start()

    def submit(self, task: dict[str, Any]) -> None:
        self._queue.put(dict(task))

    def pending_count(self) -> int:
        return int(self._queue.unfinished_tasks)

    def wait_until_idle(self) -> None:
        """Apoio determinístico para testes/manutenção; não usar no ciclo produtivo."""
        self._queue.join()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return

                json_path = self.dataset_manager.save_sample(**task)
                if json_path and self.orchestrator is not None:
                    reload_memory = getattr(
                        self.orchestrator,
                        "reload_memory",
                        None,
                    )
                    if callable(reload_memory):
                        reload_memory()
            except Exception as exc:
                print(
                    "Falha não fatal na persistência assíncrona da decisão: "
                    f"{exc}"
                )
            finally:
                self._queue.task_done()


__all__ = ["DecisionPersistenceQueue"]
