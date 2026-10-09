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
from src.services.neural_online_learning import (
    OnlineLearningQueue, eligible_online_case,
)


LIGHTING_ORDER = ("SIDE", "TOP", "MID")


def _complete_multilight_samples(task: dict) -> list[dict]:
    samples = task.get("multilight_samples", [])
    if not isinstance(samples, list) or len(samples) != len(LIGHTING_ORDER):
        return []

    by_mode = {
        str(item.get("lighting_mode", "")).strip().upper(): item
        for item in samples
        if isinstance(item, dict)
    }
    if any(mode not in by_mode for mode in LIGHTING_ORDER):
        return []
    return [by_mode[mode] for mode in LIGHTING_ORDER]


class DecisionPersistenceQueue:
    """Fila serial daemon para salvar aprendizado e atualizar a memória."""

    def __init__(self, orchestrator, dataset_manager=DatasetManager):
        self.orchestrator = orchestrator
        self.dataset_manager = dataset_manager
        self._online_learning = None
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

    def _reload_memory_once(self) -> None:
        if self.orchestrator is None:
            return
        reload_memory = getattr(
            self.orchestrator,
            "reload_memory",
            None,
        )
        if callable(reload_memory):
            reload_memory()

    def _persist_multilight(self, task: dict, samples: list[dict]) -> bool:
        """Persiste três observações da mesma peça sob um único rótulo humano."""
        base_info = dict(task.get("aoi_info", {}) or {})
        final_analysis = task.get("analysis", {}) or {}
        label = str(task.get("label", "") or "")
        source = str(task.get("source", "") or "")
        event_id = str(task.get("event_id", "") or "")

        persisted = False
        all_saved = True
        for item in samples:
            mode = str(item.get("lighting_mode", "") or "").strip().upper()
            local_analysis = item.get("analysis", {}) or {}
            local_ai_decision = (
                "NG"
                if bool(local_analysis.get("is_defect", False))
                else "OK"
            )
            info = dict(base_info)
            info["lighting_mode"] = mode

            json_path = self.dataset_manager.save_sample(
                ng_image=item.get("test_image"),
                label=label,
                sample_image=item.get("sample_image"),
                aoi_info=info,
                analysis=local_analysis,
                # Multilight é material de aprendizado explícito: preserva as
                # imagens completas de cada iluminação mesmo quando IA e
                # operador concordam. A deduplicação impede repetição.
                save_images=True,
                source=source,
                ai_decision=local_ai_decision,
                lighting_mode=mode,
                event_id=event_id,
                source_frame=item.get("source_frame"),
                final_analysis=final_analysis,
            )
            persisted = bool(json_path) or persisted
            all_saved = all_saved and bool(json_path)
        task["_online_all_persisted"] = all_saved
        return persisted

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return

                work = dict(task)
                samples = _complete_multilight_samples(work)
                partial = work.pop("shadow_partial_samples", None) or []
                if samples:
                    persisted = self._persist_multilight(work, samples)
                elif partial:
                    # O operador já deu 0/1: persistir APENAS os frames
                    # obtidos antes da decisão. Nunca aguardar TOP/MID
                    # posteriores (podem pertencer a uma outra placa).
                    persisted = self._persist_multilight(work, partial)
                    # Treino incremental de trio exige três iluminações;
                    # os pares parciais ficam no dataset para revisão futura.
                    work["_online_all_persisted"] = False
                else:
                    # Campos de orquestração não pertencem ao contrato legado
                    # do DatasetManager quando a captura é monoimagem.
                    work.pop("multilight_samples", None)
                    work.pop("event_id", None)
                    json_path = self.dataset_manager.save_sample(**work)
                    persisted = bool(json_path)

                if persisted:
                    # Nunca esperar backpropagation/validação no ciclo XP.
                    # Enfileirar DEPOIS da persistência da decisão humana,
                    # incluindo Teste, Sombra e Produção.
                    if (eligible_online_case(work)
                            and work.get("_online_all_persisted", True)):
                        try:
                            if self._online_learning is None:
                                self._online_learning = OnlineLearningQueue()
                            event_id = self._online_learning.submit_saved(work)
                            if event_id:
                                print("Aprendizado CNN incremental: job " + event_id)
                        except Exception as neural_exc:
                            print("Não foi possível enfileirar treino CNN: " + str(neural_exc))
                    self._reload_memory_once()
            except Exception as exc:
                print(
                    "Falha não fatal na persistência assíncrona da decisão: "
                    f"{exc}"
                )
            finally:
                self._queue.task_done()


__all__ = [
    "DecisionPersistenceQueue",
    "LIGHTING_ORDER",
    "_complete_multilight_samples",
]
