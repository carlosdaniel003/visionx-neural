"""Persistência serial em segundo plano para decisões humanas do VisionX.

A decisão produtiva não deve esperar gravação em disco nem recarga da memória
KNN. Esta fila executa essas tarefas fora da thread da interface e preserva a
ordem das decisões.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

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


_UNVERIFIED_AUTOMATIC_REASONS = frozenset({
    "supervised_recurrent_ok_top_mid_auto_v1",
    "supervised_mixed_light_ok_auto_v1",
    "supervised_operator_corrected_archive_ok_auto_v1",
    "supervised_operator_known_top_mid_ok_20261029_auto_v1",
    "production_uncertain_auto_ng_quarantine_v1",
    "side_strong_human_ok_supervised_probe_override_auto_ok",
})


def _is_unverified_automatic(task: dict) -> bool:
    if str(task.get("source", "")).lower() != "production_auto":
        return False
    analysis = task.get("analysis")
    detail = analysis.get("detail") if isinstance(analysis, dict) else None
    policy = detail.get("production_decision_policy") if isinstance(detail, dict) else None
    return bool(
        isinstance(policy, dict)
        and policy.get("reason") in _UNVERIFIED_AUTOMATIC_REASONS
        and policy.get("resolution") == "automatic"
    )


def _quarantine_unverified_automatic(task: dict) -> Path:
    """Archive diagnostic images without teaching KNN/CNN guessed labels."""
    now = datetime.now(timezone.utc)
    folder = (Path.cwd() / "reports/production_auto_unverified"
              / now.strftime("%Y%m%d"))
    images_dir = folder / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    analysis = task.get("analysis") if isinstance(task.get("analysis"), dict) else {}
    detail = analysis.get("detail") if isinstance(analysis.get("detail"), dict) else {}
    policy = detail.get("production_decision_policy") if isinstance(
        detail.get("production_decision_policy"), dict,
    ) else {}
    files = []
    samples = _complete_multilight_samples(task)
    if (
        policy.get("reason")
        == "side_strong_human_ok_supervised_probe_override_auto_ok"
        and not samples
    ):
        # Julgamento antecipado SIDE: arquivar somente a luz realmente
        # capturada, sem criar TOP/MID inexistentes e sem treinar a IA.
        partial = task.get("multilight_samples")
        partial = partial if isinstance(partial, list) else task.get(
            "shadow_partial_samples", []
        )
        samples = [
            row for row in partial
            if isinstance(row, dict)
            and str(row.get("lighting_mode", "")).upper() == "SIDE"
        ][:1]
    for sample in samples:
        light = sample["lighting_mode"]
        for role, key in (("reference", "sample_image"), ("test", "test_image")):
            frame = sample.get(key)
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                continue
            success, data = cv2.imencode(".png", frame)
            if not success:
                continue
            payload = data.tobytes()
            digest = sha256(payload).hexdigest()
            target = images_dir / (digest + ".png")
            if not target.exists():
                target.write_bytes(payload)
            files.append({
                "light": light, "role": role,
                "sha256": digest, "file": target.relative_to(folder).as_posix(),
            })
    audit = {
        "schema": "visionx.production_auto_unverified.v1",
        "timestamp_utc": now.isoformat(),
        "event_id": str(task.get("event_id", "")),
        "decision_sent_for_xp": str(task.get("label", "")),
        "source_verdict": analysis.get("verdict"),
        "rule": policy.get("reason"),
        "resolution": policy.get("resolution"),
        "unverified_physical_label": True,
        "used_for_knn_training": False,
        "used_for_cnn_training": False,
        "files": files,
    }
    output = folder / (now.strftime("%H%M%S_%f") + "_" + uuid4().hex[:8] + ".json")
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return output


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

    def _reload_memory_once(self, paths=()) -> None:
        if self.orchestrator is None:
            return
        incremental = getattr(self.orchestrator, "ingest_saved_memory", None)
        if callable(incremental):
            try:
                incremental(tuple(paths))
                return
            except Exception as exc:
                # Fallback raro: nunca deixar RAM parcialmente atualizada.
                print("KNN incremental indisponível; recarga segura: " + str(exc))
        reload_memory = getattr(self.orchestrator, "reload_memory", None)
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
        saved_paths = task.setdefault("_saved_memory_paths", [])
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
            if json_path:
                saved_paths.append(str(json_path))
        task["_online_all_persisted"] = all_saved
        return persisted

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return

                work = dict(task)
                if _is_unverified_automatic(work):
                    # Decision can have been sent to XP; do not reclassify
                    # any unconfirmed automatic outcome as human truth.
                    _quarantine_unverified_automatic(work)
                    continue
                samples = _complete_multilight_samples(work)
                partial = work.pop("shadow_partial_samples", None) or []
                if samples:
                    persisted = self._persist_multilight(work, samples)
                elif partial:
                    # O operador já deu 0/1: persistir somente imagens
                    # captadas ANTES do julgamento. CNN V2 aceita de 1 a 3
                    # luzes reais no journal, sem inventar TOP/MID.
                    persisted = self._persist_multilight(work, partial)
                    work["multilight_samples"] = partial
                else:
                    # Campos de orquestração não pertencem ao contrato legado
                    # do DatasetManager quando a captura é monoimagem.
                    work.pop("multilight_samples", None)
                    work.pop("event_id", None)
                    json_path = self.dataset_manager.save_sample(**work)
                    persisted = bool(json_path)
                    if json_path:
                        work.setdefault("_saved_memory_paths", []).append(str(json_path))

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
                    self._reload_memory_once(work.get("_saved_memory_paths", ()))
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
