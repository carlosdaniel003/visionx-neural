"""Aprendizado incremental de CNNs especializadas, sem bloquear a AOI.

Registro extensível por categoria: cada CNN futura adiciona seu próprio
treinador, sem alteração da fila de persistência. Somente rótulos humanos
e casos marcados como NOVOS entram na fila. Pesos nunca são trocados aqui.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4

import cv2
import numpy as np

from src.config.settings import BASE_DIR
from src.core.strict_category_memory import canonical_memory_category
from src.services.image_archive_dedup import image_fingerprint

ONLINE_SCHEMA = "visionx.specialist_online_case.v1"
HUMAN_SOURCES = frozenset({
    "button", "xp_keyboard", "keyboard", "manual", "operator",
    "physical_keyboard", "human",
})
LIGHTS = ("SIDE", "TOP", "MID")

# Futuros especialistas apenas registram categoria, entrypoint e versão.
SPECIALIST_TRAINERS = {
    "FALTANDO": "src.scripts.train_faltando_cnn_v2_online",
}


def _known_human_source(source: str) -> bool:
    normalized = str(source or "").strip().lower()
    return normalized in HUMAN_SOURCES or normalized.startswith("operator_")


def _is_novel_specialist(analysis: dict, category: str) -> bool:
    details = analysis.get("detail", {}) if isinstance(analysis, dict) else {}
    details = details if isinstance(details, dict) else {}
    if category == "FALTANDO":
        if details.get("recognition_route") == "NEW_CNN":
            return True
        lighting = details.get("recognition_light_routes", {})
        return (
            isinstance(lighting, dict)
            and "NEW_CNN" in lighting.values()
        )
    return details.get("recognition_route") == "NEW_CNN"


def eligible_online_case(task: dict) -> bool:
    """Defesa contra pseudo-rótulos e treinamento da mesma imagem repetida."""
    category = canonical_memory_category(
        (task.get("aoi_info") or {}).get("category", "")
    )
    return bool(
        category in SPECIALIST_TRAINERS
        and task.get("label") in ("OK", "NG")
        and _known_human_source(task.get("source", ""))
        and _is_novel_specialist(task.get("analysis") or {}, category)
    )


def _png_bytes(frame) -> bytes:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3 or frame.shape[2] != 3
        or min(frame.shape[:2]) < 12
    ):
        raise ValueError("Gabarito/teste RGB/BGR de 8 bits inválido")
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise ValueError("Não foi possível preservar imagem PNG")
    return encoded.tobytes()


class OnlineLearningQueue:
    """Journal em disco + trabalhador serial que usa um processo separado."""

    def __init__(
        self, root: Path = BASE_DIR, *,
        start_worker: bool = True,
        worker_command=None,
    ):
        self.root = Path(root).resolve()
        self.journal = self.root / "reports" / "neural_online"
        self.events = self.journal / "events"
        self.statuses = self.journal / "statuses"
        self.logs = self.journal / "logs"
        for path in (self.events, self.statuses, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._queue = Queue()
        self._queued: set[str] = set()
        self._worker_command = worker_command or self._run_job
        self._worker = None
        if start_worker:
            self._worker = Thread(
                target=self._work, daemon=True,
                name="VisionXNeuralOnlineWorker",
            )
            self._worker.start()
            # Recupera jobs não concluídos após reinício/crash.
            for path in sorted(self.events.glob("*.json")):
                if not (self.statuses / path.name).exists():
                    self._put(path.stem)

    def _write_latest(self, event_id: str, *, state: str,
                      promoted: bool | None = None) -> None:
        latest = self.journal / "latest_event.json"
        # Nunca sobrescrever o status de um evento posterior.
        if state != "QUEUED" and latest.is_file():
            try:
                old = json.loads(latest.read_text(encoding="utf-8"))
                if old.get("event_id") != event_id:
                    return
            except (OSError, ValueError):
                return
        info = {
            "schema": "visionx.neural_online_latest_status.v1",
            "event_id": event_id,
            "state": state,
            "promoted": promoted,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        pending = latest.with_suffix(".json.tmp")
        pending.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        pending.replace(latest)

    def _put(self, event_id: str) -> None:
        with self._lock:
            if event_id not in self._queued:
                self._queued.add(event_id)
                self._queue.put(event_id)

    def _work(self):
        while True:
            name = self._queue.get()
            try:
                event = self.events / (name + ".json")
                status_file = self.statuses / (name + ".json")
                if not event.is_file():
                    continue
                self._write_latest(name, state="TRAINING")
                result = self._worker_command(event)
                outcome_file = self.journal / "outcomes" / (name + ".json")
                outcome = (
                    json.loads(outcome_file.read_text(encoding="utf-8"))
                    if result == 0 and outcome_file.is_file()
                    else {}
                )
                promoted = outcome.get("promoted")
                final_state = (
                    "PROMOTED" if promoted is True
                    else "REJECTED" if promoted is False
                    else "COMPLETED" if result == 0 else "FAILED"
                )
                self._write_latest(name, state=final_state, promoted=promoted)
                state = {
                    "schema": "visionx.neural_online_task_status.v1",
                    "event": name,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exit_code": int(result),
                    "status": "COMPLETED" if result == 0 else "FAILED",
                }
                tmp = status_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
                tmp.replace(status_file)
            except Exception as exc:
                # Falha antes de conclusão mantém o evento no journal para
                # reexecução numa próxima inicialização da aplicação.
                print("Falha não fatal do treino neural online:", exc)
            finally:
                with self._lock:
                    self._queued.discard(name)
                self._queue.task_done()

    def _run_job(self, event: Path) -> int:
        data = json.loads(event.read_text(encoding="utf-8"))
        category = str(data.get("category", ""))
        module = SPECIALIST_TRAINERS.get(category)
        if module is None:
            return 2
        log_path = self.logs / (event.stem + ".log")
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        with log_path.open("a", encoding="utf-8") as logfile:
            proc = subprocess.run(
                [sys.executable, "-m", module, "--event", str(event),
                 "--root", str(self.root)],
                cwd=str(self.root), stdout=logfile,
                stderr=subprocess.STDOUT, check=False,
                creationflags=flags,
            )
            return proc.returncode

    def submit_saved(self, task: dict) -> str | None:
        """Somente após persistência confirmada. Retorno é ID, não êxito de treino."""
        if not eligible_online_case(task):
            return None
        category = canonical_memory_category(task["aoi_info"].get("category"))
        samples = task.get("multilight_samples", [])
        if (
            isinstance(samples, list) and len(samples) == 3
            and set(str(s.get("lighting_mode", "")).upper() for s in samples)
            == set(LIGHTS)
        ):
            inputs = samples
        else:
            inputs = [{
                "lighting_mode": task.get("aoi_info", {}).get("lighting_mode", "SIDE"),
                "sample_image": task.get("sample_image"),
                "test_image": task.get("ng_image"),
            }]
        if any(
            not isinstance(s, dict) or
            str(s.get("lighting_mode", "")).upper() not in LIGHTS
            for s in inputs
        ):
            raise ValueError("Iluminação inválida para treino incremental")
        event_id = uuid4().hex
        assets = []
        images = []
        try:
            for sample in inputs:
                light = str(sample["lighting_mode"]).strip().upper()
                ref = _png_bytes(sample.get("sample_image"))
                test = _png_bytes(sample.get("test_image"))
                for mode, role, content in ((light, "reference", ref), (light, "test", test)):
                    name = f"{event_id}_{mode}_{role}.png"
                    output = self.events / name
                    output.write_bytes(content)
                    assets.append(output)
                images.append({
                    "lighting_mode": light,
                    "reference": assets[-2].name,
                    "test": assets[-1].name,
                    "reference_sha256": sha256(ref).hexdigest(),
                    "test_sha256": sha256(test).hexdigest(),
                })
            metadata = {
                "schema": ONLINE_SCHEMA,
                "event_id": event_id,
                "source_aoi_event_id": str(task.get("event_id", "") or ""),
                "category": category,
                "label": task["label"],
                "human_source": str(task["source"]),
                "recognition_route": (
                    (task.get("analysis") or {}).get("detail", {})
                    .get("recognition_route")
                ),
                "aoi_info": {
                    key: str((task.get("aoi_info") or {}).get(key, ""))
                    for key in ("board", "parts", "category", "value")
                },
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "images": images,
                "training_requested": True,
                "automatic_production_approval": False,
            }
            journal = self.events / (event_id + ".json")
            draft = journal.with_suffix(".json.tmp")
            draft.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            draft.replace(journal)
            self._write_latest(event_id, state="QUEUED")
            self._put(event_id)
            print(
                "CNN ONLINE: confirmação humana salva. Treino enfileirado "
                f"({category}/{task['label']}, evento={event_id}, "
                f"imagens={len(images)})."
            )
            return event_id
        except Exception:
            for asset in assets:
                asset.unlink(missing_ok=True)
            raise

    def wait_until_idle(self) -> None:
        """Exclusivo para teste/manutenção; nunca bloquear a inspeção."""
        self._queue.join()


__all__ = [
    "OnlineLearningQueue", "eligible_online_case",
    "SPECIALIST_TRAINERS", "ONLINE_SCHEMA",
]
