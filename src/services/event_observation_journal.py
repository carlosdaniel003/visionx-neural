"""Diário não intrusivo de inspeções: metadados por peça, sem novos PNGs.

Não constitui truth label para decisões automáticas. Capturas podem não ser
reproduzíveis quando o arquivo visual descarta imagens por similaridade.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path
from queue import Full, Queue
import re
from threading import Thread

import numpy as np

from src.config.settings import BASE_DIR
from src.services.image_archive_dedup import image_fingerprint
from src.services.network_xp_frame import (
    network_xp_record_event_id, network_xp_frame_snapshot,
)

SCHEMA = "visionx.event_observation.v1"
LIGHTS = ("SIDE", "TOP", "MID")
OPERATOR_SOURCES = {"button", "xp_keyboard", "keyboard", "operator", "operator_button"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


def _active_snapshot(panel, label: str, source: str) -> dict | None:
    """Somente evento atual com captura íntegra: nunca reaproveitar a anterior."""
    event_id = str(network_xp_record_event_id(panel) or "").strip()
    if not SAFE_ID.fullmatch(event_id):
        return None
    analysis = getattr(panel, "current_analysis", None)
    if not isinstance(analysis, dict) or label not in {"OK", "NG"}:
        return None
    current_source = str(getattr(panel, "capture_cycle_source", "") or "").lower()
    if current_source != "network":
        return None

    category = str((getattr(panel, "current_aoi_info", {}) or {}).get(
        "category", "") or "").strip()
    if not category:
        return None
    saved_event = str(getattr(panel, "adhesive_multilight_last_event_id", "") or "")
    saved_frames = getattr(panel, "adhesive_multilight_last_source_frames", None)
    if event_id == saved_event and isinstance(saved_frames, dict) and saved_frames:
        frames = {light: saved_frames[light].copy()
                  for light in LIGHTS
                  if isinstance(saved_frames.get(light), np.ndarray)
                  and saved_frames[light].size}
    else:
        image = network_xp_frame_snapshot(panel)
        frames = {"SIDE": image.copy()} if isinstance(image, np.ndarray) and image.size else {}
    if not frames or set(frames) not in ({"SIDE"}, set(LIGHTS)):
        return None
    detail = analysis.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    lighting_detail = getattr(panel, "adhesive_multilight_last_analyses", None)
    lighting_detail = lighting_detail if isinstance(lighting_detail, dict) and saved_event == event_id else {}
    scored = lighting_detail if lighting_detail else {"SIDE": analysis}
    observed = {}
    for light in frames:
        per_light = scored.get(light) if isinstance(scored.get(light), dict) else {}
        d = per_light.get("detail")
        d = d if isinstance(d, dict) else {}
        observed[light] = {
            "checkpoint_sha256": str(d.get("cnn_v2_checkpoint_sha256") or ""),
            "cnn_ng_score_uncalibrated": d.get("cnn_v2_ng_score_uncalibrated"),
            "verdict": str(per_light.get("verdict") or ""),
        }
    origin = str(source or "").lower()
    operator = origin in OPERATOR_SOURCES
    return {
        "schema": SCHEMA, "event_id": event_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": category, "decision_label": label,
        "label_provenance": "OPERATOR_INPUT" if operator else "AUTOMATIC_DECISION",
        "expected_label": label if operator else None,
        "decision_source": origin,
        "final_verdict": str(analysis.get("verdict") or ""),
        "final_is_defect": analysis.get("is_defect"),
        "route": ("SIDE_EARLY" if detail.get("side_early_exit")
                  else "MULTILIGHT" if len(frames) == 3 else "SINGLE_FRAME"),
        "analyzed_lights": list(frames),
        "skipped_lights": [l for l in LIGHTS if l not in frames],
        "scores_by_light": observed,
        "_frames": frames,
    }


class EventObservationQueue:
    """Processa hashes e grava JSON fora da thread do controlador XP."""
    def __init__(self, root: Path | None = None, *, max_pending: int = 16):
        self.root = Path(root if root is not None else BASE_DIR).resolve()
        self._queue = Queue(maxsize=max_pending)
        self._thread = Thread(
            target=self._run, name="ODINEventJournal", daemon=True
        )
        self._thread.start()

    def submit(self, row: dict) -> bool:
        try:
            self._queue.put_nowait(row)
            return True
        except Full:
            print("Diário ODIN: fila cheia; evento não registrado, sem bloquear XP.")
            return False

    def wait_until_idle(self):
        self._queue.join()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                frames = item.pop("_frames")
                item["frames"] = {
                    light: {
                        "pixel_sha256": image_fingerprint(image),
                        "shape": list(image.shape),
                    }
                    for light, image in frames.items()
                }
                folder = self.root / "reports" / "startup_regression" / "event_observations"
                folder.mkdir(parents=True, exist_ok=True)
                filename = folder / (item["event_id"] + ".json")
                # Eventos completos nunca são substituídos por duplicações
                # da tecla XP/repetições de callback no mesmo ciclo.
                try:
                    with filename.open("x", encoding="utf-8") as stream:
                        json.dump(item, stream, ensure_ascii=False, indent=2)
                except FileExistsError:
                    pass
            except (OSError, ValueError, TypeError) as exc:
                print(f"Falha não fatal no diário de inspeção: {exc}")
            finally:
                self._queue.task_done()


def install_event_observation_journal(control_panel_cls):
    """Hook externo, depois de todos os hooks de arquivamento já instalados."""
    if getattr(control_panel_cls, "_event_observation_journal_installed", False):
        return
    original = control_panel_cls.save_label

    @wraps(original)
    def wrapped(self, user_decision: str, source="button"):
        item = _active_snapshot(self, str(user_decision or "").upper(), str(source))
        result = original(self, user_decision, source=source)
        if item is not None and result is not False:
            if source == "production_auto" and (
                getattr(self, "last_decision_command_success", None) is not True
            ):
                return result
            queue = getattr(self, "_event_observation_queue", None)
            if queue is None:
                queue = EventObservationQueue()
                self._event_observation_queue = queue
            queue.submit(item)
        return result

    control_panel_cls.save_label = wrapped
    control_panel_cls._event_observation_journal_installed = True


__all__ = ["SCHEMA", "EventObservationQueue", "install_event_observation_journal"]
