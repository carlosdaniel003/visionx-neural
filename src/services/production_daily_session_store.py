"""Persistência diária das métricas do Modo Produção.

Os dados ficam fora do repositório para sobreviver a:
- fechamento/reabertura do ODIN;
- troca entre modos;
- atualização/substituição do código.

No Windows, o diretório padrão é:
%LOCALAPPDATA%\VisionX-Neural\production_sessions\
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
import os
from pathlib import Path
from typing import Callable


SCHEMA = "visionx.production_daily_session.v1"


def _default_root_dir() -> Path:
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        return Path(local_app_data) / "VisionX-Neural" / "production_sessions"

    xdg_state = str(os.environ.get("XDG_STATE_HOME", "") or "").strip()
    if xdg_state:
        return Path(xdg_state) / "visionx-neural" / "production_sessions"

    return Path.home() / ".visionx-neural" / "production_sessions"


def _safe_int(value, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, result)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return max(0.0, result)


def empty_metrics() -> dict:
    return {
        "auto_ok": 0,
        "auto_ng": 0,
        "manual_judgments": 0,
        "analysis_count": 0,
        "analysis_time_total": 0.0,
        "analysis_time_count": 0,
        "accuracy_percent": None,
        "average_analysis_time_seconds": None,
    }


def normalize_metrics(value: dict | None) -> dict:
    source = value if isinstance(value, dict) else {}
    normalized = empty_metrics()

    normalized["auto_ok"] = _safe_int(source.get("auto_ok"))
    normalized["auto_ng"] = _safe_int(source.get("auto_ng"))
    normalized["manual_judgments"] = _safe_int(
        source.get("manual_judgments")
    )
    normalized["analysis_count"] = _safe_int(
        source.get("analysis_count")
    )
    normalized["analysis_time_total"] = _safe_float(
        source.get("analysis_time_total")
    )
    normalized["analysis_time_count"] = _safe_int(
        source.get("analysis_time_count")
    )

    completed = (
        normalized["auto_ok"]
        + normalized["auto_ng"]
        + normalized["manual_judgments"]
    )
    automatic = normalized["auto_ok"] + normalized["auto_ng"]
    normalized["accuracy_percent"] = (
        (float(automatic) / float(completed)) * 100.0
        if completed > 0
        else None
    )

    normalized["average_analysis_time_seconds"] = (
        normalized["analysis_time_total"]
        / float(normalized["analysis_time_count"])
        if normalized["analysis_time_count"] > 0
        else None
    )
    return normalized


class ProductionDailySessionStore:
    """Um arquivo JSON por dia operacional, gravado atomicamente."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        today_provider: Callable[[], date] | None = None,
    ):
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else _default_root_dir()
        )
        self.today_provider = today_provider or date.today

    def today(self) -> date:
        value = self.today_provider()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        raise TypeError("today_provider deve retornar date ou datetime")

    def today_key(self) -> str:
        return self.today().isoformat()

    def path_for_day(self, day: date | str) -> Path:
        key = day.isoformat() if isinstance(day, date) else str(day)
        return self.root_dir / f"{key}.json"

    def today_path(self) -> Path:
        return self.path_for_day(self.today_key())

    def load_today(self) -> dict:
        day_key = self.today_key()
        path = self.path_for_day(day_key)

        if not path.exists():
            return {
                "date": day_key,
                "metrics": empty_metrics(),
                "path": str(path),
            }

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return {
                "date": day_key,
                "metrics": empty_metrics(),
                "path": str(path),
            }

        if not isinstance(payload, dict):
            payload = {}

        stored_day = str(payload.get("date", "") or "").strip()
        if stored_day != day_key:
            return {
                "date": day_key,
                "metrics": empty_metrics(),
                "path": str(path),
            }

        return {
            "date": day_key,
            "metrics": normalize_metrics(payload.get("metrics")),
            "path": str(path),
        }

    def save_today(self, metrics: dict | None) -> Path:
        day_key = self.today_key()
        path = self.path_for_day(day_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        normalized = normalize_metrics(metrics)
        payload = {
            "schema": SCHEMA,
            "date": day_key,
            "updated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "metrics": normalized,
        }

        temp_path = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
        return path


__all__ = [
    "ProductionDailySessionStore",
    "SCHEMA",
    "empty_metrics",
    "normalize_metrics",
]
