"""Telemetria OCR de uma única captura AOI.

Cronometra cada chamada externa do Tesseract sem registrar imagens, texto
reconhecido ou alterar a política de parsing/erros do chamador.
"""
from __future__ import annotations

import time


class AOIOCRTiming:
    SCHEMA = "visionx.aoi_ocr_phases.v1"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.route = "not_started"
        self.parts_reason = "not_started"
        self.general_retry = False

    def call(self, phase: str, callback, *args, **kwargs):
        started = time.perf_counter()
        status = "ok"
        try:
            return callback(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            self.calls.append({
                "phase": str(phase),
                "ms": round((time.perf_counter() - started) * 1000.0, 2),
                "status": status,
            })

    def snapshot(self) -> dict:
        calls = [dict(item) for item in self.calls]
        return {
            "schema": self.SCHEMA,
            "route": self.route,
            "parts_reason": self.parts_reason,
            "general_retry": self.general_retry,
            "call_count": len(calls),
            "tesseract_calls_ms": round(
                sum(item["ms"] for item in calls), 2
            ),
            "calls": calls,
        }


__all__ = ["AOIOCRTiming"]
