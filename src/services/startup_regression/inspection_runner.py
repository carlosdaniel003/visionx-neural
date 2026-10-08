"""Replay físico SIDE da AOI, sem acesso à memória de exemplos KNN.

Não instancia ControlPanel, não carrega DatasetManager/KNN, não executa
captura MSS, não envia comandos XP e não grava imagens de debug.
O modo `_replay_without_memory` é consumido pelas fusões física e INVERTIDO.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.adhesive_multilight_analysis import analyze_lighting, build_lighting_context
from src.core.anomaly_memory_integration import install_anomaly_memory_integration
from src.core.experts.missing_component_expert import MissingComponentExpert
from src.core.experts.semantic_calibration import install_semantic_calibration
from src.core.experts.semantic_expert import SemanticExpert
from src.core.experts.shift_expert import ShiftExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.inverted_face_integration import install_inverted_face_integration
from src.core.moe_orchestrator import MoEOrchestrator
from src.core.roi_input_contract import install_roi_input_contract
from src.core.roi_visual_alignment import install_roi_visual_alignment
from src.core.semantic_roi_extension import install_semantic_roi_extension
from src.services import screen_monitor as screen_module
from src.utils.text_normalizer import CATEGORIES, normalize_aoi_text

from .replay_telemetry import decision_snapshot, geometry_snapshot


class ReplayError(Exception):
    """Falha de leitura/layout/OCR/análise; não converter em decisão OK."""


class PhysicalOnlyOrchestrator(MoEOrchestrator):
    """Mesmo MoE e extensões físicas, sem construir sequer o especialista KNN."""

    def __init__(self) -> None:
        self.experts = {
            "shift": ShiftExpert(),
            "silk": SilkExpert(),
            "ssim": SSIMExpert(),
            "semantic": SemanticExpert(),
        }
        # install_anomaly_memory_integration seleciona as rotas físicas.
        self.routing_table = {}

    def reload_memory(self) -> None:
        raise ReplayError("É proibido recarregar memória no replay offline")


def create_physical_orchestrator() -> PhysicalOnlyOrchestrator:
    """Instala apenas hooks físicos usados pela operação normal."""
    install_semantic_calibration(SemanticExpert)
    install_semantic_roi_extension(SemanticExpert)
    install_roi_visual_alignment(SilkExpert, MissingComponentExpert)
    install_anomaly_memory_integration(MoEOrchestrator)
    install_inverted_face_integration(MoEOrchestrator)
    install_roi_input_contract(MoEOrchestrator, SSIMExpert, SilkExpert)
    orchestrator = PhysicalOnlyOrchestrator()
    if "knn" in orchestrator.experts:
        raise ReplayError("Orquestrador de replay contém KNN")
    return orchestrator


def _decode_png(path: Path) -> np.ndarray:
    try:
        contents = path.read_bytes()
        frame = cv2.imdecode(
            np.frombuffer(contents, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except (OSError, cv2.error, ValueError) as exc:
        raise ReplayError(f"Falha na leitura de {path.name}: {exc}") from exc
    if frame is None or frame.size == 0:
        raise ReplayError("PNG ilegível ou vazio")
    return frame


class SideInspectionRunner:
    def __init__(self, orchestrator=None, monitor=None):
        self.orchestrator = (
            orchestrator
            if orchestrator is not None
            else create_physical_orchestrator()
        )
        self.monitor = (
            monitor if monitor is not None else screen_module.ScreenMonitor()
        )
        if "knn" in getattr(self.orchestrator, "experts", {}):
            raise ReplayError("KNN não pode estar disponível no replay")
        # Mesmo algoritmo de recorte/OCR da rede AOI, sem os writes de debug.
        self.monitor._replay_no_debug = True

    def _extract(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        if not screen_module.HAS_TESSERACT:
            raise ReplayError(
                "Tesseract/OCR indisponível; impossível validar um caso novo"
            )
        captured = []
        errors = []
        def on_layout(sample, test, info):
            captured.append((sample.copy(), test.copy(), dict(info or {})))
        def on_log(line):
            if "SUCESSO" not in str(line):
                errors.append(str(line))

        self.monitor.layout_detected.connect(on_layout)
        self.monitor.log_updated.connect(on_log)
        try:
            # As mensagens de diagnóstico do mesmo extrator operacional
            # não precisam ser repetidas para cada PNG no terminal.
            with redirect_stdout(StringIO()):
                self.monitor.process_external_image(frame)
        finally:
            self.monitor.layout_detected.disconnect(on_layout)
            self.monitor.log_updated.disconnect(on_log)
        if len(captured) != 1:
            raise ReplayError(
                "Não foi possível extrair gabarito/teste completos: "
                + "; ".join(errors[:3])
            )
        return captured[0]

    def inspect_png(self, path: Path, expected_category: str) -> dict[str, Any]:
        frame = _decode_png(Path(path))
        sample, test, info = self._extract(frame)

        if not sample.size or not test.size:
            raise ReplayError("Gabarito ou teste vazio")
        category, value = normalize_aoi_text(info.get("value", ""))
        if category not in CATEGORIES:
            raise ReplayError(
                "OCR não recuperou uma categoria válida; não usar o nome do PNG"
            )
        if expected_category in CATEGORIES and category != expected_category:
            raise ReplayError(
                f"Categoria do OCR ({category}) difere do arquivo ({expected_category})"
            )
        if not str(info.get("board", "")).strip() or not str(
            info.get("parts", "")
        ).strip():
            raise ReplayError("OCR incompleto: faltam Board ou Parts")
        info["category"] = category
        info["value"] = value
        info["lighting_mode"] = "SIDE"
        info["_replay_without_memory"] = True

        # Mesmo contexto geométrico da inspeção operacional. Ele é passado
        # ao pipeline em vez de ser recalculado para a telemetria: um único
        # par gabarito/teste, uma única decisão, sem nova inferência.
        with redirect_stdout(StringIO()):
            context = build_lighting_context(sample, test)
            analysis = analyze_lighting(
                self.orchestrator, sample, test, info, "SIDE",
                context=context,
            )
        if not isinstance(analysis, dict):
            raise ReplayError("Motor físico retornou análise inválida")

        detail = analysis.get("detail", {}) or {}
        trace = detail.get("decision_trace", {}) or {}
        weights = trace.get("weights", {}) or {}
        memory = trace.get("memory", {}) or {}
        active = list(analysis.get("active_engines", []) or [])
        if (
            any("knn" in str(name).lower() for name in active)
            or detail.get("replay_memory_consulted") is not False
            or bool(memory.get("has_memory"))
            or float(weights.get("knn", 0.0) or 0.0) != 0.0
        ):
            raise ReplayError("Isolamento de memória violado: resultado descartado")

        review = bool(
            analysis.get("production_review_required", False)
            or trace.get("operator_review_required", False)
            or detail.get("operator_review_required", False)
        )
        verdict = (
            "REVISÃO OBRIGATÓRIA"
            if review
            else str(analysis.get("verdict", "") or "").strip().upper()
        )
        if verdict not in {
            "DEFEITO REAL", "FALHA FALSA", "REVISÃO OBRIGATÓRIA"
        }:
            raise ReplayError(f"Veredito inválido: {verdict!r}")

        return {
            "verdict": verdict,
            "requires_review": review,
            "category": category,
            "ocr": {
                "board": str(info.get("board", "")),
                "parts": str(info.get("parts", "")),
                "value": str(info.get("value", "")),
            },
            "lighting_mode": "SIDE",
            "reference_dimensions": list(sample.shape[:2][::-1]),
            "test_dimensions": list(test.shape[:2][::-1]),
            "score": _finite_float(detail.get("final_score")),
            "physical_score": _finite_float(detail.get("physical_score")),
            "confidence": _finite_float(analysis.get("confidence")),
            "fusion_rule": str(detail.get("fusion_rule", "")),
            "active_engines": active,
            "memory_consulted": False,
            "knn_enabled": False,
            "telemetry": {
                **decision_snapshot(analysis),
                "geometry": geometry_snapshot(
                    context, analysis, sample, test,
                ),
            },
        }


def _finite_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (ValueError, TypeError):
        return None


__all__ = [
    "PhysicalOnlyOrchestrator",
    "ReplayError",
    "SideInspectionRunner",
    "create_physical_orchestrator",
]
