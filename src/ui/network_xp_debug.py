"""Formatação e cópia do diagnóstico da entrada recebida do Windows XP.

Este módulo é exclusivamente de observabilidade. Não participa da validação,
classificação, memória, confiança ou controle do ciclo de produção.
"""

from __future__ import annotations

import json
from typing import Any


DEBUG_SCHEMA = "visionx.network_xp_debug.v1"


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_network_debug_report(record: dict | None) -> str:
    data = _json_safe(record if isinstance(record, dict) else {})
    if not data:
        return (
            "VISIONX - DEBUG DE ENTRADA WINDOWS XP\n"
            "Nenhuma imagem recebida do XP possui diagnóstico registrado."
        )

    validation = data.get("validation", {})
    transport = data.get("transport", {})
    cycle = data.get("cycle", {})
    hints = validation.get("diagnostic_hints", []) or []

    lines = [
        "VISIONX - DEBUG DE ENTRADA WINDOWS XP",
        "=" * 72,
        f"Schema: {data.get('schema', DEBUG_SCHEMA)}",
        f"Data/hora: {data.get('timestamp', '-')}",
        f"IP de origem: {data.get('source_ip', '-')}",
        f"Etapa: {data.get('stage', '-')}",
        f"Modo: {data.get('mode', '-')}",
        "",
        "RESULTADO DA VALIDAÇÃO",
        "-" * 72,
        f"Válida: {validation.get('valid', '-')}",
        f"Motivo técnico: {validation.get('reason', '-')}",
        f"Mensagem: {data.get('validation_message', '-')}",
        f"Anomalias brutas: {validation.get('raw_anomaly_count', '-')}",
        f"Epicentros antigos: {validation.get('old_epicenter_count', '-')}",
        f"Epicentros finais: {validation.get('epicenter_count', validation.get('real_epicenter_count', '-'))}",
        f"Caixa global: {validation.get('global_box_info', '-')}",
        f"Caixa de foco: {validation.get('focus_box', '-')}",
        "",
        "TRANSPORTE / CICLO",
        "-" * 72,
        f"Imagem recebida: {transport.get('image', '-')}",
        f"Frames estáveis exigidos: {transport.get('stable_required_frames', '-')}",
        f"Gate aceitando imagens: {cycle.get('accepting_images', '-')}",
        f"Geração do gate: {cycle.get('generation', '-')}",
        f"Imagens ignoradas no gate: {cycle.get('ignored_images', '-')}",
    ]

    if hints:
        lines.extend(["", "INDÍCIOS DIAGNÓSTICOS", "-" * 72])
        for hint in hints:
            lines.append(f"- {hint}")

    lines.extend(
        [
            "",
            "REGISTRO COMPLETO (JSON)",
            "-" * 72,
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        ]
    )
    return "\n".join(lines)


def copy_network_debug_to_clipboard(panel) -> bool:
    record = getattr(panel, "network_intake_last_validation", None)
    if not isinstance(record, dict) or not record:
        return False

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    QApplication.clipboard().setText(format_network_debug_report(record))

    button = getattr(panel, "btn_copy_network_debug", None)
    if button is not None:
        try:
            button.setText("Debug XP copiado")
            button.setEnabled(True)

            def restore():
                try:
                    button.setText("Copiar debug XP")
                except Exception:
                    pass

            QTimer.singleShot(1600, restore)
        except Exception:
            pass
    return True


def set_network_debug_available(panel, available: bool = True) -> None:
    button = getattr(panel, "btn_copy_network_debug", None)
    if button is None:
        return
    try:
        button.setEnabled(bool(available))
    except Exception:
        pass


__all__ = [
    "DEBUG_SCHEMA",
    "copy_network_debug_to_clipboard",
    "format_network_debug_report",
    "set_network_debug_available",
]
