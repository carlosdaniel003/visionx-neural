"""Diagnóstico copiável da última entrada recebida do Windows XP.

Este módulo é exclusivamente de observabilidade. Não participa da validação,
classificação, memória, confiança ou controle do ciclo de produção.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.services.capture_evidence import (
    capture_debug_record,
    capture_debug_source,
    capture_image_available,
    current_copy_image_snapshot,
)
from src.ui.branding import (
    DECISION_DEBUG_TITLE,
    LOCAL_CAPTURE_DEBUG_TITLE,
    XP_DEBUG_TITLE,
)
from src.services.network_xp_frame import (
    network_xp_frame_available,
    network_xp_frame_snapshot,
)


DEBUG_SCHEMA = "visionx.network_xp_debug.v1"


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_network_debug_report(record: dict | None) -> str:
    data = _json_safe(record if isinstance(record, dict) else {})
    source = str(data.get("source", "") or "").strip().lower()
    report_title = (
        LOCAL_CAPTURE_DEBUG_TITLE
        if source == "local_mss"
        else XP_DEBUG_TITLE
    )
    if not data:
        return (
            f"{XP_DEBUG_TITLE}\n"
            "Nenhuma captura possui diagnóstico registrado."
        )

    validation = data.get("validation", {})
    transport = data.get("transport", {})
    cycle = data.get("cycle", {})
    hints = validation.get("diagnostic_hints", []) or []

    lines = [
        report_title,
        "=" * 72,
        f"Schema: {data.get('schema', DEBUG_SCHEMA)}",
        f"Evento: {data.get('event_id', '-')}",
        f"Data/hora: {data.get('timestamp', '-')}",
        f"Origem: {'Captura local MSS' if source == 'local_mss' else 'Windows XP'}",
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

    decision = data.get("decision", {})
    if isinstance(decision, dict) and decision:
        missing = decision.get("missing", {})
        memory = decision.get("memory", {})
        missing = missing if isinstance(missing, dict) else {}
        memory = memory if isinstance(memory, dict) else {}
        lines.extend(
            [
                "",
                DECISION_DEBUG_TITLE,
                "-" * 72,
                f"Categoria: {decision.get('category', '-')}",
                f"Veredito: {decision.get('verdict', '-')}",
                f"Defeito: {decision.get('is_defect', '-')}",
                f"Confiança: {decision.get('confidence', '-')}",
                f"Score final: {decision.get('final_score', '-')}",
                f"Score físico: {decision.get('physical_score', '-')}",
                f"Regra de fusão: {decision.get('fusion_rule', '-')}",
                f"Motor dominante: {decision.get('dominant_engine', '-')}",
                f"Revisão obrigatória: {decision.get('operator_review_required', '-')}",
                f"Ausência física forte efetiva: {decision.get('hard_missing_evidence', '-')}",
                f"Ausência física forte bruta: {decision.get('raw_hard_missing_evidence', '-')}",
                f"Hard missing contradito por OK quase exato: {decision.get('hard_missing_contradicted_by_exact_ok', '-')}",
                f"Missing score: {missing.get('missing_score', '-')}",
                f"Missing classe: {missing.get('missing_classification', '-')}",
                f"Missing cobertura: {missing.get('missing_changed_coverage', '-')}",
                f"Missing residual médio: {missing.get('missing_residual_mean', '-')}",
                f"Missing residual p90: {missing.get('missing_residual_p90', '-')}",
                f"Missing perda estrutural: {missing.get('missing_structure_loss', '-')}",
                f"Missing similaridade direta: {missing.get('missing_direct_similarity', '-')}",
                f"Missing perda de aparência: {missing.get('missing_appearance_loss', '-')}",
                f"Missing melhor similaridade próxima: {missing.get('missing_best_similarity', '-')}",
                f"Missing fundo exposto: {missing.get('missing_background_exposure', '-')}",
                f"Missing bordas incompatíveis: {missing.get('missing_edge_mismatch', '-')}",
                f"Corpo presente: {missing.get('missing_component_body_present', '-')}",
                f"Veto por corpo presente: {missing.get('missing_body_presence_veto', '-')}",
                f"Fonte da presença: {missing.get('missing_body_presence_source', '-')}",
                f"Caixa da presença: {missing.get('missing_body_presence_box', '-')}",
                f"Presença coarse similarity: {missing.get('missing_body_coarse_similarity', '-')}",
                f"Presença silhouette dice: {missing.get('missing_body_silhouette_dice', '-')}",
                f"Presença area ratio: {missing.get('missing_body_area_ratio', '-')}",
                f"Presença centroid shift: {missing.get('missing_body_centroid_shift', '-')}",
                f"Presença motivo: {missing.get('missing_body_presence_reason', '-')}",
                f"Missing hard reason: {missing.get('missing_hard_absence_reason', '-')}",
                f"Guarda transversal: {missing.get('missing_cross_category_guard', '-')}",
                f"Guarda política: {missing.get('missing_guard_policy', '-')}",
                f"Guarda categoria origem: {missing.get('missing_guard_source_category', '-')}",
                f"Guarda suporte físico: {missing.get('missing_guard_physical_support', '-')}",
                f"Dual-scale política: {missing.get('missing_dual_scale_policy', '-')}",
                f"Dual-scale ativo: {missing.get('missing_dual_scale_active', '-')}",
                f"Dual-scale executado: {missing.get('missing_dual_scale_triggered', '-')}",
                f"Dual-scale desacordo: {missing.get('missing_scale_disagreement', '-')}",
                f"Dual-scale razão local/global: {missing.get('missing_local_global_area_ratio', '-')}",
                f"Dual-scale caixa contexto: {missing.get('missing_context_box', '-')}",
                f"Dual-scale score contexto: {missing.get('missing_context_score', '-')}",
                f"Dual-scale cobertura contexto: {missing.get('missing_context_coverage', '-')}",
                f"Dual-scale residual contexto: {missing.get('missing_context_residual_mean', '-')}",
                f"Dual-scale perda estrutural: {missing.get('missing_context_structure_loss', '-')}",
                f"Dual-scale similaridade direta: {missing.get('missing_context_direct_similarity', '-')}",
                f"Dual-scale perda aparência: {missing.get('missing_context_appearance_loss', '-')}",
                f"Dual-scale melhor match próximo: {missing.get('missing_context_best_similarity', '-')}",
                f"Dual-scale hard absence: {missing.get('missing_context_hard_absence', '-')}",
                f"Dual-scale motivo: {missing.get('missing_context_hard_reason', '-')}",
                f"Dual-scale suporte físico: {missing.get('missing_context_physical_support', '-')}",
                f"KNN melhor rótulo: {memory.get('best_match_label', '-')}",
                f"KNN similaridade: {memory.get('best_similarity', '-')}",
                f"KNN conflito efetivo: {memory.get('memory_conflict', '-')}",
                f"KNN conflito bruto: {memory.get('raw_memory_conflict', '-')}",
                f"KNN revisão efetiva: {memory.get('operator_review_required', '-')}",
                f"KNN revisão bruta: {memory.get('raw_operator_review_required', '-')}",
                f"KNN suprimido por ausência física: {memory.get('suppressed_by_hard_missing', '-')}",
                f"Motivo final: {decision.get('reason', '-')}",
            ]
        )

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


def _debug_record(panel) -> dict:
    record = capture_debug_record(panel)
    if record:
        return record
    legacy = getattr(panel, "network_intake_last_validation", None)
    return dict(legacy) if isinstance(legacy, dict) else {}


def _legacy_xp_image_fallback_allowed(panel) -> bool:
    """Impede reutilizar frame XP quando o diagnóstico atual é local MSS."""
    return capture_debug_source(panel).strip().lower() != "local_mss"


def network_debug_image_available(panel) -> bool:
    """Aceita a evidência do evento atual sem misturar origens."""
    if capture_image_available(panel):
        return True
    if not _legacy_xp_image_fallback_allowed(panel):
        return False
    return bool(network_xp_frame_available(panel))


def _set_button_feedback(button, copied_text: str, idle_text: str) -> None:
    if button is None:
        return

    from PyQt6.QtCore import QTimer

    try:
        button.setText(copied_text)
        button.setEnabled(True)

        def restore():
            try:
                button.setText(idle_text)
            except Exception:
                pass

        QTimer.singleShot(1600, restore)
    except Exception:
        pass


def copy_network_debug_to_clipboard(panel) -> bool:
    record = _debug_record(panel)
    if not isinstance(record, dict) or not record:
        return False

    from PyQt6.QtWidgets import QApplication

    QApplication.clipboard().setText(format_network_debug_report(record))
    _set_button_feedback(
        getattr(panel, "btn_copy_network_debug", None),
        "Debug copiado",
        "Copiar debug",
    )
    return True


def _qimage_from_bgr(image: np.ndarray):
    from PyQt6.QtGui import QImage

    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        height, width = array.shape
        qimage = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        return qimage.copy()

    if array.ndim != 3:
        return None

    height, width, channels = array.shape
    if channels == 3:
        qimage = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_BGR888,
        )
        return qimage.copy()

    if channels == 4:
        bgra = np.ascontiguousarray(array[:, :, :4])
        rgba = bgra[:, :, [2, 1, 0, 3]].copy()
        qimage = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()

    return None


def network_debug_image_snapshot(panel) -> np.ndarray | None:
    """Retorna exatamente a evidência visual resolvida por Copiar imagem."""
    return current_copy_image_snapshot(panel)


def _sync_capture_image_preview(panel) -> None:
    """Atualiza somente a prévia visual, sem participar do ciclo produtivo."""
    preview = getattr(panel, "lbl_capture_evidence_preview", None)
    if preview is None:
        return

    if not bool(getattr(panel, "_inspection_images_visible", True)):
        try:
            if hasattr(preview, "clear_source_image"):
                preview.clear_source_image("Aguardando captura")
            else:
                preview.clear()
                preview.setText("Aguardando captura")
        except Exception:
            pass
        return

    image = network_debug_image_snapshot(panel)
    if image is None:
        try:
            if hasattr(preview, "clear_source_image"):
                preview.clear_source_image("Aguardando captura")
            else:
                preview.clear()
                preview.setText("Aguardando captura")
        except Exception:
            pass
        return

    qimage = _qimage_from_bgr(image)
    if qimage is None or qimage.isNull():
        return

    try:
        if hasattr(preview, "set_source_image"):
            preview.set_source_image(qimage)
        else:
            from PyQt6.QtGui import QPixmap

            preview.setPixmap(QPixmap.fromImage(qimage))
    except Exception:
        pass


def copy_network_image_to_clipboard(panel) -> bool:
    image = network_debug_image_snapshot(panel)
    if image is None:
        return False

    qimage = _qimage_from_bgr(image)
    if qimage is None or qimage.isNull():
        return False

    from PyQt6.QtWidgets import QApplication

    QApplication.clipboard().setImage(qimage)
    _set_button_feedback(
        getattr(panel, "btn_copy_network_image", None),
        "Imagem copiada",
        "Copiar imagem",
    )
    return True


def sync_network_debug_controls(panel) -> None:
    record = _debug_record(panel)
    debug_available = bool(isinstance(record, dict) and record)
    image_available = network_debug_image_available(panel)
    _sync_capture_image_preview(panel)

    debug_button = getattr(panel, "btn_copy_network_debug", None)
    image_button = getattr(panel, "btn_copy_network_image", None)
    state_label = getattr(panel, "lbl_network_debug_state", None)

    if debug_button is not None:
        try:
            debug_button.setEnabled(debug_available)
        except Exception:
            pass

    if image_button is not None:
        try:
            image_button.setEnabled(image_available)
        except Exception:
            pass

    if state_label is not None:
        try:
            validation = record.get("validation", {}) if isinstance(record, dict) else {}
            reason = str(validation.get("reason", "") or "")
            valid = validation.get("valid", None) if isinstance(validation, dict) else None
            source_ip = str(record.get("source_ip", "") or "") if isinstance(record, dict) else ""
            source = str(record.get("source", "") or "").strip().lower() if isinstance(record, dict) else ""

            if (
                debug_available
                and image_available
                and source == "local_mss"
            ):
                state_label.setText(
                    "Captura local MSS analisada • relatório + imagem"
                )
                state_label.setProperty("state", "ready")
            elif debug_available and image_available and valid is False:
                suffix = f" • {reason}" if reason else ""
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Último frame XP REJEITADO{suffix}{ip_text} • imagem preservada"
                )
                state_label.setProperty("state", "rejected")
            elif debug_available and image_available and valid is True:
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Último frame XP validado{ip_text} • relatório + imagem"
                )
                state_label.setProperty("state", "ready")
            elif debug_available and image_available:
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Frame XP recebido{ip_text} • aguardando validação"
                )
                state_label.setProperty("state", "partial")
            elif debug_available:
                state_label.setText(
                    "Relatório disponível • imagem do evento não está preservada"
                )
                state_label.setProperty("state", "partial")
            else:
                state_label.setText("Aguardando a primeira captura analisada")
                state_label.setProperty("state", "idle")

            style = state_label.style()
            style.unpolish(state_label)
            style.polish(state_label)
            state_label.update()
        except Exception:
            pass


def set_network_debug_available(panel, available: bool = True) -> None:
    """Compatibilidade com o filtro anterior; a UI deriva o estado real."""
    if not available:
        button = getattr(panel, "btn_copy_network_debug", None)
        if button is not None:
            try:
                button.setEnabled(False)
            except Exception:
                pass
    sync_network_debug_controls(panel)


__all__ = [
    "DEBUG_SCHEMA",
    "copy_network_debug_to_clipboard",
    "copy_network_image_to_clipboard",
    "format_network_debug_report",
    "network_debug_image_available",
    "network_debug_image_snapshot",
    "set_network_debug_available",
    "sync_network_debug_controls",
]
