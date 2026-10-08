"""Aprendizado humano: decisão imediata, persistência em segundo plano."""

from __future__ import annotations

import numpy as np

from src.services.decision_persistence import DecisionPersistenceQueue


def _image_snapshot(value):
    if isinstance(value, np.ndarray) and value.size > 0:
        return value.copy()
    return value


LIGHTING_ORDER = ("SIDE", "TOP", "MID")


def _multilight_learning_snapshot(panel) -> list[dict]:
    """Copia SIDE/TOP/MID antes de a interface limpar a peça julgada."""
    samples = getattr(
        panel,
        "adhesive_multilight_learning_samples",
        {},
    )
    if not isinstance(samples, dict):
        return []

    resolved = []
    for mode in LIGHTING_ORDER:
        item = samples.get(mode)
        if not isinstance(item, dict):
            return []

        sample_image = item.get("sample_image")
        test_image = item.get("test_image")
        analysis = item.get("analysis")
        if (
            not isinstance(sample_image, np.ndarray)
            or sample_image.size == 0
            or not isinstance(test_image, np.ndarray)
            or test_image.size == 0
            or not isinstance(analysis, dict)
        ):
            return []

        resolved.append(
            {
                "lighting_mode": mode,
                "sample_image": sample_image.copy(),
                "test_image": test_image.copy(),
                "source_frame": _image_snapshot(item.get("source_frame")),
                "analysis": analysis,
            }
        )
    return resolved


def _decision_task(panel, normalized: str, source: str, ai_decision: str) -> dict:
    event_id = str(
        getattr(panel, "adhesive_multilight_primary_event_id", "")
        or getattr(panel, "adhesive_multilight_last_event_id", "")
        or ""
    ).strip()
    multilight_samples = _multilight_learning_snapshot(panel)

    info = dict(getattr(panel, "current_aoi_info", {}) or {})
    analysis = getattr(panel, "current_analysis", None) or {}
    # Capturas NOVAS confirmadas por humano alimentam a CNN inclusive
    # quando o operador concorda com a IA (não só hard negatives).
    from src.services.neural_online_learning import eligible_online_case
    eligible = eligible_online_case({
        "aoi_info": info, "analysis": analysis,
        "source": source, "label": normalized,
    })

    return {
        "ng_image": _image_snapshot(getattr(panel, "current_ng", None)),
        "label": normalized,
        "sample_image": _image_snapshot(getattr(panel, "current_sample", None)),
        "aoi_info": info,
        "analysis": analysis,
        "save_images": bool(eligible or ai_decision != normalized),
        "source": source,
        "ai_decision": ai_decision,
        "event_id": event_id,
        "multilight_samples": multilight_samples,
    }


def _submit_decision_persistence(panel, task: dict) -> None:
    """Enfileira sem bloquear a thread Qt.

    Testes podem fornecer um submitter substituto para observar a submissão sem
    iniciar trabalho real em background.
    """
    submitter = getattr(panel, "_anomaly_persistence_submitter", None)
    if callable(submitter):
        submitter(task)
        return

    queue = getattr(panel, "_decision_persistence_queue", None)
    if queue is None:
        queue = DecisionPersistenceQueue(getattr(panel, "orchestrator", None))
        panel._decision_persistence_queue = queue
    queue.submit(task)


def _finish_operator_decision_ui(panel, normalized: str, source: str) -> None:
    """Conclui visualmente a peça sem esperar disco/KNN."""
    for name in ("btn_save_ok", "btn_save_ng", "btn_skip"):
        button = getattr(panel, name, None)
        if button is not None:
            button.setEnabled(False)

    panel.is_locked = False

    if getattr(panel, "capture_cycle_source", None) == "network":
        prepare_next = getattr(panel, "prepare_for_next_network_image", None)
        if callable(prepare_next):
            prepare_next()
    else:
        start = getattr(panel, "btn_start", None)
        if start is not None:
            start.setText("Capturar Local (MSS)")
            start.setEnabled(True)
        update_status = getattr(panel, "update_brain_status", None)
        if callable(update_status):
            update_status("Sistema Ocioso", False)

    update_history = getattr(panel, "update_history_status", None)
    if callable(update_history):
        update_history(normalized, source)


def install_anomaly_learning(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_anomaly_learning_installed", False):
        return

    original_save_label = control_panel_cls.save_label

    def save_label(self, user_decision: str, source="button"):
        normalized = str(user_decision or "").strip().upper()
        if normalized not in {"OK", "NG"} or self.current_ng is None:
            return None

        # Produção automática não cria rótulos humanos para si mesma.
        # "production_auto" identifica explicitamente o novo controlador visual;
        # "auto" é mantido apenas por compatibilidade com fluxos antigos.
        if str(source or "").strip().lower() in {"auto", "production_auto"}:
            return original_save_label(self, normalized, source=source)

        if source == "button":
            self.send_command_to_xp("0" if normalized == "OK" else "1")

        ai_decision = (
            "NG"
            if self.current_analysis
            and self.current_analysis.get("is_defect", False)
            else "OK"
        )

        # Copia tudo o que o worker precisa ANTES de limpar a captura atual.
        # A fila é serial: decisões sucessivas são persistidas na mesma ordem.
        task = _decision_task(
            self,
            normalized=normalized,
            source=source,
            ai_decision=ai_decision,
        )

        # Caminho crítico termina aqui: a peça já está julgada. A interface pode
        # limpar o resultado e o gate externo pode liberar a próxima imagem.
        _finish_operator_decision_ui(self, normalized, source)

        # Disco + recarga KNN deixam de bloquear o operador e o receptor XP.
        _submit_decision_persistence(self, task)
        return None

    control_panel_cls.save_label = save_label
    control_panel_cls._anomaly_learning_installed = True


__all__ = [
    "LIGHTING_ORDER",
    "_decision_task",
    "_multilight_learning_snapshot",
    "_finish_operator_decision_ui",
    "_submit_decision_persistence",
    "install_anomaly_learning",
]
