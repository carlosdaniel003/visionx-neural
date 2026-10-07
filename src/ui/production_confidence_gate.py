"""Política de intervenção humana do Modo Produção.

A versão atual não usa mais confiança mínima de 99% para autorizar OK
automático. O veredito final já renderizado é a referência:

- FALHA FALSA -> pode ser automatizado como 0/OK;
- DEFEITO REAL/NG -> intervenção humana;
- REVISÃO OBRIGATÓRIA -> intervenção humana.

A apresentação visual e o momento do envio automático pertencem ao
ProductionAutonomyController. Este módulo mantém apenas a trava operacional de
revisão e a compatibilidade com decisões 0/1 do operador.
"""

from __future__ import annotations

from typing import Any


# Mantido por compatibilidade com imports antigos. Não participa mais da política.
PRODUCTION_AUTO_CONFIDENCE_THRESHOLD = None


def normalized_confidence(analysis: dict | None) -> float:
    """Mantém a confiança como telemetria, sem usá-la como gate de decisão."""
    try:
        value = float((analysis or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    return max(0.0, min(1.0, value))


def _review_required(analysis: dict | None) -> bool:
    if not isinstance(analysis, dict):
        return True

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}

    return bool(
        analysis.get("production_review_required", False)
        or detail.get("operator_review_required", False)
        or trace.get("operator_review_required", False)
    )


def _verdict(analysis: dict | None) -> str:
    if not isinstance(analysis, dict):
        return "REVISÃO OBRIGATÓRIA"

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict in {
        "FALHA FALSA",
        "DEFEITO REAL",
        "DEFEITO",
        "REVISÃO OBRIGATÓRIA",
    }:
        return "DEFEITO REAL" if verdict == "DEFEITO" else verdict

    if _review_required(analysis):
        return "REVISÃO OBRIGATÓRIA"

    if "is_defect" in analysis:
        return "DEFEITO REAL" if bool(analysis.get("is_defect")) else "FALHA FALSA"

    return "REVISÃO OBRIGATÓRIA"


def production_decision_policy(analysis: dict | None) -> dict[str, Any]:
    """Resolve a política v1 sem limiar mínimo de confiança."""
    verdict = _verdict(analysis)
    confidence = normalized_confidence(analysis)

    if verdict == "FALHA FALSA" and not _review_required(analysis):
        proposed_decision = "OK"
        auto_allowed = True
        operator_review_required = False
        reason = "false_failure_auto_ok"
    elif verdict == "DEFEITO REAL":
        proposed_decision = "NG"
        auto_allowed = False
        operator_review_required = True
        reason = "real_defect_manual_review"
    else:
        proposed_decision = ""
        auto_allowed = False
        operator_review_required = True
        reason = "mandatory_manual_review"

    return {
        "mode": "production",
        "confidence": confidence,
        "threshold": None,
        "verdict": verdict,
        "auto_allowed": bool(auto_allowed),
        "operator_review_required": bool(operator_review_required),
        "proposed_decision": proposed_decision,
        "reason": reason,
        "operator_shortcuts": {"0": "OK", "1": "NG"},
    }


def _record_policy(panel, policy: dict, resolution: str = "pending") -> None:
    panel.production_review_policy = dict(policy)
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        detail = analysis.setdefault("detail", {})
        detail["production_decision_policy"] = {
            **policy,
            "resolution": str(resolution),
        }
        analysis["production_review_required"] = bool(
            str(policy.get("verdict", "") or "").strip().upper()
            == "REVISÃO OBRIGATÓRIA"
            and resolution == "pending"
        )


def _clear_pending(panel) -> None:
    panel.production_review_pending = False
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        analysis["production_review_required"] = False


def _show_pending_message(panel) -> None:
    policy = getattr(panel, "production_review_policy", None)
    if not isinstance(policy, dict):
        policy = production_decision_policy(
            getattr(panel, "current_analysis", None)
        )

    verdict = str(policy.get("verdict", "") or "REVISÃO OBRIGATÓRIA")
    panel.update_brain_status(
        f"Intervenção necessária: {verdict}. "
        "Aguardando operador: 0=OK | 1=NG",
        True,
    )
    if hasattr(panel, "_operational_controls"):
        panel._operational_controls.sync(force=True)


def enter_production_review(
    panel,
    analysis: dict | None = None,
    *,
    verdict_override: str = "",
) -> dict[str, Any]:
    """Congela a peça e libera somente 0/1 humano."""
    target = analysis if isinstance(analysis, dict) else getattr(
        panel,
        "current_analysis",
        None,
    )
    policy = production_decision_policy(target)

    override = str(verdict_override or "").strip().upper()
    if override:
        if override not in {"DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}:
            override = "REVISÃO OBRIGATÓRIA"
        policy = {
            **policy,
            "verdict": override,
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if override == "DEFEITO REAL" else "",
            "reason": "forced_manual_intervention",
        }

    panel.production_review_pending = True
    panel.is_locked = True
    _record_policy(panel, policy, resolution="pending")
    _show_pending_message(panel)
    return policy


def install_production_confidence_gate(control_panel_cls, presenter_cls) -> None:
    """Mantém a trava humana enquanto o controlador gerencia a autonomia."""
    if getattr(control_panel_cls, "_production_confidence_gate_installed", False):
        return

    from PyQt6.QtCore import Qt

    original_init = control_panel_cls.__init__
    original_start_monitoring = control_panel_cls.start_monitoring
    original_handle_network_image = control_panel_cls.handle_network_image
    original_skip_image = control_panel_cls.skip_image
    original_save_label = control_panel_cls.save_label
    original_handle_physical_keyboard = control_panel_cls.handle_physical_keyboard
    original_key_press = control_panel_cls.keyPressEvent
    original_presenter_sync = presenter_cls.sync

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.production_review_pending = False
        self.production_review_policy = None

    def wrapped_start_monitoring(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_start_monitoring(self, *args, **kwargs)

    def wrapped_handle_network_image(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_handle_network_image(self, *args, **kwargs)

    def wrapped_skip_image(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_skip_image(self, *args, **kwargs)

    def wrapped_save_label(self, user_decision: str, source="button"):
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        normalized_source = str(source or "").strip().lower()
        had_pending = bool(getattr(self, "production_review_pending", False))

        if is_production and normalized_source in {"auto", "production_auto"}:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            if not (
                policy["auto_allowed"]
                and str(user_decision).strip().upper() == "OK"
            ):
                enter_production_review(
                    self,
                    getattr(self, "current_analysis", None),
                )
                return None
            _record_policy(self, policy, resolution="automatic")

        elif is_production and had_pending:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            resolution = (
                "operator_ok"
                if str(user_decision).upper() == "OK"
                else "operator_ng"
            )
            _record_policy(self, policy, resolution=resolution)

        _clear_pending(self)
        result = original_save_label(self, user_decision, source=source)

        if (
            is_production
            and had_pending
            and not bool(getattr(self, "is_locked", False))
        ):
            callback = getattr(
                self,
                "production_operator_decision_completed",
                None,
            )
            if callable(callback):
                callback(user_decision, source=source)

        return result

    def wrapped_handle_physical_keyboard(self, comando_xp: str):
        command = str(comando_xp or "").strip().upper()
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(self, "production_review_pending", False))

        if command in {"0", "OK"}:
            if is_production and not pending:
                return None
            return self.save_label("OK", source="xp_keyboard")
        if command in {"1", "NG"}:
            if is_production and not pending:
                return None
            return self.save_label("NG", source="xp_keyboard")
        if pending:
            _show_pending_message(self)
            return None
        return original_handle_physical_keyboard(self, comando_xp)

    def wrapped_key_press(self, event):
        pending = bool(getattr(self, "production_review_pending", False))
        if pending:
            if event.key() == Qt.Key.Key_0 and self.btn_save_ok.isEnabled():
                self.save_label("OK", source="button")
                event.accept()
                return
            if event.key() == Qt.Key.Key_1 and self.btn_save_ng.isEnabled():
                self.save_label("NG", source="button")
                event.accept()
                return
            event.accept()
            _show_pending_message(self)
            return
        return original_key_press(self, event)

    def wrapped_presenter_sync(self, force: bool = False):
        original_presenter_sync(self, force=force)
        panel = self.panel
        mode = panel.combo_mode.currentText() if hasattr(panel, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(panel, "production_review_pending", False))

        decision_buttons_visible = (not is_production) or pending
        panel.btn_save_ok.setVisible(decision_buttons_visible)
        panel.btn_save_ng.setVisible(decision_buttons_visible)

        if not (is_production and pending):
            return

        self._set_enabled(panel.btn_start, False)
        self._set_enabled(panel.btn_skip, False)
        self._set_enabled(panel.btn_save_ok, True)
        self._set_enabled(panel.btn_save_ng, True)

        for button in (
            panel.btn_light_mid,
            panel.btn_light_side,
            panel.btn_light_top,
        ):
            self._set_enabled(button, False)

        if hasattr(panel, "btn_clear_dataset"):
            self._set_enabled(panel.btn_clear_dataset, False)

        self._set_text(
            panel.btn_start,
            "Captura congelada — aguardando operador",
        )
        self._set_text(panel.btn_save_ok, "0 - Aprovar como OK")
        self._set_text(panel.btn_save_ng, "1 - Confirmar defeito NG")

        policy = getattr(panel, "production_review_policy", {}) or {}
        verdict = str(
            policy.get("verdict", "REVISÃO OBRIGATÓRIA") or
            "REVISÃO OBRIGATÓRIA"
        )

        panel.lbl_operation_state.setText("INTERVENÇÃO NECESSÁRIA")
        panel.lbl_operation_state.setProperty("tone", "attention")
        self._refresh_style(panel.lbl_operation_state)
        panel.lbl_operation_hint.setText(
            f"{verdict}. Pressione 0 para OK ou 1 para NG."
        )
        panel.lbl_operation_actions.setText("2 AÇÕES DISPONÍVEIS")
        self.last_state_name = "production_review"

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.handle_physical_keyboard = wrapped_handle_physical_keyboard
    control_panel_cls.keyPressEvent = wrapped_key_press
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._production_confidence_gate_installed = True


__all__ = [
    "PRODUCTION_AUTO_CONFIDENCE_THRESHOLD",
    "enter_production_review",
    "install_production_confidence_gate",
    "normalized_confidence",
    "production_decision_policy",
]
