"""Trava de confiança para decisões autônomas no Modo Produção.

A IA só envia OK/NG automaticamente quando a confiança calculada pelo
orquestrador é igual ou superior a 99%. Abaixo disso, a captura permanece
congelada e somente a decisão humana 0=OK ou 1=NG é liberada.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


PRODUCTION_AUTO_CONFIDENCE_THRESHOLD = 0.99


def normalized_confidence(analysis: dict | None) -> float:
    """Obtém uma confiança finita no intervalo 0..1."""
    try:
        value = float((analysis or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if value != value:  # NaN
        value = 0.0
    return max(0.0, min(1.0, value))


def production_decision_policy(analysis: dict | None) -> dict[str, Any]:
    """Resolve se a inspeção pode ser julgada sem operador."""
    confidence = normalized_confidence(analysis)
    proposed_decision = "NG" if bool((analysis or {}).get("is_defect", False)) else "OK"
    auto_allowed = confidence + 1e-12 >= PRODUCTION_AUTO_CONFIDENCE_THRESHOLD
    return {
        "mode": "production",
        "confidence": confidence,
        "threshold": PRODUCTION_AUTO_CONFIDENCE_THRESHOLD,
        "auto_allowed": bool(auto_allowed),
        "operator_review_required": bool(not auto_allowed),
        "proposed_decision": proposed_decision,
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
            policy.get("operator_review_required", False)
            and resolution == "pending"
        )


def _clear_pending(panel) -> None:
    panel.production_review_pending = False
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        analysis["production_review_required"] = False


def install_production_confidence_gate(control_panel_cls, presenter_cls) -> None:
    """Instala a trava no controller e na apresentação dos controles."""
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
        _clear_pending(self)
        return original_start_monitoring(self, *args, **kwargs)

    def wrapped_handle_network_image(self, *args, **kwargs):
        _clear_pending(self)
        return original_handle_network_image(self, *args, **kwargs)

    def wrapped_skip_image(self, *args, **kwargs):
        _clear_pending(self)
        return original_skip_image(self, *args, **kwargs)

    def wrapped_save_label(self, user_decision: str, source="button"):
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"

        if is_production and source == "auto":
            policy = production_decision_policy(getattr(self, "current_analysis", None))
            if not policy["auto_allowed"]:
                self.production_review_pending = True
                self.is_locked = True
                _record_policy(self, policy, resolution="pending")
                confidence_pct = policy["confidence"] * 100.0
                self.update_brain_status(
                    f"Confiança {confidence_pct:.1f}% abaixo de 99% — aguardando operador: 0=OK | 1=NG",
                    True,
                )
                if hasattr(self, "_operational_controls"):
                    self._operational_controls.sync(force=True)
                return None

            _record_policy(self, policy, resolution="automatic")

        elif is_production and getattr(self, "production_review_pending", False):
            policy = production_decision_policy(getattr(self, "current_analysis", None))
            resolution = "operator_ok" if str(user_decision).upper() == "OK" else "operator_ng"
            _record_policy(self, policy, resolution=resolution)

        _clear_pending(self)
        return original_save_label(self, user_decision, source=source)

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
        return original_handle_physical_keyboard(self, comando_xp)

    def wrapped_key_press(self, event):
        pending = bool(getattr(self, "production_review_pending", False))
        if pending and event.key() in {Qt.Key.Key_0, Qt.Key.Key_1}:
            if event.key() == Qt.Key.Key_0 and self.btn_save_ok.isEnabled():
                self.save_label("OK", source="button")
                event.accept()
                return
            if event.key() == Qt.Key.Key_1 and self.btn_save_ng.isEnabled():
                self.save_label("NG", source="button")
                event.accept()
                return
        return original_key_press(self, event)

    def wrapped_presenter_sync(self, force: bool = False):
        original_presenter_sync(self, force=force)
        panel = self.panel
        mode = panel.combo_mode.currentText() if hasattr(panel, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(panel, "production_review_pending", False))

        # Em produção, OK/NG só aparecem quando a confiança exige operador.
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

        self._set_text(panel.btn_start, "Captura congelada — aguardando operador")
        self._set_text(panel.btn_save_ok, "0 - Aprovar como OK")
        self._set_text(panel.btn_save_ng, "1 - Confirmar defeito NG")

        panel.lbl_operation_state.setText("REVISÃO OBRIGATÓRIA")
        panel.lbl_operation_state.setProperty("tone", "attention")
        self._refresh_style(panel.lbl_operation_state)
        panel.lbl_operation_hint.setText(
            "Confiança abaixo de 99%. Pressione 0 para OK ou 1 para NG."
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
    "install_production_confidence_gate",
    "normalized_confidence",
    "production_decision_policy",
]
