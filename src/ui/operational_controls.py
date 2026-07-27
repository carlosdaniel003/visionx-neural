"""Camada visual e de estado dos Controles Operacionais."""

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy

from src.ui.operational_controls_model import (
    available_action_count,
    operational_state,
)


CONTROL_STYLESHEET = r"""
QFrame#operationStateBar {
    background-color: #101010;
    border: 1px solid #303030;
    border-radius: 8px;
}
QLabel#operationStateBadge {
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#operationStateBadge[tone="ready"] {
    color: #07120b;
    background-color: #4ade80;
    border: 1px solid #4ade80;
}
QLabel#operationStateBadge[tone="attention"] {
    color: #090909;
    background-color: #f5c518;
    border: 1px solid #f5c518;
}
QLabel#operationStateBadge[tone="busy"] {
    color: #f5c518;
    background-color: #2b2406;
    border: 1px solid #f5c518;
}
QLabel#operationStateHint {
    color: #d0d0d0;
    font-size: 11px;
    font-weight: 700;
}
QLabel#operationActionCount {
    color: #737373;
    font-size: 10px;
    font-weight: 800;
}

QPushButton#captureActionButton,
QPushButton#discardActionButton,
QPushButton#approveActionButton,
QPushButton#rejectActionButton,
QPushButton#cameraLightButton,
QPushButton#datasetDangerButton {
    min-height: 42px;
    border-radius: 8px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 800;
}

QPushButton#captureActionButton {
    color: #080808;
    background-color: #f5c518;
    border: 1px solid #f5c518;
}
QPushButton#captureActionButton:hover {
    background-color: #ffd84d;
    border-color: #fff0a0;
}
QPushButton#captureActionButton:pressed {
    background-color: #c99f0b;
    border: 2px solid #fff0a0;
    padding-top: 10px;
}
QPushButton#captureActionButton:focus {
    border: 2px solid #ffffff;
}

QPushButton#discardActionButton {
    color: #e2e2e2;
    background-color: #181818;
    border: 1px solid #444444;
}
QPushButton#discardActionButton:hover {
    color: #f5c518;
    background-color: #242424;
    border-color: #f5c518;
}
QPushButton#discardActionButton:pressed {
    background-color: #080808;
    border: 2px solid #f5c518;
    padding-top: 10px;
}

QPushButton#approveActionButton {
    color: #4ade80;
    background-color: #0b1710;
    border: 1px solid #4ade80;
}
QPushButton#approveActionButton:hover {
    color: #07120b;
    background-color: #4ade80;
}
QPushButton#approveActionButton:pressed {
    background-color: #2fae5c;
    border: 2px solid #b7f7cb;
    padding-top: 10px;
}

QPushButton#rejectActionButton {
    color: #ff7777;
    background-color: #1b0d0d;
    border: 1px solid #ff6262;
}
QPushButton#rejectActionButton:hover {
    color: #140606;
    background-color: #ff6262;
}
QPushButton#rejectActionButton:pressed {
    background-color: #c64141;
    border: 2px solid #ffd0d0;
    padding-top: 10px;
}

QPushButton#cameraLightButton {
    color: #d8d8d8;
    background-color: #171717;
    border: 1px solid #3b3b3b;
}
QPushButton#cameraLightButton:hover {
    color: #f5c518;
    background-color: #222222;
    border-color: #f5c518;
}
QPushButton#cameraLightButton:pressed {
    background-color: #2b2406;
    border: 2px solid #f5c518;
}
QPushButton#cameraLightButton[activeLight="true"] {
    color: #080808;
    background-color: #f5c518;
    border: 2px solid #ffd84d;
}

QPushButton#datasetDangerButton {
    color: #ff7777;
    background-color: #170b0b;
    border: 1px solid #9f3838;
}
QPushButton#datasetDangerButton:hover {
    color: #ffffff;
    background-color: #6e2020;
    border-color: #ff7777;
}
QPushButton#datasetDangerButton:pressed {
    background-color: #3e1010;
    border: 2px solid #ffb0b0;
}

QPushButton#captureActionButton:disabled,
QPushButton#discardActionButton:disabled,
QPushButton#approveActionButton:disabled,
QPushButton#rejectActionButton:disabled,
QPushButton#cameraLightButton:disabled,
QPushButton#datasetDangerButton:disabled {
    color: #565656;
    background-color: #0b0b0b;
    border: 1px dashed #292929;
}
"""


class OperationalControlsPresenter:
    """Sincroniza permissões, cursores, textos e feedback visual."""

    def __init__(self, panel):
        self.panel = panel
        self.transient_message = ""
        self.transient_until = 0.0
        self.last_state_name = None

        self._configure_buttons()
        self._build_state_bar()
        self._connect_feedback()

        self.timer = QTimer(panel)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.sync)
        self.timer.start()
        self.sync(force=True)

    @staticmethod
    def _refresh_style(widget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    @staticmethod
    def _set_text(button, text: str) -> None:
        if button.text() != text:
            button.setText(text)

    def _configure_buttons(self) -> None:
        panel = self.panel
        panel.btn_start.setObjectName("captureActionButton")
        panel.btn_skip.setObjectName("discardActionButton")
        panel.btn_save_ok.setObjectName("approveActionButton")
        panel.btn_save_ng.setObjectName("rejectActionButton")

        for button in (
            panel.btn_light_mid,
            panel.btn_light_side,
            panel.btn_light_top,
        ):
            button.setObjectName("cameraLightButton")
            button.setProperty("activeLight", False)

        if hasattr(panel, "btn_clear_dataset"):
            panel.btn_clear_dataset.setObjectName("datasetDangerButton")

        panel.btn_start.setToolTip(
            "Inicia uma captura. Durante o processamento, os demais controles ficam bloqueados."
        )
        panel.btn_skip.setToolTip(
            "Descarta a captura atual sem registrar uma decisão no dataset."
        )
        panel.btn_save_ok.setToolTip(
            "Confirma que a peça está OK. Só fica disponível após a análise."
        )
        panel.btn_save_ng.setToolTip(
            "Confirma defeito NG. Só fica disponível após a análise."
        )
        panel.btn_light_mid.setToolTip(
            "Seleciona a iluminação MID. Atalho: seta para a esquerda."
        )
        panel.btn_light_side.setToolTip(
            "Seleciona a iluminação SIDE. Atalho: seta para baixo."
        )
        panel.btn_light_top.setToolTip(
            "Seleciona a iluminação TOP. Atalho: seta para a direita."
        )

        panel.btn_skip.setEnabled(False)
        panel.btn_save_ok.setEnabled(False)
        panel.btn_save_ng.setEnabled(False)

        panel.setStyleSheet(panel.styleSheet() + CONTROL_STYLESHEET)

    def _build_state_bar(self) -> None:
        panel = self.panel
        state_bar = QFrame()
        state_bar.setObjectName("operationStateBar")
        state_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        layout = QHBoxLayout(state_bar)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(9)

        badge = QLabel("PRONTO")
        badge.setObjectName("operationStateBadge")
        badge.setProperty("tone", "ready")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("Selecione a iluminação ou inicie uma nova captura.")
        hint.setObjectName("operationStateHint")
        hint.setWordWrap(True)
        hint.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        count = QLabel("0 AÇÕES DISPONÍVEIS")
        count.setObjectName("operationActionCount")
        count.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(badge)
        layout.addWidget(hint, stretch=1)
        layout.addWidget(count)

        controls_layout = panel.controls_section.layout()
        controls_layout.insertWidget(1, state_bar)

        panel.operation_state_bar = state_bar
        panel.lbl_operation_state = badge
        panel.lbl_operation_hint = hint
        panel.lbl_operation_actions = count

    def _connect_feedback(self) -> None:
        bindings = (
            (self.panel.btn_start, "Captura solicitada."),
            (self.panel.btn_skip, "Captura atual descartada."),
            (self.panel.btn_save_ok, "Decisão OK registrada."),
            (self.panel.btn_save_ng, "Decisão NG registrada."),
            (self.panel.btn_light_mid, "Iluminação MID selecionada."),
            (self.panel.btn_light_side, "Iluminação SIDE selecionada."),
            (self.panel.btn_light_top, "Iluminação TOP selecionada."),
        )
        if hasattr(self.panel, "btn_clear_dataset"):
            bindings += (
                (self.panel.btn_clear_dataset, "Limpeza do dataset solicitada."),
            )

        for button, message in bindings:
            button.clicked.connect(
                lambda _checked=False, text=message: self.note_action(text)
            )

        self.panel.combo_mode.currentTextChanged.connect(
            lambda _text: self.sync(force=True)
        )

    def note_action(self, message: str) -> None:
        self.transient_message = str(message)
        self.transient_until = time.monotonic() + 1.25
        self.sync(force=True)

    def _set_enabled(self, button, enabled: bool) -> None:
        enabled = bool(enabled)
        if button.isEnabled() != enabled:
            button.setEnabled(enabled)
        button.setCursor(
            Qt.CursorShape.PointingHandCursor
            if enabled
            else Qt.CursorShape.ArrowCursor
        )
        button.setProperty("available", enabled)

    def _sync_lighting_selection(self) -> None:
        current = ""
        if hasattr(self.panel, "lbl_light_value"):
            current = self.panel.lbl_light_value.text().strip().upper()

        mapping = {
            self.panel.btn_light_mid: "MID",
            self.panel.btn_light_side: "SIDE",
            self.panel.btn_light_top: "TOP",
        }
        for button, name in mapping.items():
            selected = current == name
            if bool(button.property("activeLight")) != selected:
                button.setProperty("activeLight", selected)
                self._refresh_style(button)

    def _apply_texts(self, state_name: str) -> None:
        start_text = {
            "idle": "Capturar nova peça",
            "processing": "Processando captura...",
            "review_test": "Nova captura (descarta a atual)",
            "review_shadow": "Forçar nova captura",
            "production_auto": "Enviando decisão automática...",
        }.get(state_name, "Capturar nova peça")

        self._set_text(self.panel.btn_start, start_text)
        self._set_text(self.panel.btn_skip, "Descartar captura")
        self._set_text(self.panel.btn_save_ok, "Aprovar como OK")
        self._set_text(self.panel.btn_save_ng, "Confirmar defeito NG")
        self._set_text(self.panel.btn_light_mid, "Luz MID  |  ←")
        self._set_text(self.panel.btn_light_side, "Luz SIDE  |  ↓")
        self._set_text(self.panel.btn_light_top, "Luz TOP  |  →")
        if hasattr(self.panel, "btn_clear_dataset"):
            self._set_text(
                self.panel.btn_clear_dataset,
                "Excluir dataset local",
            )

    def sync(self, force: bool = False) -> None:
        panel = self.panel
        mode = panel.combo_mode.currentText()
        state = operational_state(
            mode=mode,
            is_locked=bool(getattr(panel, "is_locked", False)),
            has_analysis=getattr(panel, "current_analysis", None) is not None,
        )

        self._set_enabled(panel.btn_start, state.capture_enabled)
        self._set_enabled(panel.btn_skip, state.discard_enabled)
        self._set_enabled(panel.btn_save_ok, state.approve_enabled)
        self._set_enabled(panel.btn_save_ng, state.reject_enabled)

        for button in (
            panel.btn_light_mid,
            panel.btn_light_side,
            panel.btn_light_top,
        ):
            self._set_enabled(button, state.lighting_enabled)

        if hasattr(panel, "btn_clear_dataset"):
            self._set_enabled(
                panel.btn_clear_dataset,
                state.dataset_clear_enabled,
            )

        self._apply_texts(state.name)
        self._sync_lighting_selection()

        if force or self.last_state_name != state.name:
            panel.lbl_operation_state.setText(state.badge)
            panel.lbl_operation_state.setProperty("tone", state.tone)
            self._refresh_style(panel.lbl_operation_state)
            self.last_state_name = state.name

        hint = (
            self.transient_message
            if time.monotonic() < self.transient_until
            else state.hint
        )
        panel.lbl_operation_hint.setText(hint)

        count = available_action_count(state)
        suffix = "AÇÃO DISPONÍVEL" if count == 1 else "AÇÕES DISPONÍVEIS"
        panel.lbl_operation_actions.setText(f"{count} {suffix}")


def install_operational_controls(panel) -> None:
    """Instala a UX dos controles depois dos botões administrativos."""
    if getattr(panel, "_operational_controls_installed", False):
        return
    panel._operational_controls = OperationalControlsPresenter(panel)
    panel._operational_controls_installed = True
