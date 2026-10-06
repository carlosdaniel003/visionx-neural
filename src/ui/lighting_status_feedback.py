"""Overlay da iluminação atual sincronizado com o resultado final da análise."""

from __future__ import annotations

from functools import wraps

from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from src.ui.decision_verdict_feedback import (
    VERDICT_FEEDBACK_FADE_IN_MS,
    VERDICT_FEEDBACK_FADE_OUT_MS,
    VERDICT_FEEDBACK_SLIDE_PX,
    verdict_feedback_state,
)


LIGHTING_STATUS_WIDTH = 300
LIGHTING_STATUS_HEIGHT = 88
LIGHTING_STATUS_MARGIN = 24
LIGHTING_STATUS_TOP_OFFSET = 184
LIGHTING_STATUS_FADE_IN_MS = VERDICT_FEEDBACK_FADE_IN_MS
LIGHTING_STATUS_FADE_OUT_MS = VERDICT_FEEDBACK_FADE_OUT_MS
LIGHTING_STATUS_SLIDE_PX = VERDICT_FEEDBACK_SLIDE_PX

LIGHTING_COMMAND_BY_MODE = {
    "TOP": "LEFT",
    "SIDE": "DOWN",
    "MID": "RIGHT",
}

LIGHTING_KEY_SYMBOL = {
    "TOP": "←",
    "SIDE": "↓",
    "MID": "→",
}

LIGHTING_STATUS_STYLESHEET = """
QFrame#lightingStatusFeedback {
    background-color: rgba(13, 13, 13, 248);
    border: 1px solid #f5c518;
    border-radius: 8px;
}
QLabel#lightingStatusHeader {
    color: #f5c518;
    background: transparent;
    border: none;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#lightingStatusValue {
    color: #f5f5f5;
    background: transparent;
    border: none;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#lightingStatusKey {
    color: #f5c518;
    background: transparent;
    border: none;
    font-size: 28px;
    font-weight: 900;
}
"""


class LightingStatusOverlay(QFrame):
    """Card da iluminação que acompanha o ciclo visual do veredito final."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.current_mode = ""
        self._decision_dismiss_pending = False

        self.setObjectName("lightingStatusFeedback")
        self.setFixedSize(LIGHTING_STATUS_WIDTH, LIGHTING_STATUS_HEIGHT)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(LIGHTING_STATUS_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 9, 14, 9)
        root.setSpacing(0)

        header = QLabel("ILUMINAÇÃO ATUAL")
        header.setObjectName("lightingStatusHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        self.value_label = QLabel("-")
        self.value_label.setObjectName("lightingStatusValue")
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.key_label = QLabel("")
        self.key_label.setObjectName("lightingStatusKey")
        self.key_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        body.addStretch(1)
        body.addWidget(self.value_label)
        body.addWidget(self.key_label)
        body.addStretch(1)

        root.addWidget(header)
        root.addLayout(body, 1)

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_in = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_in.setDuration(LIGHTING_STATUS_FADE_IN_MS)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(LIGHTING_STATUS_FADE_IN_MS)
        self._slide_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_out.setDuration(LIGHTING_STATUS_FADE_OUT_MS)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_out.finished.connect(self._finish_hide)

        self.panel.installEventFilter(self)
        self.hide()

    @staticmethod
    def _normalize(mode: str) -> str:
        normalized = str(mode or "").strip().upper()
        return normalized if normalized in LIGHTING_COMMAND_BY_MODE else ""

    def _top_right_position(self) -> QPoint:
        max_x = max(0, self.panel.width() - self.width())
        max_y = max(0, self.panel.height() - self.height())
        x = max(
            0,
            self.panel.width() - self.width() - LIGHTING_STATUS_MARGIN,
        )
        y = min(LIGHTING_STATUS_TOP_OFFSET, max_y)
        return QPoint(min(x, max_x), y)

    def reposition(self) -> None:
        self.move(self._top_right_position())
        if not self.isHidden():
            self.raise_()

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
        }:
            QTimer.singleShot(0, self.reposition)
        return super().eventFilter(watched, event)

    def _stop_motion(self) -> None:
        self._fade_in.stop()
        self._slide_in.stop()
        self._fade_out.stop()

    def _finish_hide(self) -> None:
        self.hide()
        self._opacity_effect.setOpacity(0.0)
        self._decision_dismiss_pending = False

    def set_lighting(self, mode: str) -> bool:
        """Atualiza o estado conhecido sem tornar o card visível por si só."""
        normalized = self._normalize(mode)
        if not normalized:
            return False

        self.current_mode = normalized
        self.value_label.setText(normalized)
        self.key_label.setText(LIGHTING_KEY_SYMBOL[normalized])

        if not self.isHidden():
            self.reposition()
        return True

    def show_analysis(self, analysis: dict | None) -> bool:
        """Mostra a iluminação somente quando existe um veredito final válido."""
        message, _tone = verdict_feedback_state(analysis)
        if not message:
            return False

        mode = self.current_mode
        if not mode and hasattr(self.panel, "lbl_light_value"):
            mode = self.panel.lbl_light_value.text().strip().upper()
        if not self.set_lighting(mode):
            return False

        self._decision_dismiss_pending = False
        self._stop_motion()

        target = self._top_right_position()
        max_x = max(0, self.panel.width() - self.width())
        start = QPoint(
            min(max_x, target.x() + LIGHTING_STATUS_SLIDE_PX),
            target.y(),
        )

        self._opacity_effect.setOpacity(0.0)
        self.move(start)
        self._slide_in.setStartValue(start)
        self._slide_in.setEndValue(target)

        self.raise_()
        self.show()
        self._fade_in.start()
        self._slide_in.start()
        return True

    def prepare_decision_dismissal(self) -> bool:
        """Preserva o card até o mesmo fade-out usado pelo resultado e 0/1."""
        if self.isHidden() or not self.current_mode:
            return False
        self._decision_dismiss_pending = True
        return True

    def start_synchronized_fade_out(self) -> bool:
        """Inicia a saída junto com o veredito e o feedback de decisão."""
        if (
            not self._decision_dismiss_pending
            or self.isHidden()
            or not self.current_mode
        ):
            return False

        self._fade_out.stop()
        self._fade_out.setStartValue(self._opacity_effect.opacity())
        self._fade_out.setEndValue(0.0)
        self._fade_out.start()
        return True

    def clear_status(self, force: bool = False) -> bool:
        """Oculta em resets normais, mas respeita uma saída 0/1 pendente."""
        if self._decision_dismiss_pending and not force:
            return False

        self._stop_motion()
        self._finish_hide()
        return True


def install_lighting_status_feedback_hooks(control_panel_cls) -> None:
    """Sincroniza iluminação, resultado final e reset sem alterar a decisão."""
    if getattr(control_panel_cls, "_lighting_status_feedback_hooks", False):
        return

    original_change_lighting = control_panel_cls.change_lighting
    original_reference_update = getattr(
        control_panel_cls,
        "_update_reference_panel",
        None,
    )
    original_reset = getattr(
        control_panel_cls,
        "_reset_confidence_panel",
        None,
    )

    @wraps(original_change_lighting)
    def change_lighting(self, light_mode: str, source: str):
        normalized_mode = str(light_mode or "").strip().upper()
        result = original_change_lighting(self, normalized_mode, source)
        if result is False:
            return result

        update_status = getattr(self, "update_lighting_status_feedback", None)
        if callable(update_status):
            update_status(normalized_mode)

        presenter = getattr(self, "_operational_controls", None)
        note_action = getattr(presenter, "note_action", None)
        if callable(note_action):
            note_action("Iluminação {0} selecionada.".format(normalized_mode))

        normalized_source = str(source or "").strip().lower()
        if normalized_source in {
            "local",
            "button",
            "odin_control",
            "odin_keyboard",
            "network",
            "xp_keyboard",
        }:
            show_key = getattr(self, "show_operational_key_feedback", None)
            command = LIGHTING_COMMAND_BY_MODE.get(normalized_mode)
            if callable(show_key) and command:
                if normalized_source in {"network", "xp_keyboard"}:
                    visual_source = "xp_keyboard"
                elif normalized_source == "odin_keyboard":
                    visual_source = "odin_keyboard"
                else:
                    visual_source = "odin_control"
                show_key(command, source=visual_source)

        return result

    control_panel_cls.change_lighting = change_lighting

    if callable(original_reference_update):
        @wraps(original_reference_update)
        def wrapped_reference_update(self, analysis):
            result = original_reference_update(self, analysis)
            show_status = getattr(
                self,
                "show_lighting_status_feedback",
                None,
            )
            if callable(show_status):
                show_status(analysis)
            return result

        control_panel_cls._update_reference_panel = wrapped_reference_update

    if callable(original_reset):
        @wraps(original_reset)
        def wrapped_reset(self):
            result = original_reset(self)
            clear_status = getattr(
                self,
                "clear_lighting_status_feedback",
                None,
            )
            if callable(clear_status):
                clear_status()
            return result

        control_panel_cls._reset_confidence_panel = wrapped_reset

    control_panel_cls._lighting_status_feedback_hooks = True


def install_lighting_status_feedback(panel) -> None:
    """Cria o card oculto e mantém apenas o estado atual até a análise terminar."""
    if getattr(panel, "_lighting_status_feedback_installed", False):
        return

    overlay = LightingStatusOverlay(panel)
    panel.lighting_status_feedback = overlay
    panel.update_lighting_status_feedback = overlay.set_lighting
    panel.show_lighting_status_feedback = overlay.show_analysis
    panel.clear_lighting_status_feedback = overlay.clear_status
    panel.prepare_lighting_status_feedback_dismissal = (
        overlay.prepare_decision_dismissal
    )
    panel.start_lighting_status_feedback_fade_out = (
        overlay.start_synchronized_fade_out
    )
    panel._lighting_status_feedback_installed = True

    current = "SIDE"
    if hasattr(panel, "lbl_light_value"):
        current = panel.lbl_light_value.text().strip().upper() or "SIDE"
    overlay.set_lighting(current)


__all__ = [
    "LIGHTING_COMMAND_BY_MODE",
    "LIGHTING_KEY_SYMBOL",
    "LIGHTING_STATUS_FADE_IN_MS",
    "LIGHTING_STATUS_FADE_OUT_MS",
    "LIGHTING_STATUS_HEIGHT",
    "LIGHTING_STATUS_MARGIN",
    "LIGHTING_STATUS_SLIDE_PX",
    "LIGHTING_STATUS_TOP_OFFSET",
    "LIGHTING_STATUS_WIDTH",
    "LightingStatusOverlay",
    "install_lighting_status_feedback",
    "install_lighting_status_feedback_hooks",
]
