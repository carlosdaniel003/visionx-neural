"""Feedback visual temporário para teclas operacionais do ODIN e do Windows XP.

É uma camada exclusivamente visual. Não decide, não salva, não envia comandos e
não altera gate, memória, confiança ou persistência.
"""

from __future__ import annotations

import time
from functools import wraps

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
)


FEEDBACK_DURATION_MS = 800
FEEDBACK_FADE_IN_MS = 120
FEEDBACK_FADE_OUT_MS = 160
FEEDBACK_SLIDE_PX = 8
DUPLICATE_SUPPRESSION_SECONDS = 1.5
FEEDBACK_SIZE = 180
FEEDBACK_MARGIN = 24

LIGHTING_KEY_PRESENTATION = {
    "LEFT": ("←", "ESQUERDA"),
    "DOWN": ("↓", "BAIXO"),
    "RIGHT": ("→", "DIREITA"),
}


FEEDBACK_STYLESHEET = """
QFrame#decisionKeyFeedback {
    background-color: rgba(13, 13, 13, 248);
    border: 1px solid #f5c518;
    border-radius: 8px;
}
QFrame#decisionKeyFeedback[tone="ok"],
QFrame#decisionKeyFeedback[tone="ng"],
QFrame#decisionKeyFeedback[tone="light"] {
    border: 1px solid #f5c518;
}
QLabel#decisionKeyHeader {
    color: #f5c518;
    background: transparent;
    border: none;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#decisionKeyDigit {
    background: transparent;
    border: none;
    font-size: 54px;
    font-weight: 900;
}
QLabel#decisionKeyDigit[tone="ok"],
QLabel#decisionKeyLabel[tone="ok"] {
    color: #4ade80;
}
QLabel#decisionKeyDigit[tone="ng"],
QLabel#decisionKeyLabel[tone="ng"] {
    color: #ff6262;
}
QLabel#decisionKeyDigit[tone="light"],
QLabel#decisionKeyLabel[tone="light"] {
    color: #f5c518;
}
QLabel#decisionKeyLabel {
    background: transparent;
    border: none;
    font-size: 18px;
    font-weight: 900;
    letter-spacing: 1px;
}
QLabel#decisionKeySource {
    color: #b6b6b6;
    background: transparent;
    border: none;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.8px;
}
"""


class DecisionKeyFeedbackOverlay(QFrame):
    """Quadrado leve que confirma visualmente a tecla recebida/enviada."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._last_feedback = ""
        self._last_shown_at = 0.0
        self._sync_verdict_on_exit = False

        self.setObjectName("decisionKeyFeedback")
        self.setFixedSize(FEEDBACK_SIZE, FEEDBACK_SIZE)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(FEEDBACK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(2)

        self.header_label = QLabel("TECLA PRESSIONADA")
        self.header_label.setObjectName("decisionKeyHeader")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.digit_label = QLabel("-")
        self.digit_label.setObjectName("decisionKeyDigit")
        self.digit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.decision_label = QLabel("-")
        self.decision_label.setObjectName("decisionKeyLabel")
        self.decision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source_label = QLabel("")
        self.source_label.setObjectName("decisionKeySource")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.header_label)
        layout.addStretch(1)
        layout.addWidget(self.digit_label)
        layout.addWidget(self.decision_label)
        layout.addWidget(self.source_label)
        layout.addStretch(1)

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_in = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_in.setDuration(FEEDBACK_FADE_IN_MS)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(FEEDBACK_FADE_IN_MS)
        self._slide_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_out.setDuration(FEEDBACK_FADE_OUT_MS)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_out.finished.connect(self._finish_hide)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._start_fade_out)
        self.hide()

    @staticmethod
    def _normalize(decision: str) -> str:
        value = str(decision or "").strip().upper()
        return value if value in {"OK", "NG"} else ""

    @staticmethod
    def _source_text(source: str) -> str:
        normalized = str(source or "").strip().lower()
        if normalized == "xp_keyboard":
            return "TECLADO WINDOWS XP"
        if normalized == "odin_keyboard":
            return "TECLADO ODIN"
        if normalized == "odin_control":
            return "CONTROLE ODIN"
        return "AÇÃO MANUAL"

    @staticmethod
    def _header_text(source: str) -> str:
        normalized = str(source or "").strip().lower()
        if normalized == "odin_control":
            return "TECLA ENVIADA"
        if normalized == "xp_keyboard":
            return "TECLA RECEBIDA"
        return "TECLA PRESSIONADA"

    @staticmethod
    def _refresh_style(widget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _bottom_right_position(self) -> QPoint:
        width = self.width()
        height = self.height()
        x = max(0, self.panel.width() - width - FEEDBACK_MARGIN)
        y = max(0, self.panel.height() - height - FEEDBACK_MARGIN)
        return QPoint(x, y)

    def _stop_motion(self) -> None:
        self._hide_timer.stop()
        self._fade_in.stop()
        self._slide_in.stop()
        self._fade_out.stop()

    def _start_fade_out(self) -> None:
        self._fade_out.stop()
        self._fade_out.setStartValue(self._opacity_effect.opacity())
        self._fade_out.setEndValue(0.0)
        self._fade_out.start()

        if self._sync_verdict_on_exit:
            for callback_name in (
                "start_ai_verdict_feedback_fade_out",
                "start_lighting_status_feedback_fade_out",
            ):
                start_result_fade = getattr(
                    self.panel,
                    callback_name,
                    None,
                )
                if callable(start_result_fade):
                    start_result_fade()

    def _finish_hide(self) -> None:
        self.hide()
        self._opacity_effect.setOpacity(0.0)
        self._sync_verdict_on_exit = False

    def _show_feedback(
        self,
        *,
        signature: str,
        key_text: str,
        label_text: str,
        source: str,
        tone: str,
        synchronize_verdict: bool,
        before_show=None,
    ) -> bool:
        now = time.monotonic()
        if (
            signature == self._last_feedback
            and now - self._last_shown_at < DUPLICATE_SUPPRESSION_SECONDS
        ):
            return False

        if callable(before_show):
            before_show()

        self._last_feedback = signature
        self._last_shown_at = now
        self._sync_verdict_on_exit = bool(synchronize_verdict)

        self.header_label.setText(self._header_text(source))
        self.digit_label.setText(key_text)
        self.decision_label.setText(label_text)
        self.source_label.setText(self._source_text(source))

        for widget in (self, self.digit_label, self.decision_label):
            widget.setProperty("tone", tone)
            self._refresh_style(widget)

        self._stop_motion()

        target = self._bottom_right_position()
        max_y = max(0, self.panel.height() - self.height())
        start = QPoint(
            target.x(),
            min(max_y, target.y() + FEEDBACK_SLIDE_PX),
        )

        self._opacity_effect.setOpacity(0.0)
        self.move(start)
        self._slide_in.setStartValue(start)
        self._slide_in.setEndValue(target)

        self.raise_()
        self.show()
        self._fade_in.start()
        self._slide_in.start()

        hold_before_fade = max(
            0,
            FEEDBACK_DURATION_MS - FEEDBACK_FADE_OUT_MS,
        )
        self._hide_timer.start(hold_before_fade)
        return True

    def show_decision(self, decision: str, source: str = "") -> bool:
        normalized = self._normalize(decision)
        if not normalized:
            return False

        def prepare_result_overlays():
            prepared = False
            for callback_name in (
                "prepare_ai_verdict_feedback_dismissal",
                "prepare_lighting_status_feedback_dismissal",
            ):
                prepare_overlay = getattr(
                    self.panel,
                    callback_name,
                    None,
                )
                if callable(prepare_overlay):
                    prepared = bool(prepare_overlay()) or prepared
            return prepared

        digit = "0" if normalized == "OK" else "1"
        tone = "ok" if normalized == "OK" else "ng"
        return self._show_feedback(
            signature="decision:{0}".format(normalized),
            key_text=digit,
            label_text=normalized,
            source=source,
            tone=tone,
            synchronize_verdict=True,
            before_show=prepare_result_overlays,
        )

    def show_key(self, key_name: str, source: str = "odin_keyboard") -> bool:
        normalized = str(key_name or "").strip().upper()
        presentation = LIGHTING_KEY_PRESENTATION.get(normalized)
        if presentation is None:
            return False

        key_text, label_text = presentation
        return self._show_feedback(
            signature="key:{0}".format(normalized),
            key_text=key_text,
            label_text=label_text,
            source=source,
            tone="light",
            synchronize_verdict=False,
        )


def install_decision_key_feedback_hooks(control_panel_cls) -> None:
    """Mostra o feedback quando 0/1 chega fisicamente pelo Windows XP."""
    if getattr(control_panel_cls, "_decision_key_feedback_hooks", False):
        return

    original_handle_keyboard = control_panel_cls.handle_physical_keyboard

    @wraps(original_handle_keyboard)
    def handle_physical_keyboard(self, comando_xp: str):
        normalized = str(comando_xp or "").strip().upper()
        had_active_capture = bool(getattr(self, "current_ng", None) is not None)

        if normalized in {"OK", "NG"} and had_active_capture:
            show_feedback = getattr(
                self,
                "show_decision_key_feedback",
                None,
            )
            if callable(show_feedback):
                show_feedback(normalized, source="xp_keyboard")

        return original_handle_keyboard(self, comando_xp)

    control_panel_cls.handle_physical_keyboard = handle_physical_keyboard
    control_panel_cls._decision_key_feedback_hooks = True


def install_decision_key_feedback(panel) -> None:
    """Cria uma única instância do overlay e expõe os acionadores ao painel."""
    if getattr(panel, "_decision_key_feedback_installed", False):
        return

    overlay = DecisionKeyFeedbackOverlay(panel)
    panel.decision_key_feedback = overlay
    panel.show_decision_key_feedback = overlay.show_decision
    panel.show_operational_key_feedback = overlay.show_key
    panel._decision_key_feedback_installed = True


__all__ = [
    "DUPLICATE_SUPPRESSION_SECONDS",
    "FEEDBACK_DURATION_MS",
    "FEEDBACK_FADE_IN_MS",
    "FEEDBACK_FADE_OUT_MS",
    "FEEDBACK_SLIDE_PX",
    "FEEDBACK_SIZE",
    "FEEDBACK_MARGIN",
    "LIGHTING_KEY_PRESENTATION",
    "DecisionKeyFeedbackOverlay",
    "install_decision_key_feedback",
    "install_decision_key_feedback_hooks",
]
