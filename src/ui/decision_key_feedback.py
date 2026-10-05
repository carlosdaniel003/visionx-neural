"""Feedback visual temporário para decisões 0/1 do ODIN e do Windows XP.

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


FEEDBACK_STYLESHEET = """
QFrame#decisionKeyFeedback {
    background-color: rgba(13, 13, 13, 248);
    border: 1px solid #f5c518;
    border-radius: 8px;
}
QFrame#decisionKeyFeedback[tone="ok"],
QFrame#decisionKeyFeedback[tone="ng"] {
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
    """Quadrado leve que confirma visualmente a tecla de julgamento recebida."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._last_decision = ""
        self._last_shown_at = 0.0

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

        self.header_label = QLabel("DECISÃO RECEBIDA")
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
        return "DECISÃO MANUAL"

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

        # O veredito da IA deve sair no mesmo instante e com a mesma curva
        # temporal do feedback 0/1.
        start_verdict_fade = getattr(
            self.panel,
            "start_ai_verdict_feedback_fade_out",
            None,
        )
        if callable(start_verdict_fade):
            start_verdict_fade()

    def _finish_hide(self) -> None:
        self.hide()
        self._opacity_effect.setOpacity(0.0)

    def show_decision(self, decision: str, source: str = "") -> bool:
        normalized = self._normalize(decision)
        if not normalized:
            return False

        now = time.monotonic()
        if (
            normalized == self._last_decision
            and now - self._last_shown_at < DUPLICATE_SUPPRESSION_SECONDS
        ):
            return False

        self._last_decision = normalized
        self._last_shown_at = now

        # Prepare o card de veredito antes do caminho produtivo consumir a
        # decisão. Assim resets internos não o apagam antes da animação 0/1.
        prepare_verdict = getattr(
            self.panel,
            "prepare_ai_verdict_feedback_dismissal",
            None,
        )
        if callable(prepare_verdict):
            prepare_verdict()

        digit = "0" if normalized == "OK" else "1"
        tone = "ok" if normalized == "OK" else "ng"

        self.digit_label.setText(digit)
        self.decision_label.setText(normalized)
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


def install_decision_key_feedback_hooks(control_panel_cls) -> None:
    """Mostra o feedback quando 0/1 chega fisicamente pelo Windows XP."""
    if getattr(control_panel_cls, "_decision_key_feedback_hooks", False):
        return

    original_handle_keyboard = control_panel_cls.handle_physical_keyboard

    @wraps(original_handle_keyboard)
    def handle_physical_keyboard(self, comando_xp: str):
        normalized = str(comando_xp or "").strip().upper()
        had_active_capture = bool(getattr(self, "current_ng", None) is not None)

        # Exibe/prepara antes do handler produtivo para preservar o veredito
        # durante o reset interno do ciclo. O comportamento funcional do
        # comando continua delegado integralmente ao handler original.
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
    """Cria uma única instância do overlay e expõe o acionador ao painel."""
    if getattr(panel, "_decision_key_feedback_installed", False):
        return

    overlay = DecisionKeyFeedbackOverlay(panel)
    panel.decision_key_feedback = overlay
    panel.show_decision_key_feedback = overlay.show_decision
    panel._decision_key_feedback_installed = True


__all__ = [
    "DUPLICATE_SUPPRESSION_SECONDS",
    "FEEDBACK_DURATION_MS",
    "FEEDBACK_FADE_IN_MS",
    "FEEDBACK_FADE_OUT_MS",
    "FEEDBACK_SLIDE_PX",
    "FEEDBACK_SIZE",
    "FEEDBACK_MARGIN",
    "DecisionKeyFeedbackOverlay",
    "install_decision_key_feedback",
    "install_decision_key_feedback_hooks",
]
