"""Feedback visual temporário para decisões 0/1 do ODIN e do Windows XP.

É uma camada exclusivamente visual. Não decide, não salva, não envia comandos e
não altera gate, memória, confiança ou persistência.
"""

from __future__ import annotations

import time
from functools import wraps

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


FEEDBACK_DURATION_MS = 800
DUPLICATE_SUPPRESSION_SECONDS = 1.5
FEEDBACK_SIZE = 180


FEEDBACK_STYLESHEET = """
QFrame#decisionKeyFeedback {
    background-color: rgba(8, 8, 8, 238);
    border: 3px solid #5a5a5a;
    border-radius: 8px;
}
QFrame#decisionKeyFeedback[tone="ok"] {
    border-color: #4ade80;
}
QFrame#decisionKeyFeedback[tone="ng"] {
    border-color: #ff6262;
}
QLabel#decisionKeyDigit {
    background: transparent;
    border: none;
    font-size: 58px;
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
    font-size: 20px;
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
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(2)

        self.digit_label = QLabel("-")
        self.digit_label.setObjectName("decisionKeyDigit")
        self.digit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.decision_label = QLabel("-")
        self.decision_label.setObjectName("decisionKeyLabel")
        self.decision_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source_label = QLabel("")
        self.source_label.setObjectName("decisionKeySource")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(self.digit_label)
        layout.addWidget(self.decision_label)
        layout.addWidget(self.source_label)
        layout.addStretch(1)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
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

    def _center_over_panel(self) -> None:
        width = self.width()
        height = self.height()
        x = max(0, (self.panel.width() - width) // 2)
        y = max(0, (self.panel.height() - height) // 2)
        self.move(x, y)

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

        digit = "0" if normalized == "OK" else "1"
        tone = "ok" if normalized == "OK" else "ng"

        self.digit_label.setText(digit)
        self.decision_label.setText(normalized)
        self.source_label.setText(self._source_text(source))

        for widget in (self, self.digit_label, self.decision_label):
            widget.setProperty("tone", tone)
            self._refresh_style(widget)

        self._center_over_panel()
        self.raise_()
        self.show()
        self._hide_timer.start(FEEDBACK_DURATION_MS)
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

        result = original_handle_keyboard(self, comando_xp)

        if normalized in {"OK", "NG"} and had_active_capture:
            show_feedback = getattr(
                self,
                "show_decision_key_feedback",
                None,
            )
            if callable(show_feedback):
                show_feedback(normalized, source="xp_keyboard")
        return result

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
    "FEEDBACK_SIZE",
    "DecisionKeyFeedbackOverlay",
    "install_decision_key_feedback",
    "install_decision_key_feedback_hooks",
]
