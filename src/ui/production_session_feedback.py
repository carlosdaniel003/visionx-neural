"""Overlays do Modo Produção: contador da sessão e intervenção humana.

Camada visual. Não classifica, não envia comandos e não altera a análise.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
)

from src.ui.theme import ACCENT, DANGER, SUCCESS, SURFACE


SESSION_MARGIN = 24
SESSION_TOP_OFFSET = 84
SESSION_WIDTH = 250
SESSION_HEIGHT = 112

INTERVENTION_WIDTH = 360
INTERVENTION_HEIGHT = 154
INTERVENTION_TOP_OFFSET = 84


SESSION_STYLESHEET = f"""
QFrame#productionSessionFeedback {{
    background-color: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 8px;
}}
QLabel#productionSessionHeader {{
    color: {ACCENT};
    background: transparent;
    border: none;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 1px;
}}
QLabel#productionSessionOK {{
    color: {SUCCESS};
    background: transparent;
    border: none;
    font-size: 17px;
    font-weight: 900;
}}
QLabel#productionSessionNG {{
    color: {DANGER};
    background: transparent;
    border: none;
    font-size: 17px;
    font-weight: 900;
}}
"""

INTERVENTION_STYLESHEET = f"""
QFrame#productionInterventionFeedback {{
    background-color: {SURFACE};
    border: 1px solid {DANGER};
    border-radius: 8px;
}}
QLabel#productionInterventionHeader {{
    color: {ACCENT};
    background: transparent;
    border: none;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 1px;
}}
QLabel#productionInterventionReason {{
    color: {DANGER};
    background: transparent;
    border: none;
    font-size: 23px;
    font-weight: 900;
}}
QLabel#productionInterventionHint {{
    color: #d6d6d6;
    background: transparent;
    border: none;
    font-size: 11px;
    font-weight: 800;
}}
"""


class ProductionSessionFeedbackOverlay(QFrame):
    """Contador persistente somente enquanto o Modo Produção está ativo."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.ok_auto = 0
        self.ng_auto = 0

        self.setObjectName("productionSessionFeedback")
        self.setFixedSize(SESSION_WIDTH, SESSION_HEIGHT)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(SESSION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(5)

        self.header_label = QLabel("MODO PRODUÇÃO • SESSÃO")
        self.header_label.setObjectName("productionSessionHeader")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ok_label = QLabel("")
        self.ok_label.setObjectName("productionSessionOK")
        self.ok_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ng_label = QLabel("")
        self.ng_label.setObjectName("productionSessionNG")
        self.ng_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.header_label)
        layout.addWidget(self.ok_label)
        layout.addWidget(self.ng_label)

        self._refresh_text()
        self.hide()

    def _position(self) -> None:
        x = SESSION_MARGIN
        max_y = max(0, self.panel.height() - self.height())
        y = min(SESSION_TOP_OFFSET, max_y)
        self.move(x, y)

    def _refresh_text(self) -> None:
        self.ok_label.setText(f"OK AUTO     {self.ok_auto}")
        self.ng_label.setText(f"NG AUTO     {self.ng_auto}")

    def reset_session(self) -> None:
        self.ok_auto = 0
        self.ng_auto = 0
        self._refresh_text()
        self._position()
        self.raise_()
        self.show()

    def increment(self, decision: str) -> None:
        normalized = str(decision or "").strip().upper()
        if normalized == "OK":
            self.ok_auto += 1
        elif normalized == "NG":
            self.ng_auto += 1
        else:
            return
        self._refresh_text()
        self._position()
        self.raise_()
        self.show()

    def show_session(self) -> None:
        self._position()
        self.raise_()
        self.show()


class ProductionInterventionFeedbackOverlay(QFrame):
    """Mensagem persistente enquanto NG/revisão aguarda o operador."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel

        self.setObjectName("productionInterventionFeedback")
        self.setFixedSize(INTERVENTION_WIDTH, INTERVENTION_HEIGHT)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(INTERVENTION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 13, 18, 13)
        layout.setSpacing(5)

        self.header_label = QLabel("INTERVENÇÃO NECESSÁRIA")
        self.header_label.setObjectName("productionInterventionHeader")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.reason_label = QLabel("")
        self.reason_label.setObjectName("productionInterventionReason")
        self.reason_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hint_label = QLabel(
            "Aguardando operador\n0 = OK   |   1 = NG"
        )
        self.hint_label.setObjectName("productionInterventionHint")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.header_label)
        layout.addWidget(self.reason_label)
        layout.addWidget(self.hint_label)
        self.hide()

    def _position(self) -> None:
        x = max(0, (self.panel.width() - self.width()) // 2)
        max_y = max(0, self.panel.height() - self.height())
        y = min(INTERVENTION_TOP_OFFSET, max_y)
        self.move(x, y)

    def show_reason(self, reason: str) -> None:
        normalized = str(reason or "").strip().upper()
        if normalized not in {
            "DEFEITO REAL",
            "NG",
            "REVISÃO OBRIGATÓRIA",
        }:
            normalized = "REVISÃO OBRIGATÓRIA"
        self.reason_label.setText(normalized)
        self._position()
        self.raise_()
        self.show()

    def clear(self) -> None:
        self.reason_label.setText("")
        self.hide()


def install_production_session_feedback(panel) -> None:
    if getattr(panel, "_production_session_feedback_installed", False):
        return

    session = ProductionSessionFeedbackOverlay(panel)
    intervention = ProductionInterventionFeedbackOverlay(panel)

    panel.production_session_feedback = session
    panel.production_intervention_feedback = intervention
    panel.reset_production_session_feedback = session.reset_session
    panel.show_production_session_feedback = session.show_session
    panel.increment_production_session_feedback = session.increment
    panel.show_production_intervention_feedback = intervention.show_reason
    panel.clear_production_intervention_feedback = intervention.clear
    panel._production_session_feedback_installed = True


__all__ = [
    "INTERVENTION_HEIGHT",
    "INTERVENTION_TOP_OFFSET",
    "INTERVENTION_WIDTH",
    "ProductionInterventionFeedbackOverlay",
    "ProductionSessionFeedbackOverlay",
    "SESSION_HEIGHT",
    "SESSION_MARGIN",
    "SESSION_TOP_OFFSET",
    "SESSION_WIDTH",
    "install_production_session_feedback",
]
