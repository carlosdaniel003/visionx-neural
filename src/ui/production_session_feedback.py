"""Overlays do Modo Produção: desempenho da sessão e intervenção humana.

Camada visual. Não classifica, não envia comandos e não altera a análise.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from src.ui.theme import ACCENT, DANGER, SUCCESS, SURFACE


SESSION_MARGIN = 24
SESSION_TOP_OFFSET = 84
SESSION_WIDTH = 310
SESSION_HEIGHT = 190

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
QLabel#productionSessionState {{
    color: #d6d6d6;
    background: transparent;
    border: none;
    font-size: 10px;
    font-weight: 800;
}}
QLabel#productionSessionOK {{
    color: {SUCCESS};
    background: transparent;
    border: none;
    font-size: 14px;
    font-weight: 900;
}}
QLabel#productionSessionNG {{
    color: {DANGER};
    background: transparent;
    border: none;
    font-size: 14px;
    font-weight: 900;
}}
QLabel#productionSessionManual {{
    color: {ACCENT};
    background: transparent;
    border: none;
    font-size: 13px;
    font-weight: 900;
}}
QLabel#productionSessionAccuracy,
QLabel#productionSessionTime {{
    color: #f0f0f0;
    background: transparent;
    border: none;
    font-size: 13px;
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
    """Métricas persistentes da sessão, visíveis apenas com peça ativa."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.auto_ok = 0
        self.auto_ng = 0
        self.manual_judgments = 0
        self.analysis_count = 0
        self.analysis_time_total = 0.0
        self.analysis_time_count = 0
        self.paused = False

        self.setObjectName("productionSessionFeedback")
        self.setFixedSize(SESSION_WIDTH, SESSION_HEIGHT)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(SESSION_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 11, 15, 11)
        layout.setSpacing(3)

        self.header_label = QLabel("MODO PRODUÇÃO • SESSÃO")
        self.header_label.setObjectName("productionSessionHeader")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.state_label = QLabel("")
        self.state_label.setObjectName("productionSessionState")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ok_label = QLabel("")
        self.ok_label.setObjectName("productionSessionOK")
        self.ok_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ng_label = QLabel("")
        self.ng_label.setObjectName("productionSessionNG")
        self.ng_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.manual_label = QLabel("")
        self.manual_label.setObjectName("productionSessionManual")
        self.manual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.accuracy_label = QLabel("")
        self.accuracy_label.setObjectName("productionSessionAccuracy")
        self.accuracy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.time_label = QLabel("")
        self.time_label.setObjectName("productionSessionTime")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for widget in (
            self.header_label,
            self.state_label,
            self.ok_label,
            self.ng_label,
            self.manual_label,
            self.accuracy_label,
            self.time_label,
        ):
            layout.addWidget(widget)

        self._refresh()
        self.hide()

    @property
    def automatic_judgments(self) -> int:
        return int(self.auto_ok + self.auto_ng)

    @property
    def completed_judgments(self) -> int:
        return int(self.automatic_judgments + self.manual_judgments)

    @property
    def accuracy_percent(self) -> float | None:
        total = self.completed_judgments
        if total <= 0:
            return None
        return (float(self.automatic_judgments) / float(total)) * 100.0

    @property
    def average_analysis_time(self) -> float | None:
        if self.analysis_time_count <= 0:
            return None
        return self.analysis_time_total / float(self.analysis_time_count)

    def _position(self) -> None:
        x = SESSION_MARGIN
        max_y = max(0, self.panel.height() - self.height())
        y = min(SESSION_TOP_OFFSET, max_y)
        self.move(x, y)

    def _refresh(self) -> None:
        self.state_label.setText(
            "AUTOMAÇÃO PAUSADA • ESPAÇO PARA CONTINUAR"
            if self.paused
            else "AUTOMAÇÃO ATIVA • ESPAÇO PARA PAUSAR"
        )
        self.ok_label.setText(f"OK AUTO     {self.auto_ok}")
        self.ng_label.setText(f"NG AUTO     {self.auto_ng}")
        self.manual_label.setText(
            f"MANUAL      {self.manual_judgments}   •   ANÁLISES {self.analysis_count}"
        )

        accuracy = self.accuracy_percent
        accuracy_text = "—" if accuracy is None else f"{accuracy:.1f}%"
        self.accuracy_label.setText(f"PRECISÃO     {accuracy_text}")

        average = self.average_analysis_time
        average_text = "—" if average is None else f"{average:.2f} s"
        self.time_label.setText(f"MÉDIA ANÁLISE     {average_text}")

        tooltip = (
            "Desempenho da sessão atual do Modo Produção.\n\n"
            f"Análises finalizadas: {self.analysis_count}\n"
            f"Julgamentos automáticos: {self.automatic_judgments}\n"
            f"Julgamentos manuais: {self.manual_judgments}\n"
            f"Precisão autônoma: {accuracy_text}\n"
            f"Tempo médio de análise: {average_text}\n\n"
            "Precisão = julgamentos automáticos / julgamentos concluídos. "
            "Toda análise que exige decisão humana é contabilizada como falha "
            "da autonomia nesta sessão.\n\n"
            "O tempo médio usa o mesmo tempo exibido em 'Tempo de análise': "
            "do recebimento/captura da imagem até o resultado final "
            "FALHA FALSA, DEFEITO REAL ou REVISÃO OBRIGATÓRIA estar renderizado. "
            "Scroll, pausas e tempo de decisão do operador não entram na média."
        )
        self.setToolTip(tooltip)
        for child in (
            self.header_label,
            self.state_label,
            self.ok_label,
            self.ng_label,
            self.manual_label,
            self.accuracy_label,
            self.time_label,
        ):
            child.setToolTip(tooltip)

    def reset_session(self) -> None:
        self.auto_ok = 0
        self.auto_ng = 0
        self.manual_judgments = 0
        self.analysis_count = 0
        self.analysis_time_total = 0.0
        self.analysis_time_count = 0
        self.paused = False
        self._refresh()
        self.hide()

    def record_analysis(self, elapsed_seconds: float) -> None:
        self.analysis_count += 1
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed = 0.0
        if elapsed > 0.0:
            self.analysis_time_total += elapsed
            self.analysis_time_count += 1
        self._refresh()

    def record_automatic(self, decision: str) -> None:
        normalized = str(decision or "").strip().upper()
        if normalized == "OK":
            self.auto_ok += 1
        elif normalized == "NG":
            self.auto_ng += 1
        else:
            return
        self._refresh()

    def record_manual(self, decision: str) -> None:
        normalized = str(decision or "").strip().upper()
        if normalized not in {"OK", "NG"}:
            return
        self.manual_judgments += 1
        self._refresh()

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        self._refresh()

    def show_active(self) -> None:
        self._position()
        self.raise_()
        self.show()

    def hide_active(self) -> None:
        self.hide()


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
    panel.show_production_session_feedback = session.show_active
    panel.hide_production_session_feedback = session.hide_active
    panel.record_production_analysis_feedback = session.record_analysis
    panel.record_production_automatic_feedback = session.record_automatic
    panel.record_production_manual_feedback = session.record_manual
    panel.set_production_paused_feedback = session.set_paused
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
