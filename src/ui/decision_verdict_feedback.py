"""Overlay visual temporário para o veredito final da IA.

Esta camada observa somente o resultado já calculado. Não participa de score,
memória, confiança, persistência, comandos XP ou ciclo produtivo.
"""

from __future__ import annotations

from functools import wraps

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
)
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QVBoxLayout,
)

from src.ui.decision_key_feedback import FEEDBACK_FADE_OUT_MS
from src.ui.inspection_memory_feedback import memory_feedback_state
from src.ui.theme import ACCENT, DANGER, SUCCESS, SURFACE


VERDICT_FEEDBACK_FADE_IN_MS = 140
VERDICT_FEEDBACK_FADE_OUT_MS = FEEDBACK_FADE_OUT_MS
VERDICT_FEEDBACK_SLIDE_PX = 10
VERDICT_FEEDBACK_WIDTH = 300
VERDICT_FEEDBACK_HEIGHT = 88
VERDICT_FEEDBACK_MARGIN = 24
VERDICT_FEEDBACK_TOP_OFFSET = 84


VERDICT_FEEDBACK_STYLESHEET = f"""
QFrame#aiVerdictFeedback {{
    background-color: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 8px;
}}
QLabel#aiVerdictText {{
    background: transparent;
    border: none;
    font-size: 25px;
    font-weight: 900;
    letter-spacing: 0.8px;
}}
QLabel#aiVerdictText[tone="ok"] {{
    color: {SUCCESS};
}}
QLabel#aiVerdictText[tone="ng"] {{
    color: {DANGER};
}}
QLabel#aiMemoryStatusText {{
    background: transparent;
    border: none;
    color: {ACCENT};
    font-size: 11px;
    font-weight: 800;
}}
"""


def verdict_feedback_state(analysis: dict | None) -> tuple[str, str]:
    """Retorna (mensagem, tone) para decisão final ou revisão obrigatória."""
    if not isinstance(analysis, dict) or not analysis:
        return "", ""

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}
    review_required = bool(
        analysis.get("production_review_required", False)
        or trace.get("operator_review_required", False)
        or detail.get("operator_review_required", False)
    )

    # Revisão efetiva é um estado operacional próprio. Ela precisa chamar
    # atenção do operador e nunca deve ser substituída pelo is_defect bruto.
    if review_required:
        return "REVISÃO OBRIGATÓRIA", "ng"

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict == "REVISÃO OBRIGATÓRIA":
        return "REVISÃO OBRIGATÓRIA", "ng"
    if verdict == "FALHA FALSA":
        return "FALHA FALSA", "ok"
    if verdict in {"DEFEITO REAL", "DEFEITO"}:
        return "DEFEITO REAL", "ng"

    # Compatibilidade para análises antigas sem texto de veredito.
    if "is_defect" not in analysis:
        return "", ""

    return (
        ("DEFEITO REAL", "ng")
        if bool(analysis.get("is_defect", False))
        else ("FALHA FALSA", "ok")
    )


class AIVerdictFeedbackOverlay(QFrame):
    """Card leve no canto superior direito com o veredito final da IA."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._decision_dismiss_pending = False

        self.setObjectName("aiVerdictFeedback")
        self.setFixedSize(
            VERDICT_FEEDBACK_WIDTH,
            VERDICT_FEEDBACK_HEIGHT,
        )
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(VERDICT_FEEDBACK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)

        self.verdict_label = QLabel("")
        self.verdict_label.setObjectName("aiVerdictText")
        self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.verdict_label, 1)

        # Mesmo cartão, mesmo relógio de animação e sem novo overlay.
        # A memória informa se o PAR EXATO já foi confirmado por humano;
        # jamais presume que o tipo físico de defeito é inédito.
        self.memory_state_label = QLabel("")
        self.memory_state_label.setObjectName("aiMemoryStatusText")
        self.memory_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.memory_state_label.setWordWrap(True)
        self.memory_state_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self.memory_state_label, 0)
        self.memory_state_label.hide()

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_in = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_in.setDuration(VERDICT_FEEDBACK_FADE_IN_MS)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(VERDICT_FEEDBACK_FADE_IN_MS)
        self._slide_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(
            self._opacity_effect,
            b"opacity",
            self,
        )
        self._fade_out.setDuration(VERDICT_FEEDBACK_FADE_OUT_MS)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_out.finished.connect(self._finish_hide)

        self.hide()

    @staticmethod
    def _refresh_style(widget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _top_right_position(self) -> QPoint:
        x = max(
            0,
            self.panel.width()
            - self.width()
            - VERDICT_FEEDBACK_MARGIN,
        )
        max_y = max(0, self.panel.height() - self.height())
        y = min(VERDICT_FEEDBACK_TOP_OFFSET, max_y)
        return QPoint(x, y)

    def _stop_motion(self) -> None:
        self._fade_in.stop()
        self._slide_in.stop()
        self._fade_out.stop()

    def _finish_hide(self) -> None:
        self.hide()
        self._opacity_effect.setOpacity(0.0)
        self.verdict_label.setText("")
        self.memory_state_label.setText("")
        self.memory_state_label.hide()
        self.setToolTip("")
        self._decision_dismiss_pending = False

    def prepare_decision_dismissal(self) -> bool:
        """Mantém o veredito vivo até o fade-out sincronizado do 0/1."""
        if self.isHidden() or not self.verdict_label.text():
            return False
        self._decision_dismiss_pending = True
        return True

    def start_synchronized_fade_out(self) -> bool:
        """Inicia a mesma saída temporal usada pelo feedback 0/1."""
        if (
            not self._decision_dismiss_pending
            or self.isHidden()
            or not self.verdict_label.text()
        ):
            return False

        self._fade_out.stop()
        self._fade_out.setStartValue(self._opacity_effect.opacity())
        self._fade_out.setEndValue(0.0)
        self._fade_out.start()
        return True

    def clear_verdict(self, force: bool = False) -> bool:
        # Durante um julgamento 0/1, o ciclo pode resetar internamente antes
        # do feedback visual terminar. Nesse caso, preserve o card até o mesmo
        # frame lógico em que o 0/1 inicia seu fade-out.
        if self._decision_dismiss_pending and not force:
            return False

        self._stop_motion()
        self._finish_hide()
        return True

    def show_analysis(self, analysis: dict | None) -> bool:
        message, tone = verdict_feedback_state(analysis)
        if not message:
            return False

        self._decision_dismiss_pending = False
        self.verdict_label.setText(message)
        self.verdict_label.setProperty("tone", tone)
        self._refresh_style(self.verdict_label)

        # Apenas dados efetivos do roteador KNN. Em rotas desconhecidas,
        # não declarar "primeira vez" nem inventar pesquisa na memória.
        memory_title, memory_explanation, memory_tone = memory_feedback_state(analysis)
        subtitle = {"known": "JÁ VI", "new": "NUNCA VI"}.get(memory_tone, "")
        self.memory_state_label.setText(subtitle)
        self.memory_state_label.setVisible(bool(subtitle))
        if subtitle:
            self.setToolTip(
                memory_title + "\n" + memory_explanation
                + "\nNovo = sem par gabarito/teste exato na memória KNN."
            )
        else:
            self.setToolTip("")

        self._stop_motion()

        target = self._top_right_position()
        max_x = max(0, self.panel.width() - self.width())
        start = QPoint(
            min(max_x, target.x() + VERDICT_FEEDBACK_SLIDE_PX),
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


def install_ai_verdict_feedback_hooks(control_panel_cls) -> None:
    """Observa atualização/reset do painel sem modificar a decisão."""
    if getattr(control_panel_cls, "_ai_verdict_feedback_hooks", False):
        return

    original_reference_update = control_panel_cls._update_reference_panel
    original_reset = control_panel_cls._reset_confidence_panel
    original_save_label = control_panel_cls.save_label

    @wraps(original_reference_update)
    def wrapped_reference_update(self, analysis):
        result = original_reference_update(self, analysis)
        show_feedback = getattr(
            self,
            "show_ai_verdict_feedback",
            None,
        )
        if callable(show_feedback):
            show_feedback(analysis)
        return result

    @wraps(original_reset)
    def wrapped_reset(self):
        result = original_reset(self)
        clear_feedback = getattr(
            self,
            "clear_ai_verdict_feedback",
            None,
        )
        if callable(clear_feedback):
            clear_feedback()
        return result

    @wraps(original_save_label)
    def wrapped_save_label(self, *args, **kwargs):
        result = original_save_label(self, *args, **kwargs)
        if (
            getattr(self, "current_analysis", None) is None
            or not bool(getattr(self, "is_locked", False))
        ):
            clear_feedback = getattr(
                self,
                "clear_ai_verdict_feedback",
                None,
            )
            if callable(clear_feedback):
                clear_feedback()
        return result

    control_panel_cls._update_reference_panel = wrapped_reference_update
    control_panel_cls._reset_confidence_panel = wrapped_reset
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls._ai_verdict_feedback_hooks = True


def install_ai_verdict_feedback(panel) -> None:
    """Cria uma única instância do overlay de veredito."""
    if getattr(panel, "_ai_verdict_feedback_installed", False):
        return

    overlay = AIVerdictFeedbackOverlay(panel)
    panel.ai_verdict_feedback = overlay
    panel.show_ai_verdict_feedback = overlay.show_analysis
    panel.clear_ai_verdict_feedback = overlay.clear_verdict
    panel.prepare_ai_verdict_feedback_dismissal = (
        overlay.prepare_decision_dismissal
    )
    panel.start_ai_verdict_feedback_fade_out = (
        overlay.start_synchronized_fade_out
    )
    panel._ai_verdict_feedback_installed = True


__all__ = [
    "AIVerdictFeedbackOverlay",
    "VERDICT_FEEDBACK_FADE_IN_MS",
    "VERDICT_FEEDBACK_FADE_OUT_MS",
    "VERDICT_FEEDBACK_HEIGHT",
    "VERDICT_FEEDBACK_MARGIN",
    "VERDICT_FEEDBACK_SLIDE_PX",
    "VERDICT_FEEDBACK_TOP_OFFSET",
    "VERDICT_FEEDBACK_WIDTH",
    "install_ai_verdict_feedback",
    "install_ai_verdict_feedback_hooks",
    "verdict_feedback_state",
]
