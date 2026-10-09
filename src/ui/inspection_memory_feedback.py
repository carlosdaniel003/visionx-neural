"""Aviso flutuante da memória KNN: já visto ou primeira ocorrência exata.

Somente apresentação: nunca altera aprendizagem, memória, comandos XP ou
a decisão. Um caso 'novo' significa SEM PAR EXATO humano, não que
o defeito físico jamais tenha sido observado em outro formato.
"""
from __future__ import annotations

from functools import wraps

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt
from PyQt6.QtWidgets import QFrame, QGraphicsOpacityEffect, QLabel, QVBoxLayout

from src.ui.decision_key_feedback import FEEDBACK_FADE_OUT_MS
from src.ui.neural_telemetry_model import memory_panel_text, memory_seen_state


def memory_feedback_state(analysis: dict | None) -> tuple[str, str, str]:
    """Consulta binária por pares KNN verificados."""
    memory = memory_seen_state(analysis)
    if memory["status"] == "JA_VI":
        return "JÁ VI", memory_panel_text(analysis)[1], "known"
    if memory["status"] == "NUNCA_VI":
        return "NUNCA VI", memory_panel_text(analysis)[1], "new"
    return "", "", ""


class InspectionMemoryFeedbackOverlay(QFrame):
    WIDTH, HEIGHT = 332, 86
    OFFSET_TOP, RIGHT_MARGIN = 185, 24

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._decision_dismiss_pending = False
        self.setObjectName("inspectionMemoryFeedback")
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("""
            QFrame#inspectionMemoryFeedback {
                background: #131b23; border: 1px solid #f5c518;
                border-radius: 9px;
            }
            QLabel {background: transparent; border: none;}
            QLabel#inspectionMemoryTitle {
                color: #f5c518; font-size: 14px; font-weight: 900;
            }
            QLabel#inspectionMemoryDescription {
                color: #e0e9f1; font-size: 10px; font-weight: 600;
            }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(4)
        self.title_label = QLabel("")
        self.title_label.setObjectName("inspectionMemoryTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label = QLabel("")
        self.description_label.setObjectName("inspectionMemoryDescription")
        self.description_label.setStyleSheet(
            "color:#e0e9f1; font-size:10px; font-weight:600;"
        )
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setWordWrap(True)
        root.addWidget(self.title_label)
        root.addWidget(self.description_label)

        self.opacity = QGraphicsOpacityEffect(self)
        self.opacity.setOpacity(0)
        self.setGraphicsEffect(self.opacity)
        self.fade_in = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_in.setDuration(140)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_out = QPropertyAnimation(self.opacity, b"opacity", self)
        self.fade_out.setDuration(FEEDBACK_FADE_OUT_MS)
        self.fade_out.setEndValue(0.0)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_out.finished.connect(self._finish_hide)
        self.hide()

    def _position(self) -> QPoint:
        x = max(0, self.panel.width() - self.width() - self.RIGHT_MARGIN)
        y = min(self.OFFSET_TOP, max(0, self.panel.height() - self.height()))
        return QPoint(x, y)

    def _finish_hide(self) -> None:
        self.hide()
        self.opacity.setOpacity(0)
        self._decision_dismiss_pending = False

    def show_analysis(self, analysis: dict | None) -> bool:
        title, explanation, tone = memory_feedback_state(analysis)
        if not title:
            return False
        self.fade_out.stop()
        self.fade_in.stop()
        self._decision_dismiss_pending = False
        self.title_label.setText(title)
        self.description_label.setText(explanation)
        color = {"known": "#4ade80", "new": "#f5c518",
                 "mixed": "#58a6ff", "review": "#ff6262"}.get(tone, "#f5c518")
        self.title_label.setStyleSheet(f"color:{color}; font-size:14px; font-weight:900;")
        self.setStyleSheet(
            "QFrame#inspectionMemoryFeedback {background:#131b23;"
            f"border:1px solid {color}; border-radius:9px;}}"
            "QLabel {background:transparent; border:none;}"
        )
        self.setToolTip(
            memory_panel_text(analysis)[1]
            + "\nNOVO refere-se ao par exato gabarito/teste, não ao tipo físico de defeito."
        )
        self.move(self._position())
        self.opacity.setOpacity(0.0)
        self.raise_()
        self.show()
        self.fade_in.start()
        return True

    def prepare_decision_dismissal(self) -> bool:
        if self.isHidden():
            return False
        self._decision_dismiss_pending = True
        return True

    def start_synchronized_fade_out(self) -> bool:
        if not self._decision_dismiss_pending or self.isHidden():
            return False
        self.fade_out.stop()
        self.fade_out.setStartValue(self.opacity.opacity())
        self.fade_out.start()
        return True

    def clear(self, force: bool = False) -> bool:
        if self._decision_dismiss_pending and not force:
            return False
        self.fade_in.stop()
        self.fade_out.stop()
        self._finish_hide()
        return True


def install_inspection_memory_feedback_hooks(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_inspection_memory_feedback_hooks", False):
        return
    original_update = control_panel_cls._update_reference_panel
    original_reset = control_panel_cls._reset_confidence_panel

    @wraps(original_update)
    def wrapped_update(self, analysis):
        out = original_update(self, analysis)
        # SIDE intermediária não é o julgamento final das três iluminações.
        if getattr(self, "adhesive_multilight_pending_start", False) and not (
            isinstance(analysis, dict) and analysis.get("multilight_final")
        ):
            return out
        show = getattr(self, "show_inspection_memory_feedback", None)
        if callable(show):
            show(analysis)
        return out

    @wraps(original_reset)
    def wrapped_reset(self):
        out = original_reset(self)
        clear = getattr(self, "clear_inspection_memory_feedback", None)
        if callable(clear):
            clear()
        return out

    control_panel_cls._update_reference_panel = wrapped_update
    control_panel_cls._reset_confidence_panel = wrapped_reset
    control_panel_cls._inspection_memory_feedback_hooks = True


def install_inspection_memory_feedback(panel) -> None:
    if getattr(panel, "_inspection_memory_feedback_installed", False):
        return
    overlay = InspectionMemoryFeedbackOverlay(panel)
    panel.inspection_memory_feedback = overlay
    panel.show_inspection_memory_feedback = overlay.show_analysis
    panel.clear_inspection_memory_feedback = overlay.clear
    panel.prepare_inspection_memory_feedback_dismissal = (
        overlay.prepare_decision_dismissal
    )
    panel.start_inspection_memory_feedback_fade_out = (
        overlay.start_synchronized_fade_out
    )
    panel._inspection_memory_feedback_installed = True
