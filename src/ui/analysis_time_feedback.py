"""Cartão temporário com a duração final da análise, sem interferir no julgamento."""
from __future__ import annotations

from functools import wraps
from math import isfinite

from PyQt6.QtCore import QEasingCurve, QEvent, QPoint, QPropertyAnimation, Qt
from PyQt6.QtWidgets import QFrame, QGraphicsOpacityEffect, QLabel, QVBoxLayout

from src.ui.decision_verdict_feedback import (
    VERDICT_FEEDBACK_FADE_IN_MS,
    VERDICT_FEEDBACK_FADE_OUT_MS,
    VERDICT_FEEDBACK_MARGIN,
    VERDICT_FEEDBACK_SLIDE_PX,
    verdict_feedback_state,
)
from src.ui.lighting_status_feedback import (
    LIGHTING_STATUS_HEIGHT, LIGHTING_STATUS_TOP_OFFSET,
)

ANALYSIS_TIME_WIDTH = 300
ANALYSIS_TIME_HEIGHT = 86
ANALYSIS_TIME_TOP = LIGHTING_STATUS_TOP_OFFSET + LIGHTING_STATUS_HEIGHT + 12

ANALYSIS_TIME_STYLE = """
#analysisTimeFeedback {
    background-color: rgba(13, 13, 13, 248);
    border: 1px solid #f5c518;
    border-radius: 8px;
}
#analysisTimeHeader {
    color: #f5c518; background: transparent; border: none;
    font-size: 10px; font-weight: 900; letter-spacing: 1px;
}
#analysisTimeValue {
    color: #f5f5f5; background: transparent; border: none;
    font-size: 26px; font-weight: 900;
}
"""


def analysis_time_feedback_state(analysis: dict | None) -> str:
    """Usa o tempo final já medido e auditado no detail do resultado."""
    if not isinstance(analysis, dict) or not verdict_feedback_state(analysis)[0]:
        return ""
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return ""
    try:
        seconds = float(detail["analysis_time_seconds"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return ""
    return f"{seconds:.2f} s" if isfinite(seconds) and seconds >= 0.0 else ""


class AnalysisTimeFeedbackOverlay(QFrame):
    """Terceiro aviso no canto direito, sincronizado com 0/1 e descartes."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self._decision_dismiss_pending = False
        self.setObjectName("analysisTimeFeedback")
        self.setFixedSize(ANALYSIS_TIME_WIDTH, ANALYSIS_TIME_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet(ANALYSIS_TIME_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        header = QLabel("TEMPO DE ANÁLISE")
        header.setObjectName("analysisTimeHeader")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label = QLabel("")
        self.value_label.setObjectName("analysisTimeValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        layout.addWidget(self.value_label, 1)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        self._fade_in = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade_in.setDuration(VERDICT_FEEDBACK_FADE_IN_MS)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._slide_in = QPropertyAnimation(self, b"pos", self)
        self._slide_in.setDuration(VERDICT_FEEDBACK_FADE_IN_MS)
        self._slide_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_out = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade_out.setDuration(VERDICT_FEEDBACK_FADE_OUT_MS)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_out.finished.connect(self._finish_hide)
        self.panel.installEventFilter(self)
        self.hide()

    def _position(self) -> QPoint:
        # Não muda o grid principal; adapta-se inclusive a janelas estreitas.
        width = min(ANALYSIS_TIME_WIDTH, max(1, self.panel.width() - 2 * VERDICT_FEEDBACK_MARGIN))
        if width != self.width():
            self.setFixedWidth(width)
        x = max(0, self.panel.width() - self.width() - VERDICT_FEEDBACK_MARGIN)
        y = min(ANALYSIS_TIME_TOP, max(0, self.panel.height() - self.height()))
        return QPoint(x, y)

    def reposition(self) -> None:
        self.move(self._position())
        if not self.isHidden():
            self.raise_()

    def eventFilter(self, watched, event):
        if watched is self.panel and event.type() in {
            QEvent.Type.Resize, QEvent.Type.Show,
        }:
            self.reposition()
        return super().eventFilter(watched, event)

    def _stop_motion(self) -> None:
        self._fade_in.stop()
        self._slide_in.stop()
        self._fade_out.stop()

    def _finish_hide(self) -> None:
        self.hide()
        self._opacity.setOpacity(0.0)
        self.value_label.clear()
        self._decision_dismiss_pending = False

    def show_analysis(self, analysis: dict | None) -> bool:
        value = analysis_time_feedback_state(analysis)
        if not value:
            return False
        if (getattr(self.panel, "adhesive_multilight_pending_start", False)
                and analysis.get("multilight_final") is not True):
            return False
        self._decision_dismiss_pending = False
        self._stop_motion()
        self.value_label.setText(value)
        self.setToolTip(str(analysis["detail"].get("analysis_time_contract", "") or ""))
        target = self._position()
        start = QPoint(
            min(max(0, self.panel.width() - self.width()),
                target.x() + VERDICT_FEEDBACK_SLIDE_PX),
            target.y(),
        )
        self._opacity.setOpacity(0.0)
        self.move(start)
        self._slide_in.setStartValue(start)
        self._slide_in.setEndValue(target)
        self.raise_()
        self.show()
        self._fade_in.start()
        self._slide_in.start()
        return True

    def prepare_decision_dismissal(self) -> bool:
        if self.isHidden() or not self.value_label.text():
            return False
        self._decision_dismiss_pending = True
        return True

    def start_synchronized_fade_out(self) -> bool:
        if not self._decision_dismiss_pending or self.isHidden():
            return False
        self._fade_out.stop()
        self._fade_out.setStartValue(self._opacity.opacity())
        self._fade_out.start()
        return True

    def clear(self, force: bool = False) -> bool:
        if self._decision_dismiss_pending and not force:
            return False
        self._stop_motion()
        self._finish_hide()
        return True


def install_analysis_time_feedback_hooks(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_analysis_time_feedback_hooks", False):
        return
    original_reset = control_panel_cls._reset_confidence_panel

    @wraps(original_reset)
    def wrapped_reset(self):
        result = original_reset(self)
        clear = getattr(self, "clear_analysis_time_feedback", None)
        if callable(clear):
            clear()
        return result

    control_panel_cls._reset_confidence_panel = wrapped_reset
    control_panel_cls._analysis_time_feedback_hooks = True


def install_analysis_time_feedback(panel) -> None:
    if getattr(panel, "_analysis_time_feedback_installed", False):
        return
    overlay = AnalysisTimeFeedbackOverlay(panel)
    panel.analysis_time_feedback = overlay
    panel.show_analysis_time_feedback = overlay.show_analysis
    panel.clear_analysis_time_feedback = overlay.clear
    panel.prepare_analysis_time_feedback_dismissal = overlay.prepare_decision_dismissal
    panel.start_analysis_time_feedback_fade_out = overlay.start_synchronized_fade_out
    panel._analysis_time_feedback_installed = True


__all__ = [
    "ANALYSIS_TIME_HEIGHT", "ANALYSIS_TIME_TOP", "ANALYSIS_TIME_WIDTH",
    "AnalysisTimeFeedbackOverlay", "analysis_time_feedback_state",
    "install_analysis_time_feedback", "install_analysis_time_feedback_hooks",
]
