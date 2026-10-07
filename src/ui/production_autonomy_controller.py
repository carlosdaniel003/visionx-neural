"""Controlador único da autonomia do Modo Produção v1.

Contrato:
- espera a análise já renderizada;
- apresenta a tela ao operador com scroll automático não bloqueante;
- FALHA FALSA -> envia 0/OK automaticamente;
- DEFEITO REAL/NG e REVISÃO OBRIGATÓRIA -> pausa e espera decisão humana;
- decisão humana encerra a peça e rearma a autonomia para a próxima captura;
- Modo Teste e Modo Sombra não são alterados.
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QTimer,
)

from src.ui.production_confidence_gate import (
    enter_production_review,
    production_decision_policy,
)


RENDER_SETTLE_MS = 500
TOP_HOLD_MS = 450
SCROLL_DURATION_MS = 5_200
NO_SCROLL_REVIEW_MS = 1_200
POST_SCROLL_PAUSE_MS = 900
AUTO_DECISION_DELAY_MS = 300


class ProductionAutonomyController(QObject):
    """Orquestra apresentação e decisão automática somente em Produção."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.generation = 0
        self.pending_analysis = None
        self.state = "idle"
        self._scroll_animation = None
        self._in_production = False

        combo = getattr(panel, "combo_mode", None)
        if combo is not None:
            combo.currentTextChanged.connect(self._on_mode_changed)
            self._on_mode_changed(combo.currentText())

    def _mode(self) -> str:
        combo = getattr(self.panel, "combo_mode", None)
        try:
            return str(combo.currentText() or "").strip()
        except Exception:
            return ""

    def is_production(self) -> bool:
        return self._mode() == "Modo Produção"

    def _cancel_pending(self) -> None:
        self.generation += 1
        self.pending_analysis = None
        self.state = "idle"
        if self._scroll_animation is not None:
            try:
                self._scroll_animation.stop()
            except Exception:
                pass
        self._scroll_animation = None

    def _on_mode_changed(self, mode_text: str) -> None:
        production = str(mode_text or "").strip() == "Modo Produção"

        if production and not self._in_production:
            self._cancel_pending()
            reset = getattr(
                self.panel,
                "reset_production_session_feedback",
                None,
            )
            if callable(reset):
                reset()
            clear_intervention = getattr(
                self.panel,
                "clear_production_intervention_feedback",
                None,
            )
            if callable(clear_intervention):
                clear_intervention()
            self._in_production = True
            return

        if not production and self._in_production:
            self._cancel_pending()
            self._in_production = False

            session = getattr(
                self.panel,
                "production_session_feedback",
                None,
            )
            if session is not None:
                session.hide()

            clear_intervention = getattr(
                self.panel,
                "clear_production_intervention_feedback",
                None,
            )
            if callable(clear_intervention):
                clear_intervention()

    def _analysis_still_current(self, generation: int) -> bool:
        if generation != self.generation:
            return False
        if not self.is_production():
            return False
        if self.pending_analysis is None:
            return False
        return getattr(self.panel, "current_analysis", None) is self.pending_analysis

    def analysis_ready(self, analysis: dict | None) -> bool:
        """Agenda a apresentação após o resultado final já estar na interface."""
        if not self.is_production() or not isinstance(analysis, dict):
            return False

        if bool(getattr(self.panel, "production_review_pending", False)):
            return False

        self._cancel_pending()
        self.pending_analysis = analysis
        self.generation += 1
        generation = self.generation
        self.state = "rendered_wait"

        show_session = getattr(
            self.panel,
            "show_production_session_feedback",
            None,
        )
        if callable(show_session):
            show_session()

        try:
            self.panel.update_brain_status(
                "Modo Produção: análise renderizada. "
                "Preparando apresentação automática da tela...",
                True,
            )
        except Exception:
            pass

        QTimer.singleShot(
            RENDER_SETTLE_MS,
            lambda g=generation: self._go_to_top(g),
        )
        return True

    def _go_to_top(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return

        scroll = getattr(self.panel, "root_scroll", None)
        bar = scroll.verticalScrollBar() if scroll is not None else None

        if bar is None:
            QTimer.singleShot(
                NO_SCROLL_REVIEW_MS,
                lambda g=generation: self._finish_presentation(g),
            )
            return

        self.state = "presentation_top"
        try:
            bar.setValue(bar.minimum())
        except Exception:
            QTimer.singleShot(
                NO_SCROLL_REVIEW_MS,
                lambda g=generation: self._finish_presentation(g),
            )
            return

        try:
            self.panel.update_brain_status(
                "Modo Produção: apresentando a análise antes da decisão...",
                True,
            )
        except Exception:
            pass

        QTimer.singleShot(
            TOP_HOLD_MS,
            lambda g=generation: self._start_scroll(g),
        )

    def _start_scroll(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return

        scroll = getattr(self.panel, "root_scroll", None)
        bar = scroll.verticalScrollBar() if scroll is not None else None
        if bar is None:
            self._schedule_finish_without_scroll(generation)
            return

        minimum = int(bar.minimum())
        maximum = int(bar.maximum())
        if maximum <= minimum:
            self._schedule_finish_without_scroll(generation)
            return

        self.state = "scrolling"
        animation = QPropertyAnimation(bar, b"value", self)
        animation.setDuration(SCROLL_DURATION_MS)
        animation.setStartValue(minimum)
        animation.setEndValue(maximum)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.finished.connect(
            lambda g=generation: self._scroll_finished(g)
        )
        self._scroll_animation = animation
        animation.start()

    def _schedule_finish_without_scroll(self, generation: int) -> None:
        self.state = "presentation_pause"
        QTimer.singleShot(
            NO_SCROLL_REVIEW_MS,
            lambda g=generation: self._finish_presentation(g),
        )

    def _scroll_finished(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return
        self.state = "presentation_pause"
        QTimer.singleShot(
            POST_SCROLL_PAUSE_MS,
            lambda g=generation: self._finish_presentation(g),
        )

    def _finish_presentation(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return

        analysis = self.pending_analysis
        policy = production_decision_policy(analysis)

        if (
            policy.get("auto_allowed", False)
            and policy.get("proposed_decision") == "OK"
        ):
            self.state = "auto_ok_pending"
            try:
                self.panel.update_brain_status(
                    "Modo Produção: FALHA FALSA confirmada. "
                    "Enviando 0 = OK automaticamente...",
                    True,
                )
            except Exception:
                pass

            show_key = getattr(
                self.panel,
                "show_decision_key_feedback",
                None,
            )
            if callable(show_key):
                show_key("OK", source="production_auto")

            QTimer.singleShot(
                AUTO_DECISION_DELAY_MS,
                lambda g=generation: self._emit_auto_ok(g),
            )
            return

        self.state = "operator_review"
        enter_production_review(self.panel, analysis)

        reason = str(policy.get("verdict", "") or "").strip().upper()
        if reason not in {"DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}:
            reason = (
                "DEFEITO REAL"
                if policy.get("proposed_decision") == "NG"
                else "REVISÃO OBRIGATÓRIA"
            )

        show_intervention = getattr(
            self.panel,
            "show_production_intervention_feedback",
            None,
        )
        if callable(show_intervention):
            show_intervention(reason)

    def _emit_auto_ok(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return

        policy = production_decision_policy(self.pending_analysis)
        if not (
            policy.get("auto_allowed", False)
            and policy.get("proposed_decision") == "OK"
        ):
            self._finish_presentation(generation)
            return

        self.state = "emitting_auto_ok"
        try:
            self.panel.save_label("OK", source="production_auto")
        except Exception as exc:
            self.state = "operator_review"
            enter_production_review(self.panel, self.pending_analysis)
            show_intervention = getattr(
                self.panel,
                "show_production_intervention_feedback",
                None,
            )
            if callable(show_intervention):
                show_intervention("REVISÃO OBRIGATÓRIA")
            try:
                self.panel.update_brain_status(
                    f"Falha ao enviar decisão automática: {exc}. "
                    "Aguardando operador.",
                    True,
                )
            except Exception:
                pass
            return

        increment = getattr(
            self.panel,
            "increment_production_session_feedback",
            None,
        )
        if callable(increment):
            increment("OK")

        self.pending_analysis = None
        self.state = "idle"

    def operator_decision_completed(
        self,
        decision: str,
        source: str = "",
    ) -> None:
        """Rearma a autonomia após 0/1 humano em uma peça pendente."""
        self._cancel_pending()

        clear_intervention = getattr(
            self.panel,
            "clear_production_intervention_feedback",
            None,
        )
        if callable(clear_intervention):
            clear_intervention()

        if self.is_production():
            show_session = getattr(
                self.panel,
                "show_production_session_feedback",
                None,
            )
            if callable(show_session):
                show_session()

    def cancel_current(self) -> None:
        self._cancel_pending()
        clear_intervention = getattr(
            self.panel,
            "clear_production_intervention_feedback",
            None,
        )
        if callable(clear_intervention):
            clear_intervention()


def install_production_autonomy_controller(panel) -> None:
    if getattr(panel, "_production_autonomy_controller_installed", False):
        return

    controller = ProductionAutonomyController(panel)
    panel.production_autonomy_controller = controller
    panel.notify_production_analysis_ready = controller.analysis_ready
    panel.production_operator_decision_completed = (
        controller.operator_decision_completed
    )
    panel.cancel_production_autonomy = controller.cancel_current
    panel._production_autonomy_controller_installed = True


__all__ = [
    "AUTO_DECISION_DELAY_MS",
    "NO_SCROLL_REVIEW_MS",
    "POST_SCROLL_PAUSE_MS",
    "ProductionAutonomyController",
    "RENDER_SETTLE_MS",
    "SCROLL_DURATION_MS",
    "TOP_HOLD_MS",
    "install_production_autonomy_controller",
]
