"""Controlador único da autonomia do Modo Produção.

Contrato:
- análise continua sendo calculada normalmente;
- apresentação automática e julgamento podem ser pausados pela barra de espaço;
- FALHA FALSA -> 0/OK automático;
- DEFEITO REAL/NG e REVISÃO OBRIGATÓRIA -> operador;
- métricas da sessão usam somente o tempo até o resultado final renderizado;
- Modo Teste e Modo Sombra permanecem sem essa automação.
"""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QObject, QPropertyAnimation, QTimer, Qt
from PyQt6.QtGui import QKeySequence, QShortcut

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
    """Orquestra apresentação, pausa e decisão automática em Produção."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.generation = 0
        self.pending_analysis = None
        self.state = "idle"
        self.paused = False
        self._in_production = False
        self._scroll_animation = None
        self.pause_shortcut = None
        self._recorded_analysis_keys = set()

        self._stage_timer = QTimer(self)
        self._stage_timer.setSingleShot(True)
        self._stage_timer.timeout.connect(self._run_stage_callback)
        self._stage_callback = None
        self._stage_remaining_ms = 0

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

    def _analysis_key(self, analysis: dict) -> tuple:
        network_event = str(
            getattr(
                self.panel,
                "network_intake_last_image_event_id",
                "",
            )
            or ""
        ).strip()
        if network_event:
            return ("network", network_event)

        local_event = str(
            getattr(
                self.panel,
                "local_capture_debug_event_id",
                "",
            )
            or ""
        ).strip()
        if local_event:
            return ("local", local_event)

        return ("analysis", id(analysis))

    def _schedule_stage(self, delay_ms: int, callback) -> None:
        self._stage_timer.stop()
        self._stage_callback = callback
        self._stage_remaining_ms = max(1, int(delay_ms))
        if not self.paused:
            self._stage_timer.start(self._stage_remaining_ms)

    def _run_stage_callback(self) -> None:
        callback = self._stage_callback
        self._stage_callback = None
        self._stage_remaining_ms = 0
        if self.paused or not callable(callback):
            return
        callback()

    def _stop_pending_stage(self) -> None:
        if self._stage_timer.isActive():
            remaining = int(self._stage_timer.remainingTime())
            self._stage_remaining_ms = max(1, remaining)
            self._stage_timer.stop()

    def _clear_stage(self) -> None:
        self._stage_timer.stop()
        self._stage_callback = None
        self._stage_remaining_ms = 0

    def _cancel_pending(self) -> None:
        self.generation += 1
        self.pending_analysis = None
        self.state = "idle"
        self._clear_stage()
        if self._scroll_animation is not None:
            try:
                self._scroll_animation.stop()
            except Exception:
                pass
        self._scroll_animation = None

    def _set_shortcut_enabled(self, enabled: bool) -> None:
        shortcut = self.pause_shortcut
        if shortcut is not None:
            shortcut.setEnabled(bool(enabled))

    def _sync_pause_visuals(self) -> None:
        self.panel.production_autonomy_paused = bool(self.paused)

        feedback = getattr(
            self.panel,
            "set_production_paused_feedback",
            None,
        )
        if callable(feedback):
            feedback(self.paused)

        presenter = getattr(self.panel, "_operational_controls", None)
        if presenter is not None:
            try:
                presenter.sync(force=True)
            except Exception:
                pass

    def _on_mode_changed(self, mode_text: str) -> None:
        production = str(mode_text or "").strip() == "Modo Produção"
        self._set_shortcut_enabled(production)

        if production and not self._in_production:
            self._cancel_pending()
            self.paused = False
            self.panel.production_autonomy_paused = False
            self._recorded_analysis_keys = set()

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
            self._sync_pause_visuals()
            return

        if not production and self._in_production:
            self._cancel_pending()
            self.paused = False
            self.panel.production_autonomy_paused = False
            self._in_production = False

            hide_session = getattr(
                self.panel,
                "hide_production_session_feedback",
                None,
            )
            if callable(hide_session):
                hide_session()

            clear_intervention = getattr(
                self.panel,
                "clear_production_intervention_feedback",
                None,
            )
            if callable(clear_intervention):
                clear_intervention()

    def cycle_started(self) -> bool:
        """Mostra métricas somente quando existe uma peça ativa em Produção."""
        if not self.is_production():
            return False

        show_session = getattr(
            self.panel,
            "show_production_session_feedback",
            None,
        )
        if callable(show_session):
            show_session()

        self._sync_pause_visuals()

        if self.paused:
            try:
                self.panel.update_brain_status(
                    "MODO PRODUÇÃO PAUSADO • análise permitida, "
                    "julgamento automático aguardando a barra de espaço.",
                    True,
                )
            except Exception:
                pass
        return True

    def _analysis_still_current(self, generation: int) -> bool:
        if generation != self.generation:
            return False
        if not self.is_production():
            return False
        if self.pending_analysis is None:
            return False
        return (
            getattr(self.panel, "current_analysis", None)
            is self.pending_analysis
        )

    def _record_analysis_metrics(self, analysis: dict) -> None:
        key = self._analysis_key(analysis)
        if key in self._recorded_analysis_keys:
            return
        self._recorded_analysis_keys.add(key)

        elapsed = float(
            getattr(self.panel, "last_analysis_time_seconds", 0.0)
            or 0.0
        )
        record = getattr(
            self.panel,
            "record_production_analysis_feedback",
            None,
        )
        if callable(record):
            record(elapsed)

    def analysis_ready(self, analysis: dict | None) -> bool:
        """Agenda a apresentação após o resultado final já estar renderizado."""
        if not self.is_production() or not isinstance(analysis, dict):
            return False

        if bool(getattr(self.panel, "production_review_pending", False)):
            return False

        self.cycle_started()
        self._record_analysis_metrics(analysis)

        self._cancel_pending()
        self.pending_analysis = analysis
        self.generation += 1
        generation = self.generation
        self.state = "rendered_wait"

        try:
            if self.paused:
                self.panel.update_brain_status(
                    "MODO PRODUÇÃO PAUSADO • resultado pronto. "
                    "Pressione espaço para continuar a apresentação.",
                    True,
                )
            else:
                self.panel.update_brain_status(
                    "Modo Produção: análise renderizada. "
                    "Preparando apresentação automática da tela...",
                    True,
                )
        except Exception:
            pass

        self._schedule_stage(
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
            self._schedule_stage(
                NO_SCROLL_REVIEW_MS,
                lambda g=generation: self._finish_presentation(g),
            )
            return

        self.state = "presentation_top"
        try:
            bar.setValue(bar.minimum())
        except Exception:
            self._schedule_stage(
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

        self._schedule_stage(
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
        current = int(bar.value())
        if maximum <= minimum:
            self._schedule_finish_without_scroll(generation)
            return

        remaining_fraction = max(
            0.0,
            min(
                1.0,
                float(maximum - current)
                / float(maximum - minimum),
            ),
        )
        remaining_duration = max(
            450,
            int(SCROLL_DURATION_MS * remaining_fraction),
        )

        self.state = "scrolling"
        animation = QPropertyAnimation(bar, b"value", self)
        animation.setDuration(remaining_duration)
        animation.setStartValue(current)
        animation.setEndValue(maximum)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.finished.connect(
            lambda g=generation: self._scroll_finished(g)
        )
        self._scroll_animation = animation
        animation.start()

        if self.paused:
            animation.pause()

    def _schedule_finish_without_scroll(self, generation: int) -> None:
        self.state = "presentation_pause"
        self._schedule_stage(
            NO_SCROLL_REVIEW_MS,
            lambda g=generation: self._finish_presentation(g),
        )

    def _scroll_finished(self, generation: int) -> None:
        if not self._analysis_still_current(generation):
            return
        self._scroll_animation = None
        self.state = "presentation_pause"
        self._schedule_stage(
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
                    "Preparando 0 = OK automático...",
                    True,
                )
            except Exception:
                pass

            self._schedule_stage(
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
        self.panel.last_decision_command_success = None
        try:
            result = self.panel.save_label(
                "OK",
                source="production_auto",
            )
        except Exception as exc:
            result = False
            try:
                self.panel.update_brain_status(
                    f"Falha ao enviar decisão automática: {exc}. "
                    "Aguardando operador.",
                    True,
                )
            except Exception:
                pass

        command_success = getattr(
            self.panel,
            "last_decision_command_success",
            None,
        )
        if result is False or command_success is not True:
            self.state = "operator_review"
            enter_production_review(
                self.panel,
                self.pending_analysis,
                verdict_override="REVISÃO OBRIGATÓRIA",
            )
            show_intervention = getattr(
                self.panel,
                "show_production_intervention_feedback",
                None,
            )
            if callable(show_intervention):
                show_intervention("REVISÃO OBRIGATÓRIA")
            return

        record_auto = getattr(
            self.panel,
            "record_production_automatic_feedback",
            None,
        )
        if callable(record_auto):
            record_auto("OK")

        show_key = getattr(
            self.panel,
            "show_decision_key_feedback",
            None,
        )
        if callable(show_key):
            show_key("OK", source="production_auto")

        hide_session = getattr(
            self.panel,
            "hide_production_session_feedback",
            None,
        )
        if callable(hide_session):
            hide_session()

        self.pending_analysis = None
        self.state = "idle"

    def operator_decision_completed(
        self,
        decision: str,
        source: str = "",
    ) -> None:
        """Conta a intervenção como falha da autonomia e rearma a sessão."""
        record_manual = getattr(
            self.panel,
            "record_production_manual_feedback",
            None,
        )
        if callable(record_manual):
            record_manual(decision)

        self._cancel_pending()

        clear_intervention = getattr(
            self.panel,
            "clear_production_intervention_feedback",
            None,
        )
        if callable(clear_intervention):
            clear_intervention()

        hide_session = getattr(
            self.panel,
            "hide_production_session_feedback",
            None,
        )
        if callable(hide_session):
            hide_session()

    def toggle_pause(self) -> bool:
        """Espaço pausa/retoma somente a automação do Modo Produção."""
        if not self.is_production():
            return False

        self.paused = not self.paused

        if self.paused:
            self._stop_pending_stage()
            if self._scroll_animation is not None:
                try:
                    self._scroll_animation.pause()
                except Exception:
                    pass

            try:
                self.panel.update_brain_status(
                    "MODO PRODUÇÃO PAUSADO • pressione espaço para continuar.",
                    True,
                )
            except Exception:
                pass
        else:
            if self._scroll_animation is not None and self.state == "scrolling":
                try:
                    self._scroll_animation.resume()
                except Exception:
                    pass
            elif callable(self._stage_callback):
                self._stage_timer.start(
                    max(1, int(self._stage_remaining_ms or 1))
                )

            try:
                if bool(
                    getattr(
                        self.panel,
                        "production_review_pending",
                        False,
                    )
                ):
                    policy = getattr(
                        self.panel,
                        "production_review_policy",
                        {},
                    ) or {}
                    verdict = str(
                        policy.get(
                            "verdict",
                            "REVISÃO OBRIGATÓRIA",
                        )
                        or "REVISÃO OBRIGATÓRIA"
                    )
                    self.panel.update_brain_status(
                        f"Intervenção necessária: {verdict}. "
                        "Aguardando operador: 0=OK | 1=NG",
                        True,
                    )
                elif self.pending_analysis is not None:
                    self.panel.update_brain_status(
                        "Modo Produção retomado • continuando processo automático.",
                        True,
                    )
                else:
                    self.panel.update_brain_status(
                        "Modo Produção retomado • aguardando próxima imagem.",
                        True,
                    )
            except Exception:
                pass

        self._sync_pause_visuals()
        return self.paused

    def cancel_current(self) -> None:
        self._cancel_pending()
        clear_intervention = getattr(
            self.panel,
            "clear_production_intervention_feedback",
            None,
        )
        if callable(clear_intervention):
            clear_intervention()
        hide_session = getattr(
            self.panel,
            "hide_production_session_feedback",
            None,
        )
        if callable(hide_session):
            hide_session()


def install_production_autonomy_controller(panel) -> None:
    if getattr(panel, "_production_autonomy_controller_installed", False):
        return

    controller = ProductionAutonomyController(panel)

    shortcut = QShortcut(QKeySequence("Space"), panel)
    shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
    shortcut.setAutoRepeat(False)
    shortcut.activated.connect(controller.toggle_pause)
    shortcut.setEnabled(controller.is_production())
    controller.pause_shortcut = shortcut

    panel.production_autonomy_controller = controller
    panel.production_pause_shortcut = shortcut
    panel.production_autonomy_paused = False
    panel.notify_production_cycle_started = controller.cycle_started
    panel.notify_production_analysis_ready = controller.analysis_ready
    panel.production_operator_decision_completed = (
        controller.operator_decision_completed
    )
    panel.toggle_production_autonomy_pause = controller.toggle_pause
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
