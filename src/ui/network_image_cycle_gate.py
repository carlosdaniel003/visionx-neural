"""Integra a trava de ciclo de imagens ao painel de controle.

A primeira imagem válida permanece ativa até OK, NG ou descarte. Uma recepção
incompleta da rede não pode bloquear o botão de captura local: nesse caso o
VisionX cancela o ciclo de rede e inicia o MSS sem abrir a entrada para outro
frame repetido do Windows XP.
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer


NETWORK_PROCESSING_WATCHDOG_MS = 1_500


def _receiver(panel):
    return getattr(panel, "network_receiver", None)


def _mode(panel) -> str:
    combo = getattr(panel, "combo_mode", None)
    try:
        return str(combo.currentText()).strip() if combo is not None else ""
    except Exception:
        return ""


def _safe_sync(panel) -> None:
    presenter = getattr(panel, "_operational_controls", None)
    if presenter is None:
        return
    try:
        presenter.sync(force=True)
    except Exception as exc:
        print(f"Falha não fatal ao sincronizar controles: {exc}")


def _restore_window(panel) -> None:
    try:
        if hasattr(panel, "_safe_maximize"):
            panel._safe_maximize()
        else:
            panel.show()
    except Exception as exc:
        print(f"Falha não fatal ao restaurar a janela: {exc}")


def _lock_cycle(panel) -> None:
    panel.capture_cycle_active = True
    receiver = _receiver(panel)
    if receiver is not None and hasattr(receiver, "lock_image_gate"):
        try:
            receiver.lock_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao bloquear entrada de imagens: {exc}")


def _release_cycle(panel) -> None:
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0
    panel.capture_cycle_source = None
    receiver = _receiver(panel)
    if receiver is not None and hasattr(receiver, "release_image_gate"):
        try:
            receiver.release_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao liberar entrada de imagens: {exc}")
    _safe_sync(panel)


def _show_locked_message(panel, message: str | None = None) -> None:
    try:
        panel.update_brain_status(
            message
            or "Captura atual protegida — julgue como OK/NG ou descarte antes de receber outra imagem.",
            True,
        )
    except Exception as exc:
        print(f"Falha não fatal ao atualizar status da captura: {exc}")
    _safe_sync(panel)


def _safe_call(panel, method_name: str, *args, **kwargs) -> None:
    method = getattr(panel, method_name, None)
    if not callable(method):
        return
    try:
        method(*args, **kwargs)
    except Exception as exc:
        print(f"Falha não fatal em {method_name}: {exc}")


def _has_analysis(panel) -> bool:
    return getattr(panel, "current_analysis", None) is not None


def _has_captured_images(panel) -> bool:
    sample = getattr(panel, "current_sample", None)
    test = getattr(panel, "current_ng", None)
    return sample is not None and test is not None


def _schedule(panel, delay_ms: int, callback) -> None:
    scheduler = getattr(panel, "_network_cycle_scheduler", None)
    if callable(scheduler):
        scheduler(int(delay_ms), callback)
        return
    QTimer.singleShot(int(delay_ms), callback)


def _force_discard_cleanup(panel, error: Exception | None = None) -> None:
    """Restaura o estado mínimo mesmo se algum debugger falhar no descarte."""
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0
    panel.capture_cycle_source = None

    if hasattr(panel, "production_review_pending"):
        panel.production_review_pending = False

    for button_name in ("btn_save_ok", "btn_save_ng", "btn_skip"):
        button = getattr(panel, button_name, None)
        if button is not None:
            try:
                button.setEnabled(False)
            except Exception:
                pass

    start_button = getattr(panel, "btn_start", None)
    if start_button is not None:
        try:
            start_button.setText("Capturar local (MSS)")
            start_button.setEnabled(True)
        except Exception:
            pass

    _safe_call(panel, "_reset_confidence_panel")
    _safe_call(panel, "_reset_reference_panel")
    _safe_call(panel, "_reset_aoi_info")
    _restore_window(panel)

    message = "Sistema ocioso — captura descartada com segurança."
    if error is not None:
        message = (
            "Captura recuperada após uma falha interna. "
            "O sistema permanece ativo."
        )
        print(f"Erro recuperado durante limpeza da captura: {error}")
    try:
        panel.update_brain_status(message, False)
    except Exception:
        pass


def _clear_incomplete_network_cycle(panel) -> None:
    """Cancela uma recepção que não chegou a produzir imagens/análise válidas.

    A entrada do receptor permanece bloqueada. Quem chama esta função decide se
    libera o ciclo ou se inicia imediatamente o MSS local.
    """
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.capture_cycle_ignored_signals = 0
    _safe_call(panel, "_reset_confidence_panel")
    _safe_call(panel, "_reset_reference_panel")
    _safe_call(panel, "_reset_aoi_info")


def _network_watchdog(panel, generation: int) -> None:
    if generation != int(getattr(panel, "capture_cycle_network_generation", -1)):
        return
    if getattr(panel, "capture_cycle_source", None) != "network":
        return
    if not bool(getattr(panel, "capture_cycle_active", False)):
        return
    if _has_analysis(panel) or _has_captured_images(panel):
        return
    if bool(getattr(panel, "production_review_pending", False)):
        return

    print(
        "Imagem da rede não produziu uma inspeção válida; "
        "liberando o ciclo para nova captura."
    )
    _clear_incomplete_network_cycle(panel)
    _release_cycle(panel)
    try:
        panel.update_brain_status(
            "A imagem recebida não continha uma inspeção AOI válida. "
            "A captura local está disponível.",
            False,
        )
    except Exception:
        pass


def _can_replace_with_local_capture(panel) -> bool:
    """Uma análise pronta só pode ser substituída no Modo Teste."""
    return bool(
        _mode(panel) == "Modo Teste"
        and _has_analysis(panel)
        and not bool(getattr(panel, "production_review_pending", False))
        and not bool(getattr(panel, "capture_cycle_discarding", False))
        and not bool(getattr(panel, "capture_cycle_local_transition", False))
    )


def _can_take_over_incomplete_network_cycle(panel) -> bool:
    return bool(
        getattr(panel, "capture_cycle_source", None) == "network"
        and not _has_analysis(panel)
        and not _has_captured_images(panel)
        and not bool(getattr(panel, "production_review_pending", False))
        and not bool(getattr(panel, "capture_cycle_discarding", False))
        and not bool(getattr(panel, "capture_cycle_local_transition", False))
    )


def install_network_image_cycle_gate(control_panel_cls, presenter_cls) -> None:
    """Instala a trava como camada externa dos fluxos de captura e decisão."""
    if getattr(control_panel_cls, "_network_image_cycle_gate_installed", False):
        return

    original_init = control_panel_cls.__init__
    original_handle_network_image = control_panel_cls.handle_network_image
    original_start_monitoring = control_panel_cls.start_monitoring
    original_skip_image = control_panel_cls.skip_image
    original_save_label = control_panel_cls.save_label
    original_presenter_sync = presenter_cls.sync

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.capture_cycle_active = False
        self.capture_cycle_ignored_signals = 0
        self.capture_cycle_discarding = False
        self.capture_cycle_local_transition = False
        self.capture_cycle_source = None
        self.capture_cycle_network_generation = 0

    def wrapped_handle_network_image(self, img_bgr, ip: str):
        if (
            bool(getattr(self, "capture_cycle_active", False))
            or bool(getattr(self, "is_locked", False))
            or bool(getattr(self, "capture_cycle_discarding", False))
            or bool(getattr(self, "capture_cycle_local_transition", False))
        ):
            self.capture_cycle_ignored_signals = int(
                getattr(self, "capture_cycle_ignored_signals", 0)
            ) + 1
            _lock_cycle(self)
            if self.capture_cycle_ignored_signals == 1:
                _show_locked_message(self)
            return False

        _lock_cycle(self)
        self.capture_cycle_source = "network"
        self.capture_cycle_network_generation = int(
            getattr(self, "capture_cycle_network_generation", 0)
        ) + 1
        generation = self.capture_cycle_network_generation
        try:
            original_handle_network_image(self, img_bgr, ip)
        except Exception as exc:
            print(f"Falha recuperada ao aceitar imagem da rede: {exc}")
            _force_discard_cleanup(self, error=exc)
            _release_cycle(self)
            return False

        # A extração externa pode falhar sem lançar exceção. O watchdog impede
        # que essa recepção incompleta bloqueie indefinidamente o botão MSS.
        _schedule(
            self,
            NETWORK_PROCESSING_WATCHDOG_MS,
            lambda: _network_watchdog(self, generation),
        )
        _safe_sync(self)
        return True

    def _launch_local_capture(self, *args, **kwargs) -> bool:
        _lock_cycle(self)
        self.capture_cycle_source = "local"
        try:
            original_start_monitoring(self, *args, **kwargs)
            return True
        except Exception as exc:
            print(f"Falha recuperada ao iniciar captura MSS: {exc}")
            _force_discard_cleanup(self, error=exc)
            _release_cycle(self)
            return False

    def _take_over_network_with_local(self, *args, **kwargs) -> bool:
        self.capture_cycle_local_transition = True
        self.capture_cycle_discarding = True
        _lock_cycle(self)
        try:
            _clear_incomplete_network_cycle(self)
            # Invalida o watchdog pertencente à imagem de rede abandonada.
            self.capture_cycle_network_generation = int(
                getattr(self, "capture_cycle_network_generation", 0)
            ) + 1
        finally:
            self.capture_cycle_discarding = False

        launched = _launch_local_capture(self, *args, **kwargs)
        self.capture_cycle_local_transition = False
        _safe_sync(self)
        return bool(launched)

    def wrapped_start_monitoring(self, *args, **kwargs):
        if bool(getattr(self, "capture_cycle_discarding", False)) or bool(
            getattr(self, "capture_cycle_local_transition", False)
        ):
            _show_locked_message(
                self,
                "Transição de captura em andamento — aguarde alguns instantes.",
            )
            return False

        active = bool(getattr(self, "capture_cycle_active", False)) or bool(
            getattr(self, "is_locked", False)
        )

        if not active:
            return _launch_local_capture(self, *args, **kwargs)

        # O operador escolheu explicitamente o MSS. Uma imagem da rede que ainda
        # não gerou recorte/análise não pode impedir a captura local.
        if _can_take_over_incomplete_network_cycle(self):
            return _take_over_network_with_local(self, *args, **kwargs)

        if not _can_replace_with_local_capture(self):
            _show_locked_message(self)
            return False

        # Operação atômica: a rede permanece fechada entre descarte e MSS.
        self.capture_cycle_local_transition = True
        self.capture_cycle_discarding = True
        _lock_cycle(self)
        discard_error = None
        try:
            original_skip_image(self)
        except Exception as exc:
            discard_error = exc
            _force_discard_cleanup(self, error=exc)
        finally:
            self.capture_cycle_discarding = False

        if bool(getattr(self, "is_locked", False)):
            self.capture_cycle_local_transition = False
            _lock_cycle(self)
            _show_locked_message(self)
            return False

        if discard_error is None:
            self.current_analysis = None
            self.current_sample = None
            self.current_ng = None
            self.current_aoi_info = {}

        launched = _launch_local_capture(self, *args, **kwargs)
        self.capture_cycle_local_transition = False
        _safe_sync(self)
        return bool(launched)

    def wrapped_skip_image(self, *args, **kwargs):
        self.capture_cycle_discarding = True
        _lock_cycle(self)
        try:
            result = original_skip_image(self, *args, **kwargs)
        except Exception as exc:
            _force_discard_cleanup(self, error=exc)
            result = None
        finally:
            self.capture_cycle_discarding = False

        if not bool(getattr(self, "is_locked", False)):
            self.current_sample = None
            self.current_ng = None
            self.current_analysis = None
            _release_cycle(self)
        else:
            _lock_cycle(self)
            _safe_sync(self)
        return result

    def wrapped_save_label(self, user_decision: str, source="button"):
        try:
            result = original_save_label(self, user_decision, source=source)
        except Exception as exc:
            print(f"Falha ao concluir decisão {user_decision}: {exc}")
            _lock_cycle(self)
            try:
                self.update_brain_status(
                    "Falha ao concluir a decisão. A captura permanece protegida.",
                    True,
                )
            except Exception:
                pass
            return None

        if not bool(getattr(self, "is_locked", False)):
            self.current_sample = None
            self.current_ng = None
            self.current_analysis = None
            _release_cycle(self)
        else:
            _lock_cycle(self)
        return result

    def wrapped_presenter_sync(self, force: bool = False):
        try:
            original_presenter_sync(self, force=force)
        except Exception as exc:
            print(f"Falha não fatal no presenter de controles: {exc}")
            return

        panel = self.panel
        active = bool(getattr(panel, "capture_cycle_active", False))
        if not active:
            return

        has_analysis = _has_analysis(panel)
        mode = _mode(panel)
        production_pending = bool(
            getattr(panel, "production_review_pending", False)
        )
        transitioning = bool(
            getattr(panel, "capture_cycle_discarding", False)
            or getattr(panel, "capture_cycle_local_transition", False)
        )
        network_takeover = bool(
            _can_take_over_incomplete_network_cycle(panel) and not transitioning
        )
        allow_replace = bool(
            has_analysis
            and mode == "Modo Teste"
            and not production_pending
            and not transitioning
        )

        self._set_enabled(panel.btn_start, allow_replace or network_takeover)

        if network_takeover:
            self._set_text(panel.btn_start, "Capturar local (MSS)")
            panel.btn_start.setToolTip(
                "Cancela a recepção incompleta da rede, minimiza o VisionX e inicia o MSS local."
            )
            panel.lbl_operation_hint.setText(
                "A imagem recebida ainda não gerou análise. "
                "A captura local pode assumir o ciclo imediatamente."
            )
            return

        if allow_replace:
            self._set_text(
                panel.btn_start,
                "Capturar nova peça (descarta a atual)",
            )
            panel.lbl_operation_hint.setText(
                "Você pode julgar, descartar ou iniciar outra captura local. "
                "A captura atual será descartada antes do MSS."
            )
            return

        if has_analysis:
            self._set_text(
                panel.btn_start,
                "Captura bloqueada — finalize ou descarte a atual",
            )
            if not production_pending:
                panel.lbl_operation_hint.setText(
                    "A imagem atual não será substituída sem julgamento ou descarte."
                )
        else:
            self._set_text(panel.btn_start, "Processando captura...")

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._network_image_cycle_gate_installed = True


__all__ = [
    "NETWORK_PROCESSING_WATCHDOG_MS",
    "_force_discard_cleanup",
    "_lock_cycle",
    "_release_cycle",
    "install_network_image_cycle_gate",
]
