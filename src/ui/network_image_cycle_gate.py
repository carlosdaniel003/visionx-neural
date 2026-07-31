"""Integra a trava de ciclo de imagens ao painel de controle.

A primeira imagem aceita permanece como captura ativa até OK, NG ou descarte.
Imagens seguintes e novas capturas locais não substituem a captura pendente.
O descarte é protegido contra falhas de reset para não encerrar a aplicação.
"""

from __future__ import annotations


def _receiver(panel):
    return getattr(panel, "network_receiver", None)


def _safe_sync(panel) -> None:
    presenter = getattr(panel, "_operational_controls", None)
    if presenter is None:
        return
    try:
        presenter.sync(force=True)
    except Exception as exc:
        print(f"Falha não fatal ao sincronizar controles: {exc}")


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
    receiver = _receiver(panel)
    if receiver is not None and hasattr(receiver, "release_image_gate"):
        try:
            receiver.release_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao liberar entrada de imagens: {exc}")
    _safe_sync(panel)


def _show_locked_message(panel) -> None:
    try:
        panel.update_brain_status(
            "Captura atual protegida — julgue como OK/NG ou descarte antes de receber outra imagem.",
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


def _force_discard_cleanup(panel, error: Exception | None = None) -> None:
    """Restaura o estado mínimo mesmo se algum debugger falhar no descarte."""
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0

    # O descarte não deve manter uma revisão de produção pendente quando houve
    # falha interna no reset. No fluxo normal, a trava de produção bloqueia o
    # descarte antes de chegar a esta função.
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
            start_button.setText("Capturar Local (MSS)")
            start_button.setEnabled(True)
        except Exception:
            pass

    _safe_call(panel, "_reset_confidence_panel")
    _safe_call(panel, "_reset_reference_panel")
    _safe_call(panel, "_reset_aoi_info")

    message = "Sistema ocioso — captura descartada com segurança."
    if error is not None:
        message = (
            "Captura descartada com recuperação após uma falha de interface. "
            "O sistema permanece ativo."
        )
        print(f"Erro recuperado durante descarte: {error}")
    try:
        panel.update_brain_status(message, False)
    except Exception:
        pass


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

    def wrapped_handle_network_image(self, img_bgr, ip: str):
        if bool(getattr(self, "capture_cycle_active", False)) or bool(
            getattr(self, "is_locked", False)
        ) or bool(getattr(self, "capture_cycle_discarding", False)):
            self.capture_cycle_ignored_signals = int(
                getattr(self, "capture_cycle_ignored_signals", 0)
            ) + 1
            _lock_cycle(self)
            if self.capture_cycle_ignored_signals == 1:
                _show_locked_message(self)
            return None

        _lock_cycle(self)
        try:
            return original_handle_network_image(self, img_bgr, ip)
        except Exception:
            _release_cycle(self)
            raise

    def wrapped_start_monitoring(self, *args, **kwargs):
        if bool(getattr(self, "capture_cycle_active", False)) or bool(
            getattr(self, "is_locked", False)
        ) or bool(getattr(self, "capture_cycle_discarding", False)):
            _show_locked_message(self)
            return None

        _lock_cycle(self)
        try:
            return original_start_monitoring(self, *args, **kwargs)
        except Exception:
            _release_cycle(self)
            raise

    def wrapped_skip_image(self, *args, **kwargs):
        # Mantém o receptor fechado durante todo o reset da interface.
        self.capture_cycle_discarding = True
        _lock_cycle(self)
        try:
            result = original_skip_image(self, *args, **kwargs)
        except Exception as exc:
            _force_discard_cleanup(self, error=exc)
            result = None
        finally:
            self.capture_cycle_discarding = False

        # Em revisão obrigatória de produção, o wrapper interno recusa o
        # descarte e mantém is_locked=True. Nos demais casos, libera o ciclo.
        if not bool(getattr(self, "is_locked", False)):
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

        self._set_enabled(panel.btn_start, False)

        has_analysis = getattr(panel, "current_analysis", None) is not None
        if has_analysis:
            self._set_text(
                panel.btn_start,
                "Captura bloqueada — finalize ou descarte a atual",
            )

            if not bool(getattr(panel, "production_review_pending", False)):
                panel.lbl_operation_hint.setText(
                    "A imagem atual não será substituída. Julgue como OK/NG ou descarte a captura."
                )

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._network_image_cycle_gate_installed = True


__all__ = ["install_network_image_cycle_gate"]
