"""Integra a trava de ciclo de imagens ao painel de controle.

A primeira imagem aceita permanece como captura ativa até uma das resoluções:
OK, NG ou descarte. Imagens seguintes e novas capturas locais não substituem a
captura pendente.
"""

from __future__ import annotations


def _receiver(panel):
    return getattr(panel, "network_receiver", None)


def _lock_cycle(panel) -> None:
    panel.capture_cycle_active = True
    receiver = _receiver(panel)
    if receiver is not None and hasattr(receiver, "lock_image_gate"):
        receiver.lock_image_gate()


def _release_cycle(panel) -> None:
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0
    receiver = _receiver(panel)
    if receiver is not None and hasattr(receiver, "release_image_gate"):
        receiver.release_image_gate()
    if hasattr(panel, "_operational_controls"):
        panel._operational_controls.sync(force=True)


def _show_locked_message(panel) -> None:
    panel.update_brain_status(
        "Captura atual protegida — julgue como OK/NG ou descarte antes de receber outra imagem.",
        True,
    )
    if hasattr(panel, "_operational_controls"):
        panel._operational_controls.sync(force=True)


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

    def wrapped_handle_network_image(self, img_bgr, ip: str):
        if bool(getattr(self, "capture_cycle_active", False)) or bool(
            getattr(self, "is_locked", False)
        ):
            self.capture_cycle_ignored_signals = int(
                getattr(self, "capture_cycle_ignored_signals", 0)
            ) + 1
            _lock_cycle(self)
            # Evita atualizar toda a interface a cada frame repetido do XP.
            if self.capture_cycle_ignored_signals == 1:
                _show_locked_message(self)
            return None

        _lock_cycle(self)
        try:
            return original_handle_network_image(self, img_bgr, ip)
        except Exception:
            # A imagem não chegou a formar uma captura utilizável.
            _release_cycle(self)
            raise

    def wrapped_start_monitoring(self, *args, **kwargs):
        if bool(getattr(self, "capture_cycle_active", False)) or bool(
            getattr(self, "is_locked", False)
        ):
            _show_locked_message(self)
            return None

        _lock_cycle(self)
        try:
            return original_start_monitoring(self, *args, **kwargs)
        except Exception:
            _release_cycle(self)
            raise

    def wrapped_skip_image(self, *args, **kwargs):
        result = original_skip_image(self, *args, **kwargs)
        # A trava de produção pode rejeitar o descarte e manter is_locked=True.
        if not bool(getattr(self, "is_locked", False)):
            _release_cycle(self)
        return result

    def wrapped_save_label(self, user_decision: str, source="button"):
        result = original_save_label(self, user_decision, source=source)
        # Decisão automática abaixo de 99% permanece bloqueada; decisões
        # efetivamente concluídas deixam is_locked=False e liberam a próxima.
        if not bool(getattr(self, "is_locked", False)):
            _release_cycle(self)
        else:
            _lock_cycle(self)
        return result

    def wrapped_presenter_sync(self, force: bool = False):
        original_presenter_sync(self, force=force)
        panel = self.panel
        active = bool(getattr(panel, "capture_cycle_active", False))
        if not active:
            return

        # Uma captura pendente nunca pode ser substituída por uma nova captura.
        self._set_enabled(panel.btn_start, False)

        has_analysis = getattr(panel, "current_analysis", None) is not None
        if has_analysis:
            self._set_text(
                panel.btn_start,
                "Captura bloqueada — finalize ou descarte a atual",
            )

            # Não sobrescreve a orientação especial da revisão de produção.
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
