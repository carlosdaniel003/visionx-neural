"""Barreira final de segurança para todas as ações operacionais.

O botão principal arma a recepção da próxima imagem da AOI. Ele não inicia mais
a captura local MSS nem minimiza a aplicação. Todos os comandos são protegidos
contra clique duplo, reentrada e exceções de slots Qt.
"""

from __future__ import annotations

import time
from typing import Callable


ACTION_DEBOUNCE_SECONDS = {
    "start": 0.55,
    "discard": 0.55,
    "decision": 0.75,
    "lighting": 0.20,
    "keyboard": 0.15,
}


def _safe_status(panel, message: str, active: bool = False) -> None:
    try:
        panel.update_brain_status(str(message), bool(active))
    except Exception as exc:
        print(f"Falha não fatal ao atualizar status operacional: {exc}")


def _safe_network_status(panel, message: str) -> None:
    try:
        panel.update_network_status(str(message))
    except Exception as exc:
        print(f"Falha não fatal ao atualizar status de rede: {exc}")


def _safe_sync(panel) -> None:
    presenter = getattr(panel, "_operational_controls", None)
    if presenter is None:
        return
    try:
        presenter.sync(force=True)
    except Exception as exc:
        print(f"Falha não fatal ao sincronizar controles: {exc}")


def _stop_local_monitor(panel) -> None:
    """Encerra qualquer monitor MSS antigo sem bloquear indefinidamente o Qt."""
    monitor = getattr(panel, "monitor", None)
    if monitor is None:
        return
    try:
        if hasattr(monitor, "running"):
            monitor.running = False
        if hasattr(monitor, "requestInterruption"):
            monitor.requestInterruption()
        if hasattr(monitor, "isRunning") and monitor.isRunning():
            if hasattr(monitor, "quit"):
                monitor.quit()
            # Não usa wait() sem limite: isso era capaz de congelar a interface.
            if hasattr(monitor, "wait"):
                monitor.wait(250)
    except Exception as exc:
        print(f"Falha não fatal ao encerrar monitor local: {exc}")
    finally:
        panel.monitor = None


def _release_receiver(panel) -> None:
    receiver = getattr(panel, "network_receiver", None)
    if receiver is not None and hasattr(receiver, "release_image_gate"):
        try:
            receiver.release_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao liberar receptor: {exc}")
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0


def _lock_receiver(panel) -> None:
    receiver = getattr(panel, "network_receiver", None)
    if receiver is not None and hasattr(receiver, "lock_image_gate"):
        try:
            receiver.lock_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao proteger receptor: {exc}")
    panel.capture_cycle_active = True


def _clear_failed_capture(panel) -> None:
    """Volta ao estado ocioso quando uma imagem não produz uma inspeção válida."""
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.capture_cycle_active = False

    for name in ("_reset_confidence_panel", "_reset_reference_panel", "_reset_aoi_info"):
        method = getattr(panel, name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                print(f"Falha não fatal em {name}: {exc}")

    timer = getattr(panel, "lbl_timer", None)
    if timer is not None:
        try:
            timer.setText("Latência: aguardando AOI")
        except Exception:
            pass

    _release_receiver(panel)
    _safe_sync(panel)


def _debounced(panel, action: str) -> bool:
    now = time.monotonic()
    timestamps = getattr(panel, "_operational_action_times", None)
    if not isinstance(timestamps, dict):
        timestamps = {}
        panel._operational_action_times = timestamps
    last = float(timestamps.get(action, -1e9))
    minimum = float(ACTION_DEBOUNCE_SECONDS.get(action, 0.25))
    if now - last < minimum:
        return True
    timestamps[action] = now
    return False


def _enter(panel, action: str) -> bool:
    if bool(getattr(panel, "_operational_action_busy", False)):
        _safe_status(panel, "Ação anterior ainda está sendo concluída.", True)
        return False
    if _debounced(panel, action):
        return False
    panel._operational_action_busy = True
    panel._operational_action_name = action
    return True


def _leave(panel) -> None:
    panel._operational_action_busy = False
    panel._operational_action_name = ""
    _safe_sync(panel)


def _arm_next_aoi_image(panel) -> str:
    """Arma a rede sem iniciar captura MSS nem minimizar a janela."""
    if bool(getattr(panel, "is_locked", False)) or bool(
        getattr(panel, "capture_cycle_active", False)
    ):
        _safe_status(
            panel,
            "A captura atual ainda está pendente. Julgue ou descarte antes da próxima peça.",
            True,
        )
        return "blocked"

    _stop_local_monitor(panel)
    panel.is_locked = False
    panel.capture_cycle_active = False
    panel.capture_cycle_discarding = False

    # Não limpa a última imagem da tela: apenas prepara o próximo ciclo.
    _release_receiver(panel)

    timer = getattr(panel, "lbl_timer", None)
    if timer is not None:
        try:
            timer.setText("Latência: aguardando AOI")
        except Exception:
            pass

    _safe_status(panel, "Aguardando a próxima imagem nova da AOI...", True)
    _safe_network_status(panel, "Receptor armado para a próxima peça da AOI.")
    return "armed_network"


def install_operational_action_guard(control_panel_cls, presenter_cls) -> None:
    """Instala a barreira externa após todas as outras extensões de fluxo."""
    if getattr(control_panel_cls, "_operational_action_guard_installed", False):
        return

    original_init = control_panel_cls.__init__
    original_handle_network_image = control_panel_cls.handle_network_image
    original_skip_image = control_panel_cls.skip_image
    original_save_label = control_panel_cls.save_label
    original_change_lighting = control_panel_cls.change_lighting
    original_handle_keyboard = control_panel_cls.handle_physical_keyboard
    original_key_press = control_panel_cls.keyPressEvent
    original_presenter_sync = presenter_cls.sync

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._operational_action_busy = False
        self._operational_action_name = ""
        self._operational_action_times = {}
        self._capture_processing = False
        self._capture_failures = 0

        start_button = getattr(self, "btn_start", None)
        if start_button is not None:
            try:
                start_button.setToolTip(
                    "Arma o receptor para a próxima imagem nova enviada pela AOI. "
                    "Não inicia captura local nem minimiza o VisionX."
                )
            except Exception:
                pass

    def wrapped_start_monitoring(self, *args, **kwargs):
        if not _enter(self, "start"):
            return None
        try:
            return _arm_next_aoi_image(self)
        except Exception as exc:
            print(f"Erro recuperado ao armar próxima peça: {exc}")
            _clear_failed_capture(self)
            _safe_status(
                self,
                "Não foi possível armar a próxima peça. O sistema foi recuperado e continua ativo.",
                False,
            )
            return None
        finally:
            _leave(self)

    def wrapped_handle_network_image(self, img_bgr, ip: str):
        if bool(getattr(self, "_capture_processing", False)):
            _safe_network_status(self, "Uma imagem já está sendo processada; frame adicional ignorado.")
            return None

        self._capture_processing = True
        try:
            result = original_handle_network_image(self, img_bgr, ip)

            # process_external_image e layout_detected são síncronos neste fluxo.
            # Se voltaram sem análise, a imagem era inválida ou o layout não foi
            # reconhecido. Não deixamos o painel preso em PROCESSANDO.
            if getattr(self, "current_analysis", None) is None:
                self._capture_failures = int(getattr(self, "_capture_failures", 0)) + 1
                _clear_failed_capture(self)
                _safe_status(
                    self,
                    "Imagem recebida, mas a AOI não pôde ser recortada. Aguardando outra imagem.",
                    False,
                )
                return None

            self._capture_failures = 0
            return result
        except Exception as exc:
            self._capture_failures = int(getattr(self, "_capture_failures", 0)) + 1
            print(f"Erro recuperado ao processar imagem da AOI: {exc}")
            _clear_failed_capture(self)
            _safe_status(
                self,
                "Falha de análise recuperada. A aplicação continua aguardando a próxima imagem.",
                False,
            )
            return None
        finally:
            self._capture_processing = False
            _safe_sync(self)

    def wrapped_skip_image(self, *args, **kwargs):
        if not _enter(self, "discard"):
            return None
        try:
            _stop_local_monitor(self)
            result = original_skip_image(self, *args, **kwargs)
            if not bool(getattr(self, "is_locked", False)):
                _safe_status(self, "Captura descartada. Aguardando mudança da tela da AOI.", False)
            return result
        except Exception as exc:
            print(f"Erro recuperado ao descartar captura: {exc}")
            _clear_failed_capture(self)
            _safe_status(
                self,
                "Captura descartada com recuperação. O sistema permanece ativo.",
                False,
            )
            return None
        finally:
            _leave(self)

    def wrapped_save_label(self, user_decision: str, source="button"):
        if not _enter(self, "decision"):
            return None
        normalized = str(user_decision or "").strip().upper()
        try:
            if normalized not in {"OK", "NG"}:
                _safe_status(self, "Decisão inválida ignorada.", False)
                return None
            return original_save_label(self, normalized, source=source)
        except Exception as exc:
            # Uma decisão incompleta não pode liberar a peça nem fechar o app.
            print(f"Erro recuperado ao registrar decisão {normalized}: {exc}")
            self.is_locked = True
            _lock_receiver(self)
            _safe_status(
                self,
                "Falha ao registrar a decisão. A captura continua protegida; tente novamente.",
                True,
            )
            return None
        finally:
            _leave(self)

    def wrapped_change_lighting(self, light_mode: str, source: str):
        if not _enter(self, "lighting"):
            return None
        try:
            normalized = str(light_mode or "").strip().upper()
            if normalized not in {"MID", "SIDE", "TOP"}:
                return None
            return original_change_lighting(self, normalized, source)
        except Exception as exc:
            print(f"Erro recuperado ao alterar iluminação: {exc}")
            _safe_status(self, "Falha ao alterar iluminação; o sistema continua ativo.", False)
            return None
        finally:
            _leave(self)

    def wrapped_handle_keyboard(self, command: str):
        if _debounced(self, "keyboard"):
            return None
        try:
            return original_handle_keyboard(self, command)
        except Exception as exc:
            print(f"Erro recuperado no comando físico {command!r}: {exc}")
            _safe_status(self, "Comando físico inválido ou não concluído.", False)
            return None

    def wrapped_key_press(self, event):
        try:
            return original_key_press(self, event)
        except Exception as exc:
            print(f"Erro recuperado no atalho de teclado: {exc}")
            _safe_status(self, "Atalho ignorado após falha não fatal.", False)
            try:
                event.accept()
            except Exception:
                pass
            return None

    def wrapped_presenter_sync(self, force: bool = False):
        try:
            original_presenter_sync(self, force=force)
        except Exception as exc:
            print(f"Falha não fatal na atualização dos controles: {exc}")
            return None

        panel = self.panel
        state_name = str(getattr(self, "last_state_name", "") or "")
        if state_name == "idle":
            self._set_text(panel.btn_start, "Aguardar próxima peça da AOI")
        elif bool(getattr(panel, "capture_cycle_active", False)):
            self._set_text(panel.btn_start, "Captura protegida — finalize a peça atual")

        if bool(getattr(panel, "_operational_action_busy", False)):
            # Evita novo clique enquanto o slot anterior ainda está concluindo.
            action = str(getattr(panel, "_operational_action_name", "ação"))
            panel.lbl_operation_hint.setText(f"Concluindo {action}...")

    # O botão continua conectado ao nome start_monitoring, porém agora esse nome
    # arma a rede em vez de chamar o antigo MSS.
    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.change_lighting = wrapped_change_lighting
    control_panel_cls.handle_physical_keyboard = wrapped_handle_keyboard
    control_panel_cls.keyPressEvent = wrapped_key_press
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._operational_action_guard_installed = True


__all__ = [
    "ACTION_DEBOUNCE_SECONDS",
    "install_operational_action_guard",
]
