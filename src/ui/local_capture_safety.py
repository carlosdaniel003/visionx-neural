"""Supervisão da captura local por MSS.

Mantém o comportamento original de minimizar a janela e iniciar o ScreenMonitor,
mas adiciona ciclo de vida explícito, timeout e recuperação de exceções. Uma
falha de thread ou de análise nunca deve deixar o painel minimizado e bloqueado.
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QTimer

from src.services.screen_monitor import ScreenMonitor
from src.ui.network_image_cycle_gate import (
    _force_discard_cleanup,
    _lock_cycle,
    _release_cycle,
)


LOCAL_CAPTURE_TIMEOUT_MS = 20_000
LOCAL_MONITOR_FINISH_GRACE_MS = 150
LOCAL_MONITOR_STOP_TIMEOUT_MS = 1_500
LOCAL_MONITOR_TERMINATE_TIMEOUT_MS = 400


def _safe_status(panel, message: str, active: bool = False) -> None:
    try:
        panel.update_brain_status(message, active)
    except Exception as exc:
        print(f"Falha não fatal ao atualizar status MSS: {exc}")


def _safe_sync(panel) -> None:
    presenter = getattr(panel, "_operational_controls", None)
    if presenter is None:
        return
    try:
        presenter.sync(force=True)
    except Exception as exc:
        print(f"Falha não fatal ao sincronizar controles MSS: {exc}")


def _schedule(panel, delay_ms: int, callback) -> None:
    scheduler = getattr(panel, "_local_capture_scheduler", None)
    if callable(scheduler):
        scheduler(int(delay_ms), callback)
        return
    QTimer.singleShot(int(delay_ms), callback)


def _monitor_is_running(monitor) -> bool:
    if monitor is None:
        return False
    method = getattr(monitor, "isRunning", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _stop_monitor(panel) -> None:
    """Encerra o MSS sem bloquear a thread da interface indefinidamente."""
    monitor = getattr(panel, "monitor", None)
    if monitor is None or not _monitor_is_running(monitor):
        return

    try:
        if hasattr(monitor, "running"):
            monitor.running = False
        request_interruption = getattr(monitor, "requestInterruption", None)
        if callable(request_interruption):
            request_interruption()

        wait = getattr(monitor, "wait", None)
        if callable(wait):
            finished = bool(wait(LOCAL_MONITOR_STOP_TIMEOUT_MS))
            if finished or not _monitor_is_running(monitor):
                return

            print(
                "Monitor MSS não respondeu ao encerramento normal; "
                "aplicando término de emergência."
            )
            terminate = getattr(monitor, "terminate", None)
            if callable(terminate):
                terminate()
                try:
                    wait(LOCAL_MONITOR_TERMINATE_TIMEOUT_MS)
                except Exception:
                    pass
            return

        stop = getattr(monitor, "stop", None)
        if callable(stop):
            stop()
    except Exception as exc:
        print(f"Falha não fatal ao encerrar monitor MSS: {exc}")


def _restore_window(panel) -> None:
    try:
        if hasattr(panel, "_safe_maximize"):
            panel._safe_maximize()
        else:
            panel.show()
    except Exception as exc:
        print(f"Falha não fatal ao restaurar janela MSS: {exc}")


def _invalidate_local_capture(panel) -> None:
    panel.local_capture_generation = int(
        getattr(panel, "local_capture_generation", 0)
    ) + 1
    panel.local_capture_pending = False
    panel.local_capture_starting = False


def _abort_local_capture(
    panel,
    message: str,
    error: Exception | None = None,
) -> None:
    """Recupera o painel de qualquer falha ou timeout do MSS."""
    _invalidate_local_capture(panel)
    _stop_monitor(panel)
    panel.monitor = None
    panel.is_locked = False
    _force_discard_cleanup(panel, error=error)
    _release_cycle(panel)
    _restore_window(panel)
    _safe_status(panel, message, False)
    _safe_sync(panel)


def _on_local_timeout(panel, generation: int) -> None:
    if generation != int(getattr(panel, "local_capture_generation", -1)):
        return
    if not bool(getattr(panel, "local_capture_pending", False)):
        return
    _abort_local_capture(
        panel,
        "Captura MSS cancelada: a interface da AOI não foi encontrada em 20 segundos.",
    )


def _on_monitor_finished(panel, monitor, generation: int) -> None:
    def verify_after_signal_queue() -> None:
        if monitor is not getattr(panel, "monitor", None):
            return
        if generation != int(getattr(panel, "local_capture_generation", -1)):
            return
        if bool(getattr(panel, "local_capture_pending", False)):
            _abort_local_capture(
                panel,
                "O monitor MSS foi encerrado antes de localizar uma captura válida.",
            )

    _schedule(panel, LOCAL_MONITOR_FINISH_GRACE_MS, verify_after_signal_queue)


def install_local_capture_safety(control_panel_cls) -> None:
    """Instala supervisão como camada externa da trava geral de imagens."""
    if getattr(control_panel_cls, "_local_capture_safety_installed", False):
        return

    original_init = control_panel_cls.__init__
    original_start_monitoring = control_panel_cls.start_monitoring
    original_process_aoi_images = control_panel_cls.process_aoi_images
    original_skip_image = control_panel_cls.skip_image
    original_change_lighting = control_panel_cls.change_lighting
    original_close_event = control_panel_cls.closeEvent

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.local_capture_pending = False
        self.local_capture_starting = False
        self.local_capture_generation = 0
        self.local_capture_last_error = None

    def wrapped_start_monitoring(self, *args, **kwargs):
        if bool(getattr(self, "local_capture_starting", False)) or bool(
            getattr(self, "local_capture_pending", False)
        ):
            _safe_status(
                self,
                "A captura MSS local já está em andamento. Aguarde o resultado.",
                True,
            )
            return False

        self.local_capture_starting = True
        try:
            accepted = bool(original_start_monitoring(self, *args, **kwargs))
        except Exception as exc:
            self.local_capture_last_error = str(exc)
            print("Falha recuperada no botão Capturar nova peça:")
            traceback.print_exc()
            _abort_local_capture(
                self,
                "Não foi possível iniciar a captura MSS. O sistema foi restaurado.",
                error=exc,
            )
            return False
        finally:
            self.local_capture_starting = False

        if not accepted:
            return False

        self.local_capture_pending = True
        self.local_capture_generation = int(
            getattr(self, "local_capture_generation", 0)
        ) + 1
        generation = self.local_capture_generation
        _lock_cycle(self)
        _schedule(
            self,
            LOCAL_CAPTURE_TIMEOUT_MS,
            lambda: _on_local_timeout(self, generation),
        )
        return True

    def wrapped_start_radar(self):
        if not bool(getattr(self, "local_capture_pending", False)):
            return False

        existing = getattr(self, "monitor", None)
        if _monitor_is_running(existing):
            _safe_status(self, "Monitor MSS já está ativo.", True)
            return True

        factory = getattr(self, "_screen_monitor_factory", ScreenMonitor)
        generation = int(getattr(self, "local_capture_generation", 0))
        try:
            monitor = factory()
            self.monitor = monitor

            def deliver_layout(sample, test, aoi_info):
                if monitor is not getattr(self, "monitor", None):
                    return
                if generation != int(
                    getattr(self, "local_capture_generation", -1)
                ):
                    return
                if not bool(getattr(self, "local_capture_pending", False)):
                    return
                self.process_aoi_images(sample, test, aoi_info)

            monitor.layout_detected.connect(deliver_layout)
            if hasattr(monitor, "log_updated"):
                monitor.log_updated.connect(self.update_network_status)
            if hasattr(monitor, "finished"):
                monitor.finished.connect(
                    lambda: _on_monitor_finished(self, monitor, generation)
                )
            monitor.start()
            _safe_status(
                self,
                "MSS local ativo — procurando a interface da AOI.",
                True,
            )
            return True
        except Exception as exc:
            self.local_capture_last_error = str(exc)
            print("Falha recuperada ao criar o ScreenMonitor:")
            traceback.print_exc()
            _abort_local_capture(
                self,
                "Falha ao iniciar o monitor MSS. O sistema foi restaurado.",
                error=exc,
            )
            return False

    def wrapped_process_aoi_images(self, *args, **kwargs):
        was_local = bool(getattr(self, "local_capture_pending", False))
        if was_local:
            self.local_capture_pending = False
            self.local_capture_generation = int(
                getattr(self, "local_capture_generation", 0)
            ) + 1

        try:
            return original_process_aoi_images(self, *args, **kwargs)
        except Exception as exc:
            self.local_capture_last_error = str(exc)
            print("Falha recuperada durante análise da captura:")
            traceback.print_exc()
            _abort_local_capture(
                self,
                "A análise falhou, mas o sistema foi recuperado e permanece aberto.",
                error=exc,
            )
            return None

    def wrapped_skip_image(self, *args, **kwargs):
        if bool(getattr(self, "local_capture_pending", False)):
            _invalidate_local_capture(self)
            _stop_monitor(self)
            self.monitor = None
        try:
            result = original_skip_image(self, *args, **kwargs)
            if not bool(getattr(self, "is_locked", False)):
                _restore_window(self)
            return result
        except Exception as exc:
            print("Falha recuperada no botão Descartar captura:")
            traceback.print_exc()
            _abort_local_capture(
                self,
                "A captura foi descartada com recuperação de segurança.",
                error=exc,
            )
            return None

    def wrapped_change_lighting(self, light_mode: str, source: str):
        try:
            return original_change_lighting(self, light_mode, source)
        except Exception as exc:
            print(f"Falha recuperada ao selecionar iluminação {light_mode}: {exc}")
            _safe_status(
                self,
                f"Não foi possível selecionar a iluminação {light_mode}. "
                "O sistema permanece ativo.",
                False,
            )
            _safe_sync(self)
            return None

    def wrapped_close_event(self, event):
        _invalidate_local_capture(self)
        _stop_monitor(self)
        self.monitor = None
        try:
            return original_close_event(self, event)
        except Exception as exc:
            print(f"Falha não fatal ao fechar o sistema: {exc}")
            try:
                event.accept()
            except Exception:
                pass
            return None

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls._start_radar = wrapped_start_radar
    control_panel_cls.process_aoi_images = wrapped_process_aoi_images
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.change_lighting = wrapped_change_lighting
    control_panel_cls.closeEvent = wrapped_close_event
    control_panel_cls._local_capture_safety_installed = True


__all__ = [
    "LOCAL_CAPTURE_TIMEOUT_MS",
    "LOCAL_MONITOR_STOP_TIMEOUT_MS",
    "install_local_capture_safety",
]
