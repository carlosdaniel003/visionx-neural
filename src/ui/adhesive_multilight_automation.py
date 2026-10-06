"""Automação da aquisição SIDE/TOP/MID para a categoria de adesivo.

A automação controla a troca de iluminação, a coleta das imagens auxiliares e
o fechamento do ciclo multilight. TOP/MID são analisadas antes do avanço da
máquina de estados e, ao final, SIDE/TOP/MID são fundidas em um único julgamento.

SIDE já recebido -> TOP -> análise TOP -> MID -> análise MID -> restaura SIDE
-> fusão final.

As setas da AOI são seletores absolutos:
LEFT=TOP, DOWN=SIDE, RIGHT=MID.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer


LIGHTING_COMMAND = {
    "TOP": "LEFT",
    "SIDE": "DOWN",
    "MID": "RIGHT",
}

CAPTURE_SEQUENCE = ("TOP", "MID")
FRAME_TIMEOUT_MS = 8000
MAX_FRAME_RETRIES = 1


class AdhesiveMultiLightAutomation(QObject):
    """Máquina de estados não bloqueante para aquisição multilight."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.active = False
        self.completed = False
        self.expected_mode = ""
        self.captured_modes = set()
        self.retry_count = 0
        self._session_token = 0

        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._handle_timeout)

    def _set_auxiliary_receiver(self, enabled: bool) -> None:
        receiver = getattr(self.panel, "network_receiver", None)
        method = getattr(receiver, "set_auxiliary_image_mode", None)
        if callable(method):
            method(bool(enabled))

    def _set_panel_flag(self, enabled: bool) -> None:
        self.panel.adhesive_multilight_automation_active = bool(enabled)

    def _status(self, message: str, active: bool = False) -> None:
        callback = getattr(self.panel, "update_brain_status", None)
        if callable(callback):
            callback(str(message), bool(active))

    def _network_status(self, message: str) -> None:
        callback = getattr(self.panel, "update_network_status", None)
        if callable(callback):
            callback(str(message))

    def _set_visual_lighting(self, mode: str) -> None:
        """Atualiza o ODIN sem reenviar a tecla que já foi transmitida."""
        callback = getattr(self.panel, "change_lighting", None)
        if callable(callback):
            callback(mode, "adhesive_automation")

    def _send_mode(self, mode: str) -> bool:
        normalized = str(mode or "").strip().upper()
        command = LIGHTING_COMMAND.get(normalized)
        if command is None:
            return False

        sender = getattr(self.panel, "send_command_to_xp", None)
        if not callable(sender):
            self._network_status(
                "Falha na captura multilight: emissor de comandos XP indisponível."
            )
            return False

        if not bool(sender(command)):
            return False

        self._set_visual_lighting(normalized)
        return True

    def start(self) -> bool:
        """Inicia TOP/MID após SIDE já ter sido analisado e armazenado."""
        if self.active:
            return False

        if not str(getattr(self.panel, "last_xp_ip", "") or "").strip():
            self._network_status(
                "Falha na captura multilight: AOI Windows XP não identificada."
            )
            return False

        self._session_token += 1
        self.active = True
        self.completed = False
        self.expected_mode = ""
        self.captured_modes = {"SIDE"}
        self.retry_count = 0
        self._set_panel_flag(True)
        self._set_auxiliary_receiver(True)

        self._status(
            "Adesivo detectado: iniciando captura automática SIDE/TOP/MID.",
            True,
        )
        return self._request_mode("TOP", reset_retry=True)

    def _request_mode(self, mode: str, *, reset_retry: bool) -> bool:
        if not self.active:
            return False

        normalized = str(mode or "").strip().upper()
        if normalized not in CAPTURE_SEQUENCE:
            return False

        if reset_retry:
            self.retry_count = 0

        # Define a expectativa antes do envio para que qualquer retorno rápido
        # do XP já pertença ao estado correto da máquina.
        self.expected_mode = normalized

        if not self._send_mode(normalized):
            self.abort(
                "Falha ao comandar iluminação {0} no Windows XP.".format(
                    normalized
                )
            )
            return False

        self._status(
            "Captura automática de adesivo: aguardando imagem {0}.".format(
                normalized
            ),
            True,
        )
        self._timeout.start(FRAME_TIMEOUT_MS)
        return True

    def expected_frame_mode(self) -> str:
        if not self.active:
            return ""
        return self.expected_mode

    def frame_stored(self, mode: str) -> bool:
        """Avança somente quando o frame esperado foi recortado e armazenado."""
        if not self.active:
            return False

        normalized = str(mode or "").strip().upper()
        if normalized != self.expected_mode:
            self._network_status(
                "Frame multilight inesperado: esperado {0}, recebido {1}.".format(
                    self.expected_mode or "-",
                    normalized or "-",
                )
            )
            return False

        self._timeout.stop()
        self.captured_modes.add(normalized)
        self.retry_count = 0

        if normalized == "TOP":
            # Agenda no próximo giro do event loop para não aninhar comando TCP
            # dentro do callback que acabou de renderizar o frame TOP.
            QTimer.singleShot(
                0,
                lambda: self._request_mode("MID", reset_retry=True),
            )
            return True

        if normalized == "MID":
            QTimer.singleShot(0, self.finish)
            return True

        return False

    def _handle_timeout(self) -> None:
        if not self.active or not self.expected_mode:
            return

        if self.retry_count < MAX_FRAME_RETRIES:
            self.retry_count += 1
            self._network_status(
                "Imagem {0} não chegou a tempo; repetindo comando ({1}/{2}).".format(
                    self.expected_mode,
                    self.retry_count,
                    MAX_FRAME_RETRIES,
                )
            )
            self._request_mode(
                self.expected_mode,
                reset_retry=False,
            )
            return

        self.abort(
            "Captura automática de adesivo interrompida: imagem {0} não foi recebida.".format(
                self.expected_mode
            )
        )

    def _restore_side(self) -> bool:
        restored = self._send_mode("SIDE")
        if not restored:
            self._network_status(
                "ALERTA: não foi possível restaurar automaticamente a iluminação SIDE."
            )
        return restored

    def finish(self) -> bool:
        """Fecha a aquisição quando SIDE, TOP e MID já estão disponíveis."""
        if not self.active:
            return False

        self._timeout.stop()
        self.expected_mode = ""

        # Fecha a exceção do gate antes de restaurar SIDE. Assim o próximo frame
        # periódico da AOI não sobrescreve os previews já capturados.
        self._set_auxiliary_receiver(False)
        restored = self._restore_side()

        self.active = False
        self.completed = True
        self._set_panel_flag(False)

        missing = {"SIDE", "TOP", "MID"} - self.captured_modes
        if missing:
            self._network_status(
                "Captura multilight finalizada com imagens ausentes: {0}.".format(
                    ", ".join(sorted(missing))
                )
            )
            return False

        suffix = "" if restored else " SIDE precisa ser verificada manualmente."
        self._status(
            "Captura automática de adesivo concluída: SIDE, TOP e MID recebidas."
            + suffix,
            False,
        )

        # A decisão SIDE guardada durante o início da coleta deixa de ter
        # autoridade. O único resultado elegível agora é a fusão das 3 luzes.
        self.panel.adhesive_multilight_deferred_auto_decision = ""

        finalize = getattr(
            self.panel,
            "finalize_adhesive_multilight_decision",
            None,
        )
        fused = finalize() if callable(finalize) else None
        if not isinstance(fused, dict):
            self._network_status(
                "Falha ao concluir julgamento multilight de adesivo."
            )
            self.completed = False
            return False

        try:
            mode = str(self.panel.combo_mode.currentText() or "")
        except Exception:
            mode = ""

        review_required = bool(
            fused.get("production_review_required", False)
            or str(fused.get("verdict", "") or "").strip().upper()
            == "REVISÃO OBRIGATÓRIA"
        )

        if mode == "Modo Produção" and not review_required:
            final_decision = (
                "NG" if bool(fused.get("is_defect", False)) else "OK"
            )
            QTimer.singleShot(
                0,
                lambda decision=final_decision: self.panel.save_label(
                    decision,
                    source="auto",
                ),
            )

        return True

    def abort(
        self,
        reason: str = "",
        *,
        restore_side: bool = True,
        keep_manual_fallback: bool = True,
    ) -> bool:
        """Interrompe a automação sem encerrar a peça ativa."""
        was_active = self.active
        self._timeout.stop()
        self.expected_mode = ""
        self.active = False
        self.completed = False
        self._set_panel_flag(False)

        if restore_side and was_active:
            self._restore_side()

        # Em falha automática, preserva o modo auxiliar para permitir que o
        # operador ainda preencha TOP/MID manualmente na mesma peça.
        self._set_auxiliary_receiver(bool(keep_manual_fallback and was_active))

        if reason:
            self._network_status(str(reason))
            self._status(
                "Captura automática multilight requer intervenção manual.",
                False,
            )
        return was_active

    def cancel_for_cycle_end(self) -> None:
        """Cancela timer/estado e devolve a AOI para SIDE antes do próximo ciclo."""
        was_active = self.active
        self._timeout.stop()
        self.expected_mode = ""

        # Se o operador encerrou a peça no meio de TOP/MID, restauramos SIDE
        # antes de liberar o próximo ciclo. Em uma automação já concluída,
        # SIDE já foi restaurada e não reenviamos a tecla.
        if was_active:
            self._restore_side()

        self.active = False
        self.completed = False
        self.captured_modes = set()
        self.retry_count = 0
        self._set_panel_flag(False)
        self._set_auxiliary_receiver(False)


def install_adhesive_multilight_automation(panel) -> None:
    """Instala uma única máquina de estados no painel."""
    if getattr(panel, "_adhesive_multilight_automation_installed", False):
        return

    controller = AdhesiveMultiLightAutomation(panel)
    panel.adhesive_multilight_automation = controller
    panel.adhesive_multilight_automation_active = False
    panel._adhesive_multilight_automation_installed = True


__all__ = [
    "AdhesiveMultiLightAutomation",
    "CAPTURE_SEQUENCE",
    "FRAME_TIMEOUT_MS",
    "LIGHTING_COMMAND",
    "MAX_FRAME_RETRIES",
    "install_adhesive_multilight_automation",
]
