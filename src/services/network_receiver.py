# src\services\network_receiver.py
"""
Recepção contínua de imagens e comandos enviados pelo Windows XP.

O receptor mantém somente o frame candidato mais recente. Uma imagem só é
entregue à interface depois de aparecer de forma estável em envios consecutivos.
A reserva definitiva do ciclo continua acontecendo antes do sinal Qt, mas a
assinatura só é considerada uma anomalia aceita depois da validação do painel.
Telas de central/transição rejeitadas são lembradas e ignoradas até a tela mudar.
"""

from __future__ import annotations

import select
import socket
import threading
import time
import zlib

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.services.image_cycle_gate import ImageCycleGate


class NetworkReceiver(QThread):
    image_received = pyqtSignal(np.ndarray, str)
    log_updated = pyqtSignal(str)
    command_received = pyqtSignal(str)

    SIGNATURE_WIDTH = 128
    SIGNATURE_HEIGHT = 96
    SAME_FRAME_MEAN_DELTA = 0.85
    SAME_FRAME_P95_DELTA = 4.5
    SAME_FRAME_CHANGED_RATIO = 0.004
    STABLE_REQUIRED_FRAMES = 2

    def __init__(self, port=5001):
        super().__init__()
        self.port = port
        self._is_running = True
        self._image_gate = ImageCycleGate()
        self._state_lock = threading.RLock()

        self._last_ignored_log_at = 0.0
        self._last_duplicate_log_at = 0.0
        self._last_rejected_log_at = 0.0
        self._last_stability_log_at = 0.0

        # Última anomalia realmente validada pelo painel.
        self._last_accepted_signature: np.ndarray | None = None
        self._require_image_change = False
        self._duplicate_frames = 0

        # Tela que chegou ao painel, mas foi classificada como central/transição.
        self._last_rejected_signature: np.ndarray | None = None
        self._last_rejected_reason = ""
        self._rejected_frames = 0
        self._transition_observed_since_accept = False

        # Buffer latest-only e reserva ainda não confirmada pelo painel.
        self._candidate_signature: np.ndarray | None = None
        self._candidate_image: np.ndarray | None = None
        self._candidate_ip = ""
        self._candidate_stable_frames = 0
        self._reserved_signature: np.ndarray | None = None

    def lock_image_gate(self) -> None:
        """Impede que uma nova imagem seja emitida para a interface."""
        self._image_gate.lock()

    def _clear_candidate_locked(self) -> None:
        self._candidate_signature = None
        self._candidate_image = None
        self._candidate_ip = ""
        self._candidate_stable_frames = 0

    def release_image_gate(self) -> None:
        """Libera a procura pela próxima anomalia válida.

        Uma assinatura reservada que não foi confirmada pelo filtro do painel é
        automaticamente tratada como tela inválida. Isso também cobre falha de
        barras, recorte vazio e timeout do watchdog.
        """
        with self._state_lock:
            if self._reserved_signature is not None:
                self._last_rejected_signature = self._reserved_signature.copy()
                self._last_rejected_reason = (
                    self._last_rejected_reason
                    or "captura não confirmou uma inspeção AOI válida"
                )
                self._reserved_signature = None
                self._transition_observed_since_accept = True

            self._clear_candidate_locked()
            self._require_image_change = bool(
                self._last_accepted_signature is not None
                and not self._transition_observed_since_accept
            )
            self._duplicate_frames = 0
            self._rejected_frames = 0

        self._image_gate.release()
        self._last_ignored_log_at = 0.0

    def confirm_reserved_image(self) -> bool:
        """Confirma que a reserva atual contém epicentro válido."""
        with self._state_lock:
            if self._reserved_signature is None:
                return False
            self._last_accepted_signature = self._reserved_signature.copy()
            self._reserved_signature = None
            self._transition_observed_since_accept = False
            self._require_image_change = False
            self._duplicate_frames = 0
            return True

    def mark_reserved_image_rejected(self, reason: str = "") -> bool:
        """Marca a reserva atual como central/transição sem torná-la uma peça."""
        with self._state_lock:
            if self._reserved_signature is None:
                return False
            self._last_rejected_signature = self._reserved_signature.copy()
            self._last_rejected_reason = str(reason or "tela sem epicentro válido")
            self._reserved_signature = None
            self._transition_observed_since_accept = True
            self._require_image_change = False
            self._clear_candidate_locked()

        self.log_updated.emit(
            "Tela recebida ignorada: "
            f"{self._last_rejected_reason}. Aguardando a próxima anomalia válida."
        )
        return True

    def is_image_gate_open(self) -> bool:
        return self._image_gate.is_open()

    def image_gate_snapshot(self):
        return self._image_gate.snapshot()

    @classmethod
    def _frame_signature(cls, image: np.ndarray) -> np.ndarray:
        """Assinatura visual estável, ignorando bordas e pequenos textos."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            return np.zeros(
                (cls.SIGNATURE_HEIGHT, cls.SIGNATURE_WIDTH),
                dtype=np.uint8,
            )

        height, width = image.shape[:2]
        top = int(round(height * 0.08))
        bottom = int(round(height * 0.94))
        left = int(round(width * 0.02))
        right = int(round(width * 0.98))
        cropped = (
            image
            if bottom <= top or right <= left
            else image[top:bottom, left:right]
        )

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (cls.SIGNATURE_WIDTH, cls.SIGNATURE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @classmethod
    def _same_signature(
        cls,
        first: np.ndarray | None,
        second: np.ndarray | None,
    ) -> bool:
        if first is None or second is None or first.shape != second.shape:
            return False
        difference = cv2.absdiff(first, second).astype(np.float32)
        mean_delta = float(np.mean(difference))
        p95_delta = float(np.percentile(difference, 95))
        changed_ratio = float(np.mean(difference > 8.0))
        return bool(
            mean_delta <= cls.SAME_FRAME_MEAN_DELTA
            and p95_delta <= cls.SAME_FRAME_P95_DELTA
            and changed_ratio <= cls.SAME_FRAME_CHANGED_RATIO
        )

    def _stage_latest_candidate(
        self,
        image: np.ndarray,
        signature: np.ndarray,
        ip: str,
    ) -> int:
        """Mantém só o frame mais recente e retorna sua estabilidade atual."""
        with self._state_lock:
            if self._same_signature(signature, self._candidate_signature):
                self._candidate_stable_frames += 1
            else:
                self._candidate_stable_frames = 1

            self._candidate_signature = signature.copy()
            self._candidate_image = image.copy()
            self._candidate_ip = str(ip)
            return int(self._candidate_stable_frames)

    def _candidate_snapshot(self):
        with self._state_lock:
            if (
                self._candidate_signature is None
                or self._candidate_image is None
                or self._candidate_stable_frames < self.STABLE_REQUIRED_FRAMES
            ):
                return None
            return (
                self._candidate_image.copy(),
                self._candidate_signature.copy(),
                self._candidate_ip,
            )

    def _reserve_candidate(self, signature: np.ndarray) -> None:
        with self._state_lock:
            self._reserved_signature = signature.copy()
            self._last_rejected_reason = ""
            self._clear_candidate_locked()

    def _is_rejected_repeat(self, signature: np.ndarray) -> bool:
        with self._state_lock:
            return self._same_signature(
                signature,
                self._last_rejected_signature,
            )

    @staticmethod
    def _receive_payload(connection: socket.socket, total_size: int) -> bytes:
        chunks = []
        received = 0
        while received < total_size:
            packet = connection.recv(min(8192, total_size - received))
            if not packet:
                break
            chunks.append(packet)
            received += len(packet)
        return b"".join(chunks)

    @staticmethod
    def _discard_payload(connection: socket.socket, total_size: int) -> None:
        """Drena o envio para fechar a conexão sem provocar erro no XP."""
        remaining = max(0, int(total_size))
        while remaining > 0:
            packet = connection.recv(min(8192, remaining))
            if not packet:
                break
            remaining -= len(packet)

    def _log_ignored_image(self, *, already_counted: bool = False) -> None:
        ignored = (
            self._image_gate.snapshot().ignored_images
            if already_counted
            else self._image_gate.note_ignored()
        )
        now = time.monotonic()
        if ignored == 1 or now - self._last_ignored_log_at >= 3.0:
            self.log_updated.emit(
                "Captura atual aguardando julgamento/descarte — "
                f"{ignored} nova(s) imagem(ns) do XP ignorada(s)."
            )
            self._last_ignored_log_at = now

    def _log_duplicate_frame(self) -> None:
        self._duplicate_frames += 1
        now = time.monotonic()
        if self._duplicate_frames == 1 or now - self._last_duplicate_log_at >= 3.0:
            self.log_updated.emit(
                "A tela do XP ainda é igual à última anomalia julgada — "
                f"{self._duplicate_frames} frame(s) repetido(s) ignorado(s)."
            )
            self._last_duplicate_log_at = now

    def _log_rejected_frame(self) -> None:
        self._rejected_frames += 1
        now = time.monotonic()
        if self._rejected_frames == 1 or now - self._last_rejected_log_at >= 3.0:
            reason = self._last_rejected_reason or "tela sem epicentro válido"
            self.log_updated.emit(
                "Tela de central/transição ainda presente — "
                f"{self._rejected_frames} frame(s) ignorado(s). Motivo: {reason}."
            )
            self._last_rejected_log_at = now

    def _log_waiting_stability(self, count: int) -> None:
        now = time.monotonic()
        if count == 1 or now - self._last_stability_log_at >= 3.0:
            self.log_updated.emit(
                "Novo frame candidato recebido; aguardando confirmação "
                f"{count}/{self.STABLE_REQUIRED_FRAMES}."
            )
            self._last_stability_log_at = now

    def run(self):
        servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            servidor.bind(("0.0.0.0", self.port))
            servidor.listen(5)
            servidor.settimeout(1.0)
            self.log_updated.emit(
                f"Receptor de rede ativo na porta {self.port}."
            )

            while self._is_running:
                try:
                    conexao, endereco = servidor.accept()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if self._is_running:
                        self.log_updated.emit(f"Erro no socket: {exc}")
                    continue

                try:
                    ip_origem = endereco[0]
                    cabecalho_str = conexao.recv(16).decode("utf-8").strip()
                    if not cabecalho_str:
                        conexao.close()
                        continue

                    # Comandos nunca são bloqueados pela trava de imagens.
                    if cabecalho_str.startswith("CMD_"):
                        comando = cabecalho_str.split("_", 1)[1]
                        self.log_updated.emit(
                            f"Comando físico detectado no XP: {comando}"
                        )
                        self.command_received.emit(comando)
                        conexao.close()
                        continue

                    tamanho_total = int(cabecalho_str)

                    # A captura anterior ainda está em processamento ou revisão.
                    if not self._image_gate.is_open():
                        self._discard_payload(conexao, tamanho_total)
                        conexao.close()
                        self._log_ignored_image()
                        continue

                    self.log_updated.emit(
                        f"Recebendo {tamanho_total / 1024:.0f} KB do XP ({ip_origem})..."
                    )
                    buffer = self._receive_payload(conexao, tamanho_total)
                    conexao.close()

                    if len(buffer) != tamanho_total:
                        self.log_updated.emit(
                            "Imagem de rede incompleta; captura descartada."
                        )
                        continue

                    dados_originais = zlib.decompress(buffer)
                    nparr = np.frombuffer(dados_originais, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        self.log_updated.emit(
                            "Erro ao decodificar a imagem da rede."
                        )
                        continue

                    signature = self._frame_signature(img)
                    if self._require_image_change and self._same_signature(
                        signature,
                        self._last_accepted_signature,
                    ):
                        self._log_duplicate_frame()
                        continue

                    # Uma central já rejeitada não volta a ocupar o ciclo.
                    if self._is_rejected_repeat(signature):
                        self._log_rejected_frame()
                        continue

                    stable_count = self._stage_latest_candidate(
                        img,
                        signature,
                        ip_origem,
                    )
                    if stable_count < self.STABLE_REQUIRED_FRAMES:
                        self._log_waiting_stability(stable_count)
                        continue

                    # Se já existe um envio mais recente aguardando no socket,
                    # ele substitui o candidato atual: não há fila histórica.
                    conexoes_esperando, _, _ = select.select(
                        [servidor], [], [], 0.0
                    )
                    if conexoes_esperando:
                        self.log_updated.emit(
                            "Frame mais recente aguardando; candidato anterior substituível."
                        )
                        continue

                    candidate = self._candidate_snapshot()
                    if candidate is None:
                        continue
                    candidate_image, candidate_signature, candidate_ip = candidate

                    # Reserva atomicamente antes de emitir o sinal Qt.
                    if not self._image_gate.try_reserve():
                        self._log_ignored_image(already_counted=True)
                        continue

                    self._reserve_candidate(candidate_signature)
                    self._require_image_change = False
                    self._duplicate_frames = 0
                    self._rejected_frames = 0
                    self.log_updated.emit(
                        "Frame estável encaminhado para validação do epicentro."
                    )
                    self.image_received.emit(candidate_image, candidate_ip)

                except Exception as exc:
                    self.log_updated.emit(
                        f"Erro durante recebimento/processamento: {exc}"
                    )
                    try:
                        conexao.close()
                    except Exception:
                        pass

        except Exception as exc:
            self.log_updated.emit(
                f"Falha fatal ao iniciar o servidor na porta {self.port}: {exc}"
            )
        finally:
            servidor.close()
            self.log_updated.emit("Receptor de rede desligado.")

    def stop(self):
        self._is_running = False
        self.wait()
