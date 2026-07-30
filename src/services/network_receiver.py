# src\services\network_receiver.py
"""
Módulo de recepção de rede em background.

Escuta a porta 5001 para imagens e comandos do Windows XP. Depois que uma
imagem é entregue à IA, a entrada de novas imagens permanece fechada até a
captura ser julgada ou descartada. Comandos físicos continuam liberados.
"""

import select
import socket
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

    def __init__(self, port=5001):
        super().__init__()
        self.port = port
        self._is_running = True
        self._image_gate = ImageCycleGate()
        self._last_ignored_log_at = 0.0

    def lock_image_gate(self) -> None:
        """Impede que uma nova imagem seja emitida para a interface."""
        self._image_gate.lock()

    def release_image_gate(self) -> None:
        """Libera exatamente a próxima imagem enviada pelo XP."""
        self._image_gate.release()
        self._last_ignored_log_at = 0.0

    def is_image_gate_open(self) -> bool:
        return self._image_gate.is_open()

    def image_gate_snapshot(self):
        return self._image_gate.snapshot()

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
                    # Drenamos os bytes, mas não decodificamos nem emitimos sinal.
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

                    # Conserva o debounce existente: se outra conexão já chegou,
                    # processa a mais recente antes de reservar o ciclo.
                    conexoes_esperando, _, _ = select.select(
                        [servidor], [], [], 0.0
                    )
                    if conexoes_esperando:
                        self.log_updated.emit(
                            "Imagem mais recente aguardando; captura intermediária descartada."
                        )
                        continue

                    # Reserva atomicamente o ciclo antes de emitir o sinal.
                    if not self._image_gate.try_reserve():
                        self._log_ignored_image(already_counted=True)
                        continue

                    self.log_updated.emit(
                        "Imagem aceita. Entrada bloqueada até OK, NG ou descarte."
                    )
                    self.image_received.emit(img, ip_origem)

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
