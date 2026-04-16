# src\services\network_receiver.py
"""
Módulo de Recepção de Rede em Background.
Escuta a porta 5001 aguardando imagens comprimidas via zlib enviadas pelo Windows XP.
Também escuta comandos curtos de texto (CMD_OK, CMD_NG) vindos do teclado do XP.
Emite a imagem via PyQt Signal para ser processada pela IA sem travar a interface.
"""
import socket
import zlib
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

class NetworkReceiver(QThread):
    # Sinais para comunicar com a Interface Gráfica (Control Panel)
    # Envia a imagem (numpy array) e o IP do XP
    image_received = pyqtSignal(np.ndarray, str)
    
    # Envia mensagens de texto para atualizar o painel
    log_updated = pyqtSignal(str) 
    
    # NOVO: Envia um aviso de que o XP tomou uma decisão física
    command_received = pyqtSignal(str)

    def __init__(self, port=5001):
        super().__init__()
        self.port = port
        self._is_running = True

    def run(self):
        """
        Loop principal da Thread. Fica ativo o tempo todo em segundo plano.
        """
        servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            servidor.bind(('0.0.0.0', self.port))
            servidor.listen(5)
            
            # Timeout de 1 segundo para o accept(). 
            # Isso garante que o while cheque o self._is_running regularmente e possa ser fechado.
            servidor.settimeout(1.0) 
            
            self.log_updated.emit(f"📡 Receptor de Rede ATIVO na porta {self.port}...")
            
            while self._is_running:
                try:
                    conexao, endereco = servidor.accept()
                except socket.timeout:
                    # Timeout normal de 1s, volta para o início do loop
                    continue
                except Exception as e:
                    if self._is_running:
                        self.log_updated.emit(f"❌ Erro no socket: {e}")
                    continue

                try:
                    ip_origem = endereco[0]
                    
                    # 1. Recebe o cabeçalho inicial (16 bytes)
                    cabecalho_str = conexao.recv(16).decode('utf-8').strip()
                    if not cabecalho_str:
                        conexao.close()
                        continue

                    # =========================================================
                    # MUDANÇA: ROTEADOR DE FLUXO (FOTO vs COMANDO FÍSICO)
                    # =========================================================
                    
                    # Se for um comando do teclado físico do XP:
                    if cabecalho_str.startswith("CMD_"):
                        comando = cabecalho_str.split("_")[1] # Extrai apenas "OK" ou "NG"
                        self.log_updated.emit(f"⌨️ Comando físico detectado no XP: {comando}")
                        self.command_received.emit(comando)
                        conexao.close()
                        continue
                        
                    # Se não for comando, então o cabeçalho é um número (tamanho da foto)
                    tamanho_total = int(cabecalho_str)
                    self.log_updated.emit(f"🚨 ALERTA: Recebendo {tamanho_total / 1024:.0f} KB do XP ({ip_origem})...")

                    # 2. Recebe a foto compactada em pedaços
                    buffer = b""
                    while len(buffer) < tamanho_total:
                        pacote = conexao.recv(8192)
                        if not pacote: break
                        buffer += pacote
                    conexao.close()

                    # 3. Descompacta e Converte
                    dados_originais = zlib.decompress(buffer)
                    nparr = np.frombuffer(dados_originais, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        self.log_updated.emit("✅ Imagem recebida com sucesso! Iniciando IA...")
                        
                        # A MÁGICA: Em vez de usar cv2.imshow, enviamos a imagem para o ControlPanel!
                        self.image_received.emit(img, ip_origem)
                    else:
                        self.log_updated.emit("❌ Erro ao decodificar a imagem da rede.")

                except Exception as e:
                    self.log_updated.emit(f"❌ Erro durante recebimento/processamento: {e}")
                    if 'conexao' in locals():
                        conexao.close()

        except Exception as e:
            self.log_updated.emit(f"❌ Falha fatal ao iniciar o servidor na porta {self.port}: {e}")
        finally:
            servidor.close()
            self.log_updated.emit("💤 Receptor de Rede desligado.")

    def stop(self):
        """Para a thread com segurança ao fechar o programa principal."""
        self._is_running = False
        self.wait()