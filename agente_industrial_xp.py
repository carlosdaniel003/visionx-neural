# agente_industrial_xp.py (V5.3 - FULL DUPLEX C/ KERNEL HOOK + TELEMETRIA DE SETAS: COMPATIVEL PYTHON 3.4)
# Teste de envio
import socket
import win32gui
import win32api
import win32con
import win32ui
import os
import tempfile
import zlib
import time
import threading
import ctypes
from ctypes import wintypes

# --- CONFIGURACOES DA TELA E GATILHO ---
PONTO_AZUL = (238, 238)
PONTO_VERMELHO = (834, 239)

AREA_BUSCA = 15
CROP_LARGURA = 1165
CROP_ALTURA = 840

IP_IA = '169.254.87.66'
PORTA_IA = 5001
PORTA_COMANDOS_XP = 5000

# Virtual-Key Codes usados no Windows XP.
VK_0 = 0x30
VK_1 = 0x31
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_DOWN = 0x28

TECLAS_VIRTUAIS = {
    "0": VK_0,
    "1": VK_1,
    "LEFT": VK_LEFT,
    "RIGHT": VK_RIGHT,
    "DOWN": VK_DOWN,
}

user32 = ctypes.WinDLL('user32', use_last_error=True)

# Modo rapido ativado EXCLUSIVAMENTE pelo ODIN via comando TCP.
# O lease impede o XP de ficar em modo rapido apos falha/reinicio do ODIN.
DEFAULT_CAPTURE_PAUSE_SECONDS = 3.0
SHADOW_CAPTURE_PAUSE_SECONDS = 0.18
SHADOW_LEASE_SECONDS = 25.0
_shadow_capture_until = 0.0
_shadow_capture_wakeup = threading.Event()

def configurar_captura_sombra(habilitado):
    global _shadow_capture_until
    _shadow_capture_until = (
        time.time() + SHADOW_LEASE_SECONDS if habilitado else 0.0
    )
    if habilitado:
        # Acorda a primeira espera de 3s assim que o ODIN ativa Sombra.
        _shadow_capture_wakeup.set()

def pausa_pos_envio():
    if time.time() < _shadow_capture_until:
        return SHADOW_CAPTURE_PAUSE_SECONDS
    return DEFAULT_CAPTURE_PAUSE_SECONDS

def aguardar_proximo_envio():
    pausa = pausa_pos_envio()
    despertou = _shadow_capture_wakeup.wait(pausa)
    if despertou:
        _shadow_capture_wakeup.clear()
        # Nao usar captura instantanea apos mudanca de luz.
        if pausa_pos_envio() < DEFAULT_CAPTURE_PAUSE_SECONDS:
            time.sleep(SHADOW_CAPTURE_PAUSE_SECONDS)


def eh_azul(cor):
    r = cor & 0xFF
    g = (cor >> 8) & 0xFF
    b = (cor >> 16) & 0xFF
    return b > 160 and r < 100 and g < 100


def eh_vermelho(cor):
    r = cor & 0xFF
    g = (cor >> 8) & 0xFF
    b = (cor >> 16) & 0xFF
    return r > 160 and b < 100 and g < 100


def check_trigger():
    desktop_hwnd = win32gui.GetDesktopWindow()
    desktop_dc = win32gui.GetWindowDC(desktop_hwnd)

    achou_azul = False
    achou_vermelho = False

    try:
        for dx in range(-AREA_BUSCA, AREA_BUSCA):
            for dy in range(-AREA_BUSCA, AREA_BUSCA):
                cor = win32gui.GetPixel(
                    desktop_dc,
                    PONTO_AZUL[0] + dx,
                    PONTO_AZUL[1] + dy,
                )
                if eh_azul(cor):
                    achou_azul = True
                    break
            if achou_azul:
                break

        for dx in range(-AREA_BUSCA, AREA_BUSCA):
            for dy in range(-AREA_BUSCA, AREA_BUSCA):
                cor = win32gui.GetPixel(
                    desktop_dc,
                    PONTO_VERMELHO[0] + dx,
                    PONTO_VERMELHO[1] + dy,
                )
                if eh_vermelho(cor):
                    achou_vermelho = True
                    break
            if achou_vermelho:
                break
    finally:
        win32gui.ReleaseDC(desktop_hwnd, desktop_dc)

    return achou_azul and achou_vermelho


def capturar_recorte():
    hdesktop = win32gui.GetDesktopWindow()
    desktop_dc = win32gui.GetWindowDC(hdesktop)
    img_dc = win32ui.CreateDCFromHandle(desktop_dc)
    mem_dc = img_dc.CreateCompatibleDC()

    screenshot = win32ui.CreateBitmap()
    screenshot.CreateCompatibleBitmap(img_dc, CROP_LARGURA, CROP_ALTURA)
    mem_dc.SelectObject(screenshot)

    mem_dc.BitBlt(
        (0, 0),
        (CROP_LARGURA, CROP_ALTURA),
        img_dc,
        (0, 0),
        win32con.SRCCOPY,
    )

    caminho_temp = os.path.join(tempfile.gettempdir(), "aoi_trigger.bmp")
    screenshot.SaveBitmapFile(mem_dc, caminho_temp)

    mem_dc.DeleteDC()
    win32gui.DeleteObject(screenshot.GetHandle())
    win32gui.ReleaseDC(hdesktop, desktop_dc)

    with open(caminho_temp, "rb") as f:
        return zlib.compress(f.read())


# =====================================================================
# FUNCOES: TECLADO FANTASMA E COMUNICADOR BIDIRECIONAL
# =====================================================================

def apertar_tecla_fisica(tecla_str):
    tecla = str(tecla_str).strip().upper()
    codigo = TECLAS_VIRTUAIS.get(tecla)

    if codigo is None:
        print(
            "\n[FANTASMA] Comando de tecla desconhecido: '{0}'.".format(
                tecla
            )
        )
        return False

    print(
        "\n[FANTASMA] A IA comandou. Pressionando '{0}' fisicamente...".format(
            tecla
        )
    )

    win32api.keybd_event(codigo, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(codigo, 0, win32con.KEYEVENTF_KEYUP, 0)
    return True


def servidor_de_comandos():
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    servidor.bind(('0.0.0.0', PORTA_COMANDOS_XP))
    servidor.listen(1)

    while True:
        conexao = None
        try:
            conexao, _ = servidor.accept()
            comando = conexao.recv(1024).decode('utf-8').strip()

            if comando == "VISIONX_SHADOW_ON":
                configurar_captura_sombra(True)
                conexao.sendall(b"ACK_SHADOW_ON")
            elif comando == "VISIONX_SHADOW_OFF":
                configurar_captura_sombra(False)
                conexao.sendall(b"ACK_SHADOW_OFF")
            elif comando == "PRESS_0":
                apertar_tecla_fisica("0")
            elif comando == "PRESS_1":
                apertar_tecla_fisica("1")
            elif comando == "PRESS_LEFT":
                apertar_tecla_fisica("LEFT")
            elif comando == "PRESS_DOWN":
                apertar_tecla_fisica("DOWN")
            elif comando == "PRESS_RIGHT":
                apertar_tecla_fisica("RIGHT")
            else:
                print(
                    "\n[FANTASMA] Comando recebido e ignorado: '{0}'.".format(
                        comando
                    )
                )

        except Exception as e:
            print("-> Erro no servidor de comandos: {0}".format(e))
        finally:
            if conexao is not None:
                try:
                    conexao.close()
                except Exception:
                    pass


def enviar_aviso_teclado_ia(mensagem):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((IP_IA, PORTA_IA))

        pacote = mensagem.ljust(16).encode('utf-8')
        s.send(pacote)
        s.close()
    except Exception as e:
        print("-> Erro ao avisar a IA sobre o teclado: {0}".format(e))


# =====================================================================
# KERNEL HOOK GLOBAL (INTERCEPTADOR DE TECLADO INCONDICIONAL)
# =====================================================================

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100

CMPFUNC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
)


def hook_proc(nCode, wParam, lParam):
    if nCode >= 0 and wParam == WM_KEYDOWN:
        vkCode = lParam[0]

        # 0x30 = Tecla '0' | 0x31 = Tecla '1'
        # 0x60 = Numpad '0' | 0x61 = Numpad '1'
        if vkCode == 0x30 or vkCode == 0x60:
            print(
                "\n[TECLADO KERNEL] Operador apertou '0' "
                "(Falha Falsa). Avisando IA..."
            )
            threading.Thread(
                target=enviar_aviso_teclado_ia,
                args=("CMD_OK",),
            ).start()

        elif vkCode == 0x31 or vkCode == 0x61:
            print(
                "\n[TECLADO KERNEL] Operador apertou '1' "
                "(Defeito Real). Avisando IA..."
            )
            threading.Thread(
                target=enviar_aviso_teclado_ia,
                args=("CMD_NG",),
            ).start()

        elif vkCode == VK_LEFT:
            print(
                "\n[TECLADO KERNEL] Seta ESQUERDA detectada "
                "(TOP). Avisando IA..."
            )
            threading.Thread(
                target=enviar_aviso_teclado_ia,
                args=("CMD_TOP",),
            ).start()

        elif vkCode == VK_DOWN:
            print(
                "\n[TECLADO KERNEL] Seta BAIXO detectada "
                "(SIDE). Avisando IA..."
            )
            threading.Thread(
                target=enviar_aviso_teclado_ia,
                args=("CMD_SIDE",),
            ).start()

        elif vkCode == VK_RIGHT:
            print(
                "\n[TECLADO KERNEL] Seta DIREITA detectada "
                "(MID). Avisando IA..."
            )
            threading.Thread(
                target=enviar_aviso_teclado_ia,
                args=("CMD_MID",),
            ).start()

    return user32.CallNextHookEx(None, nCode, wParam, lParam)


def iniciar_hook_teclado():
    """Roda o interceptador na camada mais baixa do Windows."""
    pointer = CMPFUNC(hook_proc)
    hook_id = user32.SetWindowsHookExA(
        WH_KEYBOARD_LL,
        pointer,
        win32api.GetModuleHandle(None),
        0,
    )

    if not hook_id:
        print("-> Erro fatal ao instalar o gancho de teclado no Kernel.")
        return

    msg = wintypes.MSG()
    while user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageA(ctypes.byref(msg))

    user32.UnhookWindowsHookEx(hook_id)


# =====================================================================
# LOOP PRINCIPAL DO AGENTE
# =====================================================================

def loop_vigia_tela():
    """Movemos o loop visual para uma Thread separada."""
    while True:
        if check_trigger():
            print("!!! GATILHO DETECTADO - ENVIANDO FOTO PARA IA !!!")
            try:
                servidor_ia = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                servidor_ia.settimeout(5.0)
                servidor_ia.connect((IP_IA, PORTA_IA))

                dados_zip = capturar_recorte()

                servidor_ia.send(
                    str(len(dados_zip)).ljust(16).encode('utf-8')
                )
                servidor_ia.sendall(dados_zip)
                servidor_ia.close()

                # Sem comando do ODIN, mantem a pausa original de 3 s.
                # Sombra: captura mais frequente, SEM eliminar os dois
                # frames estaveis exigidos no PC VisionX.
                aguardar_proximo_envio()

            except Exception as e:
                print(
                    "-> Erro: PC moderno nao respondeu "
                    "(Porta 5001). {0}".format(e)
                )

        time.sleep(0.1)


def iniciar_agente():
    print("==========================================")
    print(">>> AGENTE VISIONX V5.3 - FULL DUPLEX (KERNEL HOOK + TELEMETRIA DE SETAS)")
    print(">>> Monitorando Tela, Rede e Teclado Fisico (Global)...")
    print(">>> Comandos: 0, 1, LEFT, DOWN, RIGHT | Retorno: OK, NG, TOP, SIDE, MID")
    print("==========================================")

    # 1. Liga o Ouvinte da IA (Porta 5000)
    t_comandos = threading.Thread(target=servidor_de_comandos)
    t_comandos.daemon = True
    t_comandos.start()

    # 2. Liga o Vigia Visual da Tela (Olhando as barras coloridas)
    t_tela = threading.Thread(target=loop_vigia_tela)
    t_tela.daemon = True
    t_tela.start()

    # 3. Trava a Main Thread no Loop de Mensagens do Windows Hook
    # Isso garante que mesmo minimizado ou clicando fora, o XP sentira o teclado.
    iniciar_hook_teclado()


if __name__ == "__main__":
    iniciar_agente()
