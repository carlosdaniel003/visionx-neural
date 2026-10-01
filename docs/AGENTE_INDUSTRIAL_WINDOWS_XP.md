# Agente Industrial do Windows XP — referência operacional

## Objetivo deste documento

Este documento registra o papel do arquivo `agente_industrial_xp.py` dentro da arquitetura do **VisionX Neural** e a forma correta de mantê-lo.

> **Regra importante:** o arquivo `agente_industrial_xp.py` mantido no GitHub deve ser tratado como **cópia de referência/consulta**. Alterar a cópia documentada no repositório **não altera automaticamente** o arquivo que está sendo executado na máquina Windows XP da AOI.

Quando uma alteração nesse agente for necessária, o usuário deve ser informado de que precisa **atualizar manualmente o arquivo no Windows XP** antes que a mudança passe a valer na AOI.

---

## Onde o agente realmente roda

O agente é executado **manualmente na máquina Windows XP da AOI**.

Caminho atualmente utilizado no Windows XP:

```text
C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente\agente_industrial_xp.py
```

Ambiente conhecido:

- Sistema operacional da AOI: **Windows XP**.
- Python disponível no XP: **Python 3 compatível com o ambiente legado**.
- O código atual foi escrito para compatibilidade com **Python 3.4**.
- O agente é iniciado manualmente no XP.
- O computador Windows XP e o computador moderno não dependem de internet para se comunicar.
- A comunicação entre os dois computadores acontece pela rede Ethernet/TCP-IP.

---

## Endereços atualmente utilizados

### Computador novo / VisionX

```text
169.254.87.66
```

### Computador Windows XP / AOI

```text
169.254.95.200
```

Os dois computadores conseguem se comunicar diretamente pela rede local e já respondem a `ping`.

---

## Papel do `agente_industrial_xp.py`

O agente é a ponte entre a AOI legada e o VisionX.

Ele executa três funções principais.

### 1. Captura e envio da imagem da AOI

O agente monitora a tela do Windows XP procurando os gatilhos visuais configurados.

Quando o gatilho é detectado:

1. captura um recorte da tela;
2. salva temporariamente o bitmap;
3. comprime a imagem com `zlib`;
4. abre uma conexão TCP com o computador do VisionX;
5. envia o tamanho do pacote em um cabeçalho de 16 bytes;
6. envia a imagem comprimida.

Destino atual:

```text
VisionX: 169.254.87.66
Porta: 5001
```

Fluxo:

```text
Windows XP / AOI
169.254.95.200
        │
        │ imagem comprimida via TCP
        ▼
VisionX
169.254.87.66:5001
```

---

### 2. Recebimento de comandos do VisionX

O agente mantém um servidor TCP no Windows XP:

```text
Porta 5000
```

O VisionX pode enviar:

```text
PRESS_0
PRESS_1
```

O agente converte esses comandos em pressionamentos reais de teclado no Windows XP:

```text
PRESS_0 → tecla 0 → OK / Falha Falsa
PRESS_1 → tecla 1 → NG / Defeito Real
```

Fluxo:

```text
VisionX
        │
        │ PRESS_0 / PRESS_1
        ▼
Windows XP:5000
        │
        ▼
win32api.keybd_event(...)
        │
        ├── 0 = OK
        └── 1 = NG
```

---

### 3. Captura global do teclado físico do operador

O agente instala um hook global de teclado no Windows XP.

Ele intercepta:

```text
0
1
Numpad 0
Numpad 1
```

Quando o operador pressiona:

```text
0 → CMD_OK
1 → CMD_NG
```

o agente abre uma conexão com:

```text
169.254.87.66:5001
```

e informa ao VisionX a decisão humana.

Portanto a comunicação é bidirecional:

```text
XP → VisionX

0 → CMD_OK → OK
1 → CMD_NG → NG


VisionX → XP

OK → PRESS_0 → tecla 0
NG → PRESS_1 → tecla 1
```

## Atalhos locais no VisionX

Quando uma decisão humana estiver disponível no próprio VisionX:

```text
0 / Numpad 0 → botão OK → PRESS_0 → tecla 0 na AOI → próxima imagem
1 / Numpad 1 → botão NG → PRESS_1 → tecla 1 na AOI → próxima imagem
```

Os atalhos do VisionX reutilizam exatamente os mesmos botões e travas da interface.
Eles não ignoram estados de segurança: se OK/NG estiver indisponível para o ciclo
atual, pressionar 0/1 não força uma decisão.

Essa função é implementada no computador novo. O agente XP V5.1 já entende
`PRESS_0` e `PRESS_1`, portanto essa correção de atalhos não exige alterar o
arquivo operacional do agente no Windows XP.

## Regra do ciclo rápido após julgamento

Quando o operador julga uma imagem recebida pela rede com `0 = OK` ou
`1 = NG`, a decisão produtiva deve terminar antes das tarefas de persistência.

Fluxo obrigatório no VisionX:

```text
imagem A analisada
    ↓
operador pressiona 0 ou 1
    ↓
VisionX envia PRESS_0 / PRESS_1 quando a decisão veio do computador novo
    ↓
VisionX registra a decisão em memória de trabalho
    ↓
VisionX limpa imediatamente a imagem e os painéis da captura A
    ↓
VisionX libera o gate para a próxima imagem da AOI
    ↓
VisionX mostra "Aguardando próxima imagem da AOI"
    ↓
persistência JSON/auditoria + atualização KNN continuam em background
```

### Regra de arquitetura

Não reintroduzir `DatasetManager.save_sample()` nem
`orchestrator.reload_memory()` no caminho crítico anterior à liberação do
gate. Gravação em disco, auditoria e recarga da memória são tarefas de
background e não podem manter uma peça já julgada ocupando a interface.

A fila de persistência é serial, preservando a ordem das decisões humanas. A
recarga do KNN usa troca atômica das listas de memória para que uma nova análise
não observe uma memória parcialmente recarregada.

### Estado visual esperado

Após `0` ou `1` em uma captura de rede já analisada:

- a captura anterior deve desaparecer da área de inspeção;
- OK/NG e descarte ficam indisponíveis até a próxima análise;
- o status deve indicar que o VisionX está aguardando a próxima imagem da AOI;
- o receptor de rede deve ficar livre para receber a próxima peça sem esperar
  gravação JSON, imagens de auditoria ou varredura KNN.

## Arquivo visual NG opcional no VisionX

O computador novo possui um toggle chamado:

```text
Salvar imagens NG
```

Estado padrão:

```text
DESATIVADO
```

Quando desativado, o comportamento permanece igual ao fluxo normal já
documentado e nenhuma cópia visual adicional é criada.

Quando ativado, toda **decisão final NG** de uma captura recebida do Windows XP
— humana ou automática já autorizada pelas regras de produção — gera uma cópia
em:

```text
public/ng_archive/
```

A imagem arquivada deve ser **exatamente o mesmo frame completo do Windows XP**
que o botão `Copiar imagem XP` disponibiliza naquele evento. Os dois recursos
usam a mesma fonte interna e a mesma validação de `event_id`.

Não usar `current_ng`, ROI, recorte de teste ou qualquer outra imagem como
fallback. Se o frame XP do evento atual não estiver preservado ou se o
`event_id` não coincidir com o diagnóstico atual, a evidência NG não é salva.
Isso evita associar ao julgamento uma imagem diferente da que o operador pode
copiar para auditoria.

Formato do nome:

```text
AAAA-MM-DD_HH-MM-SS-ms_CATEGORIA.png
```

Exemplo:

```text
2026-09-30_13-53-27-245_DESLOCADO.png
```

Esse arquivo é somente evidência visual. Ele é independente de
`public/dataset/`, não alimenta o KNN, não altera protótipos e não muda o
julgamento da IA. A fonte compartilhada fica em
`src/services/network_xp_frame.py`.

A gravação é feita em fila de background. Portanto, salvar a evidência NG não
deve bloquear o comando `PRESS_1`, a limpeza da captura julgada nem a
liberação do gate para a próxima imagem da AOI.

Essa função existe apenas no VisionX do computador novo e **não exige alteração
do agente industrial no Windows XP**.

### Gargalo ainda existente no agente XP V5.1

O agente operacional documentado ainda contém:

```python
print("-> Imagem enviada com sucesso. Pausa de 3s...")
time.sleep(3)
```

O VisionX mantém `STABLE_REQUIRED_FRAMES = 2` como proteção contra telas em
transição. Portanto a pausa fixa de 3 segundos do agente pode continuar
adicionando latência entre os dois frames necessários para confirmar uma nova
captura.

**Esta etapa de otimização não altera o agente XP.** A remoção ou redução dessa
pausa deve ser tratada separadamente, medida na AOI real e exigirá atualizar
manualmente o arquivo operacional no Windows XP.

---

# Regra de manutenção

## O GitHub não atualiza automaticamente o Windows XP

A cópia deste agente registrada no projeto serve para:

- consulta;
- auditoria;
- entendimento da integração;
- comparação de versões;
- planejamento de alterações;
- recuperação do código caso o arquivo operacional seja perdido.

Ela **não deve ser tratada como deploy automático**.

Se alguém modificar no GitHub:

```text
agente_industrial_xp.py
```

isso não modifica o arquivo real em:

```text
C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente\agente_industrial_xp.py
```

Para a alteração entrar em operação, deve ser solicitada uma atualização manual no computador XP.

Sempre que uma mudança futura depender do agente, informar explicitamente:

> **Esta alteração exige atualizar manualmente o `agente_industrial_xp.py` utilizado na máquina Windows XP da AOI. Alterar somente a referência no GitHub não muda o agente que está em execução.**

---

# Transferência de arquivos entre Windows XP e computador novo sem internet ou pendrive

## Cenário

Quando houver:

- um computador Windows XP;
- um computador Windows 10/novo;
- nenhum acesso a pendrive;
- nenhuma internet;
- os dois computadores conectados à mesma rede Ethernet;
- comunicação IP funcionando;
- Python 3 disponível no Windows XP;

é possível transferir arquivos diretamente por TCP/IP usando o servidor HTTP simples do Python.

Exemplo atual:

```text
Computador novo: 169.254.87.66
Windows XP:      169.254.95.200
```

---

## Trazer um arquivo do Windows XP para o computador novo

Exemplo de arquivo:

```text
agente_industrial_xp.py
```

Local no Windows XP:

```text
C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente
```

### 1. Abrir o Prompt de Comando no Windows XP

Pressionar:

```text
Win + R
```

Digitar:

```text
cmd
```

### 2. Entrar na pasta do agente

No CMD do Windows XP:

```cmd
cd /d "C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente"
```

### 3. Iniciar um servidor HTTP temporário

```cmd
python -m http.server 8000
```

A janela deverá mostrar algo semelhante a:

```text
Serving HTTP on 0.0.0.0 port 8000 ...
```

### 4. No computador novo

Abrir no navegador:

```text
http://169.254.95.200:8000/
```

A pasta compartilhada temporariamente pelo XP será exibida.

Selecionar:

```text
agente_industrial_xp.py
```

e salvar/copiar o arquivo para o computador novo.

### 5. Encerrar o compartilhamento

Quando terminar, voltar ao CMD do Windows XP e pressionar:

```text
Ctrl + C
```

O servidor HTTP temporário será encerrado.

---

## Enviar uma versão atualizada do computador novo para o Windows XP

Quando uma mudança no agente tiver sido feita no computador novo/GitHub e precisar ser instalada manualmente no XP, pode-se inverter o fluxo.

No computador novo, abrir um terminal na pasta que contém a nova versão:

```powershell
python -m http.server 8000
```

No Windows XP, abrir o navegador e acessar:

```text
http://169.254.87.66:8000/
```

Baixar a nova versão do arquivo e substituir manualmente:

```text
C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente\agente_industrial_xp.py
```

Antes de substituir, é recomendado manter uma cópia de segurança da versão operacional anterior.

Depois da substituição:

1. encerrar a instância antiga do agente;
2. iniciar manualmente o novo `agente_industrial_xp.py`;
3. confirmar que o agente abriu a porta 5000;
4. confirmar que continua enviando imagens para a porta 5001;
5. testar `0 = OK`;
6. testar `1 = NG`.

---

# Código-base atual do agente no Windows XP

Abaixo está a versão informada como base operacional atual.

> **Atenção:** este bloco é documentação. Ele não substitui automaticamente o arquivo que está no Windows XP.

```python
# agente_industrial_xp.py (V5.1 - FULL DUPLEX C/ KERNEL HOOK: COMPATIVEL PYTHON 3.4)
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

user32 = ctypes.WinDLL('user32', use_last_error=True)


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
    print(
        "\n[FANTASMA] A IA comandou. Pressionando '{0}' fisicamente...".format(
            tecla_str
        )
    )
    codigo = 0x30 if tecla_str == "0" else 0x31

    win32api.keybd_event(codigo, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(codigo, 0, win32con.KEYEVENTF_KEYUP, 0)


def servidor_de_comandos():
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    servidor.bind(('0.0.0.0', PORTA_COMANDOS_XP))
    servidor.listen(1)

    while True:
        try:
            conexao, _ = servidor.accept()
            comando = conexao.recv(1024).decode('utf-8').strip()

            if comando == "PRESS_0":
                apertar_tecla_fisica("0")
            elif comando == "PRESS_1":
                apertar_tecla_fisica("1")

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
# NOVO: KERNEL HOOK GLOBAL (INTERCEPTADOR DE TECLADO INCONDICIONAL)
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

                print("-> Imagem enviada com sucesso. Pausa de 3s...")
                time.sleep(3)

            except Exception as e:
                print(
                    "-> Erro: PC moderno nao respondeu "
                    "(Porta 5001). {0}".format(e)
                )

        time.sleep(0.1)


def iniciar_agente():
    print("==========================================")
    print(">>> AGENTE VISIONX V5.1 - FULL DUPLEX (KERNEL HOOK)")
    print(">>> Monitorando Tela, Rede e Teclado Fisico (Global)...")
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
    # Isso garante que mesmo minimizado ou clicando fora, o XP sentirá o teclado.
    iniciar_hook_teclado()


if __name__ == "__main__":
    iniciar_agente()
```

---

# Checklist para futuras alterações no agente

Antes de considerar uma mudança concluída:

- [ ] Confirmar que a alteração foi feita na cópia de referência/documentação.
- [ ] Informar ao usuário que o Windows XP não é atualizado automaticamente.
- [ ] Transferir a nova versão manualmente para o XP.
- [ ] Manter backup da versão anterior.
- [ ] Reiniciar o agente manualmente no Windows XP.
- [ ] Confirmar comunicação XP → VisionX pela porta 5001.
- [ ] Confirmar comunicação VisionX → XP pela porta 5000.
- [ ] Testar tecla física `0` no XP e confirmar `OK` no VisionX.
- [ ] Testar tecla física `1` no XP e confirmar `NG` no VisionX.
- [ ] Testar comando do VisionX `PRESS_0` e confirmar tecla `0` no XP.
- [ ] Testar comando do VisionX `PRESS_1` e confirmar tecla `1` no XP.

---

## Resumo operacional

```text
ARQUIVO OPERACIONAL REAL:
C:\Documents and Settings\CCC\My Documents\VisionX Neural\Agente\agente_industrial_xp.py

WINDOWS XP:
169.254.95.200

VISIONX / PC NOVO:
169.254.87.66

XP → VisionX:
porta 5001
imagem + CMD_OK / CMD_NG

VisionX → XP:
porta 5000
PRESS_0 / PRESS_1

0 = OK / FALHA FALSA
1 = NG / DEFEITO REAL
```

A cópia registrada no GitHub é referência de engenharia. O arquivo em execução na AOI continua sendo o arquivo local do Windows XP e só muda após atualização manual.
