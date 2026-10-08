# ODIN — Gate de regressão visual na inicialização

**Status:** Etapa 1 concluída pelo inventário real; **Etapa 2 implementada para diagnóstico SIDE sem memória**, aguardando resultado no computador da fábrica. Etapas 3–5 e gate bloqueante não implementados.  
**Registro:** 08/10/2026.  
**Escopo:** aplicação do computador novo, branch `central`; arquivos visuais `public/ok_archive/` e `public/ng_archive/`.  
**Documento-mestre do histórico:** [`estrutura_projeto.md`](../estrutura_projeto.md).

## 1. Necessidade industrial e objetivo

As pastas `C:\visionx-neural-main\public\ok_archive` e
`C:\visionx-neural-main\public\ng_archive` possuem evidências reais já
rotuladas por operador. A maioria das evidências históricas corresponde à
iluminação **SIDE**; novos eventos podem preservar **SIDE, TOP e MID** da
mesma anomalia da AOI.

**Objetivo:** em **toda inicialização**, antes de disponibilizar o painel
operacional, reanalisar o acervo completo com o **mesmo código e especialistas
físicos e modelos visuais, excluindo integralmente KNN e memória episódica**. O ODIN só pode entrar em
operação após aprovação integral da regressão. Se houver erro, divergência,
arquivo inválido ou cobertura incompleta, a entrada em operação é bloqueada,
com diagnóstico acessível em uma janela restrita de validação.

É um **teste de não regressão contra casos conhecidos**, não prova de
generalização da IA nem licença para memorizar cegamente as respostas. O
aprendizado continua vindo de rótulos humanos e do dataset separado; o gate
não treina, não retreina e não altera a verdade dos exemplos para conseguir
100%.

## 2. Contrato de aprovação

### 2.1 Acervo antigo — imagem única (legado SIDE)

Quando um screenshot histórico não possui marcador explícito de iluminação nem
manifesto multilight, classificá-lo para **replay monoimagem em SIDE**.

| Origem | Rótulo humano esperado | Único resultado aceitável |
|---|---|---|
| `ok_archive/` | OK | `FALHA FALSA`, sem revisão |
| `ng_archive/` | NG | `DEFEITO REAL`, sem revisão |

`REVISÃO OBRIGATÓRIA` não equivale a acerto, inclusive se a memória
individual for de NG ou OK. Assim, o acervo legado serve como contrato de
regressão estrito da decisão individual sob SIDE.

**Observação de implantação:** exemplos NG antigos podem ter sido arquivados
precisamente porque o ODIN de então os classificou erradamente. Esses casos
podem reprovar no primeiro diagnóstico, e isso **não autoriza inverter rótulos
automaticamente**, alterar limiares só para passar ou ignorar o arquivo.
Precisam ser corrigidos na lógica e novamente aprovados pelo teste.

### 2.2 Acervo novo — evento multilight

O evento completo compartilha:

```text
event_id
Board / Parts / Category / Value (mesmo OCR)
expected_label = OK ou NG
SIDE.png, TOP.png, MID.png
```

O replay deve executar **uma análise local por iluminação**, sob
`lighting_mode` correto, e depois usar a **mesma fusão operacional** aplicada
na AOI real:

- `MUITO ADESIVO`: fusão especializada já existente.
- Outras categorias válidas: `src/core/multilight_fusion.py`.

**A aprovação é do julgamento final do evento, não de cada PNG isolado.**

| Rótulo humano | Resultado final obrigatório |
|---|---|
| OK | `FALHA FALSA`, sem revisão |
| NG | `DEFEITO REAL`, sem revisão |

Caso NG válido:

```text
SIDE = FALHA FALSA
TOP  = DEFEITO REAL
MID  = FALHA FALSA
             ↓
fusão final = DEFEITO REAL → PASSOU
```

Exigir NG individual para SIDE, TOP e MID seria **um erro**: destruiria a
finalidade do multilight, no qual o defeito pode aparecer apenas sob uma luz.

Uma sessão incompleta que se declara multilight **não pode** ser tratada
silenciosamente como SIDE para conseguir passar; ela recebe
`INCOMPLETO/INVÁLIDO` e bloqueia a operação. O legado monoimagem explícito
continua suportado.

### 2.3 Resultado global

**Abrir o painel operacional somente quando:**

```text
total de arquivos descobertos = total de arquivos processados ou vinculados
nenhum arquivo sem rótulo/identidade válida
nenhum arquivo danificado ou ilegível
nenhum evento multilight incompleto
nenhuma classificação divergente
nenhuma revisão obrigatória em caso com rótulo OK/NG
todos os eventos e exemplos históricos aprovados
```

Estados por exemplo/evento:

- `PASSOU`: resultado final coincide com o rótulo esperado.
- `REGRESSÃO`: resultado difere do rótulo, inclusive revisão obrigatória.
- `INVÁLIDO`: screenshot corrompido, layout/recorte/OCR indispensável
  inválido, categoria não identificável, dados ausentes ou exceção de análise.
- `INCOMPLETO`: manifesto/evento declara três iluminações, mas falta alguma.
- `CONFLITO`: arquivo/identidade aparece rotulado de maneiras incompatíveis
  ou há associação contraditória comprovada.
- `SEM COBERTURA`: não há exemplos qualificados suficientes para executar o
  gate (não anunciar 100% de aprovação sobre zero casos).

Qualquer estado diferente de `PASSOU` bloqueia a **operação normal** e exige
diagnóstico/correção. A tela restrita de manutenção não é o painel do ODIN e
**não** recebe nem envia comandos à AOI.

## 3. Inventário e interpretação dos arquivos

### 3.1 Fontes e compatibilidade de nomes

Diretórios configurados em `src/config/settings.py`:

```text
settings.OK_ARCHIVE_DIR = public/ok_archive/
settings.NG_ARCHIVE_DIR = public/ng_archive/
```

Usar caminhos derivados de `settings`, **não** o caminho literal do PC do
operador; o projeto pode estar instalado em outro disco/pasta.

Inventariar recursivamente todos os PNGs, incluindo nomes antigos como:

```text
2026-10-01_07-36-23-677_FALTANDO.png
2026-10-06_1042_MUITO_ADESIVO.png
```

e novos nomes com iluminação:

```text
..._FALTANDO_SIDE.png
..._FALTANDO_TOP.png
..._FALTANDO_MID.png
```

Os nomes históricos podem variar. O parser deve tolerar os padrões conhecidos
sem inferir informação que não existe. `_SIDE/_TOP/_MID` no final identifica
iluminação **quando explícito**; ausência de sufixo no legado indica SIDE.

A **pasta** determina o rótulo esperado OK/NG. O nome e o OCR podem contribuir
para resolver categoria, mas categoria/Board/Parts/Value usados na análise
devem ser extraídos e validados com as regras atuais ou recuperados de
metadados humanos confiáveis; não inventar OCR.

Arquivos de metadados/manifesto devem ser reconhecidos explicitamente;
arquivos desconhecidos, PNG ilegível ou extensão não suportada não podem ser
ignorados silenciosamente. Não mover, renomear nem apagar arquivos no gate.

### 3.2 Manifesto de regressão por peça (novo contrato de arquivamento)

**Antes de liberar o modo multilight completo no gate**, evoluir a gravação
das evidências para guardar uma associação explícita, transacional e auditável:

```json
{
  "schema": "visionx.archive_regression.v1",
  "event_id": "id-real-da-aoi",
  "expected_label": "NG",
  "aoi_info": {
    "board": "valor-do-ocr",
    "parts": "referencia-do-ocr",
    "category": "FALTANDO",
    "value": "texto-do-ocr"
  },
  "source": "network",
  "frames": {
    "SIDE": {"path": "arquivo_SIDE.png", "sha256": "hash-dos-bytes"},
    "TOP":  {"path": "arquivo_TOP.png", "sha256": "hash-dos-bytes"},
    "MID":  {"path": "arquivo_MID.png", "sha256": "hash-dos-bytes"}
  }
}
```

O JSON acima **é especificação proposta**, não formato já implementado.
Registrar hash, origem e caminhos efetivamente persistidos (inclusive quando
a deduplicação visual reutilizar um PNG pré-existente). Gravar o manifesto
**somente quando o conjunto referenciado estiver consistente e disponível**,
com gravação atômica. Não criar manifesto que aponte para arquivos que a fila
ainda não concluiu.

Para fotos antigas, não fabricar `event_id` original ou OCR que não existe.
Criar metadado de qualificação/legado auditável apenas se necessário, sem
alterar os PNGs.

**Não agrupar SIDE/TOP/MID só por horário, sequência de nomes ou categoria.**
Isso poderia juntar peças distintas. Imagens novas sem manifesto confiável
devem ser sinalizadas para qualificação, não falsamente fundidas.

### 3.3 Conflitos entre as duas pastas

Comparar fingerprints e identidades antes do replay. A mesma evidência visual
em `ok_archive` e `ng_archive` pode refletir erro de registro,
reclassificação legítima ou contexto diferente. Registrar conflito para
investigação; **não escolher o rótulo automaticamente** e não apagar o exemplo.

## 4. Motor de replay — mesma análise, sem AOI física

```text
PNG original da AOI
    ↓
decodificação + verificação de integridade
    ↓
localização das barras azul/vermelha
    ↓
extração integral de GABARITO e TESTE
    ↓
OCR Board/Parts/Value + normalização da categoria
    ↓
detect_anomalies / EpicenterExtractor
    ↓
MoEOrchestrator.inspect + especialistas físicos SEM KNN
    ↓
resultado local SIDE/TOP/MID
    ↓
se multilight: fuse_multilight(analyses, category)
    ↓
resultado final comparado ao rótulo do arquivo/evento
```

Reutilizar/extrair a lógica de
`src/services/screen_monitor.py` (barras, extração, OCR),
`src/core/adhesive_multilight_analysis.py` (análise por luz),
`src/core/multilight_fusion.py`, módulos físicos/semânticos e memória
instalados no `main.py`. **Não manter uma segunda implementação
simplificada** de inspeção específica do startup.

O replay recebe imagens **do disco**, nunca manda
`PRESS_LEFT/PRESS_RIGHT/PRESS_DOWN` ao Windows XP, não ativa captura
MSS, não abre servidor/receptor da AOI, não envia `PRESS_0/PRESS_1`, não
incrementa métricas de produção e não usa estados de botões do operador.

**Atualização obrigatória aprovada pelo operador — replay cego à memória:**
o replay NÃO deve instanciar nem consultar KNN, dataset de exemplos,
protótipos, vizinhos, similaridades persistidas ou memórias episódicas.
A mesma imagem pode estar no dataset: sua presença não pode tornar a
aprovação mais fácil. A decisão deve se apoiar exclusivamente nos detectores,
na geometria, nas regras físicas e nos modelos visuais não baseados na busca de
amostras. O score da memória é zero; não há `memory_veto`, `best_match` ou
`memory_priority`. A regra vale para SIDE histórico e para a futura fusão
multilight, inclusive `INVERTIDO`, cuja extensão de análise consultava
novamente o KNN.

A CNN ou outro modelo de parâmetros previamente treinados continua elegível
se não procurar a imagem atual em uma base de exemplos durante a inferência;
não se permite consulta/reuso de correspondência exata com o acervo.
O replay jamais chama `save_label`, `DatasetManager.save_sample`, recarga
da memória ou arquivamento. Qualquer tentativa de acesso KNN é erro de
isolamento e invalida a execução. A operação normal mantém sua memória
habilitada; o bypass é exclusivo do modo diagnóstico offline.

**Cuidado do quadrado menor:** usar sempre o gabarito/teste **completos** da
área da AOI, com análise física local/contextual **e da área de inspeção completa**.
A terceira escala de *memória* full-frame NÃO é consultada nesse replay;
a exigência de imagem completa diz respeito aos detectores visuais,
não a uma comparação com assinaturas antigas. Não validar só a ROI pequena:
o defeito pode estar fora dela.

**Não há autocorrespondência com o dataset neste replay:** a consulta
KNN/memória é impedida por construção. Ainda assim, usar as imagens
históricas para calibrar repetidamente os mesmos limiares pode superajustar
o resultado ao acervo; manter também uma validação independente/holdout
quando houver dados qualificados.

## 5. Sequência de inicialização — bloqueio antes da produção

```text
python main.py
    ↓
QApplication / tela mínima de validação (sem produção)
    ↓
carrega configurações e especialistas físicos SEM carregar KNN/memória
    ↓
inventaria 100% dos arquivos OK/NG e manifestos
    ↓
replay completo com progresso real
    ↓
compara todos os resultados ao histórico humano
    ↓
gera relatório final
    ↓
100% PASSOU?
    ├── SIM → instancia/libera ControlPanel e serviços operacionais
    └── NÃO → mantém SOMENTE tela de diagnóstico / manutenção restrita
```

A janela `ControlPanel` e seus receptores/hook operacionais não devem ser
criadas/ativadas antes da aprovação, porque a construção do painel atual
pode preparar serviços de rede e captura. A tela de validação deve ser
independente, responsiva e mostrar `N/T`, etapa, arquivo/evento atual e
contagem de falhas. Cancelar o teste **não** libera a produção.

O gate deve ser executado **a cada partida**, mesmo sem alterações de código
ou arquivos. Hash/cache pode servir para inventário/integridade, **não
substituir a reanálise completa**. Não permitir `skip`, `force pass` ou
atualização automática do rótulo em ambiente produtivo.

**Diagnóstico/qualificação inicial:** antes de ativar o bloqueio definitivo,
executar uma fase de levantamento apenas para obter o conjunto real de falhas
do acervo antigo. Essa fase não é considerada gate aprovado nem autoriza
suprimir casos; depois da qualificação, **o modo exigido é fail-closed**.
Correções devem ser testadas e validadas antes de liberar produção. Qualquer
modo de manutenção deve manter a AOI desacoplada.

Para acervo grande, processar incrementalmente em worker separado da thread
da UI, com uso controlado de RAM e progresso por arquivo/evento. Nenhum caso
deve ser descartado por timeout silencioso.

## 6. Relatório obrigatório

Relatório no diretório próprio de diagnósticos, separado dos archives/dataset,
em **JSON estruturado e relatório legível**. Pode registrar horário, hash do
código/configuração, versão do contrato, total de arquivos, total de eventos,
contagem de rótulos e distribuição por categoria/iluminação.

Cada caso deve expor, no mínimo:

```text
status = PASSOU / REGRESSÃO / INVÁLIDO / INCOMPLETO / CONFLITO
source_path
expected_label = OK / NG
detected_category
lighting_mode(s)
event_id (quando existe)
verdict por iluminação
verdict final
operator_review_required
score, confidence, physical_score, fusion_rule
diagnóstico de OCR/recorte (quando relevante)
erro/stack trace resumido (quando houver)
```

Exemplo ilustrativo (não é resultado de execução):

```text
GATE DE INICIALIZAÇÃO ODIN

OK: 14/14 casos aprovados
NG: 17/18 casos aprovados
TOTAL: 31/32 casos aprovados

STATUS: BLOQUEADO
FALHA: ng_archive/2026-10-01_..._FALTANDO.png
TIPO: REGRESSÃO
ESPERADO: DEFEITO REAL
OBTIDO: FALHA FALSA
ILUMINAÇÃO: SIDE
```

Falha de leitura, divergência, conflito e ausência de cobertura precisam estar
visíveis. A origem dos dados sensíveis de produção não deve ser enviada
automaticamente para internet/terceiros. O relatório nunca deve modificar
as imagens e rótulos humanos.

## 7. Arquitetura proposta (separação de responsabilidades)

Sugestão de módulos para a **futura implementação**, sujeitos a refinamento
durante a primeira etapa:

```text
src/services/startup_regression/
    archive_inventory.py       # enumera arquivos, integridade, classificação
    archive_manifest.py        # escreve/lê vínculos de peças multilight
    replay_preprocessor.py     # reutiliza extração AOI + OCR existente
    inspection_runner.py       # chama motor real sem UI/rede
    verdict_contract.py        # regras de aprovação mono/multilight
    regression_runner.py       # orquestra replay e progresso
    regression_report.py       # relatório JSON/humano

src/ui/startup_regression_window.py
    # janela restrita de progresso, aprovação e bloqueio

tests/test_startup_regression_*.py
    # unitários, integração e arquivos antigos/novos
```

`main.py` deve apenas coordenar a validação antes da construção do painel,
sem absorver parser, pré-processamento ou regras de negócio. O arquivo de
evidência OK/NG continua separado de
`public/dataset/{anomalia,nao_anomalia}`.

Nenhuma alteração do `agente_industrial_xp.py` é necessária para replay
offline de imagens arquivadas.

## 8. Plano incremental de implementação

**Regra de execução:** trabalhar em **uma etapa por vez**, apresentar testes,
evidências e mudanças; avançar somente após a aprovação explícita do usuário.

### Etapa 1 — Inventário e diagnóstico do acervo existente

- Confirmar nomes, formatos, dimensões e legibilidade de todos os PNGs
  reais de OK/NG (sem excluir arquivos antigos).
- Identificar quais são LEGADO SIDE, quais já são SIDE/TOP/MID e quais possuem
  metadados confiáveis para associação por evento.
- Medir quantas imagens há por pasta/categoria/iluminação.
- Identificar conflitos, arquivos corrompidos, OCR indecifrável e possíveis
  pares não relacionáveis sem inventar vínculo.
- Gerar apenas relatório de inventário; sem abrir gate e sem alterar PNGs.

**Aceite:** relatório reprodutível do acervo real, com pendências explícitas.

#### Implementação da ferramenta de Etapa 1 — 08/10/2026

**Implementado no código:** inventariador independente, somente leitura.
**Pendente para aceitar a Etapa 1:** rodá-lo com as duas pastas reais do
computador da fábrica e examinar o relatório. Nenhuma contagem de arquivos
reais foi presumida com base no print.

Arquivos criados:

```text
src/services/startup_regression/archive_inventory.py
src/services/startup_regression/archive_inventory_report.py
src/services/startup_regression/__main__.py
tests/test_startup_regression_inventory.py
.github/workflows/startup-regression-inventory.yml
```

**Comando no PowerShell do computador com o código atualizado:**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression
```

A raiz do projeto é detectada automaticamente. Caso a execução seja iniciada
fora da pasta do projeto, usar:

```powershell
python -m src.services.startup_regression --root "C:\visionx-neural-main"
```

O comando lê `public/ok_archive/` e `public/ng_archive/`, inclusive arquivos
em subpastas; **não altera nenhum PNG, não renomeia, não cria arquivos nas
pastas e não toca no dataset**. A saída fica em:

```text
reports/startup_regression/inventory_<data>.json
reports/startup_regression/inventory_<data>.txt
```

A pasta de relatórios é ignorada pelo Git via `.gitignore`. O relatório JSON
contém o inventário completo e os fingerprints; o TXT traz resumo e pendências
para leitura rápida. Pedir esses relatórios para concluir o diagnóstico real.

Nesta etapa são verificados: arquivos descobertos, assinatura e estrutura
PNG/CRC, decodificação OpenCV, dimensões, hash do arquivo, fingerprint dos
pixels, categoria **sugerida apenas pelo nome**, iluminação explícita ou
SIDE legada, repetição exata de imagens, conflitos entre OK e NG, além de
manifestos reais quando existirem.

**Uma imagem sem manifesto não é agrupada com outras por horário/nome.**
Um único PNG deduplicado pode ser citado por dois eventos ou até mais de uma
iluminação no manifesto; a ferramenta mantém vínculos separados. Manifesto
corrompido ou incompleto nunca ganha vínculo parcial válido.

**Limite explícito:** o inventário não executa OCR, não confirma categoria
por leitura da tela, não chama o MoE/KNN nem produz qualquer veredito da IA.
Itens cuja categoria só possa ser recuperada pelo OCR ficam pendentes para
a Etapa 2. A ferramenta **não bloqueia a inicialização**, não altera
`main.py`, não treina a memória e não aciona o Windows XP.

**Próxima ação:** executar no PC real, trazer os arquivos JSON/TXT e revisar
as pendências. Somente com aceite do inventário avançar para a Etapa 2.

### Etapa 2 — Runner de replay monoimagem SIDE

- Separar pré-processamento de screenshot AOI da interface Qt.
- Reutilizar recorte completo, OCR, pipeline físico/semântico e políticas
  da operação real **sem qualquer consulta KNN ou memória episódica**.
- Executar casos antigos `OK` e `NG` em leitura e registrar veredito,
  revisão e divergências.
- Garantir que o quadrado menor não limita o campo de evidências.
- Qualificar as regressões pré-existentes sem mascará-las.

**Aceite:** todos os PNGs legados são processados ou falham com erro explícito;
replay não altera dataset/arquivos nem aciona hardware.

#### Diagnóstico real e telemetria adicional — 08/10/2026

Primeira execução industrial do `side_replay` (119 SIDE históricos):

- `OK`: 0/102 aprovados; 101 classificados como `DEFEITO REAL` e
  1 inválido por OCR ausente (imagem de 1920×1080).
- `NG`: 17/17 aprovados; todos `DEFEITO REAL`.
- `KNN`: explicitamente desabilitado em todas as observações.
- **Conclusão operacional:** replay físico ainda NÃO distingue
  adequadamente OK e NG; não ativar o gate de inicialização. Não ajustar
  limiares para fazer o histórico passar sem investigar os especialistas.

Foi adicionada a telemetria de inspeção **somente ao executor de replay**:
`src/services/startup_regression/replay_telemetry.py`.

O relatório agora traz para **cada caso**:

```text
telemetry.schema = visionx.side_replay_telemetry.v1
telemetry.reason / fusion_rule / dominant_engine / final_score / cutoff
telemetry.engines[]:
   id, label, active, triggered, selected
   raw_score, effective_score, threshold, final_influence, summary
telemetry.physical_readings:
   leituras escalares presentes em detail (sem inventar campos ausentes)
telemetry.geometry:
   dimensões completas da imagem de referência e teste
   global_box (AOI)
   old_epicenters / selected_epicenters / raw_anomaly_boxes
   final_bounding_box / specialist_boxes
   fonte coordenadas = AOI_EXTRACTED_IMAGE_XYWH
diagnostics:
   por categoria + rótulo
   por regra de fusão + status
   por motor dominante + status
   motores disparados nas regressões (não mutuamente exclusivos)
```

O `build_lighting_context` do motor existente é executado uma vez e
compartilhado com a inferência; coletar telemetria **não** executa
inferência duas vezes e **não** modifica a decisão. O registro serializa
somente campos seguros, evitando máscaras binárias, imagens e assinaturas
de memória. O modo sem KNN continua obrigatório; dados ausentes são `N/D`.

Com o repositório atualizado, executar novamente:

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

Coletar os novos `side_replay_*.json` e `side_replay_*.txt` para comparar
pontuação dos especialistas e a seleção de região em **OK e NG** da mesma
categoria. **Etapa 2 ainda aberta; Etapa 3 e gate bloqueante não iniciados.**

#### Implementação da Etapa 2 — motor físico sem memória

**Regra irrevogável no gate de regressão:** replay cego aos exemplos antigos.
O KNN não é instanciado; a extensão de fusão física e a rota específica
`INVERTIDO` recebem `_replay_without_memory=True` e não computam nem
consultam assinaturas armazenadas. A análise física continua reutilizando o
`detect_anomalies → EpicenterExtractor → MoEOrchestrator`.

```text
src/services/startup_regression/inspection_runner.py
src/services/startup_regression/side_replay.py
tests/test_startup_regression_replay.py
.github/workflows/startup-regression-side-replay.yml
```

**Comando no PowerShell com o código `central` atualizado:**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

**Saída:** `reports/startup_regression/side_replay_<data>.json` e
`side_replay_<data>.txt`. O comando avalia **somente os 119 screenshots
históricos sem sufixo**; as 90 imagens com `SIDE/TOP/MID` explícitos
aguardam o replay por evento na Etapa 3.

A leitura utiliza o mesmo `ScreenMonitor.process_external_image` que a
produção para encontrar barras, recortar gabarito/teste completos e recuperar
OCR. No replay, `_replay_no_debug=True` impede a gravação de PNGs temporários
em `public/debug_crop`. Um OCR indisponível/incompleto, divergência de
categoria ou erro físico retorna `INVALIDO`, jamais aprovação.

Para garantir o isolamento, `SideInspectionRunner` rejeita resultados que
declarem motor KNN ativo, memória consultada ou peso KNN diferente de zero.
A CPU executa os mesmos especialistas físicos, sem carregar os exemplos do
dataset. A avaliação não abre o painel, não captura MSS, não envia comandos XP,
não ensina a IA e **ainda não bloqueia** a inicialização operacional.

O resultado real do PC deve ser revisado antes de declarar esta etapa
aprovada. Falhas de Tesseract ou dependências também devem aparecer no
relatório e ser corrigidas, não ignoradas.

**Validação técnica:** workflow `Startup Regression SIDE Replay`,
execução `37780008623` em GitHub Actions Windows,
**22 testes executados, todos aprovados**, inclusive MoE físico real em
`FALTANDO` e `INVERTIDO` com KNN bloqueado por teste. Não equivale a
119/119 imagens reais aprovadas: essa medição depende dos PNGs locais.

**Observação operacional:** quando houver divergência ou falha, o comando
termina com código de saída 1 **depois de salvar os relatórios**. Isso é
correto para scripts de diagnóstico e **não** bloqueia o `main.py` nem
altera o ODIN em operação.

### Etapa 3 — Manifesto e replay por evento multilight

- Evoluir persistência futura de OK/NG para associar
  `event_id + OCR + expected_label + SIDE/TOP/MID`, com hashes/caminhos
  realmente persistidos e commit atômico do manifesto.
- Suportar screenshots com sufixos `SIDE/TOP/MID` e reconhecer
  arquivos históricos sem sufixo como SIDE.
- Reprocessar separadamente as três iluminações e passar pela fusão real,
  preservando política especializada de adesivo.
- Recusar sessão incompleta/ambígua sem tentar fusão artificial.

**Aceite:** caso NG `SIDE OK / TOP NG / MID OK` passa se a fusão resultar NG;
um evento de três imagens vale uma única decisão.

### Etapa 4 — Gate bloqueante e tela de inicialização

- Carregar modelos/memória **antes** de autorizar o painel.
- Exibir somente progresso/diagnóstico durante replay.
- Liberar `ControlPanel` apenas após 100% de aprovação.
- Impedir abertura operacional se houver regressão, revisão, arquivo inválido,
  manifesto incompleto, conflito ou ausência de cobertura.
- Persistir relatório mesmo em falha/cancelamento.

**Aceite:** testes instrumentados comprovam que nenhum receptor, captura ou
comando XP fica ativo quando o gate falha.

### Etapa 5 — Regressões automatizadas e validação na fábrica

- Unitários de parser legados, classificação, deduplicação/identidade e
  comparação de vereditos.
- Integração com screenshots completos sintéticos/anonimizados e memória real
  instalada, sem hardware XP.
- Cenários OK/NG, revisão, imagem corrompida, OCR ausente, categoria errada,
  arquivo duplicado, rótulos conflitantes, três luzes completas/incompletas.
- Testes de inicialização aprovada/bloqueada e idempotência do relatório.
- Medir tempo e RAM em acervo grande; comprovar que a UI não congela.
- Fazer diagnóstico inicial do acervo operacional e corrigir regressões,
  **sem remover a obrigatoriedade dos casos conhecidos**.

**Aceite final:** 100% do acervo qualificado passa no ambiente alvo e o
bloqueio opera corretamente diante de regressões injetadas.

## 9. Itens não abrangidos e regras de segurança

- Não instalar nova IA, retreinar CNN/KNN ou alterar a fusão só para passar
  o gate; mudanças no motor exigem tarefa e validação próprias.
- Não reclassificar automaticamente imagens humanas ao observar divergência.
- Não converter casos de revisão em OK ou NG por conveniência.
- Não supor que três imagens NG precisam ter três vereditos NG isolados.
- Não comparar apenas o quadrado menor nem reutilizar captura visual de outra
  peça.
- Não tratar 100% no acervo conhecido como garantia estatística de precisão
  em peças novas.
- Não mover/apagar arquivos e não interferir no processo físico da AOI.
- Não permitir que falha da tela de validação leve à abertura normal por
  fallback.

## 10. Estado e próximo passo

- **Etapa 1 concluída e aprovada:** inventário real em 08/10/2026,
  209 PNG válidos (119 SIDE históricos; 90 multilight explícitos sem manifesto).
  192 OK e 17 NG; nenhuma duplicata pixel a pixel. Os vínculos das 90
  iluminações multilight ficam para qualificação na Etapa 3.
- **Etapa 2: primeiro replay real diagnosticado** (17/119 aprovados,
  101 regressões e 1 OCR inválido); telemetria física detalhada adicionada
  e aguardando novo relatório industrial. KNN permanece desabilitado.
- **Etapas 3 a 5 e gate bloqueante: NÃO implementados**.
- O `main.py`, decisão em produção, dataset de aprendizado e
  `agente_industrial_xp.py` permanecem inalterados.
- **Próxima ação:** executar `python -m src.services.startup_regression.side_replay`
  no PC da fábrica, enviar os relatórios `side_replay_*.json` e
  `side_replay_*.txt` e qualificar cada divergência, sem alterar rótulos.
  Não iniciar a Etapa 3 antes do aceite explícito.


### 08/10/2026 — Verificador CNN FALTANDO v2 do acervo conhecido (isolado)

Foi criado `src/scripts/replay_faltando_cnn_v2.py` para julgar offline
todos os `FALTANDO` do `ok_archive` e `ng_archive` usando pesos
CNN v2 já treinados, com SIDE legado e SIDE/TOP/MID. Os resultados
por foto e evento ficam em
`reports/faltando_neural/replays/archive_v2_*/`.
**Esse verificador NÃO implementa o gate bloqueante de inicialização**,
não usa KNN, não altera modelo e não reconfigura Produção.
Seu sucesso integral significa somente não regressão nos casos
conhecidos, porque o próprio acervo também alimentou o treino v2.
A avaliação independente continua necessária para liberação
automática de novas peças.
