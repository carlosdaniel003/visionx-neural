# CNN especializada DESLOCADO — dataset, bootstrap e aprendizado incremental

## STATUS FINAL — CNN DESLOCADO: DESENVOLVIMENTO ABORTADO (08/10/2026)

**Decisão do operador:** suspender definitivamente esta linha de desenvolvimento
até uma eventual nova decisão expressa. **0 NG DESLOCADO reais**; não é possível
validar detecção de deslocamento físico, medir falsos OK de NG nem autorizar
substituição do motor operacional. **Não retomar treinamento, desenho de máscaras,
diagnósticos geométricos ou ativação de checkpoint como continuação deste plano.**

**Placar histórico (métricas de tarefas diferentes):** CNN v1 **0/6 OK de
desenvolvimento**; CNN v2 **31/34 OK conhecidos** no replay sem KNN, com
**3 falsos NG** e **0 NG reais avaliados**; geometria v1 **7/34 com métricas
disponíveis** (não classificação); geometria v1.1 **0/34 registros aceitos**.
As máscaras v3/v3.1 e o plano one-class **não produziram CNN v3 validada**.

**[Retrospectiva técnica, metodologia, falhas e lições](DESLOCADO_CNN_POSTMORTEM_2026-10-08.md)**

**Os comandos e as propostas das seções antigas abaixo são histórico, não
instruções para continuar o projeto.** O ODIN operacional, a CNN FALTANDO, as
imagens do acervo e os checkpoints históricos não foram alterados por esta decisão.

---


## 08/10/2026 — Solicitação

A categoria \`DESLOCADO\` deve seguir o mesmo macrofluxo da CNN FALTANDO:
**AOI → memória KNN verificada → CNN especializada se novo →
rótulo humano → treino incremental → teste de regressão → modelo ativo**.

**Estado inicial declarado pelo operador:**
- NG reais em \`public/ng_archive\`: **nenhum DESLOCADO**.
- OK SIDE histórico em \`public/ok_archive\`, por exemplo:
  \`2026-10-02_1349_DESLOCADO.png\`,
  \`2026-10-02_1350_DESLOCADO.png\`,
  \`2026-10-02_1351_DESLOCADO.png\`.
- OK SIDE/TOP/MID atuais, por exemplo:
  \`2026-10-08_0754_DESLOCADO_SIDE.png\`,
  \`2026-10-08_0754_DESLOCADO_TOP.png\`,
  \`2026-10-08_0754_DESLOCADO_MID.png\`.
- **As contagens reais ainda dependem de rodar o inventário no Windows 10**;
  estes nomes representam exemplos, não uma contagem integral verificada.

## Estrutura implementada

### Preparação do dataset

\`src/services/deslocado_neural_dataset.py\` reutiliza a extração **real**
da AOI de \`AOIPairExtractor\`: imagem integral → gabarito/teste.
Filtra somente DESLOCADO do inventário de \`ng_archive\` e \`ok_archive\`,
confere validade PNG, SHA-256 e separa SIDE legado de SIDE/TOP/MID.
Mantém candidatos de trinca somente por nome; não inventa
\`event_id\`. Nenhuma imagem do acervo original é alterada.

\`\`\`powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.deslocado_neural_dataset
\`\`\`

Gera \`reports/deslocado_neural/run_<timestamp>/manifest.json\`,
\`summary.txt\` e \`pairs/<id>/reference.png, test.png\`.
Se houver erro de extração, não prosseguir para treinamento.

### CNN DESLOCADO v1 (pesos separados)

\`src/core/neural/deslocado_cnn.py\` utiliza a **mesma arquitetura base
comparativa de duas escalas da FALTANDO v2**, mas a instância tem
checkpoint, treinamento e schema distintos:
\`visionx.deslocado_comparative_cnn.v1\`.
Entradas: gabarito/teste/diferença, imagem completa + região central,
SIDE/TOP/MID com máscara; três luzes coerentes são um evento.

**Zero NG DESLOCADO reais significa que ainda não é possível validar
supervisionadamente um classificador OK×NG real.** Para iniciar o
aprendizado da arquitetura sem inventar NG reais,
\`src/scripts/train_deslocado_cnn.py\` treina em **OK reais vs
deslocamentos locais sintéticos** da região central do teste.
Cada proxy fica identificado como \`SYNTHETIC_PROXY_SHIFT\`, separado
dos futuros NG reais. Esses proxies podem conter artefatos que não
existem em defeitos industriais; seus resultados **não medem recall NG**
nem autorizam decisões na produção. Amostras multilight são agrupadas
quando nome e OCR são coerentes; grupos board/parts são separados
no desenvolvimento quando possível.

\`\`\`powershell
python -m src.scripts.train_deslocado_cnn --epochs 15 --batch-size 4 --size 160
\`\`\`

Saída em \`reports/deslocado_neural/models/experiment_*/\`:

- \`deslocado_cnn_candidate.pt\`: checkpoint isolado,
  \`production_approved=False\`,
  \`allow_automatic_classification=False\`;
- \`training_report_deslocado.json\`: indicadores separados
  para OK real e proxy sintético, grupos e limitações;
- \`training_summary_deslocado.txt\`: resumo legível.

**Este checkpoint não é carregado em \`main.py\`.**
O ODIN continua decidindo DESLOCADO por seus motores físicos
existentes; nenhuma alteração de automação de OK/NG nesta categoria.

### Memória KNN e incremento em Teste / Produção / Sombra

O roteador KNN primeiro permanece igual: par visual
humano conhecido → memória KNN, sem especialistas.
Se o par DESLOCADO for **novo**, a rota continua
\`NEW_EXPERTS\` (especialistas físicos), com
\`detail.specialist_candidate =
DESLOCADO_CNN_V1_BOOTSTRAP_NOT_ACTIVE\`.
O tooltip explica que a CNN está em treinamento
e ainda não substitui os motores.

Quando o operador confirma **OK ou NG** no caso novo
em qualquer modo:
1. O \`DatasetManager\` salva normalmente, incluindo gabarito
   e teste mesmo quando a IA havia concordado;
2. \`DecisionPersistenceQueue\` só solicita o treino se a gravação
   humana tiver sucesso;
3. \`OnlineLearningQueue\` grava evento durável em
   \`reports/neural_online/events/\`, com hashes e até três luzes;
4. \`SPECIALIST_TRAINERS["DESLOCADO"]\` chama
   \`src.scripts.train_deslocado_cnn_online\` num processo CPU
   separado do ciclo AOI;
5. O treinador junta os OK históricos e todos os casos
   DESLOCADO humanos novos, agora incluindo **NG reais** se
   aparecerem. Futuros NG são marcados \`REAL_NG\`,
   nunca falsificados como sintéticos. O próximo candidato
   é salvo em \`reports/deslocado_neural/models/\`.

**Bloqueio intencional:** ao contrário da CNN FALTANDO v2,
\`DESLOCADO\` ainda não possui qualificação com NG reais.
Seu treinador online **não promove** checkpoint nem cria
\`reports/neural_online/live_active.json\` para esta categoria.
A ativação exigirá evidências NG reais independentes,
holdout rigoroso, replay de todos os OK/NG e validação
de NG SIDE/TOP/MID antes de liberar previsões em operação.
Mesmo quando surgirem NG online, o candidato permanece
experimental até essa etapa explícita.

O registro da fila é extensível para futuras especialidades;
os eventos de categorias distintas permanecem isolados.

## Verificações / pendências

- Testes automatizados utilizam imagens sintéticas: não avaliam o
  dataset real do computador corporativo.
- Antes de prosseguir, executar preparação local e enviar
  **\`manifest.json\` e \`summary.txt\`**.
- Executar treino local apenas se a preparação tiver extraído
  todos os pares esperados. Enviar então
  \`training_report_deslocado.json\` e
  \`training_summary_deslocado.txt\`.
- Quando houver NG real de DESLOCADO, registrá-lo com
  confirmação humana e luminosidade identificada. O dataset
  não deve incluir proxies como defeitos verificados.


## 08/10/2026 — CNN DESLOCADO v1 reprovada; correção v2 por máscara de componente

**Treino real v1 recebido da fábrica:** `training_report_deslocado.json`
e `training_summary_deslocado.txt`. O dataset contém
34 pares extraídos, dos quais **24 eventos OK reais**
(19 SIDE legados e 5 trincas candidatas SIDE/TOP/MID).
Não há qualquer NG DESLOCADO real. O treino v1 usou 18 eventos OK
mais deslocamentos sintéticos e reservou 6 eventos OK de
desenvolvimento, separados por placa/componente.

**Falha da v1:**
- No treino: 18/18 OK corretos e 18/18 proxies sintéticos corretos;
- Na validação: **0/6 OK reais corretos**, **6/6 proxies sintéticos**
  reconhecidos, acurácia combinada 50%, seis falsos NG reais;
- Os scores de deslocamento dos seis OK reais ficaram entre
  0,784709 e 0,992219; a loss de treino caiu de 0,815261
  para 0,018744 em 15 épocas. Fortes sinais de sobreajuste,
  não houve generalização para OK reservados.
- **v1 REPROVADA**: não aumentar épocas sem mudar os proxies,
  nem integrar a rede ao julgamento normal.

**Correção v2 implementada no GitHub, ainda não executada com os 34
pares locais**:

- `src/services/deslocado_proxy_v2.py` identifica uma hipótese
  de componente aproximadamente central por contraste contra
  o fundo e componentes conectados. **É uma máscara
  heurística não validada**, não uma segmentação comprovada.
  Quando não é confiável, o exemplo sintético é recusado.
- O fundo no local de origem do componente é reconstruído
  com `cv2.inpaint`, a máscara é movida e composta no
  novo local. Há **duas reconstruções com mesmo procedimento**:
  `RECOMPOSED_OK` com deslocamento zero e
  `SYNTHETIC_SHIFT_PROXY` com deslocamento, minimizando
  aprendizado apenas de artefatos de inpaint. A classe
  `REAL_OK` original permanece incluída.
  Deslocamentos variam em ângulo e distância;
  fotometria ligeiramente perturbada em ambas as classes
  e flips sincronizados entre gabarito/teste.
- `src/scripts/train_deslocado_cnn_v2.py` mantém o treino
  comparativo de duas escalas e três luzes com
  **pesos exclusivamente DESLOCADO**. Não toca a CNN FALTANDO
  nem substitui o treino v1 / arquivos originais. As trincas
  candidatas são tratadas como um evento; eventos com mesmo
  `board+parts` são mantidos juntos no split.
- Validação por época registra OK reais, OK recompostos,
  proxies e scores individuais por SIDE/TOP/MID. A escolha
  do checkpoint prioriza zero **falsos NG em OK reais**
  do conjunto reservado, depois a discriminação dos proxies.
  A verificação `dev_real_ok_zero_false_ng_gate_passed`
  só passa se **todos os OK reais** reservados forem
  reconhecidos como OK. O gate combinado também exige
  acerto em OK recompostos e proxies.
- Guarda `deslocado_cnn_v2_candidate.pt`,
  `training_report_deslocado_v2.json`,
  `training_summary_deslocado_v2.txt` e
  `holdout_predictions_deslocado_v2.json` em
  `reports/deslocado_neural/models/experiment_v2_*/`.
- Ainda **não existem NG reais**, logo recall NG real
  continua **não mensurável**. Resultados de proxy e do
  holdout de desenvolvimento (usado para escolher a melhor
  época) não são teste cego e não autorizam operação.
  Todo checkpoint declara `production_approved=False`,
  `allow_automatic_classification=False`.
  Os motores físicos da categoria continuam em uso.
  O aprendizado incremental DESLOCADO já existente mantém
  o fluxo de candidatos; esta v2 é um treino inicial
  separado, sem ativação automática no ODIN.

**Executar na estação Windows 10 que contém os pares:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.train_deslocado_cnn_v2 --epochs 25 --batch-size 4 --size 160
```

**Revisão visual dos proxies:** a v2 também gera
`proxy_previews/` dentro da pasta `experiment_v2_*`.
Cada imagem mostra lado a lado o teste com a hipótese de
contorno do componente (vermelho), o OK recomposto,
o deslocamento sintético e a diferença entre ambos.
**Conferir que o contorno realmente corresponde ao componente**,
e não a um pad/trilha/texto, antes de confiar nos exemplos
artificiais. O JSON lista os PNGs em
`development_proxy_preview_images` e os não segmentados em
`development_proxy_unresolved`.

**Após executar:** enviar os três relatórios JSON/TXT da v2.
Não há motivo para substituir o motor físico mesmo que a v2
alcance seis OK corretos, sem NG reais de deslocamento.
Se o diagnóstico mencionar `AMBIGUOUS_OR_MISSING_COMPONENT_MASK`
ou erro de segmentação, não fabricar NG fictícios: será
necessário identificar visualmente a região correta.

---



## 08/10/2026 — Resultado real da CNN DESLOCADO v2 (relatórios recebidos)

**Fontes recebidas do treinamento real no computador de fábrica:**
`training_report_deslocado_v2.json`,
`training_summary_deslocado_v2.txt` e
`holdout_predictions_deslocado_v2.json`. Manifesto de
origem `run_20261008T173116_094174Z/manifest.json`,
SHA-256 `6af07429aed97292eb47c14c12669d49936ef470e48c2f643b292c9a0421bd94`.

**Base e treino:** 34 imagens OK reais, agrupadas em 24 eventos
(18 para treino, 6 para desenvolvimento); **0 NG reais**.
Treino configurado para 25 épocas, interrompido após 10 com
paciência 8; melhor checkpoint na **época 2**. Tamanho 160,
batch 4, seed 42, 3 variantes sintéticas por evento treinável.

**Resultado v2 (modelo selecionado na própria validação de
desenvolvimento):**
- **DESENVOLVIMENTO:** 6/6 eventos REAL_OK corretos,
  0 falsos NG reais; 2/2 OK reconstruídos corretos;
  **0/2 proxies sintéticos de deslocamento detectados**.
  O gate de OK reais passou, o gate combinado REPROVOU.
  Os scores proxy sintético ficaram **abaixo**, não acima,
  dos respectivos OK originais:
  `2026-10-02_1358`: OK 0,413231, deslocado 0,405022;
  `2026-10-02_1408`: OK 0,412305, deslocado 0,408717.
  Score sigmoide NÃO é probabilidade calibrada de defeito real.
- **TREINO (checkpoint selecionado):** 15/18 OK reais,
  9/12 OK reconstruídos e 5/12 proxies detectados.
  Portanto nem mesmo os OK usados no treinamento foram
  todos preservados (3 falsos NG).
- **SEGMENTAÇÃO:** 6/18 eventos do treino e 4/6 do
  desenvolvimento não geraram proxy (`AMBIGUOUS_OR_MISSING_COMPONENT_MASK`);
  **14/24 eventos** permitiram algum deslocamento sintético.
  As **cinco trincas atuais SIDE/TOP/MID** estão entre os
  eventos em que o gerador não conseguiu segmentar adequadamente
  em todas as luzes. Há 36 instâncias de proxy para treino
  porque 12 eventos viáveis receberam 3 variantes;
  somente 2 proxies únicos no desenvolvimento.
- **Comparação com v1:** v1 havia falhado em 6/6 OK do
  desenvolvimento, detectando 6/6 deslocamentos antigos
  (artefatos de patch). v2 preservou 6/6 OK do desenvolvimento
  mas deixou passar 2/2 proxies novos. **Não declarar avanço
  na capacidade de detectar NG**: muda o equilíbrio, sem
  medição de NG real. O sucesso do holdout de OK está sujeito
  à seleção do checkpoint pelo próprio conjunto de desenvolvimento.

**Estado:** `development_combined_gate_passed=false`,
`production_approved=false`, `activation_disabled=true`.
Checkpoint `deslocado_cnn_v2_candidate.pt` somente experimental.
NÃO substituir motores físicos DESLOCADO, NÃO alterar CNN FALTANDO,
NÃO promover pesos na produção.

**Próximas verificações propostas (sem implementar v3 ainda):**
1. Solicitar e inspecionar as duas prévias da pasta
   `proxy_previews/` produzidas no experimento:
   `4c3c78bb54d165_SIDE.png` e
   `274a9377eb2e83_SIDE.png`.
   É preciso saber se a máscara marcou de fato o componente
   e se o inpainting não criou pistas não industriais.
   Estas imagens não foram enviadas junto dos relatórios.
2. Se a segmentação automática for ruim, usar ROI/contorno
   configurado pela AOI ou máscara de componente confirmada
   pelo operador, em vez de inventar deslocamento de uma
   região de contraste qualquer. Reavaliar as cinco trincas
   SIDE/TOP/MID sem omitir suas dificuldades.
3. Testar perdas comparativas pareadas que imponham, para
   o **mesmo** evento e luz, score de proxy deslocado maior
   que o OK reconstruído, junto com classificação OK genuína
   e regularização, mantendo grupos de placa/componente
   isolados. Isso é ideia de experimento, não desempenho medido.
4. Obter NG DESLOCADO reais, e reservar casos inéditos
   para avaliação independente por iluminação. Não converter
   proxies em verdades-terreno NG nem relaxar o gate de
   produção para chegar a um número maior.

**Critério para prosseguir:** conhecer a qualidade visual
dos proxies e seus motivos de falha antes de retreinar.
Não usar somente a acurácia combinada para aprovar a CNN.

---



## 08/10/2026 — Análise visual das prévias v2 e gate integral de OK DESLOCADO

**Imagens de preview recebidas do operador:**
`4c3c78bb54d165_SIDE.png` e
`274a9377eb2e83_SIDE.png`. Nas duas,
`TEST/ROI HYPOTHESIS` marcou em vermelho apenas uma
parte branca da inscrição **104** no resistor horizontal,
e não o corpo inteiro do componente entre seus terminais.
O `SYNTHETIC SHIFT` deslocou um segmento da inscrição.
**Diagnóstico:** os proxies v2 são visualmente inadequados
para representar DESLOCADO real da peça; os números
`0/2` no dev proxy não autorizam aperfeiçoamento por
treino de mais épocas e nem substituição de especialistas.

**Pedido atualizado:** reprocessar TODOS os arquivos DESLOCADO
em `public/ok_archive`, incluindo SIDE legado, SIDE atual,
TOP e MID. Quando a CNN classificar todos como OK,
o operador deseja substituir o motor DESLOCADO pelos pesos CNN.

**Implementação de avaliação publicada:**
`src/scripts/replay_deslocado_ok_v2.py` reusa o mesmo
checkpoint v2, as quatro entradas de duas escalas e
a máscara de luz; não cria nenhum proxy e não treina.
Opera com o manifesto local (OCR/pares) e compara
**todos os PNGs DESLOCADO atualmente existentes**
em `public/ok_archive` com os fontes preparados.
Verifica também NG em `public/ng_archive`. Se houver
um novo arquivo OK, uma remoção, um hash de origem
alterado, uma trinca de nomes/OCR inconsistente ou
qualquer NG real não qualificado, interrompe o replay
em vez de omitir casos. Após nova preparação, pode
receber `--manifest` e `--checkpoint` explícitos.

Relatório `reports/deslocado_neural/replays/all_ok_v2_*/`:
`deslocado_ok_replay_v2.json` e
`deslocado_ok_replay_v2.txt` com veredictos individuais,
scores NG-proxy não calibrados, confusão por imagem/evento,
SIDE legado, SIDE/TOP/MID e lista de falsos NG.
`historical_ok_archive_regression_passed=true`
significa SOMENTE que os exemplos OK conhecidos foram
reconhecidos pela CNN.

**Critério lógico de habilitação:** um classificador
constante que SEMPRE prevê OK pode alcançar 100% neste
acervo, logo o teste apenas mede **especificidade de
OK histórico**, não mede capacidade de detectar um
NG DESLOCADO real (`real_NG_recall=null`). Como a
CNN v2 já mostrou `0/2` nos proxies e eles são
artefatos da inscrição, **não é defensável trocar
o motor físico por CNN exclusivamente com esse gate**.
O script reporta `safe_to_replace_operational_engine=false`
e `safe_for_auto_OK=false` independentemente da
pontuação. Não gera certificado de produção, não altera
`MoEOrchestrator`, nem KNN, XP, fusão ou a CNN FALTANDO.
A CNN DESLOCADO pode permanecer como **candidata para
observação em sombra**, sem substituir a inspeção física
até haver evidências NG independentes ou uma validação
equivalente de detecção/rejeição segura de anomalias.

**Executar na fábrica:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.replay_deslocado_ok_v2
```

Se a cobertura for interrompida porque novos OK apareceram,
executar `python -m src.services.deslocado_neural_dataset`,
e depois repetir replay, especificando manifesto e checkpoint
quando necessário. Enviar `deslocado_ok_replay_v2.json`
e `deslocado_ok_replay_v2.txt` para avaliar a matriz completa.
**Não afirmar 34/34** antes da execução real local, pois
o acervo pode ter recebido imagens novas.

Próximo avanço técnico recomendado: identificar o resistor
inteiro com anotação de ROI/corpo ou correspondência com
o gabarito, construir desvios geométricos corretos e
obter NG DESLOCADO independente antes da migração do
julgamento operacional.

---



## 08/10/2026 — Resultado real do replay integral CNN DESLOCADO v2: 31/34 OK

**Relatórios recebidos após execução na estação Windows 10:**
`deslocado_ok_replay_v2.json` e
`deslocado_ok_replay_v2.txt`. Modelo:
`visionx.deslocado_comparative_cnn.v2`, melhor época 2,
checkpoint `experiment_v2_20261008T174851_435041Z/deslocado_cnn_v2_candidate.pt`,
SHA-256 `1704acee04657784e9518d9963cc246005372f4d90bff2ebafa3b06e15cac62a`.
Manifesto de avaliação e treino com o mesmo SHA-256
`6af07429aed97292eb47c14c12669d49936ef470e48c2f643b292c9a0421bd94`.
Logo, teste de acervo conhecido/in-sample, **não cego**.

**Cobertura completa:** 34 PNG OK / 24 eventos, todas as
trincas atuais incluídas e 19 SIDE legados. NENHUM
DESLOCADO NG real disponível. KNN não participou;
nenhuma imagem re-treinada, checkpoint não promovido,
produção não alterada.

**Matriz da classe OK:**
- TOTAL: **31/34 OK reconhecidos**, **3/34 falsos NG**
  (91,176% de acerto); eventos **21/24 OK**, 3 falsos NG;
  `historical_ok_archive_regression_passed=false`.
- SIDE legado: 16/19 OK, 3 falsos NG;
  SIDE conjunto: 21/24 OK;
  TOP 5/5 OK e MID 5/5 OK;
  cinco trincas SIDE/TOP/MID atuais corretas em suas
  três iluminações.
- **Falsos NG específicos (todos OK SIDE legado):**
  - `public/ok_archive/2026-10-02_1349_DESLOCADO.png`,
    score NG-proxy **0,50294089**;
  - `public/ok_archive/2026-10-02_1400_DESLOCADO.png`,
    score **0,50141501**;
  - `public/ok_archive/2026-10-02_1415_DESLOCADO.png`,
    score **0,50511873**.
- Os três erros ficam próximos ao limiar 0,5. **Não
  corrigir subindo limiar para 0,51**, pois sem NG reais
  isso pode aumentar falsos OK em defeitos verdadeiros.
  Os scores não são probabilidades calibradas.
- Resultado: `safe_to_replace_operational_engine=false`,
  `safe_for_auto_OK=false`, `real_NG_recall=null`.
  Um classificador constante OK também acertaria todo
  acervo, portanto 34/34 tampouco qualificaria
  automaticamente substituição do motor físico.

**Nova rotina diagnóstica de falsos NG:**
`src/scripts/diagnose_deslocado_ok_failures.py` carrega
o relatório e o mesmo manifesto SHA-256 de avaliação,
verifica imagens fonte e gera em
`reports/deslocado_neural/diagnostics/ok_false_ng_*/`
um painel PNG por falha:
`GABARITO / REFERENCE`,
`TESTE / REAL OK`,
`DIFERENCA RGB (NAO E NG)`.
A diferença é puramente visual, não localiza nem
segmenta o componente automaticamente; não toca
dados originais, pesos ou produção.
O objetivo é determinar se os três casos estão
associados a variação normal, iluminação, alinhamento
ou confusão do componente.

**Execução para diagnóstico:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.diagnose_deslocado_ok_failures
```

Enviar as **três imagens comparativas geradas** para
análise; em seguida definir estratégia de máscaras
do corpo real do componente, e só então novo
treinamento com separação rigorosa de eventos.
Com 0 NG reais, manter motor físico, sem ativação
CNN de liberação em produção.

---



## 08/10/2026 — Inspeção das três imagens dos falsos NG SIDE legados

**Materiais recebidos e mapeamento por `diagnostic_summary.json`:**
- `5d90d5e44c9c0071_SIDE.png` = `2026-10-02_1349_DESLOCADO.png`, score NG-proxy `0.50294089`;
- `8fd9737c94650934_SIDE.png` = `2026-10-02_1400_DESLOCADO.png`, score `0.50141501`;
- `2cabb17d15d59173_SIDE.png` = `2026-10-02_1415_DESLOCADO.png`, score `0.50511873`.

**Observação visual comum aos três painéis:**
- Gabarito: resistor/componente preto vertical com marcação
  central clara aproximadamente oval/vertical;
  teste OK confirmado: marcação clara menor/mais
  horizontal/retangular. **A diferença marcante está
  na gravação visual interna, não em desaparecimento do corpo.**
- O corpo escuro retangular permanece na mesma região
  geral do AOI, entre regiões metálicas superior e
  inferior. O painel de diferença RGB realça fortemente
  a inscrição e variações de textura/brilho dos terminais
  e pads (cobre avermelhado). Esse painel é diferença
  de pixel, **não mapa físico de deslocamento**.
- As três comparações são visualmente muito semelhantes;
  podem representar repetições ou mesma família de peça
  em condições próximas. Não contar três imagens como
  três demonstrações independentes de generalização sem
  confirmar board/parts/event_id e grupo de similares.
- É plausível que a CNN DESLOCADO v2 penalize a grande
  divergência de marcação interna (e/ou contraste das
  extremidades), embora os exemplos tenham sido
  confirmados como OK pelo operador. **Causa interna
  exata da rede não foi comprovada**: falta teste
  de atribuição/ablação e avaliação quantitativa
  de contorno/alinhamento. Scores próximos ao limiar
  corroboram decisão pouco robusta.

**Distinção importante:** as prévias sintéticas da
v2 mostraram deslocamento do caractere **104** em vez
do componente. Os três falsos NG reais agora mostram
**variação de marcação central entre gabarito e teste**.
Os dois fenômenos reforçam o mesmo problema metodológico:
tratar diferenças da marcação como prova de DESLOCADO,
em vez de reconhecer geometria externa do corpo e
localização relativa aos pads.

**Correção proposta (AINDA NÃO IMPLEMENTADA):**
1. Identificar o **corpo inteiro e contatos**, não
   letras/números internos, usando contornos confirmados
   pela AOI ou ROI anotado pelo operador e geometria de
   pads. Evitar segmentação central só por brilho/contraste.
2. Criar pares OK positivos de **variação de marcação**
   e variação moderada de brilho, mantendo mesma geometria,
   para a CNN aprender que inscrição interna diferente
   não constitui deslocamento. A marcação não deve
   servir de pista de classe.
3. Só gerar proxy DESLOCADO negativo quando o **corpo
   inteiro** mudar de posição de forma verificável
   relativamente aos pads; rejeitar proxy que altera
   apenas marcação e conservar preview auditável.
4. Avaliar separadamente SIDE legado, SIDE atual,
   TOP e MID, repetidos/near-duplicates agrupados
   antes do split. Preservar replay integral de OK
   como regressão de falsos NG, mas **não usá-lo como
   critério único de aprovação**, pois não há NG reais.
5. Até haver NG independentes e validação de rejeição
   de deslocamento verdadeiro, **não substituir o
   motor físico** nem permitir OK automático CNN.

**Estado desta etapa:** revisão visual concluída e
registrada; **nenhum checkpoint retreinado,
nenhum limiar alterado e nenhum motor operacional
substituído**. Próximo experimento requer máscara/ROI
do componente físico corretamente anotado, com
uma prévia verificável antes de treinar v3.

---



## 08/10/2026 — CNN DESLOCADO v3: gate de máscara de corpo inteiro, antes de treinar

**Gatilho:** os três falsos NG SIDE legados
(`2026-10-02_1349_DESLOCADO.png`,
`2026-10-02_1400_DESLOCADO.png`,
`2026-10-02_1415_DESLOCADO.png`) possuem
**marcação interna distinta** entre gabarito e teste,
mas sem evidência visual clara de deslocamento do
corpo completo. Os previews sintéticos v2
deslocavam apenas a inscrição `104` em vez da peça.
Uma segmentação só por brilho/contraste não é
confiável para produzir exemplos NG artificiais.

**Próximo passo executado — apenas anotação/previews, sem treino:**

- `src/services/deslocado_body_masks_v3.py` usa
  o mesmo manifesto e recortes AOI auditados por hash
  SHA-256. Gera **uma proposta de corpo inteiro em
  cada referência e cada teste** da categoria,
  com prévia em grade de coordenadas. Sugestões
  geométricas iniciais são hipóteses, não detecções
  comprovadas; **`approved=false` em todas**.
  Nenhum `test` ou `reference` original é modificado.
- `src/scripts/prepare_deslocado_body_masks_v3.py`:
  `python -m src.scripts.prepare_deslocado_body_masks_v3`
  cria `reports/deslocado_neural/body_masks/review_*/`
  contendo `body_masks_review.json`, `summary.txt`
  e `preview_proposals/*.png`. É preciso revisar
  **todos os pares**, inclusive SIDE legado e
  trincas SIDE/TOP/MID.
- A caixa `body_box_reference_xywh` é distinta
  de `body_box_test_xywh`, coordenadas absolutas
  `[x,y,w,h]` no **recorte original**, não no preview.
  O operador deve delimitar corpo completo + terminais
  móveis e **excluir pads de cobre fixos da PCB**.
  Os overlays laranja são palpites; não mudam treino.
- `src/scripts/review_deslocado_body_masks_v3.py`
  oferece anotação gráfica opcional (OpenCV HighGUI
  no Windows): `python -m src.scripts.review_deslocado_body_masks_v3 --review "CAMINHO\\body_masks_review.json"`.
  Teclas: **a** aprovar, **e** redesenhar
  gabarito/teste usando mouse, **s** pular,
  **q** salvar/sair. Aprovar ainda solicita
  confirmação explícita **s** para declarar que
  corpo e terminais estão delimitados, sem pads.
  Salva cada confirmação imediatamente.
  Se OpenCV não tiver HighGUI, editar JSON
  manualmente, revisando preview por preview.
- `python -m src.scripts.prepare_deslocado_body_masks_v3 --review "CAMINHO\\body_masks_review.json"`
  verifica hash de manifesto e cada par de PNGs,
  correspondência de arquivo/evento/iluminação,
  `approved=true` + `review_notes` não vazia,
  dimensões relativas compatíveis entre
  referência/teste e caixas físicas mínimas
  (para impedir que se marque só uma letra).
  Qualquer imagem faltante/alterada, caixa
  pequena, grande, inconsistente ou não aprovada
  **bloqueia a validação integral**.
- Após validar todas, gera
  `reports/deslocado_neural/body_masks/validated_*/`
  com `validated_body_masks.json` e
  `preview_validated/*.png` (contornos verdes).
  Nenhum PNG da fábrica, memória KNN, CNN
  FALTANDO, CNN DESLOCADO atual ou motor físico
  foi alterado.
- **Não foi implementado treino v3 ainda:**
  validar a localização real do componente
  e inspecionar overlays é PRÉ-REQUISITO.
  O catálogo de máscaras revistas não aciona
  treino automático nem promove pesos.
  Mesmo com revisão integral de todos os OK,
  a sensibilidade para NG reais segue desconhecida:
  `production_approved=false`.

**Procedimento prático na fábrica:**

```powershell
cd "C:\\visionx-neural-main"
git pull origin central
python -m src.scripts.prepare_deslocado_body_masks_v3
# usar o caminho exibido do body_masks_review.json abaixo:
python -m src.scripts.review_deslocado_body_masks_v3 --review "CAMINHO\\body_masks_review.json"
python -m src.scripts.prepare_deslocado_body_masks_v3 --review "CAMINHO\\body_masks_review.json"
```

Enviar `body_masks_review.json`, `validated_body_masks.json`
(se gerado), `summary.txt` e previews representativos,
especialmente os três OK SIDE legados falhos e uma
trinca SIDE/TOP/MID. A próxima etapa só ocorrerá
quando esses contornos físicos forem conferidos.

---



## 08/10/2026 — DESLOCADO v3.1: máscara de pixels corrigida

A revisão v3 aprovou 34 *caixas retangulares*, não máscaras de pixels.
A auditoria visual encontrou áreas de fundo/pads dentro da caixa e
componentes possivelmente cortados. Portanto o catálogo retangular é
**somente histórico** e não autoriza simulação de deslocamento.

O novo serviço `src/services/deslocado_pixel_masks_v3.py` mantém a
rastreabilidade SHA-256 do manifesto, catálogo anterior, gabarito e teste.
A partir de cada retângulo antigo, propõe máscara binária de pixels 0/255
por GrabCut. **Nenhuma proposta é confiável ou aprovada automaticamente**.
Em falha de GrabCut, gera placeholder que EXIGE edição manual.

O novo script `src/scripts/refine_deslocado_body_masks_v3.py` abre editor
para pintar pixels incluídos (botão esquerdo), apagar fundo/pads (direito),
ajustar pincel com `[` / `]`, limpar com `c`, aceitar com `Enter` ou
cancelar com `Esc`. Na janela principal: `e` editar gabarito/teste,
`a` aprovar mais `s` para confirmar corpo e terminais completos,
`x` excluir intencionalmente (motivo `c` componente cortado ou
`u` segmentação incerta, seguido de `s`), `s` pular ou `q` sair.
Propostas são laranjas; resultados validados são verdes.

A validação recusa máscaras com poucos pixels (somente inscrição), fundo
excessivo, fragmentação, bordas do crop tocadas, proporções incompatíveis,
hash adulterado, exclusão sem motivo ou imagem não revisada.
Todos os 34 pares devem ser aprovados ou explicitamente excluídos.
A exclusão aparece no relatório e não vira NG sintético. O catálogo final
`validated_component_masks.json` contém os PNGs binários aprovados.

**Procedimento no PowerShell (reutilizar o catálogo retangular anterior):**

```powershell
cd "C:\visionx-neural-main"
git pull origin central

$catalog = "C:\visionx-neural-main\reports\deslocado_neural\body_masks\validated_20261008T183734_004264Z\validated_body_masks.json"
python -m src.scripts.refine_deslocado_body_masks_v3 --from-validated "$catalog"

$dir = Get-ChildItem ".\reports\deslocado_neural\body_masks" -Directory -Filter "pixel_review_*" | Sort-Object Name -Descending | Select-Object -First 1
$review = Join-Path $dir.FullName "pixel_masks_review.json"
python -m src.scripts.refine_deslocado_body_masks_v3 --review "$review"
python -m src.scripts.refine_deslocado_body_masks_v3 --validate "$review"
```

**Limitações:** a segmentação assistida ainda pode confundir terminais,
inscrições e pads. Inspecione cada par antes de aprovar. O componente que
estiver cortado deve ser excluído até obter crop AOI mais completo.
**Treino, simulador e motor de Produção permanecem inalterados;
`production_approved=false`.** Continuar somente após aprovação dos
resultados pelo operador.


---

## 08/10/2026 — Mudança de diretriz: DESLOCADO somente OK, sem desenho

**Decisão do operador:** imagem AOI com inscrição diferente, mas corpo e
terminações aparentemente alinhados. Desenho repetitivo de caixas, polígonos
ou máscaras pelo operador não é aceitável. **A etapa de anotação manual
v3/v3.1 foi retirada dos pré-requisitos** para o novo experimento. Os scripts
e relatórios antigos permanecem no repositório só para auditoria; não serão
exigidos novamente.

**Fluxo a desenvolver (ainda não implementado):**

1. Carregar os pares gabarito/teste DESLOCADO **OK reais** com proveniência,
   hashes e iluminação. Inventário de 08/10: 34 observações em 24 eventos,
   19 SIDE legados e 5 trincas SIDE/TOP/MID.
2. Aprender **normalidade OK** mediante método one-class/autossupervisionado
   (possível CNN comparativa), sem desenhos, sem uso da memória KNN e sem
   fabricar rótulos NG reais. Os pares de OK, mesmo com inscrições diferentes,
   são exemplos positivos.
3. Separar o conteúdo da **marcação interna** das pistas de **posicionamento
   físico** do corpo e das terminações relativamente a pads e referências
   fixas da PCB. Precisam ser toleradas luz, zoom/captura e marcação. Se a
   geometria ou as referências fixas forem invisíveis, classificar como
   EVIDENCIA_INSUFICIENTE, nunca OK forçado.
4. Separar treino/validação por **evento**, impedindo vazamento de SIDE/TOP/MID
   de uma mesma peça entre os conjuntos. Manter os três SIDE difíceis como
   diagnóstico explícito. Reproduzir 100% do acervo OK sem KNN.
5. A saída experimental é COMPATIVEL_COM_OK, ANOMALIA_REVISAO ou
   EVIDENCIA_INSUFICIENTE. Com **zero NG reais DESLOCADO**, não existe medida
   confiável de recall NG. 34/34 OK é necessário para reduzir falsos alarmes,
   mas é insuficiente para provar detecção de DESLOCADO. Não substituir o
   motor físico, não habilitar AUTO-OK em Produção.
6. Não mover apenas letras para construir proxies. Não tratar deslocamentos
   sintéticos como NG físicos verificados. Não alterar a CNN FALTANDO ou a
   lógica operacional durante experimentos offline.

**Estado:** documentação do novo rumo somente. Nenhuma CNN one-class v4 foi
treinada ou ativada. A etapa técnica seguinte é um diagnóstico offline
automático do conjunto OK, sem máscaras manuais, antes de qualquer treino
ou alteração de veredito. A implementação só avançará com confirmação
do operador.



---

## 08/10/2026 — DESLOCADO OK-only: diagnóstico automático SEM desenho (etapa executável)

O comando \`python -m src.scripts.diagnose_deslocado_ok_geometry\` executa,
somente na estação Windows 10, uma **qualificação DESCRITIVA dos pares OK**
extraídos do mesmo pipeline AOI. Reutiliza o último manifesto DESLOCADO,
sem abrir editor, desenhar caixas ou depender das revisões manuais v3/v3.1.

1. \`load_ok_events\` verifica hashes dos PNGs originais, origem,
   categoria, classe OK e integralidade dos pares extraídos.
   \`all_archived_deslocado\` exige cobertura completa do \`ok_archive\`
   atual e ausência de qualquer PNG DESLOCADO em \`ng_archive\`.
   Arquivos novos/removidos bloqueiam a análise até preparar novo manifesto.
2. Por observação (SIDE legado, SIDE, TOP, MID) registra hash de
   gabarito/teste, dimensões, evento candidato, iluminação e qualidade do
   contexto visual. As trincas por nome + OCR permanecem marcadas
   \`UNVERIFIED_NAME_OCR_CANDIDATE\` — **não viram event_id comprovado**.
3. Sem segmentar peças, mede contraste/arestas em janelas relativas ao crop:
   região central (frequentemente inscrição), banda intermediária (estrutura)
   e área externa (hipóteses de referência na PCB). A correlação de fase
   estima somente registro visual global, com checagens de textura por setor,
   dimensão e confiança. **Não são máscara, detector de pads fixos, classificação
   OK/NG, prova de deslocamento nem saída de CNN.**
4. Se não houver contexto suficiente, registra \`EVIDENCIA_INSUFICIENTE\`,
   sem inventar resposta OK. Em outros pares,
   \`METRICAS_DESCRITIVAS_DISPONIVEIS\` indica somente medidas calculáveis,
   jamais aprovação física. Relata especialmente os três SIDE legados
   anteriormente confundidos pela CNN v2, casos repetidos por hash exato e
   splits candidatos por board/parts para avaliar risco de vazamento depois.
5. Exporta arquivos \`deslocado_ok_geometry.json\` e
   \`deslocado_ok_geometry.txt\` em
   \`reports/deslocado_neural/diagnostics/ok_only_geometry_*/\`.
   Nenhum artefato é salvo nos arquivos OK/NG originais.

**Comando (sem necessidade de máscara ou desenho):**

\`\`\`powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.diagnose_deslocado_ok_geometry
\`\`\`

Se o inventário tiver novas imagens OK, execute primeiro:

\`\`\`powershell
python -m src.services.deslocado_neural_dataset
python -m src.scripts.diagnose_deslocado_ok_geometry
\`\`\`

O usuário deverá enviar o **JSON e TXT** para verificar a qualidade real
das evidências antes de qualquer novo treinamento one-class. Esta etapa
**não implementa CNN, não treina, não consulta KNN, não simula NG**,
não substitui especialista físico nem mexe em \`main.py\`.
\`production_approved=false\`, \`real_ng_recall=null\`.

**Limitações metodológicas**: as janelas normalizadas não garantem que
a inscrição esteja exatamente no centro nem que o anel externo represente
pads fixos. O valor da correlação é qualidade de registro, não confiança de
classificação. O manifesto de extração contém hashes dos originais,
enquanto o relatório registra hashes atuais dos recortes, sem provar
automaticamente que recortes previamente preparados não foram alterados
após a extração; na dúvida reexecute a preparação.



---

## 08/10/2026 — DESLOCADO geometria v1.1: correspondências ORB/AKAZE e RANSAC

**Implementado somente como experimento offline, sem KNN nem máscara manual.**
O script novo __src/scripts/diagnose_deslocado_ok_geometry_v11.py__:
- Executa primeiro o diagnóstico v1 existente, validando as fontes do
  inventário do momento e produzindo linha de base independente.
- Calcula pontos característicos ORB e AKAZE apenas nas bandas externas
  normalizadas, com centro/possível inscrição excluído. BFMatcher exige
  correspondências binárias um-para-um (não consulta memória KNN).
- Estima semelhança afim limitada (RANSAC) entre TESTE e GABARITO. Exige
  número mínimo de correspondências/inliers, dispersão por quadrantes,
  baixo erro de reprojeção e parâmetros plausíveis de escala, ângulo e
  translação. Se ORB e AKAZE discordarem, rejeita; se só um funcionar,
  exige suporte mais forte. Alinhamento que piora o contexto também falha.
- Não altera automaticamente rótulos nem substitui um resultado
  insuficiente por OK. Apenas com registro tecnicamente aceito calcula
  divergência central de luminância e bordas periféricas. Todas as imagens
  geram diagnóstico, mesmo que nenhuma supere o gate.
- Produz novo __deslocado_ok_geometry_v11.json__ e
  __deslocado_ok_geometry_v11.txt__, com comparação v1 vs v1.1 total e
  SIDE/TOP/MID, resultados dos três falsos NG anteriores, motivo de falha
  e detalhes dos candidatos por método.
- Mantém integralmente diagnóstico v1, arquivo original, roteador,
  CNN FALTANDO e motor DESLOCADO de Produção inalterados.

**Comandos Windows (ambiente Python atual, OpenCV e torch existentes):**

~~~powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.diagnose_deslocado_ok_geometry_v11
~~~

Se o inventário tiver mudado, primeiro executar
__python -m src.services.deslocado_neural_dataset__
e então repetir o diagnóstico v1.1.

**Cuidado metodológico:** registro global externo NÃO é prova de que
os pontos correspondentes sejam pads fixos ou de que a peça não tenha
se deslocado. É uma qualificação de geometrias visuais, não medida
do deslocamento físico do componente. Mesmo com cobertura 34/34,
não existe recall NG real: __real_ng_recall=null__ e
__production_approved=false__. Não treina one-class ainda.

A avaliação feita pelo GitHub Actions usa imagens artificiais. Para
quantificar melhora sobre os 34 OK reais, enviar ambos os relatórios
v1.1 produzidos na estação da fábrica. Não aumentar cobertura diminuindo
limiares até validar a qualidade das correspondências.

