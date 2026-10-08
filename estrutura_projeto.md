# Estrutura do Projeto: VisionX Neural

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

**Após executar:** enviar os três relatórios JSON/TXT da v2.
Não há motivo para substituir o motor físico mesmo que a v2
alcance seis OK corretos, sem NG reais de deslocamento.
Se o diagnóstico mencionar `AMBIGUOUS_OR_MISSING_COMPONENT_MASK`
ou erro de segmentação, não fabricar NG fictícios: será
necessário identificar visualmente a região correta.

---


## 08/10/2026 — CNN especializada DESLOCADO: preparação, bootstrap e coleta incremental

**Situação informada:** nenhum NG DESLOCADO real no acervo local;
há OK SIDE legados e OK SIDE/TOP/MID atuais. Contagem exata
somente após o inventário no PC. **Não confundir falta de NG
com treinamento aprovado.**

Arquitetura implementada, paralela ao fluxo CNN FALTANDO:

- `src/services/deslocado_neural_dataset.py` extrai com ScreenMonitor
  as imagens gabarito/teste de DESLOCADO e cria manifesto
  por classe e iluminação, mantendo fontes intactas.
- `src/core/neural/deslocado_cnn.py` cria modelo especializado
  próprio, baseado em duas escalas comparativas e três luzes;
  pesos DESLOCADO são distintos dos da CNN FALTANDO.
- `src/scripts/train_deslocado_cnn.py` inicializa o protótipo
  usando **OK reais vs deslocamentos LOCAIS SINTÉTICOS**,
  rotulados explicitamente como proxy (não são NG reais).
  Checkpoint `production_approved=False` e
  `allow_automatic_classification=False`, não integrado em
  `main.py`. Pode receber NG **reais humanos** de forma
  separada em novos treinos online, sem autopromoção.
- `src/services/neural_online_learning.py` já registra treinadores
  por categoria. Foi adicionado DESLOCADO, ativado quando
  `recognition_route=NEW_EXPERTS` e rótulo OK/NG
  realmente fornecido por operador. O mesmo
  `DecisionPersistenceQueue` atua em Teste/Produção/Sombra,
  preservando gabarito/teste e três iluminações de uma peça
  como evento agrupado. Nunca treina com rótulo automático.
- `src/scripts/train_deslocado_cnn_online.py` carrega
  os novos eventos de DESLOCADO, valida SHA-256 e origem
  humana, reaprende com histórico + casos OK/NG adicionais
  em CPU. Modelos são **somente candidatos**.
- O roteador KNN conhecido continua intacto. Novos DESLOCADO
  continuam usando **motores físicos anteriores**, não a
  CNN incipiente, e o tooltip indica explicitamente
  `DESLOCADO_CNN_V1_BOOTSTRAP_NOT_ACTIVE`.

**Segurança:** sem exemplos reais NG, não existe recall
NG validado, e os exemplos sintéticos podem induzir
aprendizado de artefatos. Não substituir a análise em
produção até haver NG reais independentes, avaliação
adequada por luz e aprovação separada. Nenhum NG fictício
deve ser salvo nos arquivos reais nem na memória KNN.

**Execução no Windows 10** (sem mexer no computador XP):

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.deslocado_neural_dataset
python -m src.scripts.train_deslocado_cnn --epochs 15 --batch-size 4 --size 160
```

Detalhes, entradas, saídas e pendências em
`docs/DESLOCADO_CNN_DATASET_AND_TRAINING.md`.

---


## 08/10/2026 — Aprendizado incremental ao vivo de CNNs especializadas

**Solicitação operacional:** toda captura **nova**, reconhecida por rota
`NEW_CNN`, quando confirmada **OK** ou **NG** pelo operador deve disparar
treinamento incremental imediatamente, em Teste, Produção e Sombra.
Não depender somente da discordância IA × operador; mesmo se concordarem
os pares originais devem ser preservados. Implementação da primeira
especialidade: CNN FALTANDO v2. Base extensível para outras CNNs futuras.

**Pontos de integração reais:**
- `src/services/anomaly_learning.py`: `_decision_task`
  ativa `save_images=True` em **qualquer caso novo CNN confirmado**
  (inclusive quando a IA concordou com o OK/NG humano).
  É independente do modo operacional, pois todos compartilham
  `save_label` e a `DecisionPersistenceQueue`.
- `src/services/decision_persistence.py`: só depois de salvar
  com êxito o registro humano no dataset/KNN é enfileirado o treino.
  Em multilight, as **três gravações** de SIDE/TOP/MID precisam
  ter êxito antes do pedido incremental.
  A decisão na AOI NÃO espera treinamento nem escrita de pesos.
- `src/services/neural_online_learning.py`: journal durável local
  em `reports/neural_online/events/`, guarda **gabarito/teste completos
  por iluminação**, metadados, rótulo humano e hashes; trabalhador
  serial de baixa prioridade CPU, em subprocesso separado. Novo caso
  de três iluminações é **um** evento supervisionado com três pares.
  Retoma tarefas pendentes após reiniciar o ODIN. Mantém
  `statuses/`, `logs/` e `latest_event.json` para auditoria.
  `SPECIALIST_TRAINERS` registra categorias e scripts para
  futuras CNNs, sem modificar a fila/persistência.
- `src/scripts/train_faltando_cnn_v2_online.py`: carrega pesos
  ATIVOS do `FaltandoCNNV2`, os 117 screenshots históricos
  organizados em eventos, e todos os novos eventos confirmados;
  faz 3 épocas por solicitação com replay balanceado entre
  OK/NG, taxa pequena 0.00002, CPU limitada a dois threads,
  **incluindo explicitamente o novo caso em cada época**
  para não depender de amostragem aleatória.
- Todo novo checkpoint é salvo como **CANDIDATO** em
  `reports/neural_online/checkpoints/`. A validação testa
  **cada imagem/luz** dos 117 exemplos conhecidos e todas
  as novas imagens confirmadas. Para promoção requer zero
  erros em ambos, nenhum falso OK nos NG históricos,
  nenhuma perda de acerto histórico. Dados ilegíveis,
  rótulos humanos contraditórios, arquivos ausentes ou
  falhas de regressão bloqueiam ativação. Pesos antigos
  permanecem intactos.
- Quando o candidato passa, a promoção é feita com escrita
  atômica de `reports/neural_online/live_active.json` contendo
  hash SHA-256 do checkpoint. `src/core/neural/faltando_live.py`
  observa este ponteiro na próxima inspeção, verifica
  schema, caminho, hash e pesos antes de carregar.
  Um arquivo inválido resulta em `REVISÃO OBRIGATÓRIA`;
  nenhuma atualização parcial se torna modelo ativo.
- O tooltip de `CASO NOVO • CNN FALTANDO v2` explica como
  a amostra humana dispara treinamento assíncrono, e o status
  de `QUEUED/TRAINING/PROMOTED/REJECTED/FAILED`.

**Segurança contra autoaprendizado incorreto:**
- Decisão `production_auto`/`auto` **não é verdade-terreno**;
  a CNN **não aprende de sua própria previsão**.
- Casos `KNOWN_KNN` não são reensinados como exemplos novos.
- Apenas rótulos humanos OK/NG registrados por botões, 0/1
  ou fontes humanas verificadas entram na fila; a política
  produtiva atual mantém obrigatória a confirmação humana para
  novos OK de CNN FALTANDO experimental.
- A auditoria do acervo histórico é **in-sample**, uma proteção
  contra esquecimento e não teste cego de novos defeitos. A
  promoção incremental NÃO declara modelo certificado para
  liberação autônoma. Nenhuma mudança nas outras categorias.
- Não usar privilégios administrativos, não alterar XP, não
  exigir nuvem, tokens ou servidor de treinamento. Todos os
  artefatos de `reports/neural_online/` ficam locais e ignorados
  pelo Git (não enviar fotos/pesos ao repositório).

**Fluxo de produção:** atualização disponível após
`git pull origin central` e `python main.py`. Não há
treinamento síncrono no loop AOI. O primeiro treino real
precisa ser testado na máquina Windows 10 com o checkpoint
v2 e a base histórica, depois de o operador confirmar um
novo caso FALTANDO.

**Extensão para categoria futura:** adicionar
`SPECIALIST_TRAINERS[canonical_category] = "src.scripts.train_<categoria>_online"`;
cada treinador deve gerar checkpoint candidato, validar
contra regressões próprias e promover atomicamente, mantendo
rótulos humanos e isolamento por categoria. Não reutilizar
pesos do modelo FALTANDO em outra categoria.

---


## 08/10/2026 — Roteamento MEMÓRIA KNN → Especialista por ocorrência

**Regra de decisão solicitada:** a análise normal do ODIN distingue uma
inspeção **já conhecida** de uma **nova** antes de executar especialistas.

```text
AOI → gabarito/teste + OCR categoria/placa/componente/iluminação
      ↓
Verificador de memória KNN (par exato, rótulo humano validado)
      ├── CONHECIDO OK/NG → usar só o rótulo do caso KNN conhecido
      ├── CONFLITO       → REVISÃO OBRIGATÓRIA
      ├── INDISPONÍVEL   → REVISÃO OBRIGATÓRIA
      └── NOVO           → ignorar voto KNN e delegar por categoria
                           ├── FALTANDO → CNN FALTANDO v2
                           └── OUTRAS   → motores/especialistas anteriores
```

**Implementação:** `src/core/verified_memory_router.py`
(`install_memory_first_router`) é instalado em `main.py`
**depois** de `install_faltando_cnn_live`. Para evitar falso
reconhecimento de um defeito como OK, *conhecido* só existe
quando são **idênticas** as imagens RGB de gabarito e teste
(duas impressões digitais SHA-256 de pixels), no mesmo
`board`, `parts`, `category`, `lighting_mode` e
valor da AOI (`value`). Exige:
- JSON `visionx.memory.v3` carregado pelo índice da KNN;
- `label` + `decision.operator_label` coerentes com a pasta;
- `decision.source` indicando **operador humano**, nunca
  `production_auto` ou `auto`;
- PNG do gabarito e PNG do teste presentes, legíveis e coerentes
  com `storage.test_image_fingerprint`.
- Se JSON é legado/sem imagens, aproximações 90%/99%,
  conflito de rótulo, OCR diferente ou gabarito diferente:
  **não** acionar atalho de memória. Casos diferentes seguem
  para especialista; conflito exato exige revisão humana.

A rota KNN conhecida usa **recuperação da classe do registro
humano exato** da própria memória KNN, não votação de vizinhos
aproximados. Como os dois PNGs precisam estar presentes, JSONs
antigos apenas com assinatura não são promovidos artificialmente
a conhecidos. Se a UI salvar memória nova, `reload_memory`
invalida o índice e ele é reconstruído a partir dos registros
atuais da KNN.

Para casos novos de `FALTANDO`, usa a mesma CNN v2 e
checkpoint já integrados, sem consulta KNN. Para outras
categorias novas, `_replay_without_memory=True` aciona
os especialistas existentes **sem** KNN. O runner
PhysicalOnlyOrchestrator/retrospectiva offline permanece isolado.

**Interface/tooltip visíveis:**
- `CASO CONHECIDO • MEMÓRIA KNN`
- `CASO NOVO • CNN FALTANDO v2`
- `CASO NOVO • MOTORES DA CATEGORIA`
- `MULTILIGHT • KNN + ESPECIALISTAS`
- `MEMÓRIA CONTRADITÓRIA • REVISÃO`

A linha já existente `lbl_db_info` exibe a rota, enquanto
seus tooltips e os de `lbl_reason`/`lbl_verdict`
explicam o motivo, o registro reconhecido e a rota por
iluminação SIDE/TOP/MID. `fuse_multilight` preserva
`recognition_light_routes` para não confundir a rota
de uma luz com a do evento inteiro.

**Produção e segurança:** o motor da categoria FALTANDO ainda
é experimental. Se houver **qualquer luz nova analisada pela CNN**,
o gate de Produção não envia OK automático: requer operador.
Para o caso KNN conhecido por **par exato e rótulo humano**,
a política produtiva preexistente permite OK automático quando
o veredito final for `FALHA FALSA` sem revisão; NG continua
exigindo operador. O mesmo comportamento das demais categorias
já configuradas é preservado.

**Limite operacional:** pixels idênticos entre duas capturas reais
são pouco comuns. Este primeiro roteador é intencionalmente
restritivo e tende a classificar grande parte dos casos futuros
como novos. Os 117 PNGs do arquivo histórico não são
automaticamente memória KNN: é necessário que existam registros
humanos KNN com **ambos os PNGs** e metadados coerentes.
Uma futura expansão para reconhecimento aproximado requer
validação controlada de falso OK e não pode reduzir esse gate
para uma similaridade arbitrária.

**Testes**: suíte `tests/test_verified_memory_router.py`
cobre caso conhecido OK/NG, casos novos, alteração de um
único pixel, OCR/contexto, iluminação, fonte humana vs
automática, conflito OK/NG, JSON sem imagens, recarga da
memória, replay físico isolado e mistura multilight. Estes
são testes sintéticos de software; o roteador ainda precisa
ser exercitado com capturas reais no PC da fábrica.

---


## 08/10/2026 — CNN FALTANDO v2 integrada à análise normal (com revisão humana em Produção)

**Replay real recebido da fábrica:** `archive_replay_v2.json` e
`archive_replay_v2.txt`, gerados às **16:27:54 UTC** no Windows 10.
O checkpoint testado foi exatamente
`reports/faltando_neural/models/experiment_v2_20261008T155311_256670Z/faltando_cnn_v2_candidate.pt`
(`sha256=6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599`,
**época 18**). O manifesto utilizado manteve SHA-256
`049c351fa19f8afdec5dbcc23df1dbe34c0fa1a14768e268b609fe0963473245`.

**Resultado comprovado no acervo conhecido** (inferência pura, KNN desligado):
117/117 imagens corretas; 67/67 eventos corretos; nenhum falso OK,
nenhum falso NG. Breakdown:
- SIDE legado: 42/42 (32 OK, 10 NG).
- SIDE atual: 25/25 OK.
- TOP atual: 25/25 OK.
- MID atual: 25/25 OK.
- Resultado por evento: 57 OK e 10 NG corretamente identificados.
- `passed_known_archive_regression=true`; os originais não foram modificados.

**Limite:** o replay é **in-sample/known-archive**, não teste cego.
Oito NG SIDE usados em treinamento, dois NG SIDE no desenvolvimento
anterior da v1. **Não existe nenhum NG real TOP/MID**.
Resultado de 100% NÃO comprova generalização para novas ausências.
Scores de sigmoid são **não calibrados**.

**Integração autorizada e implementada no código normal:**

- `src/core/neural/faltando_live.py`: carregamento preguiçoso
  (somente quando surge FALTANDO), CPU, checkpoint fixado pelo
  caminho e SHA-256 do replay, `torch.load(weights_only=True)`,
  schema, metadados e forma de entrada verificados. Usa o mesmo
  pré-processamento da v2 (gabarito/teste; imagem total e zoom central
  70%; SIDE/TOP/MID). Não retreina.
- Hook instalado em `main.py` **após os demais wrappers de
  `MoEOrchestrator.inspect`**. `FALTANDO`/`MISSING`
  usa **somente CNN v2** para a decisão local: não executa
  MissingExpert/Shift/Silk/SSIM/Semantic nem consulta KNN.
  As outras categorias continuam exatamente na rota anterior.
  `_replay_without_memory` e o PhysicalOnlyOrchestrator do runner
  continuam fisicamente isolados da nova rota.
- A operação multilight existente executa SIDE, TOP, MID e
  `fuse_multilight` normalmente, reutilizando a evidência
  CNN de cada luz. Um evento com uma luz NG forte continua NG.
- Entradas inválidas, luz ausente, pesos ausentes, alterados ou
  erro de carregamento -> `REVISÃO OBRIGATÓRIA`, **nunca
  `FALHA FALSA`**.
  Scores NG intermediários (`0,10 < score < 0,90`) também forçam
  revisão. Esses limiares NÃO são calibração certificada.
- A interface `src/ui/control_panel.py` exibe o veredito CNN
  mesmo sem `bounding_box` físico e registra a trilha CNN no
  `detail`/`decision_trace`.
- **Trava temporária de segurança no Modo Produção**:
  `src/ui/production_confidence_gate.py` converte toda proposta
  `FALHA FALSA` proveniente desta CNN **experimental**
  para `REVISÃO OBRIGATÓRIA` na política produtiva,
  exigindo o operador `0=OK/1=NG`; não transmite `0`
  automaticamente à AOI. `DEFEITO REAL` já exigia operador.
  Somente a decisão visual da categoria FALTANDO foi substituída,
  não a política de liberação segura; demais categorias preservam
  seu comportamento autônomo pré-existente.

**Como atualizar na estação real:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python main.py
```

O `faltando_cnn_v2_candidate.pt` **não é versionado no GitHub**;
deve continuar no mesmo caminho local indicado no replay. Ao
testar, conferir `active_engines: faltando_cnn_v2.py`,
`detail.cnn_v2_checkpoint_verified=True` e
`detail.cnn_v2_status=INFERENCE_OK`. Se pesar modelo falhar,
a resposta FALTANDO será revisão obrigatória e deverá ser
investigada, nunca corrigida por aprovação automática.

**Estado:** integração de decisão CNN implementada; **não afirmamos
teste AOI em tempo real ainda**. Para qualquer futura liberação
automática de OK com CNN exigir validação independente
e controle da incerteza; dados conhecidos sozinhos não bastam.

---


## 08/10/2026 — CNN FALTANDO v2: replay de 100% do acervo conhecido (implementado)

**Pedido aprovado pelo operador:** reprocessar todos os screenshots de
`C:\visionx-neural-main\public\ng_archive` e
`C:\visionx-neural-main\public\ok_archive` com a CNN FALTANDO v2 já
treinada, nas iluminações SIDE legado e SIDE/TOP/MID. Se passar tudo,
avaliar integração no ODIN normal com preferência CNN FALTANDO sobre
especialistas físicos. Não reinterpretar sucesso em treino como
prova de generalização.

**Diagnóstico e situação antes desse pedido:** a CNN v2 foi avaliada
somente em 13 eventos (11 OK + 2 NG) e acertou os 13 na validação
de desenvolvimento. **Ainda NÃO havia sido executado o replay total
do acervo de 117 PNGs**. No último inventário real: 117 fotos de
categoria FALTANDO, incluindo 10 NG SIDE legados, 32 OK SIDE legados
e 25 trincas OK SIDE/TOP/MID (75 imagens); 67 eventos quando trincas
coerentes por nome+OCR são agrupadas. Os números deverão ser
revalidados a cada execução, pois o arquivo é dinâmico.

**Implementação:** `src/scripts/replay_faltando_cnn_v2.py`

- Seleciona o checkpoint CNN FALTANDO v2 já salvo localmente em
  `reports/faltando_neural/models/experiment_v2_*/faltando_cnn_v2_candidate.pt`
  e o `manifest.json` do staging (pode informar ambos explicitamente).
- Carrega modelo em `eval()`/`torch.inference_mode()`, sem treino,
  sem KNN, sem especialistas físicos, sem acesso a rede, sem comandos AOI/XP.
  Usa **exatamente o mesmo** `DualScaleEventDataset` do treinamento
  (gabarito/teste, full + focus, luzes e parâmetros do checkpoint).
- Exige correspondência SHA-256 do manifesto com o checkpoint,
  verifica hashes atuais dos PNGs e compara a lista atual inteira de
  PNGs FALTANDO com os itens do manifesto: novos, removidos ou
  corrompidos bloqueiam a aprovação. Não omitir casos silenciosamente.
- Reprocessa cada arquivo **individualmente por iluminação**,
  emitindo o score NG por luz para SIDE/TOP/MID, e também calcula
  **veredito final por evento** (máximo logit NG entre luzes observadas,
  como na v2). Trincas completas contam como um único evento
  multilight. Para NG multilight futuro, o veredito do evento é
  central; por segurança o relatório também destaca divergências
  individuais por luz.
- Gera `archive_replay_v2.json` e `archive_replay_v2.txt` em
  `reports/faltando_neural/replays/archive_v2_<timestamp>/` com
  contagem por PNG e evento, legacy SIDE e por luz,
  confusão FP/FN, scores e identificadores de todas as observações.
- Falha com código de retorno diferente de zero se qualquer
  acerto faltar; **não** ativa CNN automaticamente, mesmo se 100%.
  `passed_known_archive_regression` significa apenas
  acerto no acervo **conhecido**; `ready_for_automatic_production=False`
  permanece explícito.

**Execução no Windows 10 do ODIN:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.replay_faltando_cnn_v2
```

Se houver nova preparação desde o treino, pode ser necessário
informar `--manifest` com o manifesto exato registrado em
`training_report_v2.json`; o replay não deve tentar
supor automaticamente o vínculo com outro staging.

**Critério de decisão:** se o replay reprovar, **não substituir**
os especialistas atuais. Se aprovar 100% do acervo conhecido,
o resultado será um gate de **não regressão histórica** e
a integração CNN poderá ser implementada em modo observação
ou guardada por flag, mas não liberar peças automaticamente
sem avaliação externa e tratamento de incerteza. O motivo:
a maioria das 117 imagens foi usada para ajustar pesos v2,
os únicos NG reais são SIDE e apenas 2 NG não participavam
do treino. Não há NG reais em TOP/MID; acertar todas as
imagens já conhecidas não prova detecção de NG inéditos.

**Nenhuma integração operacional implementada nesta etapa:**
sem alteração de `src/core/moe_orchestrator.py`,
`main.py`, KNN, controles do XP ou startup blocking gate.
Aguardar replay executado na estação real e revisar resultados.

---


## 08/10/2026 — CNN FALTANDO v2: primeiro treino real e diagnóstico

**Arquivos recebidos:** `training_report_v2.json`,
`holdout_predictions_v2.json`, `training_summary_v2.txt`
da execução local em 08/10/2026 às 15:53:11 UTC.
**Fonte:** mesmo `manifest.json` SHA-256
`049c351fa19f8afdec5dbcc23df1dbe34c0fa1a14768e268b609fe0963473245`
usado na v1.

**Dataset:** 117 imagens / 67 eventos (57 OK, 10 NG). Split por
componente + near-duplicate, 45 grupos: 54 eventos no treino
(46 OK / 8 NG), 13 na validação de **desenvolvimento** (11 OK / 2 NG).
A v2 executou 25 épocas na CPU, batch 4, entrada 160×160 e
recorte central de foco 70%, sampler balanceado.
**Não é teste cego**, pois estes 13 eventos já tinham sido
analisados durante o desenvolvimento da v1.

**Matriz de confusão na validação de desenvolvimento v2:**

| Verdade | Predito OK | Predito NG |
|---|---:|---:|
| OK | 11 | 0 |
| NG | 0 | **2** |

Acurácia 13/13 = **100%**, recall NG 2/2 = **100%**,
especificidade OK 11/11 = **100%**, FN_NG_as_OK = **0**.
A CNN v1 havia previsto todos os 13 como OK: 11 OK corretos
e dois NG perdidos (recall NG 0%). Os dois NG SIDE antes
ignorados foram identificados agora:

- `2026-10-01_07-53-17-716_FALTANDO.png`: score NG
  **0,999785** (99,9785%).
- `2026-10-01_09-53-18-089_FALTANDO.png`: score NG
  **0,999793** (99,9793%).

Para os 11 OK, os scores NG ficaram entre
0,000016 e 0,000243. Esses são **scores de modelo não
calibrados**, não uma precisão garantida de 99,9%.
A curva apresentou NG recuperados desde a primeira época,
mas inicialmente nove falsos NG (FP); na época 3,
TP=2, TN=11, FP=0, FN=0 no mesmo conjunto de desenvolvimento.

**Achado de engenharia no seletor de checkpoint:**
o relatório original declarou `best_epoch=18` e perda
de desenvolvimento **0,000099**, embora a época 23
tenha atingido **0,000070** (época 25: 0,000081).
Isso ocorreu porque o código misturava uma tolerância fixa
`min_delta=0,0001` para early stopping com a seleção da
melhor época. A correção `dd092fa` separou seleção
por **menor perda efetiva** e tolerância apenas para
`patience`; adicionou teste específico de pequenas
melhorias. **O checkpoint local antigo continua com
os pesos da época 18**, pois a correção só vale para
novos treinamentos. A época 23 também acertou 13/13,
mas só uma nova execução salvará seu estado como melhor.

**Estado operacional:** CNN v2 permanece experimental
(`production_approved=False`), sem tocar KNN,
fusão física, modo Produção, comando XP, interface ou
gate de regressão. Há somente **dois NG avaliados**,
todos SIDE, e não há NG reais TOP/MID. O conjunto
de desenvolvimento já foi usado para orientar a evolução
da v1 para v2, portanto o resultado de 100% **não comprova
generalização** nem autoriza liberação automática de peças.
Próxima etapa técnica recomendada: **teste cego e shadow
inference com novos casos reais independentes, incluindo
NG TOP/MID**, sem retreinar nos casos destinados ao teste.

---


## 08/10/2026 — CNN FALTANDO v1: treino real reprovado; evolução v2

**Resultado real da CNN v1 recebido do operador:**
`training_report.json` e `training_summary.txt` (gerados localmente
08/10/2026 às 15:40:35 UTC). Treino de 25 épocas em CPU, tamanho
160 × 160, batch 4, seed 42, holdout 20%, sem memória KNN.
A preparação original continha **117 frames**, correspondentes a
**67 eventos** (57 OK / 10 NG): 25 trincas OK SIDE/TOP/MID por
nome+OCR e 42 monoimagens históricas (32 OK/10 NG SIDE).
O split agrupado por board/parts e proximidade visual resultou em
54 eventos de treinamento (46 OK / 8 NG) e 13 de validação
(11 OK / 2 NG), em 45 grupos de separação.

**Matriz de confusão v1 na validação:**

| Real | Previsto OK | Previsto NG |
|---|---:|---:|
| OK | 11 | 0 |
| NG | **2** | **0** |

Acurácia 84,62%, recall NG **0/2 = 0%**, especificidade OK
11/11 = 100%. O valor de acurácia é enganoso por desbalanceamento,
porque **todas as 13 observações foram previstas como OK**. Os dois
NG não detectados ficaram no holdout:

- `2026-10-01_07-53-17-716_FALTANDO.png` (R475);
- `2026-10-01_09-53-18-089_FALTANDO.png` (R475).

A perda de treinamento caiu de **1,337919** (época 1)
para **0,000206** (época 25), enquanto a validação final teve
perda **0,637349**. Nas 25 épocas, o recall NG no holdout
permaneceu **0%**. Isto é consistente com sobreajuste e
ausência de generalização do detector NG, não prova de que
somente mais épocas resolveriam o problema. A v1 é uma
**linha de base reprovada**, não modelo para produção.

**Evolução v2 implementada, ainda não treinada com os 117 PNGs reais:**

- `src/core/neural/faltando_cnn_v2.py`: CNN comparativa
  de **nove canais** por escala (`RGB_gabarito`,
  `RGB_teste`, `|diferença|`) e duas escalas;
  imagem integral + zoom de 70% da região central,
  com mapa final espacial 2×2, encoder compartilhado,
  dropout e fusão do maior logit NG em SIDE/TOP/MID.
  A região central é hipótese operacional a verificar,
  **não** um localizador infalível de componente.
- `src/scripts/train_faltando_cnn_v2.py`: aceita as pastas
  e o `manifest.json` existentes, rótulos OK/NG
  confirmados pelo operador e mesmo agrupamento multilight
  por nome+OCR; revalida hashes e pares antes de treinar.
  Ajustes: sampler balanceado OK/NG, BCE binária,
  perda auxiliar por iluminação, pequenas augmentações
  sincronizadas, AdamW, weight decay e early stopping
  por perda em validação de desenvolvimento.
- **Diagnóstico por evento:** grava probabilidade de NG por
  evento e por iluminação, FP/FN, identificador do evento,
  matriz de confusão, curva por época, melhor época,
  seed e identificações exatas do split.
- **Arquivos locais:** `reports/faltando_neural/models/experiment_v2_*/`
  com `faltando_cnn_v2_candidate.pt`,
  `training_report_v2.json`,
  `holdout_predictions_v2.json`,
  `training_summary_v2.txt`.
- O split seed 42 permite comparar o mesmo conjunto de desenvolvimento
  com a v1, mas **esse conjunto já foi consultado e NÃO é teste cego**.
  Nenhuma conclusão de generalização pode ser tirada sem novos
  eventos NG independentes. Não há NG TOP/MID reais na base.
- O comando executa somente no Windows 10 novo. **Nunca** conecta
  a CNN à produção, roteador KNN, XP ou gate de startup;
  checkpoint possui `experimental=True` e
  `production_approved=False`. v1 preservada.

**Comando da próxima execução no computador do ODIN:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.train_faltando_cnn_v2 --epochs 25 --batch-size 4 --size 160 --device cpu
```

Depois enviar `training_report_v2.json`,
`holdout_predictions_v2.json` e `training_summary_v2.txt`
para comparação caso a caso (principalmente R475).
A execução precisa ocorrer na máquina com o staging real:
o GitHub só testou dados sintéticos. O modelo não pode
liberar NG automático sem testes cegos novos.

---


## 08/10/2026 — CNN FALTANDO: treinamento experimental com dataset confirmado

**Decisão do operador:** confiar nos rótulos locais de OK/NG e iniciar treinamento
sem exigir a conclusão do painel de qualificação visual. O painel permanece
opcional; esta autorização vale para **treino experimental**, não para
declaração automática de modelo aprovado em Produção.

**Fontes e limitação:** `reports/faltando_neural/run_*/manifest.json` já
existe no computador novo com 117 pares AOI extraídos, 107 OK e 10 NG SIDE
históricos. 25 trincas SIDE/TOP/MID OK reconhecidas por nome e OCR
board/parts/value coerentes; nenhum manifesto AOI com event_id verificável.
O treinamento pode agrupar essas trincas **provisoriamente** como
observações da mesma peça, nunca considera as 75 fotos como 75 eventos
independentes. O modelo não dispõe de NG TOP/MID para validar generalização.

**Novos módulos:**

- `src/core/neural/faltando_cnn.py`: CNN pequena e comparativa com
  extrator visual compartilhado (gabarito/teste) e fusão multilight
  conservadora por máximo logit NG entre as luzes disponíveis;
  aceita SIDE histórico sem TOP/MID. Não há consulta KNN.
- `src/scripts/train_faltando_cnn.py`: treinamento PyTorch offline,
  pad/resizing sem esticar a geometria, augmentação de pares sincronizada,
  BCE ponderada por desbalanceamento, objetivo auxiliar por iluminação,
  split holdout por componente e pares visualmente quase repetidos.
  Valida hashes SHA-256 das fontes, cria checkpoint local
  `faltando_cnn_candidate.pt`, `training_report.json` e
  `training_summary.txt` em `reports/faltando_neural/models/experiment_*/`.
  Sempre marca `experimental=True` e `production_approved=False`.
- `tests/test_faltando_cnn_training.py` e workflow Linux CPU:
  valida tensor CNN, split sem vazamento, 1 epoch sintético real,
  proteção de fontes e ausência de autorização à produção.

**Execução no PC novo (após git pull):**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -c "import torch; print(torch.__version__)"
python -m src.scripts.train_faltando_cnn --epochs 25 --batch-size 4 --size 160 --device cpu
```

**Segurança:** nenhum modelo é carregado no `main.py`; nenhum especialista
físico foi retirado ainda; os resultados do holdout e o risco de NG
classificado como OK precisam ser examinados antes de integração à Produção.
É possível começar o treino, mas não afirmar precisão operacional a partir
de dez NG SIDE. O gate de regressão também permanece independente.

---


## 08/10/2026 — Qualificação visual assistida CNN FALTANDO (offline)

**Escopo autorizado:** criar ferramenta local para revisar pares gabarito/teste
e as trincas SIDE/TOP/MID preparadas anteriormente, sem qualquer treinamento
ou integração à produção. A preparação real confirmou **117 pares extraídos**
(107 OK, 10 NG) e **25 trincas candidatas por nome**, nenhuma com manifesto
`event_id` comprovado; `training_ready=0`. A contagem de 17 trincas
anterior ignorava arquivos `_SIDE_2/_TOP_2/_MID_2` por causa do sufixo.

**Implementação:**

- `src/services/faltando_neural_qualification.py`: carrega
  `reports/faltando_neural/run_*/manifest.json`, valida caminhos/hash
  SHA-256 da origem antes de confirmar, persiste
  `qualification.json` atomicamente em staging com decisão explícita
  do operador (OK presente, NG ausente, recorte rejeitado), observação
  e divergência contra o rótulo do archive, sem tocar nas fontes.
  A qualificação nunca muda automaticamente `training_ready`.
- `src/ui/faltando_neural_review.py`: aplicativo PyQt6 independente
  do ODIN, com lista pesquisável, gabarito/teste completos responsivos,
  inspeção de três pares para trincas, rótulo original e observações.
  Exige confirmação visual explícita antes de salvar. O vínculo
  multilight humano exige que os três pares sejam revisados individualmente
  e convergentes; armazena `human_group_id`, não inventa `event_id`.
- dHash em **ambos** gabarito e teste sugere proximidade visual por
  iluminação sem remover/treinar com imagens ou reclassificar rótulos.
- `tests/test_faltando_neural_qualification.py`: fluxos de revisão,
  recarga, integridade da fonte, caminhos seguros, trincas, OCR e GUI
  sem AOI/XP. Workflow Windows de testes da ferramenta.

**Como abrir no computador do ODIN:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.ui.faltando_neural_review
```

Os botões da ferramenta só geram `qualification.json` dentro do run;
não treinam CNN, não alteram rótulos/hashes/PNG nem reconfiguram o ODIN.
Para avançar ao treino exigir aprovação explícita após revisão dos dados.

---


## Preparação neural FALTANDO — qualificação offline (08/10/2026)

**Status da etapa:** preparador implementado, sem treinar/classificar automaticamente.
Após contagem manual do operador em `public/ok_archive` e `public/ng_archive`:
117 screenshots `FALTANDO`: **107 OK** (56 SIDE legados, 17 SIDE,
17 TOP, 17 MID explícitos) e **10 NG SIDE legados**.
Nenhum arquivo possui hash SHA-256 igual a outro e nenhum hash aparece em
ambos os rótulos. Existem 17 trincas por nome, não 17 eventos comprovados.
Os dez NG podem ser visões de poucos defeitos independentes; TOP/MID NG não
possuem cobertura comprovada.

**Implementação desta etapa:**

- `src/services/faltando_neural_dataset.py`: usa o
  `ScreenMonitor.process_external_image` operacional, sem interface nem
  comandos XP, extrai gabarito e teste integrais de cada PNG, e gera pares
  derivados isolados em `reports/faltando_neural/run_*/pairs/`.
- `manifest.json` e `summary.txt`: hash e origem, rótulo provisório da
  pasta, iluminação, OCR observado, dimensão do par, status, possível
  `event_id` **somente via manifesto verificado** e pendências de
  qualificação humana. Sufixo e horário viram apenas candidatos a grupo;
  nunca são fundidos automaticamente em uma peça.
- `tests/test_faltando_neural_dataset.py`: segurança de diretório, integridade
  das fontes, formação de pares, falha de recorte e grupos multilight
  não comprovados. Workflow Windows específico.
- `.gitignore` exclui integralmente o staging derivado de `reports/faltando_neural/`.
- Documentação operacional: `docs/FALTANDO_CNN_DATASET_PREPARATION.md`.

**Executar no Windows 10 do ODIN:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.faltando_neural_dataset
```

**Não é treino e não habilita decisão neural:** `training_ready=False`
para todos até validar rótulos humanos e contexto das peças. Não carrega KNN,
não altera `public`, dataset, modelo, decisão ou startup gate. Próximas
etapas só mediante novo OK do operador, depois de revisar o relatório real.

---


## 08/10/2026 — Etapa 2: telemetria física do replay SIDE sem KNN

**Contexto:** o primeiro replay real do acervo histórico SIDE teve
**119** casos: **17 PASSOU** (17/17 NG), **101 REGRESSÃO**
(101/102 OK julgados como `DEFEITO REAL`) e **1 INVÁLIDO**
(OCR incompleto em `2026-10-02_1542_FALTANDO.png`).
Os 90 PNGs multilight novos continuam fora do replay legado.
Todos os registros declararam `memory_consulted=False` e
`knn_enabled=False`. **O gate bloqueante permanece DESATIVADO.**

**Objetivo desta correção:** diagnosticar por que o MoE físico decide NG
para praticamente todos os OK, **sem** alterar scores, pesos, thresholds,
rótulos, especialistas, dataset, CNN, fusão ou decisão de Produção.
Não presumir que os NG aprovados representem discriminação real até
confrontar evidências físicas dos OK e NG por categoria.

**Implementação isolada no replay:**

- `src/services/startup_regression/replay_telemetry.py`: normalização
  somente leitura dos campos reais de `detail.decision_trace`: por motor,
  `id`, `active`, `triggered`, `selected`, `raw_score`,
  `effective_score`, `threshold`, `final_influence` e `summary`;
  score/limiar final, motor dominante, regra de fusão, razão física,
  `physical_readings` disponíveis. Arrays de máscaras, imagens,
  assinaturas KNN e objetos não seriais não entram no relatório.
- `src/services/startup_regression/inspection_runner.py`: extrai o
  `build_lighting_context` **uma única vez**, entrega o mesmo contexto
  para a análise já existente e registra `geometry` com resolução do
  gabarito/teste completos, caixa global, candidatos da AOI, epicentros
  selecionados, anomalias brutas e caixas dos especialistas. Não executa
  uma segunda análise para explicar o veredito.
- `src/services/startup_regression/side_replay.py`: JSON inclui
  `cases[].telemetry` e `diagnostics` agregados por
  `expected_label + category`, regra da fusão, motor dominante e motores
  disparados em regressões. TXT apresenta as mesmas evidências de **cada**
  caso OK e NG, inclusive os aprovados, para comparação lado a lado.
- `tests/test_startup_regression_telemetry.py`: valida fidelidade da
  trilha, geometria, JSON sem NaN/array/KNN, agregações, casos NG/OK e
  não mutação dos resultados; incorporado ao workflow Windows dedicado.

**Invariantes:**
`_replay_without_memory=True`; especialista KNN não é criado/consultado;
peso KNN continua zero. Se o motor físico não informar um campo, o
relatório indica `N/D` (não inventa zero). Nenhum replay altera a
operação normal, grava PNGs ou autoriza inicialização bloqueante.

**Como reproduzir no PC real (branch `central` atualizada):**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

Enviar os novos `reports/startup_regression/side_replay_*.json` e
`side_replay_*.txt`. A análise seguinte será **diagnóstica**: comparar
escores e caixas físicos que disparam em OK versus NG, depois propor
correções com salvaguardas dos 17 NG. Não avançar à Etapa 3 até aceite.

---

## 08/10/2026 — C2 FALTANDO: revisão indevida apesar de testemunhas OK

**Evento AOI:** `17eee0c0e2b446208d2a12ea5b806ef5`, peça `C2`,
categoria `FALTANDO`, operador validou como `FALHA FALSA`.
ODIN retornou `REVISÃO OBRIGATÓRIA` sob
`multilight_missing_physical_disagreement`.

**Evidência por luz, extraída do debug real:**

- **SIDE:** `FALHA FALSA`; melhor KNN OK 0.9113768; corpo
  detectado `missing_component_body_present=True`, Dice 0.876121,
  coarse 0.722954, envelope de massa invariável preservada e
  `fusion_rule=hard_missing_invariant_presence_ok_witness`.
- **TOP:** `DEFEITO REAL`; `missing_hard_absence=True`,
  `missing_score=0.907439`, KNN OK 0.926029. A hipótese OK
  possui similaridade de **quadro completo** 0.956968,
  **contexto** 0.950198 e **epicentro** 0.904621.
- **MID:** `FALHA FALSA`; KNN OK 0.957204; especialista físico
  `missing_is_defect=False`, score 0.26867 (limite 0.36),
  envelope contextual preservado e sem exposição de fundo.

**Correção no núcleo:** nova rota
`multilight_missing_verified_presence` em
`src/core/multilight_fusion.py`, depois de detectar hard missing
isolado contradito por outras iluminações. Para liberar
`FALHA FALSA` exige **simultaneamente**:

1. Uma única luz acusa `missing_hard_absence`, nenhuma outra acusa
   defeito nem solicita revisão, e todas as memórias são OK da categoria
   `FALTANDO` e das respectivas iluminações.
2. Na luz discordante há memória OK 0.92+, margem 0.08+ e evidência
   de correspondência *multiescala* registrada: quadro completo 0.95+,
   contexto 0.94+ e epicentro 0.90+.
3. **Outra iluminação** comprova presença física de corpo com a rota
   `hard_missing_invariant_presence_ok_witness`, coarse 0.70+,
   Dice 0.84+, envelope de presença invariável, fundo exposto baixo
   e auditoria ROI consistente.
4. **Terceira iluminação distinta** apresenta `missing_is_defect=False`,
   score abaixo da tolerância, envelope contextual preservado e
   melhor memória OK 0.93+. Os dois vereditos não-NG precisam ter score
   final até 0.20.
5. Em falta de qualquer testemunha, permanece a fusão conservadora
   anterior (`REVISÃO OBRIGATÓRIA` ou `DEFEITO REAL`).
   Não liberar OK simplesmente por estar no dataset.

Ao confirmar a rota, o `raw_hard_missing_evidence` permanece
`True` para auditoria, mas `hard_missing_evidence` efetivo é
`False`, `operator_review_required=False`, o score NG
provisório da TOP não contamina o score final OK, e os três
papéis (`suspect_mode`, `body_mode`, `clear_mode`) são gravados
em `multilight_missing_presence_witnesses`.

**Testes:** regressão de telemetria real compactada de C2/17eee
e variações negativas removendo individualmente cada evidência
para garantir que o ODIN não libere OK sem contraprova.
Reproduções anteriores SIDE/NG real mantidas. Os arquivos AOI
brutos e o dataset local não são modificados. Status: correção
de código preparada; **aguardando replay real na AOI para validar o
veredito automático**. Nenhuma alteração do agente Windows XP.

---

## 08/10/2026 — evento C6~3 FALTANDO e Copiar imagens SIDE/TOP/MID

**Evento:** `691e95c58dab47ed95a233bc08c563b6`, `C6~3`, operador
identificou `FALHA FALSA`; ODIN classificou `DEFEITO REAL` (99%).

**Debug:** epicentro SIDE correto `[279,56,190,166]`. A fusão
`multilight_strong_single` foi decidida pela **SIDE**, que acusou
`missing_hard_absence=True` com score `0.97815`, KNN OK `0.91249`.
TOP foi OK com KNN OK `0.92595`, `missing_global_envelope_support=True`
e `missing_hard_absence=False`. MID foi OK com KNN OK `0.94149`,
`missing_is_defect=False`, `missing_score=0.12`.
O mecanismo anterior tratava apenas TOP forte isolada, de modo que a
contradição na SIDE era ignorada.

**Implementado:** `src/core/multilight_fusion.py` trata evidência
`missing_hard_absence` forte isolada de **SIDE, TOP ou MID** com três
memórias OK fortes e testemunha de estrutura contextual independente
de outra iluminação combinada a uma iluminação sem sinal físico local.
Quando essas condições de contradição ocorrem, o veredito é
`REVISÃO OBRIGATÓRIA` — nunca OK automático. Na ausência de contraprova
independente, o NG forte prevalece, inclusive quando só uma luz consegue
mostrar um defeito real. Não foram alterados pesos de treinamento nem
limiares físicos do especialista.

**Explicabilidade:** o debug e o cabeçalho passam a distinguir
`dominant_engine=multilight` de `multilight_dominant_mode=SIDE` e
`multilight_dominant_local_engine=missing`. No painel, o texto passa
de "Nenhum motor dominante" para "Fusão multilight • SIDE (origem:
Sinal de ausência)" e a barra de missing é chamada **Sinal de ausência**.
Os 98% são score de **divergência/ausência**, não 98% de presença física;
pesos locais da SIDE não são descritos como fórmula final multilight.

**Clipboard:** o botão único `Copiar imagens SIDE/TOP/MID` indica quando
existem os três frames completos do mesmo evento. O preview e a cópia
usam a mesma composição SIDE | TOP | MID, identificada por título e sem
perder resolução; a captura MSS/legada permanece monoimagem. Não há
fallback silencioso para frame isolado durante evento multilight parcial.

**Cobertura de teste:** cenário do evento SIDE NG/TOP+MID OK,
ausência real sem testemunha corroborada, motor dominante no painel e
clipboard + prévia multilight. O caso ainda requer validação operacional
real na AOI; a correção elimina NG automático indevido por contradição,
mas **não atesta ainda FALHA FALSA automática** sem uma prova de presença
independente dos sinais de iluminação. O agente XP permanece inalterado.

---

## Correções implementadas — epicentro profundo e contradição multilight FALTANDO (08/10/2026)

**Estado:** implementação técnica; aguarda teste da AOI e confirmação do operador.
Evento de regressão: `2eac69d849b04b5bb205ec191d9f130c`.

1. `EpicenterExtractor.select_radar_candidate`: quando existem três molduras
   (`[2,2,549,274]` global, `[24,24,522,230]` intermediária e
   `[76,56,189,166]` interna), descarta como foco qualquer caixa que
   contenha outra candidata claramente menor. O desempate de IoU contra
   o TESTE é aplicado **somente entre candidatas do nível mais profundo**.
   Mantém o Radar, o fallback por linhas cruzadas, a análise de contexto
   maior e os testes de ROI alta/estreita.
2. `src/core/multilight_fusion.py`: em `FALTANDO`, uma **única TOP**
   `missing_hard_absence=True` não produzirá automaticamente NG quando
   SIDE/MID foram negativas, **as três iluminações têm melhor memória OK
   forte (>=0.88)** e o envelope TOP mantém estrutura (coarse >=0.70,
   fundo global exposto <=0.03). Nessas condições a divergência é
   `REVISÃO OBRIGATÓRIA` (`multilight_missing_physical_disagreement`),
   **não é liberada como OK automaticamente**. A evidência TOP e o
   `missing_hard_absence` bruto continuam no debug. Esse tratamento
   impede a falsa confirmação automática com evidência contraditória,
   mas não substitui uma prova independente da presença física.
3. Uma TOP com ausência forte e sem a tripla contradição de memória/contexto
   continua `DEFEITO REAL`. `MUITO ADESIVO` e demais categorias
   preservam suas regras de fusão.

**Regressões novas:** molduras em três níveis com IoU favorecendo caixa
intermediária, conflito de ausência física TOP versus SIDE/MID e três
memórias OK, ausência real/sem suporte contextual e isolamento por categoria.

**Consistência do debug:** a fusão registra
`decision_trace.raw_hard_missing_evidence=True` para o sinal local TOP e
`decision_trace.hard_missing_evidence=False` quando a contradição
multilight coloca a peça em revisão. O payload final exibe ausência
**bruta** separada da **efetiva**, sem apagar a evidência TOP.

**Limitação explícita:** o operador informou que a peça está OK, mas esta
proteção inicialmente devolve `REVISÃO OBRIGATÓRIA`, não um OK automático.
Para automatizar OK de maneira segura será necessário reproduzir a decisão
física com imagens originais gabarito/teste (não somente o screenshot de debug),
validar presença do corpo em TOP e executar regressões NG reais. Não forçar
OK apenas pelo KNN, nem reduzir indiscriminadamente os limiares.

---

## Validação em AOI real e novo falso positivo — 08/10/2026

### Caso anterior C6~2 / DESLOCADO — confirmação do operador

**Situação: VALIDADO NA FÁBRICA para a captura anteriormente reportada.**
O operador confirmou que a revisão da hierarquia de molduras na branch
`central`, commit `92555f1`, resolveu a escolha incorreta do quadrado
externo naquele teste. Preservar essa regressão: quando o recorte apresenta
a moldura global `[25,25,525,230]` e a interna `[76,56,189,166]`,
o epicentro correto é a **interna**. A área global e a imagem inteira
continuam disponíveis como contexto para análise física e memória.

Esta validação confirma o caso específico testado na AOI, não todos os
arranjos de molduras; o caso novo abaixo mostra uma variante ainda não
coberta.

### Novo caso C5~2 (OCR aproximado) / FALTANDO — falso positivo

**Situação: DIAGNOSTICADO; correção ainda NÃO implementada.**
**Rótulo humano:** OK / `FALHA FALSA`. **Saída ODIN:**
`DEFEITO REAL` com confiança de 99% e score 100%.
Evidência: screenshot multilight SIDE/TOP/MID e debug
`Texto colado(20261008-120524).txt`, evento
`2eac69d849b04b5bb205ec191d9f130c`, horário
`2026-10-08T08:04:05.625`, categoria `FALTANDO`.
O OCR do componente no debug aparece como `CS~2`; conferir o
identificador na AOI antes de usar como chave de regressão.

**Evidência visual:** a captura mostra corpo do componente e terminais
metálicos em gabarito e teste (especialmente SIDE/TOP); existem diferenças
de luminosidade, contraste e posição, mas o componente **não está faltando**,
conforme julgamento do operador. Na MID há saturação/clipping claro, não
evidência independente de ausência.

**Saídas por iluminação:**

| Iluminação | Veredito local | Score final | Física | KNN |
|---|---|---:|---:|---|
| SIDE | `FALHA FALSA` | 0% | 88% | Melhor OK 92,34% |
| TOP | `DEFEITO REAL` | 100% | 100% | Melhor OK 90,75%, suprimido pelo hard missing |
| MID | `FALHA FALSA` | 13,16% | 85% | Melhor OK 89,76% |

A fusão `multilight_strong_single` tratou **uma única TOP positiva
forte** como suficiente para `DEFEITO REAL`, sem confirmação de SIDE/MID.
O erro nasce na evidência física da TOP; não resolver diminuindo
indiscriminadamente o peso da fusão nem permitindo que o KNN anule
qualquer ausência física real.

**Indícios técnicos da TOP que precisam ser investigados:**

- O `missing_expert` gerou `missing_hard_absence=True`,
  `missing_score≈1.0`, `missing_changed_coverage=0.6616`,
  `missing_background_exposure=0.4967`,
  `missing_structure_loss=0.8736` e
  `missing_body_presence_veto=False`, afirmando que o componente
  foi substituído pelo fundo, apesar da evidência visual de presença.
- O alinhamento visual tentou corrigir `dx=12 px, dy=9 px`,
  `alignment_score≈0.673` e ganho `≈0.178`;
  na ROI pequena uma variação de iluminação/posição pode estar
  sendo confundida com perda física. Esta explicação é **hipótese
  técnica**, não causa definitivamente comprovada por replay.
- O corpo preservado não passou nos critérios geométricos:
  `coarse_similarity≈0.477`, `silhouette_dice≈0.583`,
  `area_ratio≈1.688`, `centroid_shift≈0.151`.
  O envelope global também não confirmou presença
  (`coarse_similarity≈0.773`, perfis horizontais/verticais
  `≈0.809/0.801`, `invariant_support=False`).
- `dual_scale_triggered=False` porque o motor declarou
  `escala local já confirmou ausência física`; o segundo exame
  não pôde contradizer o hard missing local. A trilha TOP registra
  `roi_consistent=False`; verificar se a diferença é apenas
  transformação de alinhamento do SSIM ou perda de identidade do
  recorte entre especialistas.
- O melhor exemplo de memória TOP estava rotulado OK
  (`similarity=0.907532`), mas foi usado só para auditoria,
  pela regra que dá precedência à ausência física forte.

**Outra variante do problema do epicentro na entrada SIDE:**

O debug de validação do mesmo evento informa
`global_box_info=[2,2,549,274]`,
`focus_box=[24,24,522,230]`,
`candidate_selected_by_radar=[24,24,522,230]`
e `selection_rule=inner_frame_confirmed_by_test`, enquanto os
contornos revelam também a ROI realmente interna
`[76,56,189,166]`, com a correspondente no TESTE
`[79,57,188,166]`.
A detecção da moldura ainda mais externa `[2,2,...]`
faz `[24,24,522,230]` parecer interna; o desempate por IoU
com `old_epicenters` pode premiar a moldura intermediária,
em vez do **menor retângulo independente**. Portanto a
correção C6~2 foi validada, mas a seleção de **três níveis de
moldura** permanece problemática. Esse foco equivocado ocorreu
na validação/SIDE; a TOP efetivamente analisou a ROI
`[76,56,189,166]`. Não atribuir automaticamente o hard missing
da TOP ao problema da SIDE.

**Critérios para futura correção, pendente de autorização:**

1. Reproduzir os dois contextos com os frames reais e criar
   regressões independentes: molduras 2 níveis e 3 níveis,
   `FALTANDO` presente sob TOP com variação visual, e
   `FALTANDO` verdadeiramente ausente.
2. Distinguir moldura global/contextual, moldura intermediária
   e ROI de epicentro com evidência geométrica robusta;
   não assumir que a caixa mais central nem o melhor IoU
   sempre define a ROI. Preservar leitura da imagem inteira.
3. Auditar o alinhamento e os sinais de presença física da TOP,
   usando estrutura/corpo/pads do componente e confirmação
   contextual independentemente da variação de iluminação.
   Ausência física forte só deve superar memória OK quando
   houver evidência física suficientemente confiável; incerteza
   deve conservar `REVISÃO OBRIGATÓRIA` conforme segurança do
   processo, em vez de forçar OK.
4. Verificar o `roi_audit` dos especialistas e o tempo real
   de processamento (ciclo informado: `26.143 s`).
5. Preservar as regressões já aprovadas e casos NG reais,
   antes de alterar a lógica de julgamento. Não mexer em
   rótulos do dataset ou no agente XP como atalho.

**Escopo desta atualização:** somente documentação e diagnóstico.
Não há correção de código nem resultado validado para este novo
falso positivo.

---




## Correção 08/10/2026 — moldura global confundida com epicentro (C6~2)

**Ocorrência:** inspeção `DESLOCADO`, 3 iluminações SIDE/TOP/MID.
Após correção da captura cortada, o radar aceitava a moldura maior como
foco, embora o contorno interno fosse detectado. O debug comprovou:

- Global: `[25,25,525,230]`;
- Foco incorreto: `[25,25,520,228]`;
- Epicentro interno real nos contornos do gabarito:
  `[76,56,189,166]` (corresponde ao candidato legado TESTE `[78,57,189,166]`);
- `missing_local_global_area_ratio=0.9819`, desativando a verificação
  contextual `Dual-Scale Presence` por foco quase igual à moldura global.

**Causa:** o Radar Euclidiano selecionava apenas por distância até o centro.
O quadro global possuía largura >85% e altura <85% do recorte, passando no
filtro de descarte e vencendo a caixa interna deslocada para a esquerda.

**Correção localizada:** `EpicenterExtractor.select_radar_candidate` passou
a reconhecer `global_box_info.detected` e priorizar contornos internos com
área distinta e inclusão geométrica no quadro externo. Quando possível,
usa concordância da ROI entre TESTE e GABARITO para desempate, sem tratar a
moldura global como candidata. Sem relação hierárquica confiável, mantém
recuperação por linhas cruzadas, fallback legado e a rejeição de telas sem
epicentro. O painel de debug chama exatamente a mesma seleção, exibindo
`candidate_selected_by_radar` e `selection_rule` coerentes com o motor.

A análise do quadro completo continua ativa, mas o foco local recebe apenas
o retângulo menor. Sem alteração de OCR, KNN, categorias, fusão multilight,
pipeline dos especialistas ou agente XP.

**Regressão:** recorte `570×276` reproduzindo a geometria do debug,
moldura global isolada e integração de validação de entrada da AOI.

---



## Correção 08/10/2026 — epicentro oculto por molduras verdes cruzadas

**Escopo:** validação AOI/Windows XP, sem alteração do agente XP ou de decisões OK/NG.

**Caso real:** componente D6~2, categoria `DESLOCADO`, iluminação `SIDE`.
O gabarito e o teste mostram duas molduras verdes muito próximas, que se
cruzam na parte superior e são cortadas na borda inferior. Os contornos
tradicionais se fundiam; o Radar de `EpicenterExtractor` via uma caixa gigante
(378×524 no recorte 394×540), a descartava por largura e altura superiores
a 85%, e os fragmentos restantes não passavam pelo mínimo de 15 px.
A validação encerrava como `missing_epicenter`, apesar do epicentro visível.

**Correção:** `src/core/epicenter_line_recovery.py` complementa o Radar somente
quando ele não seleciona nenhuma caixa. Usa aberturas morfológicas direcionais
para identificar laterais verticais e topos horizontais, exige duas molduras
independentes e seleciona a caixa de laterais internas; permite que o topo da
caixa menor esteja acima do topo externo e que as bordas inferiores saiam da
imagem. Se existirem pequenas interrupções, faz uma segunda passagem de
reconexão limitada. Sem provas de duas molduras, não gera epicentro fictício.

A escolha original do Radar e o fallback legado permanecem preservados.
As ROIs de análise continuam coexistindo com contexto do componente e
quadro completo; a correção não altera fusão, KNN, OCR, dataset ou replay
de inicialização planejado.

**Regressões:** foram acrescentados cenários de caixas cruzadas/cortadas,
linhas interrompidas, moldura isolada, linha espúria e integração com
`validate_network_inspection`. O agente no Windows XP não precisa ser atualizado.

---


## Planejado — 08/10/2026 — gate de regressão visual antes da inicialização

**Status em 08/10/2026:** Etapa 1 concluída e aprovada (209 PNG);
    Etapa 2 replay SIDE sem KNN implementada e aguardando diagnóstico real;
    gate operacional bloqueante ainda NÃO implementado.

Documento técnico completo (fonte única deste plano):
[`docs/ODIN_STARTUP_REGRESSION_GATE.md`](docs/ODIN_STARTUP_REGRESSION_GATE.md).

**Objetivo:** a cada inicialização do ODIN, reprocessar integralmente os casos
confirmados pelo operador em `public/ok_archive/` e `public/ng_archive/`,
utilizando o pipeline real de inspeção, OCR, especialistas físicos e fusão,
mas **sem consultar KNN, protótipos ou qualquer memória episódica**.
O painel operacional só poderá abrir depois da validação **100% aprovada**.
Regressão, revisão obrigatória, arquivo inválido, conflito ou evento incompleto
bloquearão a operação e produzirão relatório técnico, sem mandar comandos à AOI.

**Regra de avaliação por geração de dados:**

- PNGs históricos sem iluminação identificada: `SIDE` monoimagem;
  `ok_archive` exige `FALHA FALSA`; `ng_archive` exige
  `DEFEITO REAL`, sem `REVISÃO OBRIGATÓRIA`.
- Novos eventos SIDE/TOP/MID: três análises locais independentes e **um
  veredito final da fusão**. As três iluminações não precisam ter o mesmo
  resultado local; o resultado final precisa corresponder ao rótulo OK/NG.
- Novo manifesto associará `event_id`, OCR, rótulo humano e arquivos
  SIDE/TOP/MID, sem agrupar peças por coincidência de horário/nome.
- O replay preservará o **gabarito/teste completos**. O quadrado menor e a
  caixa contextual não limitam a evidência global; a imagem completa pode
  ser usada pelos motores físicos, sem consulta à memória `full_frame`.
- O gate será **somente leitura**: não retreina KNN/CNN, não altera rótulos,
  não limpa o dataset, não arquiva novas imagens e não aciona o Windows XP.
- O relatório distinguirá `PASSOU`, `REGRESSÃO`, `INVÁLIDO`,
  `INCOMPLETO`, `CONFLITO` e `SEM COBERTURA`.

**Ordem aprovada para implementação futura, em etapas independentes:**

1. Inventariar e qualificar os arquivos locais existentes (SIDE e multilight).
2. Reutilizar o pipeline real em replay monoimagem SIDE, sem Qt produtivo.
3. Criar manifesto e replay de eventos multilight com fusão final.
4. Instalar gate bloqueante e tela restrita de progresso/diagnóstico.
5. Adicionar regressões automatizadas e validar o acervo real na fábrica.

**Etapa 1 implementada (sem alterar o motor operacional):**

- `src/services/startup_regression/archive_inventory.py`: inspeção
  somente leitura dos PNGs, dimensões, hashes, duplicatas, categorias sugeridas
  pelo nome, iluminação SIDE/TOP/MID e manifestos reais.
- `src/services/startup_regression/archive_inventory_report.py`: relatório
  JSON integral e resumo TXT, gravados fora de archives/dataset.
- `src/services/startup_regression/__main__.py`: CLI de execução manual.
- `tests/test_startup_regression_inventory.py`: testes com casos sintéticos,
  arquivos antigos, manifestos, conflitos e PNG inválido.
- `.github/workflows/startup-regression-inventory.yml`: validação no Windows.

Execução no computador real, com o código atualizado:

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression
```

Arquivos produzidos (não subir no Git):

```text
reports/startup_regression/inventory_<data>.json
reports/startup_regression/inventory_<data>.txt
```

O inventário **não** executa OCR, especialistas, MoE, KNN ou valida OK/NG.
O usuário aprovou a Etapa 1 após o inventário real de 08/10/2026:
**209 PNG válidos** (192 OK, 17 NG), incluindo **119 SIDE históricos**
e **90 imagens multilight sem manifesto**. Nenhuma duplicação pixel a pixel.

**Etapa 2 implementada (aguardando resultado real):** replay físico SIDE sem
KNN/memória. A ordem do replay é screenshot → barras azul/vermelha →
gabarito/teste completos → OCR → `detect_anomalies` →
`EpicenterExtractor` → especialistas físicos/semânticos →
fusão física, sem `memory_veto`, sem similaridade de vizinhos e sem
usar os próprios arquivos arquivados como memória.

```text
src/services/startup_regression/inspection_runner.py
src/services/startup_regression/side_replay.py
tests/test_startup_regression_replay.py
.github/workflows/startup-regression-side-replay.yml
```

Comando no PC da fábrica:

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

Relatórios: `reports/startup_regression/side_replay_*.json` e
`side_replay_*.txt`. O KNN não é carregado pelo executor e a operação
normal mantém a memória intacta. Não há treinamento, alteração do dataset,
escrita de PNGs em `public/debug_crop`, comunicação XP ou bloqueio
de inicialização nesta etapa. A extensão de `INVERTIDO` também não
pode consultar memória. Não avançar à Etapa 3 antes da revisão do
relatório real e aprovação do usuário.

**Regra de execução:** trabalhar em **uma etapa por vez**, avançando somente
após aceite expresso do operador.

---

## Atualização operacional — 07/10/2026 — multilight geral para todas as categorias AOI

O padrão multilight que foi validado primeiro em `MUITO ADESIVO` passa a ser o
**contrato geral de inspeção para todo ciclo recebido do Windows XP com categoria
AOI válida**. Referências posteriores neste documento que descrevem o recurso
como “exclusivo do adesivo” devem ser lidas como histórico da implantação; o
escopo atual é geral.

Contrato atual:

```text
mesma peça / mesmo event_id
        ↓
SIDE recebido + análise completa da categoria
        ↓
TOP automático + análise completa da mesma categoria
        ↓
MID automático + análise completa da mesma categoria
        ↓
restaurar SIDE
        ↓
fusão final única das três iluminações
        ↓
um único julgamento da peça
```

Regras obrigatórias:

- SIDE/TOP/MID nunca viram três peças ou três `event_id`;
- cada iluminação executa o mesmo pipeline técnico e os mesmos especialistas
  roteados pela categoria AOI;
- TOP/MID permanecem não elegíveis para decisão enquanto isolados;
- `MUITO ADESIVO` continua usando `fuse_adhesive_multilight` e o detector MID
  `mid_bright_resin_v1`, sem regressão da política já validada;
- as demais categorias usam a fusão geral de
  `src/core/multilight_fusion.py`: evidência forte isolada ou duas iluminações
  positivas confirmam defeito; uma única positiva moderada ou revisão local
  exige `REVISÃO OBRIGATÓRIA`; somente três análises sem defeito resultam em
  `FALHA FALSA`;
- `Copiar debug` registra SIDE/TOP/MID e a fusão final para qualquer categoria;
- `Copiar imagem` usa a composição SIDE/TOP/MID sem redimensionar os frames;
- arquivos visuais OK/NG persistem SIDE, TOP e MID separadamente com o sufixo da
  iluminação;
- a captura local MSS permanece monoimagem, pois não possui a máquina de estados
  que comanda as iluminações da AOI;
- nenhuma mudança no `agente_industrial_xp.py` é necessária: os comandos
  LEFT/DOWN/RIGHT já existentes são reutilizados.

### Implementação concluída — generalização do padrão de adesivo

A implementação foi concluída na branch `central` usando como base o fluxo que
já estava validado para `MUITO ADESIVO`. Em vez de criar um segundo mecanismo
de aquisição, a máquina de estados existente foi reaproveitada e generalizada.

O ciclo de uma peça recebida pela rede passa a ser:

```text
1. AOI envia a imagem SIDE
2. ODIN valida a captura e identifica a categoria
3. ODIN executa a análise completa da SIDE
4. ODIN envia PRESS_LEFT ao XP
5. AOI muda para TOP
6. XP envia a imagem TOP da mesma peça
7. ODIN executa a análise completa da TOP
8. ODIN envia PRESS_RIGHT ao XP
9. AOI muda para MID
10. XP envia a imagem MID da mesma peça
11. ODIN executa a análise completa da MID
12. ODIN envia PRESS_DOWN ao XP
13. AOI volta para SIDE
14. ODIN funde SIDE + TOP + MID
15. somente a fusão recebe autoridade de julgamento final
```

As três imagens compartilham obrigatoriamente:

```text
mesma peça
mesmo event_id
mesma categoria AOI
mesmo ciclo operacional
```

TOP e MID são frames auxiliares do ciclo ativo. Eles **não podem** abrir uma nova
peça, liberar o gate principal ou produzir uma decisão autônoma isolada.

#### Análise por iluminação

Cada iluminação usa o mesmo pipeline técnico da categoria:

```text
imagem
  ↓
detect_anomalies
  ↓
EpicenterExtractor
  ↓
MoEOrchestrator.inspect
  ↓
especialistas roteados pela categoria
  ↓
análise local SIDE / TOP / MID
```

Cada resultado local permanece disponível para auditoria, mas contém:

```text
eligible_for_final_decision = False
```

enquanto ainda for somente uma análise isolada.

Somente o objeto produzido pela fusão das três iluminações recebe:

```text
lighting_mode = MULTILIGHT
multilight_final = True
eligible_for_final_decision = True
```

#### Política de fusão geral

Foi criado:

```text
src/core/multilight_fusion.py
```

Esse módulo é responsável por decidir qual política de fusão deve ser utilizada
sem misturar regras específicas entre categorias.

Para categorias diferentes de `MUITO ADESIVO`:

```text
1 evidência local forte
        → DEFEITO REAL

2 ou mais iluminações positivas
        → DEFEITO REAL

1 única iluminação positiva moderada
        → REVISÃO OBRIGATÓRIA

nenhuma positiva, mas alguma análise pediu revisão
        → REVISÃO OBRIGATÓRIA

SIDE + TOP + MID sem defeito
        → FALHA FALSA
```

A decisão final registra também:

- iluminação dominante;
- iluminações positivas;
- iluminações com evidência forte;
- iluminações que exigiram revisão;
- score final;
- score físico;
- regra de fusão aplicada;
- telemetria local das três análises.

A memória/KNN continua pertencendo à análise individual de cada iluminação e não
é transformada em uma memória compartilhada entre SIDE/TOP/MID.

#### Exceção preservada — MUITO ADESIVO

`MUITO ADESIVO` **não foi migrado para a política genérica**.

A categoria continua delegando para:

```text
src/core/adhesive_multilight_fusion.py
fuse_adhesive_multilight(...)
```

e mantém:

- regras físicas já validadas para excesso de adesivo;
- telemetria `adhesive_multilight_*`;
- perfil MID `mid_bright_resin_v1`;
- testemunha clara/creme da iluminação MID;
- comportamento anteriormente validado em produção.

`src/core/multilight_fusion.py` apenas detecta que a categoria é
`MUITO ADESIVO` e delega a decisão para a fusão especializada.

#### Interface multilight geral

A visualização que antes era exclusiva de adesivo passou a ser utilizada durante
qualquer ciclo multilight recebido da AOI.

A tela apresenta, para a mesma peça:

```text
SIDE
  - imagem recebida
  - caixa maior
  - epicentro

TOP
  - imagem recebida
  - caixa maior
  - epicentro

MID
  - imagem recebida
  - caixa maior
  - epicentro
```

O painel de especialistas também exibe as três análises em paralelo.

Os textos visuais foram generalizados de `ADESIVO` para `MULTILIGHT`, sem
remover os nomes internos legados `adhesive_multilight_*` quando eles são
necessários para compatibilidade com módulos e testes antigos.

Durante a coleta:

```text
MULTILIGHT • AGUARDANDO TOP/MID
```

é mostrado no lugar de um veredito prematuro da SIDE.

#### Copiar debug

`Copiar debug` passou a anexar ao relatório técnico, para qualquer categoria
multilight:

```text
categoria
event_id

ILUMINAÇÃO SIDE
  análise local completa

ILUMINAÇÃO TOP
  análise local completa

ILUMINAÇÃO MID
  análise local completa

JULGAMENTO FINAL MULTILIGHT
  veredito
  score
  score físico
  regra de fusão
  iluminação dominante
  iluminações positivas
  iluminações fortes
  iluminações em revisão
  papel da memória
```

O relatório só reutiliza uma sessão multilight quando o `event_id` coincide com
o evento atual, impedindo que imagens ou análises da peça anterior apareçam no
debug da peça seguinte.

#### Copiar imagem

`Copiar imagem` usa os três frames originais:

```text
SIDE | TOP | MID
```

em uma única imagem composta.

Os frames não são redimensionados nem sobrepostos. O compositor apenas cria uma
área comum e posiciona as três imagens lado a lado, preservando a evidência
original de cada iluminação.

Se SIDE/TOP/MID não pertencem ao mesmo `event_id` ou o conjunto não está
completo, a composição multilight não é considerada válida.

#### Arquivo visual OK/NG

O resolvedor compartilhado:

```text
src/services/image_archive_candidates.py
```

também foi generalizado.

Quando existe uma sessão multilight completa para o evento atual, um único
julgamento OK ou NG pode persistir três arquivos independentes:

```text
<CATEGORIA>_SIDE
<CATEGORIA>_TOP
<CATEGORIA>_MID
```

Exemplo para `FALTANDO`:

```text
..._FALTANDO_SIDE.png
..._FALTANDO_TOP.png
..._FALTANDO_MID.png
```

Isso vale para:

```text
public/ng_archive/
public/ok_archive/
```

A deduplicação visual por conteúdo continua ativa.

Um conjunto multilight incompleto nunca é arquivado parcialmente como se fosse
um conjunto completo. Nessa situação, o resolvedor retorna para a evidência
monoimagem disponível do ciclo.

#### Dataset e memória de aprendizado multilight

O multilight também passa a enriquecer o **Active Learning/KNN**, e não somente
os arquivos visuais OK/NG.

Um ciclo completo continua representando **uma única peça e um único julgamento
humano**, porém a persistência do aprendizado pode produzir três memórias
relacionadas:

```text
mesmo event_id
mesmo OCR
mesmo rótulo humano OK ou NG

├── memória SIDE
├── memória TOP
└── memória MID
```

Cada registro recebe explicitamente:

```text
lighting_mode = SIDE | TOP | MID

Board
Parts
Category
Value

event_id
análise local da iluminação
resumo da fusão multilight final
rótulo humano da peça
```

Portanto TOP e MID **não viram peças independentes**. São observações adicionais
da mesma anomalia, sob condições de iluminação diferentes.

A fila de persistência em
`src/services/decision_persistence.py` grava as três observações e recarrega a
memória KNN somente uma vez ao final do conjunto.

Decisões automáticas de Produção continuam sem ensinar a IA a partir da própria
resposta. O Active Learning permanece baseado em confirmação humana, evitando
feedback autorreferente.

##### Proteção contra epicentro/quadrado menor incorreto

O dataset não depende somente do menor retângulo detectado.

Uma situação real possível é:

```text
quadrado menor
  → está em uma região local

defeito verdadeiro
  → está evidente em outra região
  → ou só fica evidente olhando a imagem completa
```

Para evitar perda dessa informação, a memória passa a trabalhar com três
escalas quando os dados novos estão disponíveis:

```text
1. EPICENTRO
   detalhe local / quadrado menor

2. CONTEXTO DO COMPONENTE
   caixa maior / região estrutural

3. QUADRO COMPLETO
   toda a área de inspeção gabarito x teste
```

A terceira escala é implementada em:

```text
src/core/full_frame_memory.py
```

Ela é independente da posição do epicentro e da caixa maior. Assim, uma memória
pode continuar carregando evidência global mesmo quando a proposta local estiver
deslocada da região onde o defeito é visualmente mais explícito.

Quando as três escalas existem, a comparação preserva a política anterior como
base e acrescenta o quadro completo como evidência global. JSONs antigos
continuam compatíveis: quando não possuem a terceira escala, o comparador volta
para a política anterior sem invalidá-los.

O frame bruto recebido da AOI também pode ser salvo como `*_source.png` para
auditoria. Ele não substitui o par técnico gabarito/teste usado pela memória.

##### Isolamento por iluminação

A memória KNN passa a consultar:

```text
mesma categoria
+
mesma iluminação
```

Exemplo:

```text
consulta FALTANDO / TOP
  → memória FALTANDO / TOP

não consulta SIDE como se fosse TOP
não consulta MID como se fosse TOP
```

Isso é necessário porque a aparência física da mesma anomalia muda
substancialmente entre SIDE, TOP e MID.

Registros antigos sem `lighting_mode` são tratados como:

```text
SIDE
```

porque todo o dataset histórico anterior à generalização multilight foi formado
a partir da iluminação padrão SIDE.

##### Deduplicação do dataset

A deduplicação do Active Learning é separada da deduplicação dos arquivos
visuais OK/NG.

A chave de identidade do aprendizado considera:

```text
rótulo OK/NG
categoria
Board
Parts
iluminação
conteúdo visual exato
```

O conteúdo visual recebe fingerprint SHA-256 determinístico.

Consequência para o dataset histórico:

```text
SIDE antigo idêntico já existe
  → não duplica a imagem SIDE

TOP novo
  → salva memória TOP

MID novo
  → salva memória MID
```

Mesmo que duas iluminações produzam pixels idênticos por alguma condição
anormal, suas identidades não são fundidas: SIDE/TOP/MID permanecem escopos
diferentes.

Para OK, uma duplicata exata pode enriquecer/reutilizar o registro/protótipo
existente.

Para NG, permanece a regra de segurança já existente:

```text
cada observação NG = memória JSON protegida individual
```

Porém a imagem pesada repetida não é escrita novamente. O novo JSON aponta que
o conteúdo visual já existe, preservando a observação NG sem desperdiçar disco.

##### REVISÃO OBRIGATÓRIA

`REVISÃO OBRIGATÓRIA` continua não sendo uma terceira classe do dataset.

O fluxo é:

```text
ODIN → REVISÃO OBRIGATÓRIA
        ↓
operador decide
        ↓
0 = OK ou 1 = NG
        ↓
SIDE/TOP/MID recebem o mesmo rótulo humano final
```

Os JSONs preservam que a fusão havia solicitado revisão, mas a classe usada para
aprendizado continua sendo a verdade humana OK/NG.

#### Captura local MSS

A generalização é aplicada ao fluxo que vem da AOI pela rede.

A captura local MSS permanece:

```text
1 captura
1 análise
1 evidência
```

porque o MSS não possui a máquina de estados que solicita ao Windows XP/AOI a
troca física entre SIDE, TOP e MID.

Portanto não existe espera artificial por TOP/MID em uma captura local.

#### Agente Windows XP

Nenhuma alteração adicional foi necessária no agente XP.

A implementação reutiliza os comandos já existentes:

```text
PRESS_LEFT  → TOP
PRESS_DOWN  → SIDE
PRESS_RIGHT → MID
```

e o modo auxiliar já usado pelo adesivo permite que TOP/MID atravessem o receptor
sem abrirem um novo ciclo principal.

#### Arquivos centrais envolvidos

A generalização alterou ou passou a depender diretamente de:

```text
src/core/multilight_fusion.py
src/core/adhesive_multilight_fusion.py
src/core/adhesive_multilight_analysis.py
src/core/full_frame_memory.py
src/core/dual_scale_memory.py
src/core/strict_category_memory.py
src/core/prototype_memory.py
src/core/experts/knn_expert.py

src/ui/adhesive_multilight_automation.py
src/ui/adhesive_multilight_inspection.py
src/ui/adhesive_multilight_analysis.py
src/ui/network_xp_debug.py
src/ui/control_panel.py
src/ui/control_panel_ui.py

src/services/image_archive_candidates.py
src/services/ng_image_archive.py
src/services/ok_image_archive.py
src/services/anomaly_learning.py
src/services/decision_persistence.py
src/services/dataset_manager.py
```

Os nomes `adhesive_multilight_*` foram mantidos onde necessário para evitar uma
renomeação transversal de alto risco. O comportamento, porém, passou a ser geral.

#### Testes e validação

Foi criado o workflow:

```text
.github/workflows/multilight-generalization-tests.yml
```

e o teste dedicado:

```text
tests/test_multilight_generalization.py
tests/test_multilight_learning.py
```

A suíte específica valida:

- máquina SIDE → TOP → MID → SIDE;
- TOP/MID como frames auxiliares;
- uma única decisão final;
- fusão geral para categoria não adesiva;
- defeito forte em uma iluminação;
- defeito corroborado por duas iluminações;
- revisão para evidência isolada moderada;
- FALHA FALSA quando as três iluminações estão limpas;
- preservação da política especializada de `MUITO ADESIVO`;
- três imagens independentes no arquivo visual;
- bloqueio de conjunto multilight incompleto;
- debug com SIDE/TOP/MID;
- composição de `Copiar imagem`;
- captura de SIDE/TOP/MID completos para Active Learning;
- mesmo OCR/event_id/rótulo humano nas três persistências;
- isolamento KNN por iluminação;
- compatibilidade de memórias antigas como SIDE;
- terceira escala de quadro completo;
- defeito fora do quadrado menor preservado na assinatura global;
- deduplicação do SIDE legado;
- independência de TOP/MID;
- NG protegido em JSON sem duplicação da imagem pesada.

Resultado da validação específica em 07/10/2026:

```text
General Multilight Tests
50 testes executados
resultado: SUCCESS
```

O workflow amplo `Network XP Debug Tests` possui contratos antigos que já
estavam incompatíveis com mudanças anteriores do projeto — principalmente UI
responsiva, gate de imagens, arquivo OK, Produção e uma expectativa antiga de
`FALTANDO`. Essas falhas foram separadas da validação desta implementação para
não confundir regressões preexistentes com a generalização multilight.

### Estado final desta atualização

```text
ADESIVO
  → continua multilight especializado

FALTANDO
DESLOCADO
EMBORCADO
INVERTIDO
e demais categorias AOI válidas
  → passam a usar multilight geral

SIDE/TOP/MID
  → mesma peça
  → mesmo event_id
  → três análises independentes
  → uma fusão final
  → um único julgamento
```

A partir desta atualização, o padrão arquitetural do ODIN para imagens recebidas
da AOI é **multilight por padrão**, e não mais uma exceção exclusiva de adesivo.


**Módulos Existentes:**
- `src/core/multilight_fusion.py`: Fusão final SIDE/TOP/MID para categorias AOI gerais, delegando `MUITO ADESIVO` à política especializada existente.
- `src/core/full_frame_memory.py`: Terceira escala da memória; compara o quadro completo para preservar evidências fora do epicentro/contexto local.
- `src/services/anomaly_learning.py`: Captura o snapshot humano e, em ciclo multilight completo, preserva SIDE/TOP/MID antes da limpeza da interface.
- `src/services/decision_persistence.py`: Persiste as três observações multilight sob um único julgamento humano e recarrega o KNN uma vez ao final.
- `src/services/dataset_manager.py`: Persistência `visionx.memory.v3`, identificação de iluminação, OCR/event_id compartilhados e deduplicação visual do dataset.
- `src/config/settings.py`: Centralização de todas as variáveis de ambiente, caminhos e constantes mágicas.
- `src/services/ng_image_archive.py`: Arquivo visual opcional de decisões finais NG em fila de background, independente do dataset e da memória KNN.
- `src/services/ok_image_archive.py`: Arquivo visual opcional de decisões humanas OK em fila de background, usando a mesma evidência de `Copiar imagem`.
- `src/services/image_archive_naming.py`: Formato compartilhado de nomes dos arquivos visuais OK/NG.
- `src/services/production_daily_session_store.py`: Persistência diária e atômica das métricas do Modo Produção em diretório de dados do usuário, fora do repositório.
- `src/services/aoi_ocr_fields.py`: Normalização específica de Board/Parts/Value e releitura dirigida do campo de componente quando o OCR geral é inconsistente.

**Fluxos Principais (Planejados):**
1. **Pilar 1 (Extrator Visual):** Monitoramento contínuo da tela usando `mss` para detectar a janela da AOI.
2. **Pilar 2 (Cérebro Comparativo):** Rede siamesa avaliando propostas de defeitos.
3. **Pilar 3 (Display HUD):** Janela transparente sobreposta sinalizando as anomalias detectadas.
4. **Pilar 4 (Active Learning):** Salvamento local de recortes aprovados/rejeitados em `public/dataset/`.

**Dependências Base:**
- PyTorch (Redes Neurais)
- OpenCV (Visão Clássica / Tratamento de Imagem)
- mss (Captura de tela ultrarrápida)
- PyQt6 (Criação do HUD transparente)


## Melhoria em andamento — inspeção mult-iluminação para adesivo

### Objetivo

A próxima melhoria do ODIN para a categoria de excesso de adesivo passa a usar
as três iluminações disponíveis na AOI sobre a **mesma peça**.

A motivação visual observada é:

```text
SIDE → adesivo pouco evidente; útil como base geométrica/contextual
TOP  → adesivo muito escuro e bem marcado
MID  → adesivo muito claro/branco e bem marcado
```

A hipótese de trabalho é que `TOP` e `MID` fornecem evidências fotométricas
complementares do adesivo, enquanto `SIDE` continua servindo como imagem
inicial e referência de contexto.

### Mapeamento operacional confirmado da AOI

```text
← seta esquerda → TOP
↓ seta para baixo → SIDE
→ seta direita → MID
```

### Estado do agente Windows XP

Em 06/10/2026, a V5.2 foi copiada manualmente para o Windows XP e validada para
os comandos enviados pelo ODIN. Depois disso, a referência da branch `central`
evoluiu para:

```text
agente_industrial_xp.py V5.3
```

A V5.3 mantém:

```text
PRESS_0
PRESS_1
PRESS_LEFT
PRESS_DOWN
PRESS_RIGHT
```

e acrescenta a telemetria XP → ODIN das setas físicas por
`CMD_TOP/CMD_SIDE/CMD_MID`.

Até nova confirmação operacional, o estado documentado é: **GitHub em V5.3 e
Windows XP ainda precisa receber manualmente essa versão e reiniciar o agente**.

### Controle manual do ODIN compatível com a AOI

Em 06/10/2026, o controle manual de iluminação do ODIN foi corrigido para usar
o mesmo mapeamento operacional do Windows XP:

```text
← → TOP  → PRESS_LEFT
↓ → SIDE → PRESS_DOWN
→ → MID  → PRESS_RIGHT
```

O estado inicial exibido pelo ODIN passa a ser `SIDE`, que é a iluminação
padrão da captura recebida da AOI.

Os três botões de iluminação também seguem a ordem visual e os atalhos corretos:

```text
Luz TOP  | ←
Luz SIDE | ↓
Luz MID  | →
```

A troca manual só atualiza o estado visual depois que o envio TCP do comando ao
agente XP retorna com sucesso. Se o comando falhar, o ODIN não deve fingir que a
iluminação mudou.

O feedback temporário de tecla no canto inferior direito foi ampliado:

- `0/1` continuam exibindo a tecla e `OK/NG`;
- `←/↓/→` reutilizam o mesmo card e mostram explicitamente qual seta foi
  pressionada/enviada;
- feedback de seta não pode disparar o fade-out do veredito da IA.

Existe ainda um terceiro card flutuante em
`src/ui/lighting_status_feedback.py`. Ele fica no canto superior direito,
abaixo do card de veredito, e mostra:

```text
ILUMINAÇÃO ATUAL
TOP  ←
SIDE ↓
MID  →
```

O estado de iluminação é atualizado internamente durante os comandos, mas o
card **permanece oculto enquanto a análise ainda não terminou**. Ele só aparece
quando existe um veredito final válido, no mesmo evento visual em que o card
`FALHA FALSA / DEFEITO REAL / REVISÃO OBRIGATÓRIA` é exibido.

Ao ocorrer o julgamento `0/1`, o card de iluminação é preservado durante o
reset produtivo e inicia o mesmo fade-out sincronizado do veredito e do feedback
de tecla. Portanto os três elementos encerram juntos o ciclo visual da peça.

Esse card é somente de apresentação: não envia comandos, não altera análise,
gate, KNN, dataset ou decisão.

Após o primeiro teste operacional em 06/10/2026 foram encontrados dois pontos:

- os botões `Luz TOP/SIDE/MID` funcionaram e a AOI respondeu corretamente;
- as setas do teclado do próprio ODIN não disparavam de forma confiável porque
  dependiam de `keyPressEvent` do painel e o foco podia estar em widgets filhos;
- as setas físicas do Windows XP mudavam a AOI, mas o agente V5.2 não enviava
  essa mudança de volta ao ODIN.

Correção implementada:

- `src/ui/lighting_shortcuts.py` instala `QShortcut` com
  `WindowShortcut`, tornando `←/↓/→` válidas em toda a janela do ODIN e
  preservando as travas dos botões;
- o agente foi evoluído para V5.3 e passa a enviar `CMD_TOP`, `CMD_SIDE` e
  `CMD_MID` quando as setas são detectadas pelo hook global;
- comandos recebidos da rede atualizam o card fixo e mostram o mesmo feedback
  temporário de tecla, com origem `TECLADO WINDOWS XP`;
- o eco de uma seta que foi originalmente enviada pelo ODIN continua coberto
  pela supressão temporal do feedback.

A correção do ODIN está implementada. A telemetria XP → ODIN depende de copiar a
V5.3 do `agente_industrial_xp.py` para o Windows XP e reiniciar o agente.

### Etapas A e B — interface e alimentação visual implementadas

O ODIN agora possui uma área de inspeção específica para adesivo em
`src/ui/adhesive_multilight_inspection.py`.

Ativação:

```text
ADESIVO
ADHESIVE
MUITO ADESIVO
MUCH ADHESIVE
EXCESS ADHESIVE
ADESIVO EM EXCESSO
        ↓
categoria canônica = MUITO ADESIVO
        ↓
layout multilight
```

As entradas `ADESIVO` e `ADHESIVE` também passam a ser aliases oficiais de
`MUITO ADESIVO` no normalizador. Para qualquer outra categoria, a pilha visual
retorna ao painel normal já existente e o comportamento anterior é preservado.

Cada iluminação possui exatamente três visões:

```text
SIDE
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor

TOP
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor

MID
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor
```

O retângulo maior vem de `global_box_info` produzido por
`detect_anomalies()`. O retângulo menor usa o epicentro selecionado por
`EpicenterExtractor.extract_focus()`. Portanto os recortes visuais reutilizam
a mesma geometria já empregada pelo pipeline atual e não introduzem uma segunda
regra de detecção de caixas.

#### Alimentação das três iluminações

A primeira imagem de uma peça de adesivo continua entrando no pipeline normal e
é registrada visualmente como `SIDE`, conforme o contrato operacional da AOI.

Depois que a análise inicial está ativa, o receptor pode aceitar frames
auxiliares da mesma peça mesmo com o gate principal fechado. Essa exceção é
explicitamente visual:

- `NetworkReceiver.set_auxiliary_image_mode(True)` não reabre o gate principal;
- o frame auxiliar não cria novo `event_id`;
- o frame auxiliar não substitui `current_sample`, `current_ng` ou
  `current_analysis`;
- o frame auxiliar executa uma análise MoE isolada da iluminação para alimentar
  os painéis de especialistas, mas essa análise não substitui
  `current_analysis` nem participa do resultado final;
- a iluminação atribuída ao preview vem do estado atual
  `SIDE/TOP/MID` já comandado/confirmado pelo ODIN;
- ao julgar ou descartar a peça, o modo auxiliar é desligado antes da liberação
  do próximo ciclo.

Assim, quando novas imagens chegarem após o operador mudar a iluminação, elas
preenchem o card correspondente sem transformar TOP/MID em novas peças.

A alimentação visual agora está conectada à automação de aquisição. Depois que a
primeira inspeção válida de adesivo em `SIDE` termina, o ODIN inicia uma máquina
de estados não bloqueante em `src/ui/adhesive_multilight_automation.py`. A
fusão das três iluminações na decisão continua separada e ainda não foi
implementada.

#### Análise dos especialistas por iluminação

A categoria de adesivo também possui uma página visual própria na seção
`ANÁLISE DOS ESPECIALISTAS`, implementada em:

```text
src/ui/adhesive_multilight_analysis.py
```

Ela apresenta três grupos independentes:

```text
ANÁLISE SIDE ↓
ANÁLISE TOP  ←
ANÁLISE MID  →
```

Cada grupo já possui a mesma estrutura de especialistas da interface normal:

- SSIM • textura e calor;
- XOR • tinta e epicentro;
- DNA • assinatura semântica;
- SHIFT • deslocamento;
- FUSÃO • score final como fallback quando não há motor ativo.

**Contrato atual:** `SIDE`, `TOP` e `MID` possuem análises visuais
independentes. SIDE continua usando o `current_analysis` principal produzido
pelo fluxo original. Quando os frames auxiliares TOP e MID chegam, cada um
percorre o mesmo pipeline técnico:

```text
detect_anomalies
        ↓
EpicenterExtractor.extract_focus
        ↓
MoEOrchestrator.inspect
        ↓
painéis de especialistas daquela iluminação
```

O contexto geométrico calculado para TOP/MID é compartilhado entre o preview
visual e o MoE para evitar executar novamente `detect_anomalies` e
`EpicenterExtractor` sobre o mesmo frame.

As análises são armazenadas separadamente em
`adhesive_multilight_analyses["SIDE"|"TOP"|"MID"]`. TOP e MID recebem
metadados explícitos:

```text
multilight_visual_analysis = True
eligible_for_final_decision = False
lighting_mode = TOP | MID
```

Contrato atual:

- TOP não sobrescreve a análise SIDE durante a coleta;
- MID não sobrescreve a análise SIDE durante a coleta;
- os `is_defect`, `confidence` e `verdict` locais continuam sendo saídas
  auditáveis de cada iluminação;
- depois que SIDE, TOP e MID terminam, uma quarta análise lógica é criada:
  a fusão final de `src/core/adhesive_multilight_fusion.py`;
- somente essa análise fundida recebe `eligible_for_final_decision = True` e
  substitui `current_analysis` para o julgamento da peça;
- a decisão 0/1 automática de Produção usa exclusivamente o resultado fundido.

A automação só avança da captura TOP para MID, e de MID para conclusão, depois
que a imagem **e a análise visual** da iluminação esperada foram concluídas.

#### Debug técnico e Copiar imagem para adesivo

O diagnóstico copiável também passa a respeitar o contrato multilight somente
quando o evento atual pertence à categoria canônica `MUITO ADESIVO`.

**Copiar debug**

O relatório técnico original do evento SIDE é preservado e recebe uma seção
adicional:

```text
ANÁLISES MULTILIGHT - ADESIVO

ILUMINAÇÃO SIDE
ILUMINAÇÃO TOP
ILUMINAÇÃO MID
```

Para cada iluminação são registrados, entre outros:

- motores ativos;
- veredito local do MoE, identificado explicitamente como não sendo o resultado
  final multilight;
- flag local de defeito;
- confiança local;
- score final local;
- score físico local;
- regra de fusão local;
- motor dominante local;
- motivo local;
- detalhes técnicos compactos em JSON.

Matrizes e imagens internas não são despejadas pixel a pixel no clipboard.
Arrays NumPy são resumidos por `shape`, `dtype`, mínimo, máximo e média, e
listas muito grandes são resumidas. Isso mantém o debug técnico copiável sem
perder a estrutura necessária para diagnóstico.

O relatório agora inclui também uma seção `JULGAMENTO FINAL MULTILIGHT`
com veredito, score final, score físico máximo, regra de fusão, iluminação
dominante, iluminações positivas, auxiliares fortes e o papel da memória KNN.

**Copiar imagem**

Para o mesmo `event_id` de adesivo, o ODIN preserva também os frames completos
de origem de cada iluminação:

```text
SIDE | TOP | MID
```

O botão `Copiar imagem` só considera o conjunto multilight pronto quando os
três frames estão disponíveis. Em vez de copiar apenas SIDE, ele cria uma única
imagem composta:

```text
┌────────────┬────────────┬────────────┐
│    SIDE    │    TOP     │    MID     │
├────────────┼────────────┼────────────┤
│ frame SIDE │ frame TOP  │ frame MID  │
└────────────┴────────────┴────────────┘
```

As imagens ficam lado a lado, com separadores próprios e sem sobreposição.
Os frames não são redimensionados para montar a composição; o canvas apenas
acomoda as alturas e larguras originais e adiciona um cabeçalho externo com o
nome da iluminação.

A evidência multilight persistida para debug é vinculada ao `event_id` da
captura SIDE original. Se o evento atual mudar, um conjunto multilight antigo
não pode ser reutilizado por `Copiar debug` ou `Copiar imagem`.

Essa mudança é exclusiva do computador novo e não altera
`agente_industrial_xp.py`.

A página normal de especialistas permanece como padrão para todas as categorias
que não sejam adesivo. Imagens multilight e especialistas multilight são
trocados juntos através do mesmo modo condicional de adesivo.

#### Responsividade

As áreas de imagens e especialistas usam páginas empilhadas: o painel antigo
permanece como página padrão e as páginas multilight só são selecionadas para
adesivo.

O layout multilight refluí em dois níveis:

**Imagens**

- largura interna abaixo de `1000 px`: uma iluminação por linha;
- largura interna a partir de `1000 px`: SIDE, TOP e MID em três colunas;
- os nove viewports preservam proporção com
  `KeepAspectRatio + SmoothTransformation`.

**Especialistas**

- abaixo de `1500 px`: SIDE, TOP e MID ficam empilhados verticalmente;
- a partir de `1500 px`: SIDE, TOP e MID ficam em três colunas;
- dentro de cada iluminação, os especialistas usam scroll horizontal para não
  serem comprimidos abaixo de uma largura útil.

Durante uma inspeção de adesivo o splitter principal fica vertical, dando a
largura inteira primeiro para as imagens e depois para os especialistas. Em
notebooks o conteúdo cresce verticalmente dentro do `root_scroll`; em
monitores grandes a largura adicional é aproveitada para três colunas.

Ao sair da categoria de adesivo, o splitter e as páginas normais voltam ao
comportamento responsivo anterior. Portanto a mudança não altera visualmente as
outras categorias.

### Automação implementada da captura mult-iluminação

A aquisição automática de imagens para adesivo está implementada em:

```text
src/ui/adhesive_multilight_automation.py
```

Fluxo atual:

```text
mesma peça / mesmo ciclo
        ↓
SIDE recebido e analisado pelo pipeline atual
        ↓
categoria canônica = MUITO ADESIVO?
   ├── NÃO → fluxo normal atual
   └── SIM
        ↓
preservar SIDE
        ↓
habilitar recepção auxiliar sem reabrir o gate principal
        ↓
PRESS_LEFT
        ↓
AOI seleciona TOP
        ↓
aguardar frame TOP
        ↓
armazenar TOP na mesma sessão
        ↓
PRESS_RIGHT
        ↓
AOI seleciona MID
        ↓
aguardar frame MID
        ↓
armazenar MID na mesma sessão
        ↓
fechar recepção auxiliar
        ↓
PRESS_DOWN
        ↓
restaurar SIDE
        ↓
captura automática concluída
```

Não existe retorno intermediário para SIDE entre TOP e MID, porque as setas da
AOI são tratadas como seletores absolutos:

```text
LEFT  = TOP
DOWN  = SIDE
RIGHT = MID
```

### Validação operacional da automação

Em 06/10/2026, o fluxo automático de troca de iluminação e recebimento das
imagens foi testado na AOI real pelo operador e confirmado como funcional.

O comportamento validado foi:

```text
SIDE inicial
→ TOP automático + foto recebida
→ MID automático + foto recebida
→ retorno automático para SIDE
```

Essa validação confirma a infraestrutura de aquisição na AOI real. A fusão
multilight foi implementada depois dessa validação operacional e ainda precisa
ser validada em novos casos reais OK/NG antes de qualquer calibração adicional.

A máquina de estados somente avança após o frame esperado ter sido recortado e
armazenado. Para cada `TOP` ou `MID`:

- timeout: `8000 ms`;
- após o primeiro timeout, o mesmo seletor absoluto é enviado uma vez novamente;
- após uma segunda falha, a automação é interrompida, SIDE é restaurada e o
  status de rede informa a falha;
- em falha automática, a recepção auxiliar permanece disponível para fallback
  manual da mesma peça.

Enquanto a sequência está ativa, os botões/setas de iluminação do próprio ODIN
não podem trocar manualmente o modo no meio de TOP/MID. Isso evita que uma foto
seja armazenada sob a iluminação errada.

Ao julgar ou descartar a peça antes da conclusão, qualquer timer pendente é
cancelado, a recepção auxiliar é fechada e a AOI é devolvida para SIDE antes do
próximo ciclo.

#### Modo Produção durante a automação

O pipeline SIDE continua calculando seu resultado normalmente. Entretanto, em
`Modo Produção`, o `PRESS_0/PRESS_1` automático não pode avançar a AOI antes
das fotos auxiliares terminarem.

Contrato:

```text
resultado local SIDE calculado
        ↓
não promover SIDE a julgamento final
        ↓
capturar + analisar TOP
        ↓
capturar + analisar MID
        ↓
restaurar SIDE
        ↓
fuse_adhesive_multilight(SIDE, TOP, MID)
        ↓
resultado final único
        ↓
Modo Produção sem revisão?
        ├── NÃO → REVISÃO OBRIGATÓRIA, sem PRESS_0/PRESS_1 automático
        └── SIM → save_label(resultado fundido, source="auto")
                    ↓
                 PRESS_0 / PRESS_1
```

Assim a mesma peça permanece na tela durante toda a coleta. Em `Modo Teste` ou
`Modo Sombra`, ações de julgamento iniciadas no próprio ODIN são recusadas
enquanto a automação está ativa, com mensagem para aguardar a sequência.

O teclado físico do Windows XP continua sendo um controle externo à aplicação;
o operador não deve julgar a peça fisicamente com `0/1` enquanto a sequência
automática TOP/MID estiver em andamento, pois a própria AOI pode avançar antes
que o ODIN consiga impedir a ação.

### Fusão final multilight de adesivo

A primeira política de fusão está implementada em:

```text
src/core/adhesive_multilight_fusion.py
```

Ela é exclusiva da categoria canônica `MUITO ADESIVO` e não usa votação
majoritária nem média simples entre SIDE/TOP/MID.

Motivo: uma iluminação pode ocultar o adesivo sem isso significar que o adesivo
não existe. Portanto um resultado local negativo não tem o mesmo significado de
uma evidência física positiva forte.

Política inicial:

```text
TOP ou MID com:
  adhesive_is_defect = True
  adhesive_score >= 0.80
  physical_score >= 0.80
        ↓
DEFEITO REAL

ou

duas iluminações com adhesive_score >= tolerância do motor
        ↓
DEFEITO REAL

ou

somente uma iluminação positiva, sem força auxiliar suficiente
        ↓
REVISÃO OBRIGATÓRIA

ou

nenhuma das três com evidência física positiva
        ↓
FALHA FALSA
```

TOP e MID são as testemunhas fotométricas prioritárias; SIDE continua útil como
testemunha contextual/corroboradora. O KNN de cada iluminação é preservado para
auditoria, mas a memória local recebe papel `audit_only` na fusão e não pode
vetar evidência física multilight forte.

A análise dominante da fusão é a iluminação com maior `adhesive_score`, usando
`physical_score` e prioridade auxiliar como desempate. O resultado final
registra:

- `lighting_mode = MULTILIGHT`;
- `multilight_final = True`;
- `eligible_for_final_decision = True`;
- `adhesive_multilight_positive_modes`;
- `adhesive_multilight_strong_auxiliary_modes`;
- `adhesive_multilight_dominant_mode`;
- `fusion_rule`;
- `memory_role = audit_only`.

O tempo de análise de adesivo também passa a terminar somente após SIDE, TOP e
MID terem sido analisadas, a fusão final ter sido calculada e o resultado ter
sido pintado na interface.

A etapa seguinte continua separada: melhorar a capacidade específica do motor de
adesivo na iluminação MID. A política de fusão não modifica internamente o
detector MID.

#### Caso real usado como referência da primeira política

No teste de 06/10/2026 que motivou esta fusão, a mesma peça apresentou:

```text
SIDE
  veredito local = FALHA FALSA
  adhesive_score ≈ 0.792
  physical_score = 0.85
  motor de adesivo físico = positivo
  KNN = OK ≈ 96.4%

TOP
  veredito local = DEFEITO REAL
  adhesive_score ≈ 0.972
  physical_score ≈ 0.972
  KNN = NG 100%

MID
  veredito local = FALHA FALSA
  adhesive_score = 0.0
  physical_score ≈ 0.190
```

Pela nova fusão, TOP é uma testemunha auxiliar forte e SIDE também fornece
corroboração física. O único resultado final esperado para esse conjunto é
`DEFEITO REAL`, independentemente do veto KNN que anteriormente fazia a
primeira imagem SIDE encerrar a peça como `FALHA FALSA`.

Esse caso foi usado como referência funcional da regra inicial. Após a
implementação da fusão, o operador repetiu o fluxo na AOI real e confirmou que o
novo julgamento multilight funcionou corretamente. A fusão inicial está,
portanto, **validada operacionalmente** para esse caso real.

Isso não encerra a calibração estatística: novos casos reais OK/NG ainda devem
ser usados para avaliar os limiares e reduzir risco de falso positivo/falso
negativo.

### Detector MID claro — `mid_bright_resin_v1`

A análise real mostrou que a iluminação MID deixava o adesivo visualmente muito
evidente, porém o motor físico retornava:

```text
adhesive_score = 0
adhesive_is_defect = False
reference_area_pct = 0
test_area_pct = 0
```

A causa era fotométrica: o detector original de
`src/core/experts/adhesive_shift_expert.py` foi construído para material
quente, saturado e relativamente escuro. Na MID, a mesma película pode aparecer
quase branca/creme/amarelada e, por isso, desaparecer da máscara antiga.

Foi implementado um segundo perfil, ativado **somente quando
`lighting_mode == "MID"`**:

```text
perfil padrão SIDE/TOP = dark_warm_v1
perfil MID             = mid_bright_resin_v1
```

O perfil MID não classifica simplesmente pixels claros como adesivo. Ele compara
gabarito e teste no espaço LAB e exige uma mudança cromática local coerente:

```text
gabarito MID + teste MID
        ↓
ΔE em LAB
        +
ganho amarelo/vermelho (b*/a*)
        +
presença quente no teste
        +
brilho alto
        -
supressão de cobre saturado
        ↓
mid_bright_resin_witness
```

A testemunha é **diferencial**: uma região branca estável presente nas duas
imagens não gera evidência. Uma mudança apenas neutra de brilho também não deve
ser suficiente. O perfil foi criado para destacar a película clara/creme que
aparece no TESTE e não está no GABARITO.

Integração com o motor existente:

- o detector escuro/quente original continua funcionando e não foi removido;
- SIDE e TOP permanecem no perfil legado, sem novos limiares;
- MID combina o material legado com a nova testemunha clara;
- o limiar de material da máscara TESTE em MID passa a `0.18`; o gabarito
  continua usando `0.22`;
- a cobertura coerente da testemunha MID reforça o `adhesive_score`, mas não
  substitui as métricas físicas já existentes de excesso, padding, expansão,
  espalhamento e vazamento;
- a tolerância final do motor permanece `0.32`.

Novas telemetrias:

```text
adhesive_lighting_mode
adhesive_detector_profile
adhesive_material_threshold
mid_bright_witness_coverage
mid_bright_witness_peak
mid_bright_witness_score
mid_bright_witness_mask
```

Esses valores aparecem no **Copiar debug** da iluminação MID e também no painel
`FLUXO DE ADESIVO`, permitindo verificar se o ODIN realmente passou a enxergar
a película clara.

Regressões adicionadas:

- película clara/amarelada diferencial em MID deve ser detectada;
- a mesma aparência clara não ativa o perfil novo em TOP;
- MID idêntica entre gabarito e teste continua estável;
- mudança neutra de luminosidade sem ganho amarelo/vermelho não deve virar
  adesivo;
- uma MID fisicamente forte pode ser a testemunha dominante da fusão final.

Esta implementação foi validada operacionalmente na AOI real com o caso que
antes deixava a iluminação MID matematicamente cega. O operador confirmou que,
após a inclusão do perfil `mid_bright_resin_v1`, o fluxo passou a funcionar
corretamente também na MID.

Estado atual validado:

```text
SIDE → análise física
TOP  → análise física
MID  → análise física com mid_bright_resin_v1
        ↓
fusão SIDE/TOP/MID
        ↓
um único julgamento final
```

A validação confirma o comportamento funcional do detector MID no caso real
testado. Isso ainda não substitui uma calibração estatística ampla: novos casos
OK/NG devem continuar sendo coletados antes de alterar limiares ou pesos.

### Restrições arquiteturais da melhoria

A implementação atual e as próximas etapas devem preservar os seguintes contratos:

- `SIDE`, `TOP` e `MID` pertencem à mesma peça e não podem virar três
  inspeções independentes;
- as capturas auxiliares não podem ganhar `event_id` de peças diferentes;
- o gate de rede precisa distinguir imagens auxiliares da sessão
  mult-iluminação de uma nova peça real;
- não confiar apenas na ordem temporal; uma captura precisa corresponder ao
  estado de iluminação solicitado antes de ser aceita;
- o ODIN deve restaurar `SIDE` ao final da sequência para não deixar a AOI em
  uma iluminação inesperada;
- falha ao obter `TOP` ou `MID` não deve fabricar evidência ausente nem
  reutilizar silenciosamente um frame anterior;
- cada motor local continua monoimagem e auditável; a fusão acontece em uma
  camada posterior, sem alterar os cálculos internos de SIDE/TOP/MID;
- a memória KNN local permanece disponível para diagnóstico, mas não pode vetar
  evidência física multilight forte;
- a política atual não recalibra ainda o detector específico da iluminação MID.

### Ordem de trabalho

A melhoria será executada por etapas, sem avançar automaticamente:

1. **Concluído — agente XP:** comandos de setas disponíveis; a referência
   atual do GitHub é V5.3 e a atualização operacional continua manual no XP.
2. **Correção implementada — controle manual do ODIN:** botões e
   `QShortcut` usam `← TOP / ↓ SIDE / → MID`; V5.3 suporta retorno
   `CMD_TOP/CMD_SIDE/CMD_MID`.
3. **Concluído — Etapa A:** layout multilight condicional e responsivo para
   adesivo, com nove viewports.
4. **Concluído — Etapa B visual:** SIDE usa a primeira inspeção normal e frames
   auxiliares recebidos depois podem preencher TOP/MID sem criar nova peça nem
   substituir a análise ativa.
5. **Concluído — especialistas por iluminação:** somente para adesivo, a seção
   de especialistas possui SIDE/TOP/MID; SIDE mostra a análise real atual e
   TOP/MID permanecem aguardando análise, sem cálculo artificial.
6. **Concluído e validado na AOI — automação de aquisição:** após SIDE, o
   ODIN comanda TOP, aguarda a foto, comanda MID, aguarda a foto e restaura
   SIDE; timeout, repetição única e cancelamento de ciclo são tratados pela
   máquina de estados.
7. **Concluído — análise visual por iluminação:** SIDE, TOP e MID executam
   análises independentes dos especialistas e alimentam seus próprios painéis;
   TOP/MID não substituem `current_analysis` e não entram no veredito final.
8. **Concluído — debug/evidência multilight:** `Copiar debug` reúne as três
   análises e `Copiar imagem` gera uma única composição SIDE/TOP/MID sem
   sobreposição, vinculada ao mesmo `event_id`.
9. **Concluído e validado operacionalmente — fusão multilight inicial:**
   SIDE/TOP/MID formam um único julgamento final físico; TOP/MID fortes têm
   autoridade, duas iluminações positivas corroboram defeito, caso intermediário
   exige revisão e o KNN local permanece apenas como auditoria na fusão.
10. **Concluído e validado operacionalmente — detector MID claro v1:** o
    perfil `mid_bright_resin_v1` adiciona testemunha diferencial LAB para
    película clara/creme/amarelada exclusivamente na iluminação MID,
    preservando SIDE/TOP no detector legado. O caso real que antes gerava
    `adhesive_score = 0` na MID foi repetido e o comportamento foi confirmado
    como correto pelo operador.
11. **Próxima etapa — calibração com mais casos reais:** coletar novos exemplos
    OK/NG de adesivo em SIDE/TOP/MID antes de reajustar qualquer limiar ou peso.

A aquisição, as três análises visuais, a fusão final e o detector específico de
MID estão implementados e validados operacionalmente nos casos reais testados.
O próximo trabalho recomendado é aumentar a base de validação antes de novas
mudanças de regra.


**Arquivo visual NG opcional:**
- Toggle **ativado por padrão** em toda inicialização do ODIN. O operador pode desativá-lo manualmente durante a sessão.
- Para categorias comuns, um julgamento final `NG` de captura XP arquiva o frame completo do evento atual em `public/ng_archive/`.
- Para a categoria canônica `MUITO ADESIVO`, quando SIDE/TOP/MID pertencem ao mesmo `event_id`, o mesmo julgamento pode arquivar **até três imagens completas separadas**:
  - `..._MUITO_ADESIVO_SIDE.png`
  - `..._MUITO_ADESIVO_TOP.png`
  - `..._MUITO_ADESIVO_MID.png`
- A resolução dessas imagens fica em `src/services/image_archive_candidates.py` e usa `adhesive_multilight_last_source_frames` somente quando o `event_id` do conjunto coincide com o evento julgado.
- Não existe fallback para `current_ng` ou outro recorte. Se a evidência completa do evento não estiver disponível, nenhuma imagem substituta é arquivada.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.
- Deduplicação de eco por `event_id` continua obrigatória: o mesmo julgamento não pode ser arquivado duas vezes se o comando voltar pelo hook do XP.
- **Deduplicação persistente por conteúdo visual:** antes de gravar, a fila indexa os PNGs já existentes e calcula SHA-256 do conteúdo visual exato. Se uma imagem pixel a pixel idêntica já existir em `public/ng_archive/`, ela é ignorada mesmo que venha de outro evento ou após reiniciar o ODIN.
- Imagens realmente diferentes continuam sendo preservadas. Se duas imagens diferentes caírem no mesmo minuto/categoria, o ODIN cria um nome alternativo `_2`, `_3`, etc., em vez de sobrescrever o arquivo anterior.
- O arquivamento só é permitido enquanto existe uma captura de rede ativa, com análise ativa e categoria AOI não vazia.
- `SEM_CATEGORIA` não é um nome de arquivo válido para o fluxo automático de evidências NG.


**Arquivo visual OK opcional:**
- Existe um segundo toggle **`Salvar imagens OK`**, exibido imediatamente abaixo de **`Salvar imagens NG`**.
- O toggle inicia **ATIVADO por padrão** em toda abertura do ODIN e o operador pode desativá-lo durante a sessão.
- Visualmente, o bloco OK deve manter o mesmo layout, dimensões, tipografia, hover, focus e estado checked do bloco NG.
- Quando ativado, cada julgamento humano final `OK` arquiva a evidência do evento em `public/ok_archive/`; para adesivo multilight, o mesmo julgamento pode gerar SIDE/TOP/MID como três PNGs separados.
- Julgamentos humanos aceitos: botão/atalho do ODIN (`source="button"`) e teclado físico do XP (`source="xp_keyboard"`).
- Decisão automática de Produção (`source="auto"`) **não** gera arquivo OK.
- Para categoria comum, a imagem salva continua sendo a evidência completa resolvida por `Copiar imagem`. Para adesivo multilight, em vez da composição visual, são preservados os três frames completos individuais SIDE/TOP/MID da mesma peça.
- O contrato compartilhado de evidência fica em `src/services/capture_evidence.py`, por meio de `current_copy_image_snapshot()` e `current_copy_image_event_id()`.
- O arquivo OK aceita tanto captura recebida do **Windows XP** quanto captura local **MSS**, desde que exista análise ativa, `event_id` válido e categoria AOI válida.
- Uma captura local MSS nunca pode usar como fallback um frame XP anterior.
- Não usar `current_ng`, ROI, foco ou outro recorte como imagem substituta.
- O formato do nome é o mesmo do arquivo NG: `YYYY-MM-DD_HHmm_CATEGORIA.png`.
- A implementação de nome compartilhada fica em `src/services/image_archive_naming.py`.
- `SEM_CATEGORIA` não é permitido no arquivamento automático OK.
- A gravação é assíncrona em fila daemon e não pode bloquear julgamento, envio de tecla, limpeza da interface ou recepção da próxima captura.
- Deduplicação obrigatória por `event_id`: um mesmo evento não pode ser salvo duas vezes caso o julgamento retorne pelo hook do XP.
- **Deduplicação persistente por conteúdo visual em OK e NG:** se uma imagem pixel a pixel idêntica já existir no respectivo arquivo visual, um novo julgamento dessa mesma imagem não cria outro PNG, mesmo que apareça muitos eventos depois ou após reiniciar o ODIN.
- A verificação é feita pelo conteúdo da imagem, não pelo nome do arquivo nem pelo `event_id`. Portanto arquivos antigos com o padrão de nome legado também contam como duplicatas se contiverem exatamente os mesmos pixels.
- A fila OK indexa os PNGs já existentes em background para não bloquear o julgamento. Novas imagens realmente diferentes continuam sendo salvas normalmente.
- O mesmo utilitário compartilhado em `src/services/image_archive_dedup.py` é usado pelas filas OK e NG.
- O arquivo é somente evidência visual/auditoria e não participa do dataset, KNN, protótipos, score, confiança ou decisão.

Fluxo:

```text
captura XP ou MSS analisada
        ↓
operador julga OK
        ↓
Salvar imagens OK está ATIVADO?
        ├── NÃO → não arquiva
        └── SIM
              ↓
event_id + categoria + análise ativa válidos?
              ├── NÃO → não arquiva
              └── SIM
                    ↓
mesma evidência de Copiar imagem
                    ↓
public/ok_archive/
```

Regressões obrigatórias do arquivo OK:

- toggle inicia ativado;
- operador pode desativar durante a sessão;
- NG nunca é salvo pelo arquivo OK;
- `source="auto"` nunca gera arquivo OK;
- `source="button"` e `source="xp_keyboard"` podem gerar arquivo OK;
- XP salva exatamente o mesmo frame de `Copiar imagem`;
- MSS salva exatamente o mesmo frame de `Copiar imagem`;
- MSS nunca reutiliza frame XP anterior;
- mesmo `event_id` é salvo no máximo uma vez;
- mesma imagem OK reaparecendo em outro `event_id` não cria outro PNG;
- mesma imagem OK já existente antes de reiniciar o ODIN também não é duplicada;
- um PNG antigo com nome legado bloqueia nova cópia quando o conteúdo visual é idêntico;
- imagens visualmente diferentes continuam sendo preservadas separadamente;
- novo `event_id` com imagem diferente pode ser salvo normalmente;
- categoria vazia não cria `SEM_CATEGORIA`;
- fila grava PNG com o mesmo formato de nome do NG;
- o bloco visual OK permanece imediatamente abaixo do bloco NG e usa o mesmo padrão responsivo.


### Regra de não duplicar a mesma imagem OK

A necessidade operacional é manter apenas uma evidência quando a **mesma imagem**
for julgada como OK repetidas vezes.

Exemplo:

```text
imagem A → operador julga OK → salva 1 PNG
10 outras imagens passam
imagem A reaparece → operador julga OK → NÃO salva outro PNG
imagem A reaparece novamente → operador julga OK → NÃO salva outro PNG
```

A identidade usada nessa regra é o conteúdo exato dos pixels. O nome do arquivo,
horário e `event_id` podem mudar; se os pixels forem idênticos, a evidência já
existe e o novo salvamento é ignorado.

A deduplicação deve sobreviver a reinicializações do ODIN porque a fila
`OKImageArchiveQueue` indexa os PNGs já existentes em `public/ok_archive/`.

Essa regra é exclusiva do arquivo visual OK. O arquivo visual NG não deve adotar
automaticamente essa deduplicação por conteúdo.


#### Validação operacional da deduplicação OK em 02/10/2026

O operador validou em uso real o comportamento de não duplicar a mesma imagem OK.

Foi confirmado que:

- uma imagem julgada OK é salva na primeira ocorrência;
- se a mesma imagem reaparecer vários eventos depois e for julgada OK novamente,
  nenhum novo PNG é criado;
- outras imagens podem passar entre as ocorrências sem quebrar a deduplicação;
- imagens visualmente diferentes continuam sendo salvas normalmente;
- a regra permanece exclusiva do arquivo OK;
- o arquivo NG mantém o comportamento anterior;
- a deduplicação por conteúdo não altera julgamento, dataset, KNN ou ciclo
  produtivo.

Essa validação passa a ser a referência operacional para o arquivo visual OK:
**uma mesma evidência visual exata deve existir apenas uma vez em
`public/ok_archive/`**.

### Padrão de nome dos arquivos visuais OK/NG

OK e NG usam obrigatoriamente o mesmo gerador compartilhado em
`src/services/image_archive_naming.py`.

Formato atual:

```text
YYYY-MM-DD_HHmm_CATEGORIA.png
```

Exemplo:

```text
2026-10-02_0811_FALTANDO.png
```

Esse padrão substitui o formato anterior com dia/mês textual, segundos e
milissegundos. Refatorações futuras não devem criar formatos diferentes entre
`public/ng_archive/` e `public/ok_archive/`.


#### Validação operacional do padrão de nomes em 02/10/2026

Após reiniciar o ODIN com a versão atualizada, o operador confirmou em uso real
que novos arquivos OK e NG passaram a ser gravados no padrão correto:

```text
YYYY-MM-DD_HHmm_CATEGORIA.png
```

Foi confirmado que:

- o formato antigo deixou de ser usado para novos arquivos;
- OK e NG usam o mesmo padrão;
- a categoria continua sendo normalizada no nome;
- arquivos antigos já existentes não são renomeados retroativamente;
- a aplicação precisa carregar a versão atual do módulo de nomeação para usar o
  novo padrão;
- a gravação efetiva de OK e NG chama diretamente o gerador compartilhado
  `build_archive_filename()`.

Essa configuração passa a ser a referência operacional validada para nomes dos
arquivos visuais de auditoria.

### Arquivamento multilight de adesivo após julgamento 0/1

Para a categoria de adesivo, o conjunto SIDE/TOP/MID já existe no computador
novo antes do julgamento final. Ao ocorrer `0 = OK` ou `1 = NG`, e com o
respectivo arquivo visual habilitado, o ODIN usa o mesmo `event_id` da peça
para resolver as três evidências completas.

Fluxo:

```text
SIDE + TOP + MID da mesma peça
        ↓
julgamento 0 ou 1
        ↓
resolver imagens do mesmo event_id
        ↓
para cada imagem:
  fingerprint SHA-256 do conteúdo
        ↓
já existe no arquivo visual?
   ├── SIM → não salvar novamente
   └── NÃO → salvar PNG
```

Os arquivos usam a iluminação no nome:

```text
YYYY-MM-DD_HHmm_MUITO_ADESIVO_SIDE.png
YYYY-MM-DD_HHmm_MUITO_ADESIVO_TOP.png
YYYY-MM-DD_HHmm_MUITO_ADESIVO_MID.png
```

Se duas imagens realmente diferentes produzirem o mesmo nome base no mesmo
minuto, a segunda recebe sufixo incremental `_2`, depois `_3`, evitando
sobrescrita silenciosa.

A deduplicação é exata, por conteúdo pixel a pixel, e é persistente entre
reinicializações porque os PNGs já existentes são indexados ao iniciar a fila.
OK e NG mantêm índices separados, pois representam arquivos de auditoria com
rótulos distintos.

Essa mudança ocorre apenas no computador novo e não exige alteração no agente
Windows XP.

### Validação operacional do arquivo visual OK em 02/10/2026

O recurso foi validado em uso real pelo operador após a implementação.

Foi confirmado que:

- o bloco **`Salvar imagens OK`** aparece corretamente logo abaixo de
  **`Salvar imagens NG`**;
- o toggle inicia **ATIVADO** ao abrir o ODIN;
- o operador pode desativar e reativar o recurso durante a sessão;
- julgamentos humanos `OK` geram corretamente a evidência visual;
- a imagem salva corresponde à mesma evidência completa de **`Copiar imagem`**;
- o comportamento funciona sem alterar o fluxo normal de julgamento;
- o visual do bloco permanece consistente com o arquivo NG;
- a gravação não interfere no ciclo produtivo, KNN, dataset ou decisão.

Essa validação passa a ser a referência operacional do arquivo visual OK.
Refatorações futuras devem preservar o mesmo contrato de evidência, o estado
ativado por padrão e a posição visual imediatamente abaixo do arquivo NG.


**Regra visual do painel KNN:**
- A existência de memória e a força do match são conceitos diferentes na interface.
- `PRIMEIRA OCORRÊNCIA` só pode ser exibido quando não existir nenhum registro da categoria consultada.
- Se existirem JSONs da categoria, mas não houver assinatura visual válida/comparável, mostrar `MEMÓRIA CARREGADA` e a quantidade de registros; não representar como primeira ocorrência.
- Se houver comparação visual, a barra amarela `Melhor match` deve usar a melhor similaridade realmente calculada. Caso `best_similarity` não tenha sido propagado por uma camada de telemetria, a UI pode recuperar o mesmo valor a partir da melhor hipótese OK/NG já calculada, sem alterar a decisão.
- Match abaixo do limiar continua visível como `MEMÓRIA ENCONTRADA • abaixo do limiar`; isso não concede influência ao KNN.
- Essa regra é exclusivamente de apresentação. Limiares, classificação, pesos, conflito OK/NG e regra de melhor correspondência permanecem no núcleo.


**Regra visual da barra "Influência dos Motores" para KNN:**
- No KNN, `raw_score`/voto mede direção da decisão: `0 = OK` e `1 = NG`. Esse valor não representa força da memória.
- A barra grande da linha `Memória local KNN` deve representar `best_similarity` (força do melhor match visual), nunca o percentual de voto NG.
- Portanto um caso conhecido como falha falsa pode ter `voto 0% NG` e, ao mesmo tempo, uma barra de match quase cheia.
- O texto da linha deve separar `match`, `voto OK/NG`, `peso` e `efeito`.
- A barra fina amarela continua representando somente o peso efetivo do KNN na fusão.
- O marcador de referência da barra grande do KNN usa o limiar de match da memória, não o corte de decisão física.
- Essa alteração é apenas de apresentação/telemetria; não muda voto, peso, limiar de memória ou resultado da fusão.


## Regra crítica da categoria FALTANDO — ausência física prevalece sobre memória

### Problema observado em 01/10/2026

Foi identificado um caso real em que a AOI enviou uma inspeção válida da categoria
`FALTANDO`: o gabarito mostrava o componente presente e a imagem NG mostrava a
região fisicamente vazia. O intake estava correto:

- `valid_epicenter = True`;
- 6 anomalias brutas;
- epicentro final presente;
- ROI de foco aproximadamente `63 × 122 px`;
- a falha ocorria depois do intake, durante a fusão dos motores.

O problema arquitetural era que o especialista físico `MissingComponentExpert`
podia detectar uma quebra forte da expectativa visual, mas uma correspondência
KNN antiga rotulada como `OK` ainda podia receber peso dominante/100% e vetar
o defeito físico, produzindo `FALHA FALSA`. Além disso, a camada de contraste
OK × NG podia transformar um conflito de memória em `REVISÃO OBRIGATÓRIA`,
mesmo quando o componente estava fisicamente ausente.

Para `FALTANDO`, memória visual é evidência histórica; ela não pode provar a
presença de um componente que o comparador físico confirmou que desapareceu.

### Solução: hard physical absence

`MissingComponentExpert` acrescenta agora:

- `missing_hard_absence`;
- `missing_hard_absence_reason`;
- `missing_hard_absence_thresholds`.

A ausência física forte só é habilitada quando o caso já é defeito físico e não
foi classificado como `DESLOCAMENTO PROVÁVEL`.

Existem três rotas de confirmação:

1. **substituição pelo fundo**
   - score físico >= 72%;
   - área divergente >= 25%;
   - exposição de fundo >= 28%;
   - melhor correspondência próxima < 60%.

2. **colapso estrutural forte**
   - score físico >= 85%;
   - área divergente >= 30%;
   - residual médio >= 45%;
   - perda estrutural >= 30%;
   - melhor correspondência próxima < 60%.

3. **componente removido com footprint/base escura**
   - score físico >= 90%;
   - área divergente >= 45%;
   - residual médio >= 60%;
   - perda de aparência >= 50%;
   - similaridade direta <= 50%;
   - melhor correspondência próxima < 35%;
   - não pode estar classificado como `DESLOCAMENTO PROVÁVEL`.

A terceira rota cobre componentes que, ao desaparecerem, deixam uma área escura,
footprint, cola ou base com aparência parcialmente semelhante ao corpo original.
Nesses casos, `background_exposure` pode permanecer em 0% e
`structure_loss` pode ficar abaixo de 30%, mesmo quando a aparência esperada
foi destruída.

Caso real registrado em 01/10/2026:
- evento `bf6b6a2f1e844cc796ae355ae7ceb7e8`;
- componente R375;
- `missing_score ≈ 96.4%`;
- cobertura ≈ 54.7%;
- residual médio ≈ 67.2%;
- `missing_structure_loss ≈ 24.0%`;
- `background_exposure = 0%`;
- melhor correspondência próxima ≈ 13.2%;
- conflito KNN: NG 90.6% × OK 90.0%;
- consequência antiga: revisão obrigatória.

Esse padrão não deve depender de exposição do fundo vermelho nem exigir perda
estrutural >= 30%. A combinação de score alto, grande cobertura, residual alto,
perda da aparência original e ausência de correspondência próxima é suficiente
para caracterizar ausência física forte.

Esses limites são deliberadamente mais fortes que o limiar comum do motor. O
objetivo é reservar o override apenas para desaparecimento inequívoco, não para
variação de iluminação, ruído, deslocamento ou divergência parcial.

### Hierarquia de decisão quando hard missing = true

Quando `missing_hard_absence == True`:

```text
MissingComponentExpert
        ↓
ausência física forte confirmada
        ↓
fusion_rule = missing_hard_absence
        ↓
motor dominante = missing
        ↓
peso físico = 100%
peso KNN = 0%
        ↓
final_score = 1.0
confidence = 0.99
        ↓
DEFEITO REAL / NG
```

A memória KNN continua sendo calculada e exibida para auditoria, inclusive seu
melhor rótulo e similaridade, mas recebe:

```text
role = AUDITORIA — SEM VETO SOBRE AUSÊNCIA FÍSICA
suppressed_by_hard_missing = true
```

Ela não pode alterar o veredito nesse caso.

### Conflito de memória

Se as melhores memórias OK e NG estiverem quase empatadas, a regra normal ainda
é revisão humana. Porém, quando a ausência física forte já foi confirmada, esse
conflito não pode rebaixar a confiança nem criar revisão obrigatória. O conflito
permanece apenas como telemetria/auditoria.

### Regra de segurança

Não transformar toda categoria `FALTANDO` em NG automático.

Casos abaixo dos critérios de hard absence continuam usando exatamente o fluxo
normal:

- motor físico;
- KNN;
- contraste OK × NG;
- limiares existentes;
- revisão humana quando aplicável.

Um componente deslocado deve continuar sendo distinguido de um componente
ausente. A existência de uma correspondência próxima forte é uma das travas que
impedem o hard missing.

### Debug obrigatório para FALTANDO

O `Copiar debug XP` deve registrar, após a análise:

- categoria normalizada;
- veredito;
- confiança;
- score final e score físico;
- regra de fusão;
- motor dominante;
- `missing_score`;
- `missing_changed_coverage`;
- `missing_structure_loss`;
- `missing_background_exposure`;
- `missing_direct_similarity`;
- `missing_appearance_loss`;
- `missing_edge_mismatch`;
- `missing_residual_p90`;
- `missing_best_similarity`;
- `missing_hard_absence`;
- motivo do hard absence;
- melhor rótulo/similaridade KNN;
- conflito de memória;
- se o KNN foi suprimido pelo hard missing.

Assim um futuro caso de componente realmente ausente não deve ser diagnosticado
somente pelo intake. O debug precisa mostrar também a decisão final e qual camada
teve autoridade sobre ela.

### Regressões obrigatórias

Manter testes que garantam:

- componente completamente removido → `missing_hard_absence=True`;
- componente removido deixando footprint/base escura → `missing_hard_absence=True`;
- diferença parcial com correspondência próxima plausível → não vira hard missing;
- componente deslocado → não vira hard missing;
- memória OK forte não veta hard missing;
- conflito OK × NG não força revisão sobre hard missing;
- hard missing alcança 99% de confiança e pode passar pela trava existente de
  Produção sem reduzir globalmente o limiar de confiança;
- casos ambíguos continuam seguindo a política normal.


### Caso observado em 02/10/2026 — componente presente confundido com ausência

Evento operacional: `a997255818ea489b8afddf3d90b470fd`.

A AOI classificou a ocorrência como `FALTANDO`, mas visualmente o componente
continuava presente. O corpo físico permanecia no mesmo local e com dimensões
compatíveis; a principal diferença estava em aparência, brilho, contraste e
serigrafia interna.

O debug mostrou:

- `missing_score ≈ 100%`;
- cobertura divergente ≈ `56,3%`;
- residual médio ≈ `61,4%`;
- perda estrutural ≈ `86,6%`;
- exposição de fundo ≈ `55,3%`;
- incompatibilidade de bordas ≈ `78,8%`;
- melhor correspondência próxima do motor físico ≈ `34,3%`;
- KNN melhor rótulo = `OK`;
- similaridade OK ≈ `99,43%`;
- melhor NG ≈ `89,68%`;
- sem conflito de memória;
- resultado incorreto antes da correção: `missing_hard_absence=True` e
  `DEFEITO REAL`.

Este caso reforça que **não é suficiente apenas baixar ou subir limiares**.
Uma mudança grande de aparência pode produzir métricas típicas de ausência mesmo
quando o corpo físico continua presente.

#### Nova testemunha de presença em duas escalas

O `MissingComponentExpert` passa a procurar presença física em duas regiões:

1. a ROI interna usada pelo motor `FALTANDO`;
2. o envelope do componente fornecido pelo epicentro final da AOI.

A segunda escala é obrigatoriamente geométrica e procura:

- similaridade de baixa frequência do corpo;
- Dice da silhueta;
- razão de área;
- razão de largura e altura da massa principal;
- deslocamento do centróide.

Existem duas rotas de presença:

1. **coarse + geometria**
   - usa similaridade de baixa frequência junto com silhueta, área e centro;
2. **geometry-only**
   - não exige correlação tonal forte;
   - exige silhueta muito compatível;
   - área compatível;
   - caixa principal com largura/altura compatíveis;
   - centro praticamente preservado.

A rota `geometry-only` existe para componentes como o D5 do caso real, em que
o corpo continua presente, mas acabamento, brilho, cor aparente e serigrafia
mudam o suficiente para derrubar a correlação tonal.

A serigrafia/texto interno e variações globais de brilho não devem, sozinhos,
ser usados como prova de falta física.

Se a ROI interna divergir muito, mas o epicentro final confirmar massa,
silhueta, área e centro coerentes, o sistema deve registrar:

```text
missing_component_body_present = True
missing_body_presence_source = aoi_epicenter
missing_body_presence_veto = True
missing_hard_absence = False
```

Nesse estado, o motor `FALTANDO` não pode classificar o componente como
fisicamente ausente. A divergência de aparência continua disponível para os
outros motores e para a memória KNN.

#### Telemetria obrigatória

O debug de captura deve registrar:

- `missing_component_body_present`;
- `missing_body_presence_veto`;
- `missing_body_presence_source`;
- `missing_body_presence_box`;
- `missing_body_coarse_similarity`;
- `missing_body_silhouette_dice`;
- `missing_body_area_ratio`;
- `missing_body_centroid_shift`;
- `missing_body_box_width_ratio`;
- `missing_body_box_height_ratio`;
- `missing_body_presence_policy`;
- `missing_body_presence_reason`.

A presença geométrica só bloqueia o hard missing quando o motor local está
tentando classificar a ROI como conteúdo ausente/quebra da expectativa. Uma ROI
local conforme, por si só, não pode impedir a análise dual-scale do contexto
maior; isso preserva os casos reais em que um patch pequeno parece normal mas o
componente desapareceu fora dele.

#### Validação operacional da testemunha geométrica em 02/10/2026

O caso real do componente D5 foi retestado após a implementação da testemunha
de presença em duas escalas e o operador confirmou que o falso positivo foi
corrigido.

Foi validado que:

- o componente fisicamente presente deixou de ser tratado como ausência forte;
- a análise do corpo completo pelo epicentro final complementa a ROI interna;
- a rota `geometry_only` consegue preservar a evidência de presença mesmo com
  mudança forte de brilho, acabamento e serigrafia;
- a correção não depende de simplesmente baixar o `missing_score` ou afrouxar
  globalmente os limiares de ausência;
- o hard missing continua disponível para componentes realmente ausentes;
- a análise dual-scale continua preservada nos casos em que a ROI local, sozinha,
  não é suficiente para decidir;
- a memória KNN continua atuando separadamente da testemunha física de presença.

Contrato operacional validado:

```text
componente presente no envelope AOI
        ↓
silhueta/área/caixa/centro coerentes
        ↓
missing_component_body_present = True
missing_body_presence_source = aoi_epicenter
missing_body_presence_policy = geometry_only
        ↓
hard missing bloqueado
        ↓
não classificar como componente fisicamente ausente
```

Essa regra passa a ser a referência operacional para casos de `FALTANDO` em
que a aparência muda fortemente, mas o corpo físico do componente continua
presente.


### Caso observado em 02/10/2026 — ROI interna estreita escondia o envelope do componente

Evento: `4a05d237e8c44c44a1c2dbd996afbb51`.

A ocorrência foi classificada pela AOI como `FALTANDO`, porém visualmente o
componente continuava presente. O ODIN concluiu incorretamente `DEFEITO REAL`
por `missing_hard_absence`.

O debug mostrou:

- `missing_score ≈ 96,50%`;
- cobertura divergente ≈ `57,43%`;
- residual médio ≈ `59,22%`;
- perda estrutural ≈ `52,61%`;
- similaridade direta ≈ `44,85%`;
- exposição de fundo = `0%`;
- melhor similaridade próxima do motor físico ≈ `26,16%`;
- melhor memória KNN = `OK` com ≈ `91,85%`;
- melhor memória NG ≈ `87,66%`;
- sem conflito de memória efetivo.

A testemunha de presença havia sido executada somente na caixa estreita:

```text
[128, 60, 101, 480]
```

Nessa região interna, que concentra serigrafia/conteúdo do componente, as
métricas ficaram incompatíveis:

- `silhouette_dice ≈ 17,3%`;
- `area_ratio ≈ 3,81`;
- `box_height_ratio ≈ 3,31`.

Entretanto, a própria AOI também havia detectado a caixa verde externa do
componente, aproximadamente:

```text
[26, 26, 307, 514]
```

Essa caixa global não chegava ao especialista porque `detect_anomalies()`
preservava somente `w/h` em `global_box_info`, descartando `x/y`.

#### Correção arquitetural

`src/core/inspection.py` passa a preservar:

```text
global_box_info = {
    x,
    y,
    w,
    h,
    detected
}
```

O `MissingComponentExpert` ganhou uma testemunha complementar de **envelope
global**.

Ela não declara OK e não apaga a divergência física. Sua única função é impedir
que uma ROI interna estreita tenha autoridade absoluta para declarar
`missing_hard_absence` quando o envelope completo do componente permanece
estruturalmente coerente.

A testemunha compara, no envelope global:

- perfil horizontal de baixa frequência;
- perfil vertical de baixa frequência;
- similaridade coarse do corpo;
- exposição de fundo.

Contrato:

```text
ROI interna sugere hard missing
        ↓
envelope global AOI detectado
        ↓
perfis horizontal + vertical preservados
+ coarse similarity compatível
+ sem exposição relevante de fundo
        ↓
missing_global_envelope_support = True
missing_global_envelope_veto = True
        ↓
missing_hard_absence = False
        ↓
divergência física continua existente
        ↓
decisão retorna à fusão normal física + KNN
```

Importante: o envelope global **não produz FALHA FALSA diretamente**. Ele apenas
remove a autoridade especial de ausência física forte. A memória e os demais
motores continuam responsáveis pelo veredito final.

Isso é importante neste caso porque o KNN já possui correspondência OK forte
(acima de 90%) e só estava impedido de atuar pela supressão de hard missing.

#### Segurança

A caixa global só é usada quando foi realmente detectada pela AOI
(`global_box_info.detected=True`). O fallback de frame inteiro não pode servir
como testemunha de presença.

Uma remoção real do componente deve continuar produzindo baixo suporte do
envelope e manter `missing_hard_absence=True`.

#### Telemetria obrigatória

O debug passa a registrar:

- `missing_global_envelope_active`;
- `missing_global_envelope_support`;
- `missing_global_envelope_veto`;
- `missing_global_envelope_box`;
- `missing_global_envelope_row_profile`;
- `missing_global_envelope_col_profile`;
- `missing_global_envelope_coarse_similarity`;
- `missing_global_envelope_background_exposure`;
- `missing_global_envelope_reason`.

#### Validação operacional em 02/10/2026

O caso foi retestado na interface real após a correção e o operador confirmou
que o falso positivo foi resolvido.

Foi validado que:

- o componente presente deixou de ser tratado como ausência física forte;
- o envelope global da AOI passou a participar da verificação de presença;
- a ROI interna estreita não possui mais autoridade isolada para impor
  `missing_hard_absence`;
- o hard missing foi rebaixado quando o envelope completo permaneceu coerente;
- a divergência física continuou registrada, sem ser apagada;
- a decisão retornou à fusão normal física + KNN;
- a memória OK forte pôde voltar a participar da decisão;
- o veredito final passou corretamente para **FALHA FALSA**;
- a correção não exigiu redução global de limiares;
- componentes realmente ausentes continuam protegidos pelas regressões que
  exigem baixo suporte do envelope antes de manter `missing_hard_absence=True`.

Contrato operacional validado:

```text
ROI interna estreita indica ausência
        +
envelope global do componente permanece preservado
        ↓
missing_global_envelope_support = True
missing_global_envelope_veto = True
        ↓
hard missing perde autoridade especial
        ↓
fusão normal física + KNN
        ↓
FALHA FALSA neste caso validado
```

Esse comportamento passa a ser a referência operacional para casos de
`FALTANDO` em que a ROI interna concentra serigrafia/conteúdo e não representa
corretamente o corpo completo do componente.

Status em 02/10/2026: **validado operacionalmente na mesma peça**.

### Caso observado em 02/10/2026 — componente presente com posição interna diferente

Evento: `fb7de76ab04843c3b2ab4ad3e16da3f6`.

A AOI classificou a ocorrência como `FALTANDO`, mas o componente físico estava
presente tanto no gabarito quanto no teste. O ODIN concluiu incorretamente
`DEFEITO REAL` por `missing_hard_absence`.

O caso mostrou um limite diferente do falso positivo anterior:

- caixa global detectada: aproximadamente `[26, 26, 308, 514]`;
- ROI/foco local: aproximadamente `[39, 105, 279, 147]`;
- `missing_score ≈ 99,62%`;
- cobertura local ≈ `68,58%`;
- residual médio ≈ `78,36%`;
- exposição de fundo local ≈ `62,69%`;
- melhor KNN = `OK` com ≈ `92,30%`;
- melhor NG ≈ `91,01%`;
- margem OK × NG ≈ `1,30 p.p.`;
- sem conflito de memória.

Visualmente, o componente continuava presente, porém sua massa escura,
serigrafia e posição interna estavam diferentes dentro do envelope global.

#### Erro de escopo espacial identificado

Foi identificado que `missing_global_envelope_background_exposure` estava
reutilizando diretamente `missing_background_exposure`, calculado na ROI local.

Isso é incorreto porque exposição de fundo é uma métrica espacial:

```text
background_exposure da ROI local
!=
background_exposure do envelope global
```

A partir desta correção, a exposição de fundo do envelope é recalculada
diretamente sobre a própria caixa global com `_background_replacement_signal()`.

Nenhuma métrica espacial da ROI local pode ser apresentada como se tivesse sido
medida no envelope global.

#### Presença global invariável a deslocamento interno

Os perfis horizontal/vertical alinhados do envelope continuam úteis, mas podem
cair quando o mesmo componente muda de posição ou orientação interna.

Foi adicionada uma testemunha auxiliar que mede a distribuição de massa escura
no envelope sem exigir que essa massa esteja nas mesmas linhas/colunas.

Ela registra:

- limiar de massa escura;
- fração escura no gabarito;
- fração escura no teste;
- retenção de massa escura;
- perfil horizontal invariável;
- perfil vertical invariável;
- `missing_global_envelope_invariant_support`.

Os perfis invariáveis são comparados após ordenação de suas distribuições. Isso
reduz a dependência de translação interna e da orientação da serigrafia.

#### Regra de segurança: massa escura sozinha não derruba hard missing

Uma regressão de segurança mostrou que um componente realmente ausente pode
deixar footprint/base escuro e, portanto, também preservar parte da massa
escura.

Por isso:

```text
missing_global_envelope_invariant_support = True
```

**não é suficiente sozinho** para vetar `missing_hard_absence`.

A evidência invariável permanece auxiliar no especialista físico.

#### Nova regra combinada

Para `FALTANDO`, um hard missing bruto pode perder autoridade pela nova rota
somente quando todas as condições abaixo forem satisfeitas:

- `missing_global_envelope_invariant_support=True`;
- memória disponível e confiável;
- melhor rótulo = `OK`;
- melhor OK >= `90%`;
- vantagem OK sobre NG >= `1 ponto percentual`;
- sem conflito de memória.

Contrato:

```text
hard missing bruto
        +
massa física global invariável preservada
        +
melhor memória OK >= 90%
        +
OK - NG >= 1 p.p.
        +
sem conflito
        ↓
hard_missing_contradicted_by_invariant_ok = True
fusion_rule = hard_missing_invariant_presence_ok_witness
hard_missing_evidence = False
        ↓
FALHA FALSA
```

Essa combinação é deliberadamente mais restrita que simplesmente permitir que
uma memória OK forte anule ausência física.

As duas proteções são obrigatórias:

- **OK forte sem presença global invariável** não veta hard missing;
- **massa global invariável sem OK forte dominante** não veta hard missing.

Isso preserva os casos reais de componente removido com footprint escuro.

#### Telemetria adicionada

O debug passa a registrar também:

- `missing_global_envelope_dark_threshold`;
- `missing_global_envelope_reference_dark_fraction`;
- `missing_global_envelope_test_dark_fraction`;
- `missing_global_envelope_dark_retention`;
- `missing_global_envelope_invariant_row_profile`;
- `missing_global_envelope_invariant_col_profile`;
- `missing_global_envelope_invariant_support`;
- `hard_missing_contradicted_by_invariant_ok`.

Status em 02/10/2026: **correção implementada e protegida por regressões;
aguardando validação operacional nesta mesma peça antes de considerar a regra
validada**.

### Caso observado em 05/10/2026 — ROI pequena preservava ocupação, mas hard missing encerrava a decisão

Evento: `c5a70a299d2349fabd0cd078998a70d1`.

A AOI classificou a ocorrência como `FALTANDO`, mas visualmente o componente
estava presente. O ODIN concluiu incorretamente `DEFEITO REAL` por
`missing_hard_absence`.

O debug mostrou:

- caixa global detectada: `[25, 25, 525, 285]`;
- ROI/foco local: `[335, 36, 137, 260]`;
- a ROI local representa aproximadamente `23,8%` da área global;
- `missing_score ≈ 89,38%`;
- cobertura local ≈ `46,34%`;
- residual médio ≈ `69,30%`;
- exposição de fundo local ≈ `47,10%`;
- envelope global com `background_exposure ≈ 5,69%`;
- massa escura do gabarito ≈ `43,38%`;
- massa escura do teste ≈ `50,52%`;
- retenção de massa escura ≈ `116,48%`;
- perfil invariável horizontal ≈ `90,56%`;
- perfil invariável vertical ≈ `92,52%`;
- `missing_global_envelope_invariant_support=True`;
- melhor memória KNN = `OK` com ≈ `89,34%`;
- melhor NG ≈ `87,47%`;
- sem conflito de memória.

A testemunha rígida de corpo não passou, porém a geometria local permaneceu
fortemente ocupada:

- `silhouette_dice ≈ 73,97%`;
- `area_ratio ≈ 88,58%`;
- `centroid_shift ≈ 8,59%`;
- `box_width_ratio = 1,0`;
- `box_height_ratio = 1,0`.

Mesmo com esses sinais, a ROI local encerrava o caso como ausência física forte
e o dual-scale não era executado porque a escala local já havia confirmado
`hard missing`.

#### Correção arquitetural

Foi adicionada uma testemunha composta de presença para `FALTANDO`.

Ela não altera os limiares rígidos existentes e não baixa o corte global da
memória. A nova rota só pode atuar quando todas as condições abaixo forem
verdadeiras:

- a ROI local ocupa no máximo `25%` do envelope global;
- `missing_global_envelope_invariant_support=True`;
- `silhouette_dice >= 70%`;
- `area_ratio` entre `70%` e `135%`;
- `centroid_shift <= 10%`;
- razões de largura e altura entre `75%` e `133%`.

Contrato:

```text
ROI local pequena (<= 25% do envelope)
        +
massa física global invariável preservada
        +
ocupação geométrica local ainda coerente
        ↓
missing_invariant_occupancy_support = True
missing_invariant_occupancy_veto = True
        ↓
hard missing local perde autoridade especial
        ↓
divergência física permanece registrada
        ↓
decisão retorna à fusão normal física + memória
```

A nova testemunha não declara `OK` diretamente. Ela apenas impede que uma ROI
local pequena represente sozinha o desaparecimento do componente inteiro.

No vetor deste evento, após o rebaixamento do hard missing, a memória OK de
aproximadamente `89,34%` volta a participar como `best_match_intermediate`.
Como o peso da memória é superior a 80%, o resultado esperado é
`FALHA FALSA` sem alterar o corte de 90% da regra
`hard_missing_invariant_presence_ok_witness`.

#### Segurança

A presença composta exige simultaneamente evidência global e ocupação local.
Uma massa escura global preservada sem geometria local coerente não pode vetar
hard missing. Isso protege componentes realmente removidos que deixam footprint
escuro.

A telemetria adicionada inclui:

- `missing_invariant_occupancy_support`;
- `missing_invariant_occupancy_veto`;
- `missing_invariant_occupancy_reason`;
- `missing_local_global_area_ratio` mesmo quando o dual-scale não executa.

Status em 05/10/2026: **correção implementada e coberta por regressões;
aguardando validação operacional nesta peça antes de considerar o caso validado**.

### Validação operacional — testemunha OK quase exata em FALTANDO em 02/10/2026

Foi validado em uso real um caso da categoria `FALTANDO` em que o detector
físico bruto levantava `missing_hard_absence=True`, porém a memória encontrava
uma ocorrência OK praticamente idêntica ao evento atual.

No caso validado:

- melhor similaridade OK ≈ `99,999999%`;
- melhor similaridade NG ≈ `87,25%`;
- sem conflito de memória efetivo;
- o hard missing bruto continuou registrado para auditoria;
- o hard missing efetivo foi descartado;
- o motor dominante passou a ser o KNN;
- o veredito final passou corretamente para **FALHA FALSA**.

Contrato validado:

```text
raw_hard_missing_evidence = True
hard_missing_evidence = False
hard_missing_contradicted_by_exact_ok = True
fusion_rule = hard_missing_exact_ok_witness
dominant_engine = knn
weights = physical 0% / knn 100%
veredito = FALHA FALSA
```

A causa técnica corrigida estava na camada `best_match_memory.py`: ela recebia
novamente o `missing_hard_absence` bruto depois da fusão base e restaurava
indevidamente a prioridade física, anulando a testemunha OK quase exata.

A regra validada permanece restrita:

- aplica-se à categoria `FALTANDO`;
- exige memória OK quase exata;
- exige vantagem clara sobre a melhor hipótese NG;
- não vale para memória apenas "forte";
- não vale quando existe conflito relevante;
- não enfraquece a guarda transversal de ausência física em outras categorias;
- não altera o detector físico bruto, apenas a autoridade final de fusão.

Essa validação passa a ser a referência operacional para falsos positivos de
hard missing quando existir uma recorrência OK praticamente idêntica.

### Regra de consistência visual — hard missing não pode reaparecer como revisão

Foi identificado um segundo ponto após a proteção de hard missing: a camada de
decisão podia corretamente concluir `missing_hard_absence`, zerar o peso do
KNN e remover a revisão, enquanto a camada visual de memória ainda lia o
`memory_conflict` bruto do KNN e sobrescrevia o texto do veredito com
`CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA`.

Essa divergência entre núcleo e interface é proibida.

Quando qualquer um dos sinais abaixo estiver presente:

- `decision_trace.hard_missing_evidence == True`;
- `decision_trace.fusion_rule == "missing_hard_absence"`;
- `decision_trace.memory.suppressed_by_hard_missing == True`;
- `detail.missing_hard_absence == True`;

a UI deve considerar:

```text
conflito bruto KNN = somente auditoria
revisão visual = False
veredito exibido = o veredito final do núcleo
```

O conflito bruto continua preservado na telemetria para investigação, mas deve
ser distinguido do conflito efetivo que possui autoridade para exigir operador.

O painel KNN deve mostrar nesse estado algo equivalente a:

```text
AUSÊNCIA FÍSICA FORTE
KNN SOMENTE AUDITORIA
```

e nunca `REVISÃO OBRIGATÓRIA`.

A camada `memory_status_model.py` é responsável por separar:

- `raw_conflict`: conflito bruto calculado pela memória;
- `conflict`: conflito efetivo para apresentação;
- `raw_review_required`: pedido bruto de revisão da memória;
- `review_required`: revisão efetiva;
- `hard_missing_override`: autoridade física que suprimiu a memória.

A telemetria do hard missing deve preservar também `best_ok_similarity`,
`best_ng_similarity`, `hypothesis_margin` e `memory_conflict`, mesmo quando
o KNN recebe peso zero, para que o debug continue auditável.

Regressão obrigatória: hard missing + conflito KNN bruto deve manter
`DEFEITO REAL`, confiança 99%, zero peso KNN e nenhuma mensagem visual de
revisão obrigatória.


## Guarda transversal de ausência física — categoria AOI não define a realidade visual

### Problema observado em 01/10/2026 — EMBORCADO com componente ausente

Foi registrado o evento `1ef8605d905e4fac998cee708c89cbf6`.
A AOI classificou a ocorrência como `EMBORCADO`, mas visualmente o componente
do gabarito estava presente e a imagem de teste mostrava somente a região/footprint
onde o componente deveria existir.

Antes desta correção:

```text
Categoria AOI = EMBORCADO
        ↓
MissingComponentExpert não executava
        ↓
motores físicos = defeito forte
        ↓
KNN encontra OK parecido com ~90,6%
        ↓
best_match_strong
        ↓
peso KNN = 100%
        ↓
FALHA FALSA
```

O debug desse caso mostrava:

- `physical_score = 1.0`;
- divergência estrutural ≈ 54%;
- evidência semântica ≈ 71%;
- memória OK ≈ 90,6%;
- `fusion_rule = best_match_strong`;
- `dominant_engine = knn`;
- resultado incorreto: `FALHA FALSA`.

A causa arquitetural era assumir que ausência física só poderia existir quando
o texto da AOI fosse `FALTANDO`. A categoria da AOI é um rótulo do equipamento,
não uma prova de que o componente está presente.

### Solução

Foi criado `src/core/experts/physical_absence_guard.py`.

A `PhysicalAbsenceGuard` é uma guarda visual independente da categoria
`FALTANDO`. Ela atua somente nas categorias:

- `EMBORCADO`;
- `DESLOCADO`;
- `INVERTIDO`.

`MUITO ADESIVO` fica explicitamente fora desta guarda porque possui física e
região de interesse próprias. `FALTANDO` continua usando seu especialista
dedicado e não passa pela guarda transversal.

A guarda não renomeia a categoria. Um evento `EMBORCADO` continua sendo
persistido, consultado e auditado como `EMBORCADO`.

### Contrato mais restritivo

Fora de `FALTANDO`, a ausência física só é aceita quando todas as condições
abaixo são satisfeitas:

- score da guarda >= 82%;
- cobertura alterada >= 45%;
- residual médio >= 38%;
- perda de aparência >= 40%;
- similaridade direta <= 60%;
- incompatibilidade de bordas >= 40%;
- melhor correspondência próxima < 25%;
- classificação não pode ser `DESLOCAMENTO PROVÁVEL`;
- comparador estrutural >= 35%;
- motor semântico >= 60%;
- score físico agregado >= 85%.

A concordância entre a guarda visual, o comparador estrutural e o motor
semântico é obrigatória. Assim um exemplo OK da memória não é ignorado apenas
porque uma única métrica visual subiu.

### Reprodução do caso real

Usando o mesmo frame e a mesma ROI do evento EMBORCADO, a guarda transversal
produziu aproximadamente:

- score da guarda: 84,9%;
- cobertura: 52,4%;
- residual médio: 40,8%;
- perda de aparência: 46,2%;
- similaridade direta: 53,8%;
- incompatibilidade de bordas: 51,0%;
- melhor correspondência próxima: 9,3%;
- estrutural da decisão original: 54%;
- semântico da decisão original: 71%;
- score físico agregado: 100%.

Esse vetor satisfaz o contrato transversal e caracteriza desaparecimento físico
mesmo que a AOI tenha usado o rótulo `EMBORCADO`.

### Hierarquia da decisão

Quando a guarda transversal confirma `missing_hard_absence=True`:

```text
categoria original permanece EMBORCADO/DESLOCADO/INVERTIDO
        ↓
ausência física forte confirmada
        ↓
fusion_rule = missing_hard_absence
        ↓
motor dominante = missing
        ↓
peso físico = 100%
peso KNN = 0%
        ↓
KNN continua visível somente para auditoria
        ↓
DEFEITO REAL / NG
confiança = 99%
```

Essa regra agora também existe na fusão base, antes da ponderação da memória.
Portanto a proteção não depende da ordem de instalação de wrappers de KNN.

### Memória continua isolada por categoria

A guarda transversal não pode:

- mudar `EMBORCADO` para `FALTANDO`;
- consultar memória de outra categoria;
- adicionar `missing_mask` à assinatura KNN de EMBORCADO/DESLOCADO/INVERTIDO;
- treinar o KNN como se o evento pertencesse a FALTANDO.

A memória continua usando a categoria AOI original. A guarda é somente uma
trava física de segurança contra veto incorreto de um desaparecimento
inequívoco.

### Debug obrigatório

Quando a guarda transversal for avaliada, `Copiar debug XP` deve registrar:

- `missing_cross_category_guard`;
- `missing_guard_policy`;
- `missing_guard_source_category`;
- `missing_guard_physical_support`;
- todas as métricas de ausência física;
- se o KNN foi suprimido por hard missing.

O debug deve permitir distinguir um `EMBORCADO` aprendido normalmente pela
memória de um evento rotulado como `EMBORCADO` pela AOI, mas com componente
fisicamente ausente.

### Regressões obrigatórias

Manter testes que garantam:

- vetor real do evento EMBORCADO ausente → guarda transversal confirma ausência;
- memória OK ~90,6% não veta ausência transversal;
- fusão base aplica `missing_hard_absence` antes do peso KNN;
- match próximo plausível bloqueia o override;
- suporte semântico/estrutural fraco bloqueia o override;
- `DESLOCAMENTO PROVÁVEL` nunca é convertido pela guarda;
- `MUITO ADESIVO` e `FALTANDO` não usam essa guarda;
- máscara da guarda transversal não altera assinatura KNN da categoria original.


## INVERTIDO — fusão única, conflito de memória e colapso visual extremo

### Caso real de 01/10/2026

Evento: `299285f21c8448b0bff8a9757fa382af`.

A AOI classificou a ocorrência como `INVERTIDO`, mas visualmente o corpo do
componente presente no gabarito desapareceu no teste. O debug mostrou:

- score físico ≈ 85,7%;
- `missing_score ≈ 94,5%`;
- cobertura alterada ≈ 75,6%;
- residual médio ≈ 59,6%;
- perda de aparência ≈ 66,2%;
- similaridade direta ≈ 33,8%;
- melhor correspondência próxima ≈ 14,6%;
- estrutural ≈ 45,7%;
- semântico ≈ 57,1%;
- match KNN OK ≈ 87,7%.

A guarda transversal executou, mas a rota primária de hard missing não passou
porque o semântico ficou abaixo de 60% e a incompatibilidade de bordas ficou
abaixo de 40%. Esse padrão é típico de um epicentro estreito: quase todo o
conteúdo esperado desaparece, porém parte das bordas locais permanece.

### Rota de colapso visual extremo

A `PhysicalAbsenceGuard` possui uma segunda rota para esse caso. Ela exige
simultaneamente:

- score >= 92%;
- cobertura alterada >= 70%;
- residual médio >= 55%;
- perda de aparência >= 60%;
- similaridade direta <= 40%;
- incompatibilidade de bordas >= 30%;
- melhor correspondência próxima < 20%;
- estrutural >= 40%;
- semântico >= 55%;
- score físico agregado >= 85%;
- classificação diferente de `DESLOCAMENTO PROVÁVEL`.

Essa rota não substitui a rota primária. Ela existe somente para colapso visual
extremo, onde a quantidade de evidências independentes compensa um epicentro
local estreito.

### Causa arquitetural descoberta no INVERTIDO

O módulo `inverted_face_integration.py` mantinha uma segunda implementação de
fusão e importava `_dynamic_fusion` diretamente no carregamento do módulo.
Depois, `best_match_memory` e `memory_hypothesis_contrast` substituíam a
função de fusão no módulo central, mas o módulo INVERTIDO continuava apontando
para a referência antiga.

Isso permitia este fluxo incorreto:

```text
fusão central atualizada
        ↓
hard missing / best-match / conflito OK×NG tratados
        ↓
INVERTIDO executa uma segunda fusão privada antiga
        ↓
resultado e telemetria podem divergir
```

Essa duplicação foi removida.

### Regra obrigatória de arquitetura

Nenhum especialista de categoria pode possuir uma segunda implementação de
pesos KNN/físico.

`INVERTIDO`, `EMBORCADO`, `DESLOCADO`, `FALTANDO` e futuras categorias
devem terminar na mesma função de fusão central.

O módulo INVERTIDO agora:

1. calcula a assinatura específica da face;
2. adiciona suas métricas ao `detail`;
3. consulta a memória da categoria INVERTIDO;
4. chama dinamicamente `anomaly_memory_module._dynamic_fusion`;
5. herda automaticamente best-match, contraste OK×NG, hard missing e futuras
   extensões da fusão.

Não importar `_dynamic_fusion` por valor para manter uma referência antiga.

### Conflito bruto x conflito efetivo no debug

Foi identificada outra divergência: a UI podia ler `detail.memory_conflict`
enquanto o debug lia apenas `decision_trace.memory.memory_conflict`. Quando o
trace não continha a chave, o debug mostrava `False` por padrão mesmo que a UI
estivesse mostrando `CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA`.

O debug deve agora registrar separadamente:

- `raw_memory_conflict`: conflito produzido pela comparação de memórias;
- `memory_conflict`: conflito efetivo depois das regras de autoridade física;
- `raw_operator_review_required`: revisão pedida pela memória;
- `operator_review_required`: revisão efetiva;
- `suppressed_by_hard_missing`: indica que a ausência física retirou o poder
  de veto/revisão do KNN.

O debug e a UI devem sempre concordar sobre o estado efetivo.

### Regressões obrigatórias

Manter testes que garantam:

- vetor real INVERTIDO de colapso extremo → hard missing verdadeiro;
- `INVERTIDO` usa a fusão central atual, nunca uma cópia privada antiga;
- hard missing continua ativo depois do especialista INVERTIDO;
- memória OK forte não veta ausência física;
- conflito bruto pode permanecer na auditoria, mas não cria revisão efetiva sob
  hard missing;
- debug não pode reportar `memory_conflict=False` quando a UI estiver usando
  um conflito bruto verdadeiro sem supressão;
- assinatura e memória continuam isoladas pela categoria original.


### Caso observado em 06/10/2026 — ROI testemunha mínima não pode significar inversão por si só

Evento: `bc8b227c749241fcb72fd7e2fd7e0a48`.

A AOI classificou a ocorrência como `INVERTIDO`, porém a peça era OK e deveria
ser tratada como falha falsa após confirmação do operador.

O debug mostrou que o problema não vinha de `FALTANDO` nem da guarda
transversal de ausência física:

- `missing_hard_absence=False`;
- dual-scale executado;
- contexto não confirmou ausência física forte;
- similaridade direta contextual ≈ `84,5%`.

A decisão incorreta estava concentrada no motor `INVERTIDO` e na forma como o
estado de conflito era apresentado:

- caixa global do componente: aproximadamente `278 × 527 px`;
- ROI testemunha: aproximadamente `125 × 44 px`;
- a ROI representa apenas ≈ `3,75%` da área global;
- score bruto do especialista INVERTIDO ≈ `67%`;
- retenção da marca ≈ `44%`;
- perda da marca/feature ≈ `58%`;
- topologia divergente ≈ `29%`;
- orientação divergente ≈ `4%`;
- melhor memória OK ≈ `90,01%`;
- melhor memória NG ≈ `89,36%`;
- margem ≈ `0,65 p.p.`;
- conflito de memória efetivo = `True`;
- revisão obrigatória = `True`.

Apesar disso, o resultado exposto era `DEFEITO REAL`.

#### Causa 1 — piso artificial de 90% no motor INVERTIDO

Na fusão central, qualquer disparo do especialista INVERTIDO era convertido em:

```text
effective_score = max(90%, inverted_score)
```

Portanto:

```text
score bruto ≈ 67%
        ↓
motor disparou
        ↓
score físico efetivo = 90%
```

Isso dava autoridade de defeito forte para uma divergência de marca localizada,
mesmo quando não existia uma segunda evidência clara de inversão.

#### Autoridade da ROI testemunha

O `InvertedWitnessExpert` passa a registrar:

- `inverted_local_global_area_ratio`;
- `inverted_small_witness_roi`;
- `inverted_high_authority`;
- `inverted_corroborated`;
- `inverted_corroboration_reason`.

Uma ROI é considerada pequena para fins de autoridade quando ocupa no máximo
`10%` da caixa global detectada.

Uma ROI pequena só mantém a antiga autoridade física elevada quando existe ao
menos um corroborador forte independente:

1. orientação >= `20%` **e** topologia >= `20%`;
2. sinal de deslocamento >= `40%`;
3. ganho de transformação >= `10%` com similaridade transformada >= `56%`;
4. face alternativa >= `58%`;
5. perda da marca >= `62%`;
6. perda estrutural da marca >= `65%`.

Sem esses corroboradores:

```text
ROI testemunha <= 10% do componente
        +
marca local divergiu
        +
sem evidência forte independente de inversão
        ↓
inverted_high_authority = False
        ↓
não aplicar piso físico de 90%
        ↓
usar o score bruto real do especialista
```

Essa regra **não desativa** o motor INVERTIDO. A divergência continua registrada
e pode participar da fusão. A mudança remove apenas a amplificação automática
de uma evidência local ambígua.

ROIs maiores continuam com o comportamento anterior. ROIs pequenas com
corroboração forte também preservam o piso físico existente.

#### Causa 2 — revisão obrigatória não é um veredito binário

A camada de contraste de memória já concluía corretamente:

```text
fusion_rule = memory_conflict_operator_review
operator_review_required = True
confidence = 50%
```

e a justificativa informava que a decisão automática estava bloqueada.

Mesmo assim, as integrações convertiam `is_defect=True` diretamente em
`DEFEITO REAL`, criando uma contradição semântica.

Foi criado o contrato:

```text
operator_review_required = True
        ↓
verdict = REVISÃO OBRIGATÓRIA
```

Somente quando não houver revisão pendente:

```text
is_defect=True  → DEFEITO REAL
is_defect=False → FALHA FALSA
```

A função central `resolved_analysis_verdict()` é a fonte desse mapeamento nas
integrações de memória e INVERTIDO.

O estado de revisão:

- não significa OK;
- não significa NG;
- não pode ser apresentado como `FALHA FALSA` nem `DEFEITO REAL`;
- deve exibir no overlay superior a mensagem `REVISÃO OBRIGATÓRIA` em vermelho;
- deve permanecer amarelo no painel técnico de decisão, distinguindo alerta operacional de veredito NG;
- mantém `0=OK` e `1=NG` disponíveis para o operador;
- continua preservando `is_defect` e scores brutos para auditoria interna.

No vetor deste evento, a memória continua em conflito por margem inferior a
`1 p.p.`. Portanto o resultado automático esperado após esta correção é
**REVISÃO OBRIGATÓRIA**, e não um OK artificial.

Ao o operador confirmar `0=OK`, a ocorrência pode fortalecer a hipótese OK da
categoria INVERTIDO para recorrências futuras.

#### Debug obrigatório para INVERTIDO

O `Copiar debug` passa a expor explicitamente:

- `inverted_score`;
- classificação;
- retenção/perda da testemunha;
- perda de feature;
- topologia;
- orientação;
- face alternativa;
- transformação e similaridade;
- relocação;
- razão ROI/global;
- se a ROI é pequena;
- se possui alta autoridade;
- se existe corroborador;
- motivo da autoridade/corroborador.

Isso permite distinguir uma marca local diferente de uma evidência física
corroborada de face invertida.

#### Regressões obrigatórias

Manter testes que garantam:

- ROI ≈ `3,75%` do componente, orientação ≈ `4%`, perdas moderadas e sem
  corroborador → `inverted_high_authority=False`;
- esse vetor não recebe piso físico automático de `90%`;
- ROI pequena com orientação + topologia fortes mantém alta autoridade;
- evidência INVERTIDO corroborada continua preservando o piso físico existente;
- conflito OK × NG sem hard missing efetivo → `REVISÃO OBRIGATÓRIA`;
- revisão obrigatória nunca é apresentada como `DEFEITO REAL` ou
  `FALHA FALSA`;
- hard missing verdadeiro continua tendo precedência e não vira revisão;
- debug mostra os campos de autoridade do INVERTIDO.

Status em 06/10/2026: **correção implementada; aguardando validação operacional
nesta mesma peça**.

## Dual-Scale Presence — epicentro local + contexto físico do componente

### Motivação

Foi identificado que a ROI/epicentro fornecida pela AOI pode representar apenas
uma pequena fração do componente. Nesses casos, uma região interna escura pode
continuar parecida mesmo depois que o corpo inteiro do componente desapareceu.

Evento real de referência: `912c92754da3433d8a2a0980052e2b78`, categoria
`FALTANDO`, em 01/10/2026.

O evento apresentava:

- caixa global aproximadamente `547 × 261 px`;
- caixa de foco aproximadamente `140 × 108 px`;
- área local equivalente a aproximadamente 10,6% da área global;
- `missing_score` local ≈ 34,9%;
- cobertura local ≈ 4,1%;
- similaridade direta local ≈ 87,5%;
- classificação local: `ROI CONFORME`;
- score físico agregado ≈ 85%;
- divergência estrutural ≈ 56%;
- evidência semântica ≈ 56%;
- KNN encontrou memória `OK` ≈ 89,4%;
- consequência antiga: `best_match_intermediate` com KNN dominante e
  resultado incorreto `FALHA FALSA`.

O problema não era somente calibração de thresholds. A escala local estava
respondendo à pergunta 'este pequeno patch ainda parece parecido?', enquanto a
decisão de presença precisava responder 'o componente como um todo ainda
existe?'.

### Arquitetura

Foi criado:

`src/core/experts/dual_scale_presence.py`

A `DualScalePresenceAnalyzer` adiciona uma segunda escala sem remover o
epicentro original.

```text
imagem AOI
    ↓
epicentro / ROI local
    ├── escala local: precisão do defeito
    │
    └── quando necessário
         ↓
       ROI contextual expandida
         ↓
       presença física do componente
         ↓
       hard missing?
         ├── SIM → KNN somente auditoria
         └── NÃO → fluxo normal de fusão/memória
```

Essa funcionalidade deve ser chamada de **Dual-Scale Presence** para não ser
confundida com `dual_scale_memory.py`, que trata da representação/memória KNN.

### Quando a segunda escala é executada

A análise contextual não roda indiscriminadamente em toda inspeção.

Ela é ativada quando:

1. a área do epicentro representa <= 25% da área global do componente; ou
2. a escala local diz que não há defeito, mas os motores estrutural/semântico
   apresentam suporte físico independente suficiente.

O estado de contradição é registrado como `missing_scale_disagreement`.

Para suporte físico contextual, a política atual exige:

- estrutural >= 45% E semântico >= 45%; ou
- um dos dois >= 65%.

### Construção da ROI contextual

A ROI contextual é centrada no epicentro e cresce sem sair da imagem:

- expansão mínima em torno do foco: 2,5× por eixo;
- referência mínima adicional: 55% das dimensões da caixa global;
- largura/altura finais limitadas ao recorte real recebido.

No evento de referência:

```text
local:    140 × 108
contexto: 350 × 270
```

Essa expansão mantém o epicentro como âncora, mas passa a incluir o corpo do
componente e sua relação com pads/footprint ao redor.

### Métricas contextuais

A segunda escala reutiliza o mesmo pipeline determinístico de visão, porém em
uma caixa maior. Ela registra:

- `missing_context_score`;
- `missing_context_coverage`;
- `missing_context_residual_mean`;
- `missing_context_residual_p90`;
- `missing_context_structure_loss`;
- `missing_context_edge_mismatch`;
- `missing_context_direct_similarity`;
- `missing_context_appearance_loss`;
- `missing_context_best_similarity`;
- `missing_context_box`;
- `missing_local_global_area_ratio`;
- `missing_context_hard_absence`;
- `missing_context_hard_reason`.

### Contrato para ausência contextual forte

A rota contextual normal exige simultaneamente:

- score contextual >= 72%;
- cobertura contextual >= 30%;
- residual médio contextual >= 50%;
- perda de aparência contextual >= 35%;
- melhor correspondência próxima < 35%;
- perda estrutural >= 20% OU incompatibilidade de bordas >= 30%;
- suporte físico independente estrutural/semântico conforme regra acima.

Existe também uma rota contextual extrema, reservada para desaparecimento
inequívoco mesmo quando o suporte global não estiver disponível:

- score >= 85%;
- cobertura >= 45%;
- residual médio >= 60%;
- perda de aparência >= 50%;
- melhor correspondência próxima < 25%.

A rota extrema não deve ser usada para simples diferença parcial ou
deslocamento.

### Reprodução do evento real

Com o mesmo frame e a mesma geometria do evento
`912c92754da3433d8a2a0980052e2b78`, a análise contextual produz
aproximadamente:

- score contextual ≈ 79,5%;
- cobertura ≈ 38,8%;
- residual médio ≈ 68,8%;
- P90 ≈ 84,0%;
- perda estrutural ≈ 32,7%;
- incompatibilidade de bordas ≈ 39,2%;
- similaridade direta ≈ 56,2%;
- perda de aparência ≈ 43,8%;
- melhor correspondência próxima ≈ 19,8%.

Com estrutural ≈ 56% e semântico ≈ 56%, esse vetor confirma ausência física
contextual, apesar de a pequena ROI local ter sido classificada como conforme.

### Autoridade sobre memória

Quando `missing_context_hard_absence == True`, o resultado é promovido para:

```text
missing_hard_absence = True
fusion_rule = missing_hard_absence
motor dominante = missing
peso físico = 100%
peso KNN = 0%
confidence = 0.99
```

A memória continua sendo consultada e exibida para auditoria, mas não pode
anular uma ausência física contextual confirmada.

Isso vale tanto para o especialista dedicado de `FALTANDO` quanto para a
`PhysicalAbsenceGuard` transversal de `EMBORCADO`, `DESLOCADO` e `INVERTIDO`.

### Isolamento e segurança

A Dual-Scale Presence:

- não altera a categoria recebida da AOI;
- não mistura memórias entre categorias;
- não injeta máscara contextual na assinatura KNN transversal;
- não muda regras de adesivo;
- não substitui o epicentro local;
- não transforma automaticamente todo `FALTANDO` em NG;
- somente ganha autoridade quando o contrato contextual de ausência física é
  satisfeito.

Se o contexto não confirmar ausência, a decisão continua no fluxo normal com
motores físicos, KNN, contraste OK×NG e revisão humana quando aplicável.

### Debug obrigatório

`Copiar debug XP` deve registrar:

- política dual-scale;
- se a análise contextual foi ativada/executada;
- razão área local/global;
- desacordo entre escalas;
- caixa contextual;
- score/cobertura/residual contextual;
- perda estrutural e incompatibilidade de bordas contextual;
- similaridade/perda de aparência contextual;
- melhor match próximo contextual;
- resultado de hard absence contextual;
- suporte físico independente e motivo final.

O debugger visual de presença deve mostrar explicitamente `LOCAL + CONTEXTO` e
as principais métricas contextuais quando a segunda escala tiver sido
executada.

### Regressões obrigatórias

Manter testes que garantam:

- geometria real 140×108 dentro de 547×261 dispara a segunda escala;
- evento real reproduzido possui razão local/global ≈ 10,6%;
- contexto do evento real confirma hard missing;
- contexto sem suporte físico suficiente não promove ausência, exceto pela
  rota extrema;
- match próximo plausível bloqueia hard missing contextual;
- FALTANDO localmente conforme pode ser promovido por contexto confirmado;
- guarda transversal pode ser promovida pelo contexto;
- hard missing contextual continua zerando o peso KNN na fusão;
- debug e UI expõem claramente as duas escalas.



### Regressão de segurança — DESLOCADO presente não pode virar hard missing

Em 01/10/2026 foi registrado o evento `4128dec4a02f423fbdbcd47fca777108` da categoria `DESLOCADO`. Visualmente o componente `104` estava presente tanto no gabarito quanto no teste; a ocorrência foi validada pelo operador como **falha falsa** do ODIN.

O debug anterior mostrava:

- `missing_score ≈ 98,9%`;
- cobertura local ≈ 78,8%;
- residual local ≈ 70,1%;
- dual-scale contextual ≈ 97,8%;
- cobertura contextual ≈ 69,8%;
- estrutural independente ≈ 48,7%;
- semântico independente ≈ 30,1%;
- `missing_context_physical_support.supported = False`;
- melhor memória KNN `OK ≈ 98,18%`;
- melhor memória NG ≈ 89,76%;
- resultado incorreto antigo: `missing_hard_absence=True`, KNN suprimido e `DEFEITO REAL`.

A causa era a rota contextual de **colapso visual extremo** permitir hard missing mesmo quando os motores físicos independentes não confirmavam a ausência. Em uma categoria de deslocamento, comparar referência e teste em posições fixas pode produzir grande residual apenas porque o mesmo componente mudou de posição/registro.

#### Regra corrigida

Para a guarda transversal usada por `EMBORCADO`, `DESLOCADO` e `INVERTIDO`:

- a rota extrema do Dual-Scale Presence **não pode** promover `missing_hard_absence` sem `missing_context_physical_support.supported=True`;
- `FALTANDO` mantém sua política própria e pode continuar usando a rota contextual extrema conforme seu contrato dedicado;
- a categoria original continua preservada;
- memória KNN continua sendo consultada normalmente quando o hard missing transversal é bloqueado.

No evento de referência, como o suporte transversal era falso, a ausência física deve permanecer falsa e o KNN `OK ≈ 98,18%` volta a participar da fusão. O resultado esperado é **FALHA FALSA / OK**, sem suprimir a memória.

Essa proteção existe para impedir que **deslocamento, mudança de registro, pequena variação geométrica ou iluminação** sejam confundidos com desaparecimento físico apenas porque a comparação fixa local/contextual apresenta residual alto.

Regressões obrigatórias:

- evento DESLOCADO presente + suporte físico transversal falso → `missing_hard_absence=False`;
- rota extrema transversal sem suporte independente → não promove ausência;
- KNN OK forte permanece elegível quando o hard missing foi corretamente bloqueado;
- os casos reais anteriores de ausência física em EMBORCADO/INVERTIDO continuam passando quando possuem suporte físico independente suficiente.

#### Validação operacional da correção

Em 01/10/2026, após o endurecimento da guarda transversal, o mesmo fluxo foi
retestado na AOI real e o operador confirmou que o comportamento ficou correto.

Esse resultado valida especificamente a regra introduzida para o evento
`4128dec4a02f423fbdbcd47fca777108`:

- componente presente em categoria `DESLOCADO`;
- suporte físico transversal insuficiente;
- `missing_hard_absence` não deve ser promovido;
- memória `OK` forte continua elegível;
- o resultado final deve permanecer `FALHA FALSA / OK`.

Essa validação deve ser preservada como regressão operacional. Qualquer mudança
futura no Dual-Scale Presence ou na `PhysicalAbsenceGuard` não pode reintroduzir
o comportamento antigo de transformar deslocamento/variação de registro em
ausência física forte sem confirmação independente.

### Validação operacional em AOI real

Em 01/10/2026, após a implementação da **Dual-Scale Presence**, o fluxo foi
retestado na AOI real e o operador confirmou que o comportamento esperado
funcionou corretamente.

Essa validação operacional complementa as regressões automatizadas e deve ser
preservada como referência de engenharia para futuras alterações no motor de
presença.

Regra de manutenção:

- não remover a análise contextual apenas porque a ROI local apresenta alta
  similaridade;
- não retornar ao modelo de decisão baseado exclusivamente no epicentro para
  presença/ausência física;
- qualquer refatoração futura deve preservar a hierarquia:
  `ROI local → contexto quando necessário → hard missing → KNN somente
  auditoria`;
- alterações de thresholds devem manter as travas contra
  `DESLOCAMENTO PROVÁVEL`, match próximo plausível e ausência de suporte físico;
- se um caso futuro voltar a produzir `FALHA FALSA` com componente fisicamente
  ausente, registrar o debug completo e verificar primeiro se o dual-scale foi
  ativado, qual caixa contextual foi usada e qual métrica bloqueou o
  `missing_context_hard_absence`.


## Identidade visual da interface — ODIN

A identidade exibida ao operador foi padronizada para:

```text
ODIN - Observador Digital Inteligente
```

Essa é uma alteração **visual/de apresentação**. Ela não renomeia o repositório,
schemas, módulos Python, arquivos de pesos, caminhos, protocolos de rede nem
identificadores persistidos que já usam o nome técnico `visionx`.

Superfícies visuais obrigatórias:

- título da janela principal:
  `ODIN - Observador Digital Inteligente - Monitoramento IA`;
- título principal do cabeçalho:
  `ODIN - Observador Digital Inteligente`;
- HUD inicial:
  `ODIN - Observador Digital Inteligente: Inicializando...`;
- janela de calibração:
  `ODIN - Observador Digital Inteligente - Calibrar Zona de Interesse Avançado`;
- relatório técnico copiado pela interface:
  `ODIN - Observador Digital Inteligente - DEBUG DE ENTRADA WINDOWS XP`;
- seção de decisão do relatório:
  `DECISÃO ODIN - Observador Digital Inteligente`.

A fonte única da identidade visual fica em:

```text
src/ui/branding.py
```

Não espalhar novamente strings de marca diretamente pelos widgets. Novas telas
devem reutilizar as constantes de `branding.py`.

### Regra de compatibilidade

Manter inalterados, salvo migração específica e planejada:

- `visionx.network_xp_debug.v1`;
- nome do repositório `visionx-neural`;
- nomes de arquivos/pesos como `visionx_neural_weights.pth`;
- nomes de classes, módulos e APIs já existentes;
- protocolos TCP e comandos usados pelo agente Windows XP.

A troca para ODIN não pode quebrar integração, persistência ou histórico.

### Responsividade da marca

Como `ODIN - Observador Digital Inteligente` é maior que o nome anterior, o
título principal deve aceitar quebra de linha e largura mínima zero para
continuar responsivo em notebooks e monitores menores.


## Fundo neutro permanente do ODIN

O fundo geral do ODIN permanece **sempre no tema escuro neutro**, independentemente
do veredito final da IA.

Contrato atual:

```text
sem análise
→ fundo neutro

FALHA FALSA / OK
→ fundo neutro

DEFEITO REAL / NG
→ fundo neutro

REVISÃO OBRIGATÓRIA
→ fundo neutro
```

O fundo não comunica mais estado operacional. A comunicação visual do resultado
fica concentrada no card de veredito do canto superior direito.

Cores neutras de referência:

- canvas: `#050505`;
- superfícies principais: `#0d0d0d`.

### Alteração de contrato em 02/10/2026

O comportamento anterior, já validado em operação, fazia o fundo principal
mudar para verde-escuro em `FALHA FALSA` e vermelho-escuro em
`DEFEITO REAL`.

Esse comportamento foi **substituído deliberadamente**.

A partir desta alteração:

- o fundo não acompanha `analysis["is_defect"]`;
- o `main.py` não instala mais `install_decision_background(ControlPanel)`;
- o módulo `src/ui/decision_background.py` permanece apenas como compatibilidade
  defensiva e normaliza qualquer solicitação para `neutral`;
- verde/vermelho continuam permitidos no texto do card de veredito;
- botões, hover, focus, checked e demais componentes mantêm o tema original.

A decisão de remover o fundo dinâmico evita que grandes áreas coloridas disputem
atenção com a inspeção e deixa o veredito explícito em um único elemento visual.

### Regra crítica de arquitetura

Nenhum veredito pode alterar a cor geral do canvas ou das superfícies principais.

Regressões obrigatórias:

- `None` → `neutral`;
- `is_defect=False` → `neutral`;
- `is_defect=True` → `neutral`;
- uma chamada defensiva `apply_decision_background(..., "ok")` ou
  `apply_decision_background(..., "ng")` também resulta em `neutral`;
- `main.py` não instala o hook de fundo dinâmico;
- o stylesheet global não é substituído;
- um `sectionPanel` permanece em `#0d0d0d` mesmo se alguém tentar aplicar
  estado NG pelo módulo legado.

## Feedback visual temporário de teclas operacionais

O ODIN possui um overlay exclusivamente visual para confirmar imediatamente ao
operador qual tecla operacional foi pressionada ou enviada.

Comportamento:

```text
0 → OK
1 → NG
← → seta esquerda / TOP
↓ → seta para baixo / SIDE
→ → seta direita / MID
```

Fontes cobertas:

- teclado do próprio ODIN: `0`, `Num+0`, `1` e `Num+1`;
- teclado físico do Windows XP recebido pela rede como `CMD_OK` ou `CMD_NG`.

A apresentação é um quadrado temporário de aproximadamente `180 × 180 px`, posicionado no **canto inferior direito** da interface, com margem aproximada de `24 px` das bordas e acima dos demais componentes.

O visual segue a identidade industrial do ODIN:

- fundo escuro `#101010`;
- borda-base discreta `#303030`;
- cabeçalho `TECLA PRESSIONADA`, `TECLA ENVIADA` ou `TECLA RECEBIDA` em amarelo ODIN `#f5c518`;
- `0 / OK`: detalhe, borda e tipografia de estado em verde `#4ade80`;
- `1 / NG`: detalhe, borda e tipografia de estado em vermelho `#ff6262`;
- origem exibida como `TECLADO ODIN` ou `TECLADO WINDOWS XP`;
- duração aproximada total: `800 ms`;
- entrada suave: fade-in + deslocamento vertical de apenas `8 px` em aproximadamente `120 ms`;
- saída suave: fade-out em aproximadamente `160 ms`;
- desaparece automaticamente.

Implementação:

```text
src/ui/decision_key_feedback.py
```

### Regra crítica de arquitetura

Esse recurso é **somente apresentação**. Ele não pode:

- alterar `save_label()`;
- enviar comandos ao XP;
- decidir OK/NG;
- modificar confiança, score ou memória KNN;
- bloquear o gate de imagens;
- gravar dataset ou evidências;
- capturar foco ou cliques do mouse;
- criar espera ativa, `sleep` ou animação pesada no caminho produtivo.

O overlay reutiliza um único widget, um único `QTimer` single-shot e animações curtas de propriedades Qt. A animação só existe enquanto o alerta está visível: não há loop, animação contínua, thread adicional ou `sleep`. O fade usa `QGraphicsOpacityEffect` somente sobre o pequeno widget de `180 × 180 px`, e o slide altera apenas sua posição em `8 px`.

Ele possui `WA_TransparentForMouseEvents` e `NoFocus`, portanto pode aparecer sobre outros componentes sem impedir interação. A escolha de animações curtas e locais é obrigatória para manter o custo de renderização desprezível diante do pipeline de visão computacional.

### Ordem de acionamento

No teclado local, o feedback aparece imediatamente **antes** do caminho normal do botão OK/NG. Assim a confirmação visual não espera o envio TCP `PRESS_0/PRESS_1`.

No teclado XP, o feedback é exibido somente quando existe uma captura ativa e o ODIN recebe `CMD_OK` ou `CMD_NG`.

### Supressão de eco visual

O agente XP pode devolver pelo hook global a mesma tecla que o ODIN acabou de enviar por `PRESS_0/PRESS_1`. Para não mostrar dois alertas para uma única decisão, repetições do mesmo julgamento dentro de aproximadamente `1,5 s` são suprimidas **somente na camada visual**.

Essa deduplicação não altera nem descarta comandos produtivos; ela apenas impede um segundo flash do overlay.

### Regressões obrigatórias

Manter testes que garantam:

- tecla local `0` aciona `0 / OK`;
- tecla local `1` aciona `1 / NG`;
- `CMD_OK` e `CMD_NG` do XP acionam o mesmo overlay com origem XP;
- botões desabilitados não são burlados pelos atalhos;
- comando XP sem captura ativa não produz confirmação visual de julgamento;
- o eco da mesma decisão não gera um segundo alerta imediato;
- o overlay permanece click-through e sem foco;
- o overlay permanece ancorado no canto inferior direito em telas de tamanhos diferentes;
- fade-in/fade-out são curtos e sem repetição;
- o movimento de entrada permanece pequeno e não desloca outros componentes;
- não existe animação em background quando o alerta está oculto;
- o recurso não altera nenhuma regra de negócio do ciclo.

### Validação operacional

Em 01/10/2026, o feedback visual `0 = OK` / `1 = NG` foi validado em operação e o comportamento esperado foi confirmado.

Após essa validação, a posição visual foi refinada do centro da tela para o canto inferior direito para reduzir interferência visual sobre a inspeção principal. Essa posição passa a fazer parte do contrato da interface.


### Refinamento visual do feedback 0/1 em 02/10/2026

O card de confirmação de tecla continua no canto inferior direito e mantém a
mesma função operacional, porém sua linguagem visual passa a ficar mais próxima
do restante do ODIN:

- fundo escuro próximo ao `SURFACE`;
- borda amarela ODIN, em vez de usar verde/vermelho como moldura principal;
- cabeçalho continua amarelo;
- verde/vermelho ficam restritos ao conteúdo do estado `OK` / `NG`;
- animação, posição, click-through, duração e deduplicação de eco permanecem
  inalterados.

O objetivo é evitar um alerta visual que pareça pertencer a outro sistema ou a
um componente genérico de IA.

## Feedback visual do veredito final da IA

Além do feedback de tecla `0/1`, o ODIN possui um segundo overlay exclusivamente
visual para comunicar o veredito final calculado pela IA.

Esse overlay tem função diferente do feedback de tecla:

```text
feedback 0/1
= confirma ação/comando recebido

feedback de veredito
= comunica o resultado final da análise da IA
```

### Mensagens permitidas

O card de veredito deve exibir somente:

```text
FALHA FALSA
```

ou:

```text
DEFEITO REAL
```

Não exibir nesse card:

- porcentagem;
- score;
- confiança;
- motor dominante;
- regra de fusão;
- categoria;
- justificativa;
- origem da tecla.

Essas informações continuam disponíveis nos painéis detalhados e no debug.

### Posição

O overlay do veredito fica no **canto superior direito** da janela principal.

Essa escolha é deliberada:

- separa visualmente o veredito da IA do feedback de tecla `0/1`, que fica no
  canto inferior direito;
- não compete com a área central de decisão;
- segue um padrão natural de notificação sem ocupar o centro da inspeção;
- mantém leitura imediata em telas largas.

Contrato atual:

- largura aproximada: `300 px`;
- altura aproximada: `88 px`;
- margem direita: `24 px`;
- offset superior aproximado: `84 px`.

### Linguagem visual

O card deve seguir a identidade industrial do ODIN:

- base escura `SURFACE`;
- borda fina amarela `ACCENT`;
- nenhum texto auxiliar: o único texto renderizado é o próprio veredito;
- `FALHA FALSA` em verde;
- `DEFEITO REAL` em vermelho;
- tipografia forte e limpa;
- sem ícones decorativos genéricos;
- sem pills;
- sem gradientes;
- sem barras laterais coloridas;
- sem aparência de componente gerado por IA.

O verde/vermelho é usado apenas para o texto do estado. A estrutura do card
continua escura/amarela. Não exibir cabeçalho, subtítulo ou legenda dentro do
card: a moldura amarela já comunica que o elemento pertence ao ODIN.

### Relação com o fundo neutro

O card de veredito é agora o **único elemento global de alto nível** que comunica
o resultado binário da IA.

```text
FALHA FALSA
        ↓
fundo permanece neutro
        +
card superior direito com texto verde

DEFEITO REAL
        ↓
fundo permanece neutro
        +
card superior direito com texto vermelho
```

Não existe mais dependência de ordem entre overlay e fundo dinâmico, porque o
fundo não reage ao resultado.

Quando o ODIN volta para `AGUARDANDO PEÇA`, o card deve desaparecer e o fundo
continua neutro.

### Persistência, entrada e saída sincronizada

Enquanto existe uma análise ativa, o card de veredito permanece fixo no canto
superior direito.

Entrada:

- fade-in curto de aproximadamente `140 ms`;
- pequeno slide horizontal de aproximadamente `10 px`;
- depois da entrada, permanece estático e visível;
- não existe `QTimer` próprio de auto-hide;
- sem thread adicional;
- sem `sleep`;
- sem loop contínuo;
- `WA_TransparentForMouseEvents`;
- `NoFocus`.

#### Saída sincronizada com o julgamento 0/1

Quando o operador confirma a decisão por `0 / OK` ou `1 / NG`, o card de
veredito não pode desaparecer imediatamente por causa do reset interno do ciclo.

O contrato é:

```text
operador pressiona 0 ou 1
        ↓
feedback 0/1 aparece
        +
card FALHA FALSA / DEFEITO REAL é marcado para saída
        ↓
ciclo produtivo pode salvar/resetar normalmente
        ↓
card de veredito permanece visível durante a confirmação 0/1
        ↓
feedback 0/1 inicia fade-out
        ↓ mesmo evento
card de veredito inicia fade-out
        ↓
ambos desaparecem juntos
```

A saída dos dois overlays usa:

- início no mesmo callback lógico;
- duração de fade-out: aproximadamente `160 ms`;
- easing `InOutQuad`;
- nenhuma animação contínua.

O `QTimer` continua pertencendo somente ao feedback temporário de `0/1`. O
card de veredito não cria um segundo timer: ele apenas aguarda o sinal de
fade-out emitido pelo overlay de tecla.

Durante essa janela, chamadas internas de reset/`save_label()` não podem
apagar o veredito instantaneamente. Elas respeitam o estado de saída pendente.

Se não existir feedback `0/1` ativo, um reset normal para
`AGUARDANDO PEÇA` continua limpando o card imediatamente.

Para comandos físicos vindos do Windows XP, o feedback visual é preparado antes
do handler produtivo consumir `CMD_OK/CMD_NG`. Isso garante a mesma
sincronização mesmo quando o processamento do comando encerra o ciclo
imediatamente.

### Estado de revisão obrigatória

Se a análise exigir revisão humana, o overlay não deve inventar uma decisão
binária. Em vez disso, ele deve comunicar explicitamente o estado operacional:

```text
operator_review_required = True
        ↓
REVISÃO OBRIGATÓRIA
        ↓
texto vermelho no card superior direito
```

Portanto:

- `REVISÃO OBRIGATÓRIA` → mostrar exatamente `REVISÃO OBRIGATÓRIA` em vermelho;
- nunca substituir esse estado por `FALHA FALSA` ou `DEFEITO REAL`, mesmo que
  `is_defect` bruto ainda esteja presente para auditoria;
- análises legadas sem texto de veredito só podem usar `is_defect` como
  fallback quando não existir revisão humana pendente.

### Arquitetura

Implementação:

```text
src/ui/decision_verdict_feedback.py
```

O recurso é exclusivamente visual e não pode:

- alterar `analysis`;
- alterar `is_defect`;
- alterar confiança ou score;
- alterar KNN;
- alterar dataset;
- alterar arquivos OK/NG;
- enviar `PRESS_0/PRESS_1`;
- responder pelo operador;
- bloquear o ciclo de imagens.

### Regressões obrigatórias

Manter testes que garantam:

- `FALHA FALSA` → texto verde, sem percentual;
- `DEFEITO REAL` → texto vermelho, sem percentual;
- posição no canto superior direito;
- card click-through e sem foco;
- entrada curta e não bloqueante;
- ausência de `QTimer` próprio no card de veredito;
- card permanece visível após a animação de entrada;
- julgamento `0/1` prepara a saída antes do reset produtivo;
- reset/`save_label()` não apagam o veredito enquanto a saída sincronizada está pendente;
- fade-out do veredito e do feedback `0/1` iniciam no mesmo evento;
- ambos usam aproximadamente `160 ms` e easing `InOutQuad`;
- comando `CMD_OK/CMD_NG` do XP prepara o feedback antes do handler produtivo;
- revisão obrigatória exibe `REVISÃO OBRIGATÓRIA` em vermelho e não inventa um veredito binário;
- reset sem feedback `0/1` ativo continua limpando o overlay imediatamente;
- o fundo permanece neutro durante toda a análise;
- feedback de tecla continua independente no canto inferior direito;
- o visual do feedback `0/1` usa moldura amarela/escura e conserva
  verde/vermelho apenas no conteúdo do estado.

### Validação operacional do fundo neutro e card persistente em 02/10/2026

O comportamento atualizado foi validado em uso real pelo operador.

Foi confirmado que:

- o fundo do ODIN permanece escuro/neutro durante `FALHA FALSA`;
- o fundo permanece escuro/neutro durante `DEFEITO REAL`;
- o fundo também permanece neutro em `AGUARDANDO PEÇA`;
- o card do canto superior direito continua exibindo somente o veredito;
- `FALHA FALSA` permanece visível em verde enquanto a análise atual continua
  ativa;
- `DEFEITO REAL` permanece visível em vermelho enquanto a análise atual
  continua ativa;
- o card não desaparece mais por tempo;
- não existe `QTimer` de auto-hide nem fade-out automático;
- ao encerrar/resetar o ciclo e retornar para `AGUARDANDO PEÇA`, o card é
  removido;
- uma nova análise pode substituir o conteúdo do card pelo novo veredito;
- o feedback temporário de tecla `0/1` continua independente no canto inferior
  direito;
- nenhum desses elementos altera decisão, confiança, KNN, dataset ou comandos XP.

Essa configuração continua sendo a referência para o fundo neutro e a posição do
card. Em 05/10/2026, o contrato de **saída** do card foi refinado: após um
julgamento `0/1`, ele deixa de ser removido imediatamente pelo reset e passa a
desaparecer sincronizado com o feedback temporário de tecla.

A sincronização de saída está **implementada e aguardando validação operacional
na interface real**. A validação anterior do fundo neutro permanece válida.


## Tempo de análise end-to-end

O card `TEMPO DE ANÁLISE` mede o tempo operacional percebido entre a entrada
da imagem no ODIN e o resultado já atualizado visualmente na interface.

### Contrato

Para imagem recebida do Windows XP:

```text
payload completo terminou de chegar ao ODIN
        ↓
descompressão / decode
        ↓
confirmação de estabilidade
        ↓
validação da AOI / epicentro
        ↓
extração / OCR / normalização
        ↓
motores físicos + memória + fusão
        ↓
widgets de resultado atualizados
        ↓
Qt processa a pintura pendente
        ↓
fim do Tempo de análise
```

O marco inicial da rede usa `time.perf_counter()` imediatamente após o payload
completo ser recebido, antes de descompressão e `cv2.imdecode()`.

O `NetworkReceiver` associa esse timestamp ao candidato estável efetivamente
entregue ao painel através de `last_delivered_image_received_at`.

Para captura local MSS:

```text
frame MSS terminou de ser capturado
        ↓
detecção da interface / recortes / OCR
        ↓
análise
        ↓
resultado pintado
        ↓
fim do Tempo de análise
```

O clique em `Capturar local (MSS)` e a espera anterior ao primeiro frame válido
não fazem parte do tempo de análise.

### Relógio

Usar exclusivamente relógio monotônico de alta resolução (`time.perf_counter()`)
para duração. `time.time()` não deve ser usado para calcular esse intervalo,
porque alterações no relógio do sistema podem distorcer a duração.

### Momento final

Depois de atualizar veredito, motivo, imagens, painéis técnicos e overlay, o
ODIN chama `QApplication.processEvents()` com input do operador excluído.
Somente depois desse processamento de pintura é registrado
`analysis_displayed_at`.

Assim, o valor não termina apenas quando a IA retorna: inclui a preparação e a
entrega visual do resultado ao operador.

### Telemetria

`analysis["detail"]` registra:

- `analysis_time_seconds`;
- `analysis_time_start_source`;
- `analysis_time_contract`.

Origens válidas:

- `network_payload_received`;
- `local_mss_frame_received`;
- `process_entry_fallback` apenas como proteção de compatibilidade.

O `Copiar debug` também deve exibir esses campos para auditoria.

### Regra de apresentação

O título permanece `TEMPO DE ANÁLISE` e o valor é mostrado em segundos, por
exemplo:

```text
1.27 s
```

Não rotular esse valor como simples `Latência`, pois ele representa o ciclo
end-to-end descrito acima.

Status em 06/10/2026: **contrato corrigido e implementação concluída;
aguardando validação operacional com medições reais no ODIN**.

## Diagnóstico e cópia de evidência por origem

Os controles visuais:

- `Copiar debug`;
- `Copiar imagem`;

não são exclusivos do Windows XP.

Eles devem ficar disponíveis para a **última captura analisada**, independentemente
da origem:

```text
Windows XP / rede
        ou
Captura local MSS
```

A fonte genérica da evidência fica em:

```text
src/services/capture_evidence.py
```

Contrato:

- cada captura possui um `event_id` próprio;
- o relatório e a imagem copiada devem pertencer ao mesmo `event_id`;
- uma captura local nunca pode reutilizar silenciosamente o último frame XP;
- uma captura XP nunca pode reutilizar silenciosamente um frame MSS;
- para rede, `Copiar imagem` continua usando exatamente o frame completo recebido do XP;
- para captura local, `ScreenMonitor` preserva exatamente o frame completo MSS que originou os recortes analisados;
- o relatório identifica a origem como `Windows XP` ou `Captura local MSS`;
- a interface usa o título genérico `DIAGNÓSTICO DA CAPTURA`.

Para captura local, o relatório usa o schema de observabilidade
`visionx.capture_debug.v1` e registra, quando disponíveis:

- frame MSS completo;
- recorte gabarito;
- recorte teste;
- informações AOI;
- decisão final;
- categoria;
- confiança;
- memória/KNN e métricas de ausência física já expostas pelo debug.

Essa camada é somente de observabilidade e não altera classificação, memória,
dataset, gate, confiança ou decisão.

### Estado padrão dos arquivos visuais NG e OK

Os controles `Salvar imagens NG` e `Salvar imagens OK` iniciam em:

```text
ATIVADO
```

em toda abertura do ODIN.

O operador continua podendo desativar cada arquivo independentemente durante a
sessão.

O NG mantém seu contrato atual: arquivamento automático restrito às evidências
de captura recebidas do Windows XP.

O OK possui contrato próprio: arquiva somente julgamentos humanos `OK`
(`button` ou `xp_keyboard`) e pode usar XP ou MSS, sempre através da mesma
evidência completa de `Copiar imagem`.

O arquivo OK não transforma decisões automáticas de Produção em evidência de
operador e não participa do dataset/KNN.

### Estado padrão do arquivo visual NG

O controle `Salvar imagens NG` inicia em:

```text
ATIVADO
```

em toda abertura do ODIN.

A mudança é somente do estado inicial do toggle. O operador continua podendo
desativá-lo a qualquer momento durante a sessão. O arquivo permanece assíncrono
e independente do dataset/KNN.

A regra de arquivamento automático continua restrita às evidências de captura
recebidas do Windows XP, conforme o contrato existente. Tornar o arquivamento
NG local/MSS automático exige uma decisão de produto separada e não deve ser
introduzido implicitamente.

### Regressões obrigatórias

Manter testes que garantam:

- captura XP habilita `Copiar debug` e `Copiar imagem`;
- captura local MSS analisada também habilita os dois controles;
- `Copiar imagem` local copia o frame MSS completo preservado, não o último frame XP;
- relatório local identifica `Captura local MSS`;
- `event_id` da imagem e do relatório sempre coincide;
- o toggle `Salvar imagens NG` inicia marcado/ativado;
- o operador ainda pode desativar o arquivamento durante a sessão.

### Validação operacional em 02/10/2026

O comportamento foi validado em uso real pelo operador após a implementação.

Foi confirmado que:

- imagens obtidas por **`Capturar nova peça (descarta a atual)`** também disponibilizam
  corretamente **`Copiar debug`** e **`Copiar imagem`**;
- a evidência copiada pertence à captura local MSS atual e não reutiliza um frame
  anterior recebido do Windows XP;
- o diagnóstico e a imagem permanecem associados ao mesmo `event_id`;
- o controle **`Salvar imagens NG`** inicia **ATIVADO** ao abrir o ODIN;
- o operador continua podendo desativar o arquivamento durante a sessão;
- o arquivamento automático de NG permanece restrito ao fluxo Windows XP, sem
  introduzir salvamento automático de capturas MSS.

Essa validação deve ser preservada como regressão operacional. Refatorações futuras
na camada de evidência não podem voltar a tornar os botões exclusivos do XP nem
permitir mistura silenciosa entre frames XP e MSS.

### Validação operacional da prévia da captura em 02/10/2026

A nova visualização **`CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA`**, posicionada na
seção **`IMAGENS DA INSPEÇÃO`** antes dos cards de gabarito/teste, foi validada
em uso real pelo operador.

Foi confirmado que:

- a prévia exibe corretamente a captura recebida do **Windows XP**;
- a prévia também exibe corretamente a captura local **MSS**;
- a imagem mostrada é exatamente a mesma evidência utilizada por
  **`Copiar imagem`**, sem criar uma segunda fonte visual independente;
- a proporção original da imagem é preservada durante o redimensionamento;
- o componente se adapta de forma responsiva ao espaço disponível na interface;
- o visual permanece consistente com os demais cards do ODIN;
- a prévia não altera classificação, KNN, decisão, ciclo produtivo nem regras de
  captura;
- permanece proibido reutilizar silenciosamente um frame XP antigo quando a
  evidência atual pertence a uma captura local MSS.

Essa prévia deve continuar sendo apenas uma camada de visualização da evidência
já validada pelo `event_id`. Refatorações futuras não devem duplicar a fonte da
imagem nem desacoplar a prévia do mesmo contrato usado por **`Copiar imagem`**.


### Limpeza visual ao entrar em AGUARDANDO PEÇA

Quando o painel de decisão retorna para:

```text
AGUARDANDO PEÇA
```

nenhuma imagem da inspeção anterior pode permanecer visível na seção de
inspeção.

Devem ser limpos visualmente:

- `CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA`;
- gabarito em visão completa;
- foco do gabarito;
- teste em visão completa;
- foco do teste.

A limpeza é **somente visual**. Ela não apaga a evidência técnica da última
captura e não invalida, por si só:

- `Copiar debug`;
- `Copiar imagem`;
- `event_id`;
- frame completo preservado para auditoria.

O controller mantém o estado visual `_inspection_images_visible`:

- `False` em `AGUARDANDO PEÇA`;
- `True` quando uma nova inspeção válida começa a ser processada.

Enquanto esse estado estiver falso, `resizeEvent` não pode reconstruir os
pixmaps a partir de `current_sample` ou `current_ng`, e a sincronização do
diagnóstico não pode repopular a prévia completa com a evidência anterior.

Essa separação é obrigatória: **evidência preservada para auditoria não significa
imagem antiga visível na inspeção atual**.

Regressões obrigatórias:

- `_reset_confidence_panel()` limpa todos os visuais da peça anterior;
- a prévia completa mostra `Aguardando captura`;
- gabarito/teste mostram apenas placeholders de espera;
- resize durante espera não faz a peça anterior reaparecer;
- `Copiar imagem` continua disponível quando existe evidência válida;
- nova inspeção válida reabilita a exibição das imagens.


## Responsividade dos painéis inferiores do ODIN

A partir da seção **DECISÃO E CONFIANÇA**, todo o restante da interface principal
usa os mesmos breakpoints definidos em `src/ui/responsive_layout.py`.

Objetivo operacional:

```text
notebook / largura < 1500 px
        ↓
leitura vertical, sem compressão lateral

monitor padrão / 1500–1799 px
        ↓
mais colunas sem perder espaçamento

monitor grande / >= 1800 px
        ↓
uso amplo da largura disponível
```

### DECISÃO E CONFIANÇA

Os três cards:

```text
VEREDITO DA IA
MÉTRICAS DE DECISÃO
MEMÓRIA LOCAL • KNN
```

refluem conforme o perfil:

- notebook: 1 coluna;
- monitor padrão: 2 colunas;
- monitor grande: 3 colunas.

Os cards usam largura mínima zero e política `Expanding`, evitando que um
`sizeHint` interno force scroll horizontal.

### CONTROLES OPERACIONAIS

Os grupos de **ILUMINAÇÃO DA CÂMERA** e **CAPTURA E DECISÃO DO OPERADOR**
deixam de depender de uma pilha vertical fixa.

- notebook: os dois grupos ficam empilhados;
- monitor padrão/grande: os grupos ficam lado a lado, com mais espaço reservado
  para captura e decisão;
- as três luzes TOP/SIDE/MID permanecem em três colunas;
- as quatro ações principais ficam em grade 2×2 no notebook e em uma linha de
  quatro botões nos perfis maiores.

A barra dinâmica `PRONTO / PROCESSANDO / ...` também é responsiva:

- notebook: badge e contador ocupam a primeira linha e a mensagem fica abaixo;
- monitor maior: badge, mensagem e contador ficam na mesma linha.

Nenhuma regra de habilitação, decisão, atalhos ou comunicação com o XP é
alterada por esse reflow.

#### Correção de regressão — botão Excluir dataset local

Após a primeira implementação responsiva, o container raiz dos controles passou
de `QVBoxLayout` para `QGridLayout`. O instalador de
`src/ui/test_mode_dataset_controls.py` ainda chamava
`action_layout.addLayout(button_row)`, assinatura válida para layout vertical,
mas inválida para `QGridLayout`, causando falha na inicialização do ODIN.

O botão administrativo agora é inserido preferencialmente dentro de
`decision_controls_group`, que é o grupo semântico de captura/decisão e mantém
layout vertical. Existe ainda fallback defensivo para `QGridLayout`, fornecendo
explicitamente linha, coluna e spans.

Regressões cobrem os dois contratos: hierarquia responsiva atual e fallback em
grade. A correção é somente estrutural/visual e não altera a limpeza do dataset,
KNN ou regras do Modo Teste.

### ARQUIVO VISUAL NG / OK

Os dois painéis já possuem contrato adaptativo e continuam assim:

- notebook: título, estado e botão ficam empilhados;
- monitor padrão/grande: ficam distribuídos horizontalmente.

### DIAGNÓSTICO DA CAPTURA

No notebook, título, estado e ações são empilhados e **Copiar debug** /
**Copiar imagem** passam a ocupar uma coluna cada, eliminando compressão.
Nos perfis maiores, as duas ações permanecem lado a lado.

### Três indicadores SVG da barra inferior

Os indicadores de:

```text
rede / AOI
estado do cérebro
última peça
```

também participam do reflow.

- notebook: os três grupos ficam em três linhas, alinhados à esquerda;
- monitor padrão/grande: ficam em três colunas equivalentes, com rede à
  esquerda, estado ao centro e histórico à direita;
- ícone e texto pertencem ao mesmo grupo responsivo e não são separados durante
  redimensionamento.

A implementação envolve:

```text
src/ui/control_panel_ui.py
src/ui/responsive_layout.py
src/ui/operational_controls.py
src/ui/iconography.py
```

e é exclusivamente visual. Não altera análise, KNN, arquivos OK/NG, dataset,
fusão multilight, teclas 0/1, setas ou protocolo Windows XP.


## Implementado — Modo Produção autônomo v1

Esta etapa se aplica **somente ao Modo Produção**. O **Modo Teste deve manter
exatamente o comportamento atual**, sem scroll automático, contador de decisões
autônomas ou julgamento automático adicional.

### Objetivo

Status: **implementação concluída e validada operacionalmente na AOI real em 07/10/2026.**

O ODIN poderá julgar peças automaticamente, mas o processo deve continuar
visível para o operador. A prioridade desta fase não é velocidade; é permitir
acompanhar o que o ODIN analisou antes que ele envie uma decisão à AOI.

Contrato visual e operacional:

```text
imagem recebida
      ↓
análise completa
      ↓
renderizar imagens, especialistas, KNN, decisão, confiança,
debug e demais painéis
      ↓
resultado final já visível no ODIN
      ↓
scroll automático da interface
do topo até a parte inferior
      ↓
pequena pausa visual no final
      ↓
avaliar política de autonomia
```

O scroll deve ser não bloqueante, usando mecanismos do Qt
(`QTimer` / animação da scrollbar) e **nunca `sleep` na thread da interface**.

### Política de decisão nesta fase de testes

Não existe mais requisito de confiança mínima de 99% para permitir uma decisão
automática OK. O estado final apresentado pelo ODIN é a referência operacional.

A política inicial fica:

```text
FALHA FALSA
      ↓
Modo Produção
      ↓
scroll/apresentação concluídos
      ↓
ODIN envia 0 = OK automaticamente

DEFEITO REAL / NG
      ↓
NÃO enviar 1 automaticamente nesta fase
      ↓
pausar autonomia
      ↓
aguardar operador: 0 = OK ou 1 = NG

REVISÃO OBRIGATÓRIA
      ↓
não enviar 0 nem 1
      ↓
pausar autonomia
      ↓
aguardar operador: 0 = OK ou 1 = NG
```

Portanto, embora o mapeamento operacional continue sendo:

```text
FALHA FALSA → 0 / OK
DEFEITO REAL → 1 / NG
```

**somente o caminho FALHA FALSA → 0 é autônomo nesta primeira versão**.
`DEFEITO REAL/NG` e `REVISÃO OBRIGATÓRIA` permanecem deliberadamente sob
decisão humana enquanto o sistema está em fase de testes.

### Intervenção humana e retomada automática

Quando houver `DEFEITO REAL/NG` ou `REVISÃO OBRIGATÓRIA`, o ODIN deve:

- congelar o ciclo atual;
- manter toda a análise renderizada;
- exibir mensagem visual persistente indicando que há intervenção necessária;
- aceitar julgamento humano tanto pelo teclado do Windows XP quanto pelos
  controles/atalhos do ODIN;
- não receber uma nova peça enquanto a atual estiver pendente.

Depois que o operador decidir `0` ou `1`, o ciclo é encerrado normalmente e
a autonomia fica automaticamente habilitada novamente para a **próxima peça**.
Não deve existir uma etapa manual extra de "retomar automação".

### Métricas diárias persistentes do Modo Produção

A sessão deixou de ser vinculada à entrada/saída do modo. Ela agora é vinculada
à **data local do computador**.

Contrato:

```text
07/10/2026
   ↓
todos os julgamentos e tempos acumulam no mesmo arquivo diário
   ↓
trocar Produção → Teste → Produção
   ↓
mantém os mesmos números

fechar ODIN / abrir novamente
   ↓
restaura os mesmos números

atualizar/substituir o código do repositório
   ↓
mantém os mesmos números

08/10/2026
   ↓
nova sessão diária começa em zero
```

O card continua oculto enquanto não existe peça ativa. Quando uma imagem entra
no ciclo ele mostra os dados acumulados do dia e desaparece novamente depois
que a peça é julgada.

Conteúdo atual:

```text
MODO PRODUÇÃO • DIA DD/MM/AAAA
AUTOMAÇÃO ATIVA / PAUSADA

OK AUTO        N
NG AUTO        N
MANUAL         N   •   ANÁLISES N
PRECISÃO       N%
MÉDIA ANÁLISE  N.NN s
```

Persistência:

- serviço: `src/services/production_daily_session_store.py`;
- schema: `visionx.production_daily_session.v1`;
- um arquivo JSON por data;
- gravação atômica por arquivo temporário + `os.replace`;
- no Windows, diretório padrão:
  `%LOCALAPPDATA%\\VisionX-Neural\\production_sessions\\`;
- exemplo:
  `%LOCALAPPDATA%\\VisionX-Neural\\production_sessions\\2026-10-07.json`;
- por ficar fora do checkout Git, um `git pull`, troca de branch, substituição
  da pasta do projeto ou atualização do código não apaga a sessão diária;
- arquivos de dias anteriores são preservados como histórico e o ODIN carrega
  somente o arquivo correspondente à data local atual.

Dados persistidos:

```text
auto_ok
auto_ng
manual_judgments
analysis_count
analysis_time_total
analysis_time_count
accuracy_percent
average_analysis_time_seconds
```

`accuracy_percent` e `average_analysis_time_seconds` são gravados como
snapshot para inspeção humana, mas são recalculados a partir dos acumuladores
ao carregar o arquivo.

Regras de cálculo permanecem:

- `OK AUTO` incrementa somente quando o ODIN realmente envia `0`
  automaticamente;
- `NG AUTO` permanece em zero nesta fase porque NG não é automatizado;
- `MANUAL` incrementa quando uma peça que exigiu intervenção é julgada pelo
  operador, seja pelo XP ou pelo ODIN;
- precisão =
  `julgamentos automáticos / julgamentos concluídos × 100`;
- 100 julgamentos concluídos com 98 automáticos e 2 manuais = `98,0%`;
- `ANÁLISES` conta resultados finais renderizados;
- a média usa `last_analysis_time_seconds` de cada resultado final;
- scroll, espera de apresentação, pausa e tempo do operador não entram na média.

O tooltip do card mostra a data operacional, os números restaurados, a fórmula,
o contrato temporal e o caminho do arquivo persistido. Se uma gravação falhar,
a inspeção não é interrompida; o tooltip passa a informar a falha de
persistência.

A pausa por `Space` é transitória e não é restaurada após reiniciar o
programa. Somente as métricas diárias são persistentes.

### Feedback visual da decisão automática

Quando o ODIN enviar um `0` sozinho, deve existir feedback visual equivalente
ao feedback já usado para teclas, deixando a origem explícita. Exemplo:

```text
TECLA ENVIADA AUTOMATICAMENTE

0
OK

ODIN • MODO PRODUÇÃO
```

Esse feedback deve ocorrer somente **depois** de:

1. toda a análise estar pronta;
2. o veredito estar renderizado;
3. o scroll automático ter apresentado a interface;
4. a pausa final ter terminado.

### Mensagem para intervenção

Para `DEFEITO REAL/NG` ou `REVISÃO OBRIGATÓRIA`, usar um card/overlay
persistente e independente, por exemplo:

```text
INTERVENÇÃO NECESSÁRIA

DEFEITO REAL
Aguardando operador

0 = OK
1 = NG
```

ou:

```text
INTERVENÇÃO NECESSÁRIA

REVISÃO OBRIGATÓRIA
Aguardando operador

0 = OK
1 = NG
```

O card só desaparece quando a decisão humana da peça atual for consumida.

### Separação entre modos

Contrato obrigatório:

```text
Modo Produção
→ pode usar apresentação automática + scroll
→ FALHA FALSA pode gerar 0 automático
→ NG/DEFEITO REAL e REVISÃO aguardam operador
→ contador de decisões automáticas da sessão

Modo Teste
→ permanece exatamente como está hoje
→ nenhuma nova automação de julgamento
→ nenhum scroll automático obrigatório
→ nenhum contador de decisões autônomas
```

Nenhuma dessas regras deve alterar o comportamento do Modo Teste por efeito
colateral.

### Arquitetura implementada

A autonomia foi centralizada em:

```text
src/ui/production_autonomy_controller.py
```

Responsabilidades atuais:

```text
ProductionAutonomyController
├── esperar a renderização final
├── registrar o tempo final da análise na sessão
├── posicionar a interface no topo
├── executar scroll visível e não bloqueante
├── pausar/retomar o fluxo pela barra de espaço
├── aplicar pausa final
├── avaliar o veredito final
├── enviar somente 0/OK autorizado
├── pausar em NG/revisão
├── contar julgamentos automáticos e manuais
├── observar decisão humana
└── manter o ciclo pronto para a próxima peça
```

O contador e a mensagem de intervenção ficam em:

```text
src/ui/production_session_feedback.py
```

O antigo `production_confidence_gate.py` permanece como módulo de compatibilidade
e trava de intervenção, mas **não usa mais 99% como requisito**. A confiança
continua disponível apenas como telemetria.

O fluxo normal em `control_panel.py` também não chama mais
`save_label(..., source="auto")` imediatamente ao terminar a análise. Ele
entrega a análise ao controlador. Para adesivo, a automação SIDE/TOP/MID entrega
somente a fusão final multilight ao mesmo controlador.

### Temporização da apresentação

Constantes atuais:

```text
espera após renderização = 500 ms
pausa no topo            = 450 ms
scroll vertical total    = 6300 ms
scroll horizontal experts = 2800 ms
pausa no final           = 900 ms
espera antes do 0        = 300 ms
```

Quando não existe scroll vertical útil, o ODIN mantém uma janela de apresentação
de aproximadamente `1200 ms` antes de aplicar a política.

Todo o fluxo usa `QTimer` e `QPropertyAnimation`; não existe `sleep` na
thread da interface.

### Pausa pela barra de espaço e overlays

O atalho `Space` é um `QShortcut` de janela habilitado **somente no Modo
Produção**. No Modo Teste e no Modo Sombra ele fica desabilitado e não altera o
uso normal da barra de espaço.

Contrato:

```text
ESPAÇO durante Modo Produção
        ↓
pausa timers da apresentação
pausa a animação de scroll no ponto atual
bloqueia o envio automático 0
        ↓
ESPAÇO novamente
        ↓
retoma o estágio pendente
continua o scroll do ponto atual
ou continua a espera/decisão pendente
```

A pausa é da **autonomia de julgamento/apresentação**. Uma análise matemática
que já está executando pode terminar e renderizar normalmente; o controlador
fica aguardando até o operador despausar. Isso evita interromper o pipeline de
visão no meio de uma inferência ou da coleta multilight.

A barra operacional e o card da sessão passam a mostrar
`PRODUÇÃO PAUSADA / ESPAÇO PARA CONTINUAR`. Se a pausa ocorrer antes de uma
nova peça, o card continua oculto; quando a peça entrar, ele aparece já no estado
pausado e o resultado aguarda a retomada.

Somente um comando automático realmente confirmado como enviado ao XP incrementa
`OK AUTO/NG AUTO`. Se o envio de `0` falhar, a placa não é contabilizada como
automática e passa para intervenção humana.

O feedback de tecla automática reutiliza o card existente:

```text
TECLA ENVIADA AUTOMATICAMENTE
0
OK
ODIN • MODO PRODUÇÃO
```

Decisões humanas em Produção incrementam `MANUAL`, não os contadores
automáticos. Essa intervenção reduz a precisão autônoma calculada da sessão.

### Intervenção humana

Depois da apresentação completa:

- `FALHA FALSA` → `production_auto` envia somente `0/OK`;
- `DEFEITO REAL` → nenhuma tecla automática; exibe intervenção;
- `REVISÃO OBRIGATÓRIA` → nenhuma tecla automática; exibe intervenção;
- falha no envio automático de `0` → intervenção operacional.

Durante intervenção:

- a captura permanece protegida;
- os botões `0 - Aprovar como OK` e `1 - Confirmar defeito NG` ficam
  disponíveis no ODIN;
- `0/1` físicos recebidos do XP também são aceitos;
- fora de intervenção, `0/1` físicos em Produção são ignorados;
- após uma decisão humana válida, o overlay de intervenção some e a autonomia
  fica armada para a próxima placa sem botão adicional de retomada.

### Separação preservada

O **Modo Teste não foi alterado** por essa implementação. Ele continua com
julgamento manual, possibilidade de substituir/descartar capturas, limpeza do
dataset e controles já existentes.

O **Modo Sombra também mantém o contrato anterior**.

### Validação operacional em 07/10/2026

O operador executou o novo **Modo Produção autônomo v1** em uso real e confirmou
que o fluxo funcionou corretamente.

Foram validados operacionalmente:

- análise completa antes do julgamento automático;
- renderização do resultado antes da tomada de decisão;
- scroll automático da interface até a parte inferior;
- envio automático de `0 = OK` para `FALHA FALSA`;
- bloqueio de decisão automática para `DEFEITO REAL/NG`;
- bloqueio de decisão automática para `REVISÃO OBRIGATÓRIA`;
- decisão manual pelo operador quando há intervenção;
- retomada automática do fluxo na peça seguinte, sem botão extra;
- pausa do processo automático com a barra de espaço;
- retomada do mesmo estágio ao pressionar espaço novamente;
- atualização visual do estado `PRODUÇÃO PAUSADA`;
- contador `OK AUTO / NG AUTO / MANUAL / ANÁLISES`;
- cálculo da precisão autônoma com penalização de julgamentos manuais;
- atualização da média do tempo de análise;
- tooltip dinâmico com fórmula da precisão e contrato temporal;
- card da sessão visível somente enquanto existe peça ativa;
- preservação do comportamento existente do **Modo Teste**.

Com essa confirmação, pausa/retomada, métricas da sessão e política conservadora
de intervenção passam a ser consideradas **validadas operacionalmente** para a
versão atual.

A validação operacional não libera `NG AUTO`. Nesta fase, qualquer
`DEFEITO REAL/NG` ou `REVISÃO OBRIGATÓRIA` continua exigindo julgamento
humano e conta como intervenção manual na precisão da sessão.

### Aprendizado e persistência

A origem `production_auto` é tratada como decisão automática e não cria
rótulo humano no Active Learning. Ela também não é confundida com uma decisão
de operador no histórico da interface.

A implementação não libera NG automático nesta fase. Portanto
`NG AUTO` permanece em zero até uma etapa futura explicitamente aprovada.


### Ajuste visual — imagens integrais e especialistas horizontais

Implementado após a validação inicial do Modo Produção.

#### IMAGENS DA INSPEÇÃO

Os quatro viewports normais:

```text
GABARITO • VISÃO COMPLETA
GABARITO • EPICENTRO
TESTE • VISÃO COMPLETA
TESTE • EPICENTRO
```

passaram de `QLabel` simples para um viewport responsivo que preserva o pixmap
fonte e o redesenha sempre que **o próprio card** muda de tamanho.

Isso resolve o caso em que mover a divisória do `QSplitter` alterava o espaço
do label sem disparar `resizeEvent` da janela principal. Antes, o pixmap podia
continuar maior que o viewport e parecer cortado até o operador aumentar
manualmente a seção.

Contrato atual:

```text
mudou largura/altura do viewport
        ↓
recalcular pixmap a partir da fonte
        ↓
KeepAspectRatio
        ↓
imagem inteira dentro do card
```

A prévia completa e os viewports multilight de adesivo já possuíam lógica
equivalente e continuam preservados.

#### ANÁLISE DOS ESPECIALISTAS

A navegação horizontal fica explícita:

- categoria normal: a barra horizontal do `scroll_area` fica sempre visível;
- adesivo: SIDE/TOP/MID possuem uma **barra horizontal mestre única**;
- a barra mestre distribui proporcionalmente a posição para os scrolls internos
  das três iluminações, mantendo os mesmos especialistas visualmente alinhados;
- as barras internas de cada lane ficam ocultas para evitar três controles
  horizontais concorrentes.

No Modo Produção, a apresentação automática passa a executar:

```text
topo da página
   ↓
scroll vertical até ANÁLISE DOS ESPECIALISTAS
   ↓
scroll horizontal dos especialistas
   ↓
continua scroll vertical até o final
   ↓
pausa final
   ↓
política de decisão
```

A pausa por `Space` continua válida durante qualquer uma dessas animações.

#### Velocidade da apresentação

O movimento vertical foi desacelerado de `5200 ms` para `6300 ms` no total.
Quando existe overflow horizontal nos especialistas, é acrescentado um passeio
de aproximadamente `2800 ms`.

A duração vertical é distribuída proporcionalmente entre o trecho até os
especialistas e o trecho restante até o final; não são adicionados dois ciclos
verticais completos.

Esses ajustes são de interface/apresentação e não alteram o tempo matemático de
análise, a fusão multilight, o KNN ou a política de julgamento.


## OCR da interface AOI — correção contextual dos campos Board / Parts / Value

### Caso real observado em 07/10/2026

Na captura real da AOI, o OCR geral do Tesseract identificou corretamente a
estrutura da tela, mas introduziu três erros nos campos exibidos pelo ODIN:

~~~text
Board
OCR anterior: [P22-22200 (PRINCIPAL) L13
correto:      P22-22200 (PRINCIPAL) L13

Parts
OCR anterior: [RI~5
correto:      R3~5

Value
OCR anterior: [io <= 2 <= 80 FALTANDO
correto:      10 <= 2 <= 80 FALTANDO
~~~

O problema não era a extração das imagens de inspeção. O texto bruto continha
os rótulos Board, Parts e Value, porém a fonte clássica do Windows/AOI fazia o
Tesseract confundir bordas de células e alguns caracteres.

### Implementação

Foi criado:

~~~text
src/services/aoi_ocr_fields.py
~~~

O ScreenMonitor continua responsável pela leitura geral da região textual.
Depois da extração inicial, os três campos passam por políticas específicas.

#### Board

Remove somente ruído de borda no começo da célula:

~~~text
[P22-22200 (PRINCIPAL) L13
↓
P22-22200 (PRINCIPAL) L13
~~~

Não existe substituição global de letras ou números no nome da placa.

#### Value

Confusões entre letras e números são corrigidas somente quando aparecem no
primeiro token de uma expressão comparativa numérica:

~~~text
io <= 2 <= 80 FALTANDO
↓
10 <= 2 <= 80 FALTANDO
~~~

Mapeamento contextual:

~~~text
I / i / L / l → 1
O / o         → 0
~~~

Esse mapeamento não é aplicado livremente ao restante do texto, evitando
corromper palavras e categorias.

#### Parts

O campo de componente possui uma validação sintática inicial, por exemplo:

~~~text
R3~5
C120
CN12
~~~

Quando a leitura geral já é válida, nenhuma chamada OCR adicional é feita.

Quando a leitura não possui uma referência coerente, como:

~~~text
RI~5
~~~

o ODIN executa uma segunda leitura somente da célula Parts:

1. image_to_data localiza o rótulo Parts e a próxima coluna da mesma linha;
2. é criada uma ROI estreita apenas com o valor da célula;
3. ocorre uma leitura PSM 7 com whitelist alfanumérica;
4. ocorre uma segunda passada numérica com whitelist 0123456789~-;
5. os resultados são combinados somente se formarem uma referência válida.

Isso permite recuperar o caso real:

~~~text
OCR geral:          RI~5
leitura numérica:   3~5
prefixo confiável:  R
resultado:           R3~5
~~~

A estratégia é deliberadamente condicional para não aumentar o custo do OCR em
todas as capturas: a releitura dirigida só ocorre quando Parts falha na
validação de formato.

### Resultado esperado para o caso de 07/10/2026

~~~text
Placa / Máquina: P22-22200 (PRINCIPAL) L13
Componente:      R3~5
Valor / OCR:     10 <= 2 <= 80 FALTANDO
~~~

Essa alteração afeta metadados OCR e apresentação dos dados da AOI. Não altera
a classificação visual dos especialistas, a fusão, o KNN, o protocolo XP ou a
política do Modo Produção.


## Ajuste de 07/10/2026 — imagens sem divisor manual e segundo refinamento do OCR

### IMAGENS DA INSPEÇÃO

Foi identificado que o QSplitter entre IMAGENS DA INSPEÇÃO e ANÁLISE DOS
ESPECIALISTAS ainda permitia arraste manual. Em notebook, especialmente quando
o splitter estava na orientação vertical, arrastar a barra alterava
continuamente a área das imagens e dava a impressão de "zoom infinito".

O contrato foi alterado:

~~~text
antes
QSplitter visível e arrastável
↓
operador podia aumentar/reduzir indefinidamente a seção

agora
QSplitter continua existindo apenas como mecanismo interno de reflow
↓
handle com largura 0
↓
handle desabilitado
↓
tamanhos definidos pelo perfil responsivo
~~~

No perfil compact/notebook:

~~~text
IMAGENS DA INSPEÇÃO  mínimo 560 px
ANÁLISE ESPECIALISTAS mínimo 430 px
estágio completo      mínimo 1000 px
~~~

O root_scroll da página absorve a altura adicional. Portanto não é necessário
sacrificar metade das imagens nem arrastar a antiga divisória.

Em telas standard/wide, a largura de IMAGENS DA INSPEÇÃO passa a acompanhar a
largura da janela de forma limitada:

~~~text
28% da largura disponível
mínimo 420 px
máximo 520 px
~~~

Os pixmaps continuam usando KeepAspectRatio e são redesenhados a partir da
imagem-fonte quando o viewport muda de tamanho.

### Segundo caso real de OCR

Outra captura real apresentou:

~~~text
Componente
OCR:      u2~s
correto:  U2~5

Valor / OCR
OCR:      fo <= $4.872 <= 10 FALTANDO
correto:  0 <= 54.872 <= 10 FALTANDO
~~~

O debug confirmou que esses valores já estavam presentes no aoi_info após a
leitura geral do Tesseract.

#### Parts

A normalização de Parts passou a separar prefixo alfabético e corpo numérico
somente quando existe um dígito explícito.

Exemplo:

~~~text
u2~s
↓
prefixo = U
corpo numérico = 2~s
↓
s em posição numérica → 5
↓
U2~5
~~~

Se não existe um dígito explícito confiável antes do separador, como RI~5, a
normalização não inventa o número e mantém a releitura dirigida da célula.

#### Value

A normalização não trata apenas o primeiro número. Quando o campo possui duas
comparações, os três operandos são analisados como números e somente eles
recebem correções OCR.

Mapeamentos numéricos atuais incluem:

~~~text
I / i / L / l / | → 1
O / o / Q / q     → 0
S / s / $         → 5
Z / z             → 2
G / g             → 6
B / b             → 8
~~~

Caracteres sem significado numérico são descartados somente dentro do operando
numérico. Assim:

~~~text
fo <= $4.872 <= 10 FALTANDO
↓
0 <= 54.872 <= 10 FALTANDO
~~~

O sufixo textual FALTANDO permanece intacto.

Esses dois refinamentos estão implementados e aguardam nova validação
operacional na AOI real.

## Caso real pendente — FALTANDO fisicamente ausente classificado como FALHA FALSA

### Registro de 07/10/2026

Evento:

~~~text
6a45d7509bf8444fa5e529645cab1236
~~~

Origem:

~~~text
Windows XP
categoria AOI = FALTANDO
~~~

Validação do operador:

~~~text
DEFEITO REAL / NG
motivo: componente fisicamente ausente
~~~

Resultado incorreto do ODIN:

~~~text
FALHA FALSA
confidence = 0.99
fusion_rule = best_match_strong
dominant_engine = knn
~~~

### Evidência do debug

O caso é importante porque o especialista dedicado de ausência física já marcou defeito, mas essa evidência não ganhou autoridade final:

~~~text
missing_is_defect = True
missing_score ≈ 0.4223
physical_score = 0.88

missing_hard_absence = False
missing_context_hard_absence = False
~~~

A memória encontrou:

~~~text
best_match_label = OK
best_similarity ≈ 0.9039
~~~

e, como não existia missing_hard_absence, a fusão escolheu:

~~~text
best_match_strong
↓
motor dominante = KNN
↓
FALHA FALSA
~~~

### Sinais que bloquearam a promoção para ausência física forte

A ROI local foi interpretada como corpo preservado principalmente por geometria:

~~~text
missing_component_body_present = True
missing_body_presence_policy = geometry_only
silhouette_dice ≈ 0.9526
area_ratio ≈ 0.9111
centroid_shift ≈ 0.0327
coarse_similarity ≈ -0.0235
~~~

O ponto de atenção é que a forma/ocupação do footprint permaneceu parecida mesmo com o componente ausente. Portanto a testemunha geométrica conseguiu descrever a região como corpo presente, apesar de a similaridade visual coarse ser praticamente nula/negativa.

O envelope global também registrou massa invariável:

~~~text
missing_global_envelope_invariant_support = True
dark_retention ≈ 0.7580
invariant_row_profile ≈ 0.8477
invariant_col_profile ≈ 0.8477
~~~

Nesse caso, pads, cobre, fundo e estruturas vizinhas preservadas podem dominar a massa global e não provar que o componente central continua presente.

A análise contextual dual-scale encontrou divergência relevante:

~~~text
missing_context_score ≈ 0.7764
coverage ≈ 0.3703
residual_mean ≈ 0.6748
appearance_loss ≈ 0.4162
direct_similarity ≈ 0.5838
~~~

mas não confirmou suporte físico independente:

~~~text
missing_context_physical_support.supported = False
structural ≈ 0.3811
semantic ≈ 0.5496
~~~

Com isso:

~~~text
missing_context_hard_absence = False
~~~

e a memória OK voltou a ter autoridade total.

### Diagnóstico arquitetural provisório

Este caso não deve ser tratado simplesmente como "KNN errou".

A sequência observada foi:

~~~text
componente realmente ausente
↓
especialista missing detecta defeito
↓
geometria local confunde footprint preservado com corpo presente
↓
contexto não alcança contrato de hard missing
↓
missing_hard_absence permanece False
↓
KNN OK forte continua elegível
↓
best_match_strong
↓
FALHA FALSA
~~~

Portanto o ponto de investigação é a fronteira entre:

- missing_is_defect=True;
- presença geométrica local;
- promoção para missing_hard_absence;
- autoridade do KNN quando existe evidência física relevante, mas ainda abaixo do contrato de hard absence.

### Regra de segurança para a futura correção

Qualquer correção futura deste evento deve preservar as regressões já validadas contra falsos positivos de ausência física.

Em especial, não se deve simplesmente transformar todo missing_is_defect=True em hard missing.

A correção precisa distinguir:

~~~text
footprint/pads permanecem, componente sumiu
~~~

de:

~~~text
componente realmente presente, porém deslocado / com mudança de registro / iluminação / serigrafia
~~~

O caso real DESLOCADO já validado, em que o componente está presente e a memória OK forte deve continuar elegível, permanece uma regressão obrigatória.

### Correção implementada — footprint_absence dedicado

A correção foi implementada especificamente dentro do especialista de
`FALTANDO`, sem alterar a guarda transversal usada por
`DESLOCADO/EMBORCADO/INVERTIDO`.

Nova rota:

~~~text
missing_is_defect = True
        ↓
não é DESLOCAMENTO PROVÁVEL
        ↓
presença não foi confirmada por aparência
ou geometry_only possui coarse similarity <= 10%
        ↓
ROI local:
residual >= 60%
estrutura >= 55%
bordas incompatíveis >= 42%
melhor match próximo < 30%
        ↓
contexto dual-scale:
score >= 72%
cobertura >= 32%
residual >= 60%
appearance loss >= 38%
structure loss >= 35%
direct similarity <= 62%
melhor match próximo < 50%
        ↓
motores independentes:
estrutural >= 35%
semântico >= 50%
        ↓
missing_dedicated_footprint_absence = True
        ↓
missing_hard_absence = True
~~~

O evento real `6a45d7509bf8444fa5e529645cab1236` satisfaz esse
contrato:

~~~text
body coarse similarity ≈ -0.023
local residual ≈ 0.691
local structure loss ≈ 0.629
local edge mismatch ≈ 0.473
local nearby similarity ≈ 0.216

context score ≈ 0.776
context coverage ≈ 0.370
context residual ≈ 0.675
context appearance loss ≈ 0.416
context structure loss ≈ 0.397
context direct similarity ≈ 0.584
context nearby similarity ≈ 0.446

motor estrutural ≈ 0.381
motor semântico ≈ 0.550
~~~

Essa rota reconhece que footprint, pads e massa escura podem conservar uma
silhueta semelhante mesmo depois que o corpo central desaparece. Portanto
`geometry_only` deixa de ser suficiente para proteger presença quando há
contradição multiescala independente.

### Autoridade da memória nesta rota

Quando `missing_dedicated_footprint_absence=True`:

- a massa global invariável continua registrada para auditoria;
- ela não pode acionar a exceção
  `hard_missing_invariant_presence_ok_witness`;
- o KNN continua visível no debug, porém recebe peso zero;
- a exceção de memória OK **quase exata** permanece válida somente a partir do
  contrato já existente de aproximadamente 99,5% + margem mínima.

Para o evento real, o KNN OK ≈ 90,4% não é suficiente para anular a ausência
física dedicada.

Resultado esperado:

~~~text
fusion_rule = missing_hard_absence
dominant_engine = missing
physical weight = 100%
KNN weight = 0%
DEFEITO REAL
confidence = 0.99
~~~

### Observabilidade

O debug passa a incluir:

~~~text
FALTANDO footprint dedicado: True/False
FALTANDO footprint motivo: ...
~~~

e o payload completo preserva:

~~~text
missing_dedicated_footprint_absence
missing_dedicated_footprint_reason
~~~

### Regressões adicionadas

Foram adicionados testes para:

- reproduzir as métricas reais do evento `6a45...`;
- promover esse vetor para hard missing;
- impedir que KNN OK ≈ 90,4% + massa invariável vetem a ausência dedicada;
- manter seguro um caso de `geometry_only` com coarse similarity ainda
  plausível para corpo presente;
- expor a nova rota no debug técnico.

### Status

~~~text
registrado em 07/10/2026
correção implementada na branch central
aguardando validação operacional na AOI real
~~~

A regressão real de `DESLOCADO` com componente presente continua obrigatória:
esta nova rota não é compartilhada com a guarda transversal e não deve
transformar deslocamento/registro em componente faltando.
