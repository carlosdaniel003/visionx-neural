## 09/10/2026 — CNN FALTANDO v2 como especialista de ausência em três modos

**Decisão de implementação:** ODIN usa a CNN FALTANDO v2 para as categorias
AOI **FALTANDO, EMBORCADO, INVERTIDO e DESLOCADO**. O rótulo original é
mantido no debug/memória; não transforma a imagem em categoria FALTANDO no
arquivo. **MUITO ADESIVO, MUCH ADHESIVE e demais sinônimos ficam
EXCLUSIVAMENTE no especialista adesivo**. Categorias não previstas continuam
no MoE legado. A CNN DESLOCADO especializada continua cancelada/arquivada.

**KNN e conjunto OK:** Antes da CNN, o ODIN consulta somente a memória
**humana de par exato**, segregada por categoria original, placa, peça, valor
e iluminação. Igualdade aproximada com imagens da pasta OK **não libera
nenhum componente automaticamente**. Caso novo usa a CNN de ausência. A
CNN não é retreinada automaticamente com outras categorias por esta
alteração; o treinamento incremental original de FALTANDO não mudou.

**Três modos usam o mesmo roteamento de inspeção:**
- **Modo Teste:** CNN no resultado visual; rótulo 0/1 por decisão do humano,
  seguindo as regras já existentes de captura e salvamento.
- **Modo Sombra:** avalia a CNN e registra o diagnóstico, sem enviar 0/1 ao XP.
- **Modo Produção:** em evento com SIDE/TOP/MID, a fusão registra as três
  inferências e o SHA-256 do checkpoint. O controlador considera um 0/1
  automático **somente** se todas as iluminações forem da CNN, tiverem a
  **mesma categoria**, o **mesmo checkpoint íntegro**, score finito e
  conclusivo (**OK <= 0.10** ou **NG >= 0.90**), e os **três votos forem
  idênticos**, sem revisão individual/final. Todos os demais casos com
  CNN exigem operador — imagem mono-SIDE isolada, conflito TOP/MID/SIDE,
  KNN + CNN misturados, carregamento incompleto, pontuação intermediária,
  inversão ou troca do checkpoint, categoria adesivo/desconhecida.

**Automação supervisionada:** depois da apresentação visual, aguarda
**2000 ms** antes do comando; **Space** pausa imediatamente o agendamento
e só continua após novo Space. 0=OK e 1=NG usam os mesmos caminhos
existentes de salvamento, transmissão e feedback; apenas comandos com
confirmação de envio contam em métricas AUTO. Falha de transmissão exige
intervenção humana. Troca de modo, ciclo ou imagem invalida envio pendente.
**O operador deve estar presente e pode intervir antes do envio**.
Após o envio ao XP, Space não desfaz a decisão já transmitida.

**Limite da qualificação:** a auditoria transversal histórica registrou
199/199 resultados concordantes com os arquivos fora de adesivo,
mas pode haver exemplos vistos no treino. DESLOCADO dispõe apenas de OK
(sem NG reais), e os NG das demais categorias nesse histórico são SIDE.
Portanto **esse teste não certifica segurança de detecção de NG inéditos**.
Mesmo com consenso de três luzes, existem riscos residuais de falha
sistemática comum às três visões; a implantação exige observação técnica
e validação supervisionada nas placas da produção.

**Verificação automatizada de software:**
- __tests/test_faltando_shared_production.py__
- __tests/test_faltando_cnn_v2_live.py__
- __tests/test_verified_memory_router.py__
- __tests/test_production_autonomy_controller.py__
- __tests/test_production_confidence_gate.py__
- Workflow __.github/workflows/faltando-shared-production-tests.yml__.

**Diagnóstico e rollback:** se uma peça suspeita receber OK, pausar o
Modo Produção com Space, voltar ao Modo Teste para avaliação humana e
preservar as imagens e debug. Para reverter esta integração, restaurar
o commit anterior na branch central via revisão/rollback Git (não
excluir dataset nem memórias). Não tratar sucesso dos testes unitários
como aprovação industrial de uma nova categoria.


---

# Preparação do dataset neural — FALTANDO

**Etapa atual (08/10/2026):** CNN FALTANDO v2 integrada à inspeção normal, com aprendizado incremental disparado por decisões humanas novas em Teste/Produção/Sombra, promoção de checkpoint condicionada a replay de regressão. Validação na estação real deste fluxo incremental ainda pendente.
**Branch:** central. **Máquina de execução:** Windows 10 do ODIN (não o XP).
**Fonte:** `public/ok_archive` e `public/ng_archive`, somente leitura.

## Inventário informado pelo operador (08/10/2026)

- 117 PNGs FALTANDO: 107 OK e 10 NG.
- NG: 10 sem sufixo de luz (SIDE histórico presumido); nenhum NG TOP/MID.
- OK: 32 sem sufixo + 25 SIDE, 25 TOP e 25 MID explícitos.
- SHA-256 distintos: 117; nenhuma duplicata exata, nenhum hash comum OK/NG.
- 25 trincas por **nome** (sufixos _2/_3 incluídos), sem manifesto event_id.
- Isso **não prova** 117 observações independentes, rótulos perfeitos ou cobertura
  suficiente para treinamento seguro de NG multilight.

## Objetivo desta etapa

Converter os screenshots AOI em pares gabarito/teste completos, sem treino e
sem mudar os rótulos. Reutilizar exclusivamente
`ScreenMonitor.process_external_image`, o extrator da produção. Proteger
a imagem inteira, evitar recortes limitados ao epicentro, e gerar relatório
de qualificação humana antes de qualquer divisão treino/validação/teste.

**Não chamar:** MoEOrchestrator, KNN, CNN, pesos, replay do startup, comando XP
ou API de Produção. A extração não executa julgamento e não publica modelo.

### Comando

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.faltando_neural_dataset
```

Saída local sob `reports/faltando_neural/run_<timestamp>/`:

- `manifest.json`: hash SHA-256 da origem, rótulo **provisório** da pasta,
  iluminação, event_id se comprovado por manifesto real, status, pendências de
  qualificação, caminhos de imagens derivadas, tamanhos e OCR observado;
- `summary.txt`: total extraído, falhas, vínculos de evento confirmados e
  candidatos por nome;
- `pairs/<label>_<hash>/reference.png`: gabarito completo;
- `pairs/<label>_<hash>/test.png`: teste completo.

Nenhum arquivo em `public` é editado, renomeado ou substituído. A pasta
`reports/faltando_neural` é excluída do Git. A saída é um staging técnico,
**não** um dataset de treinamento aprovado.

## Qualificação obrigatória antes do treino

- Confirmar que a pasta OK/NG corresponde a **julgamento humano**, não
  etiqueta automática, em cada imagem selecionada.
- Inspecionar se gabarito e teste foram corretamente recortados (sem
  barras/textos da interface misturados às regiões físicas).
- Conferir a categoria via OCR quando disponível; preservar o nome como
  pista, não reclassificar automaticamente.
- Validar que arquivos nomeados SIDE/TOP/MID pertencem ao mesmo evento:
  **sufixo e horário não bastam**. Um manifesto com hashes/event_id permite
  vínculo confiável; os demais são somente sugestões de grupo.
- Identificar duplicatas **perceptuais**, capturas sucessivas da mesma peça
  e casos de PCB/componentes equivalentes; SHA distinto não garante
  independência.
- Identificar diversidade de defeitos NG. Com apenas 10 PNGs NG SIDE, o
  conjunto ainda não sustenta declarar CNN multilight segura para operação.
- Somente depois da qualificação, definir splits por **evento/peça/lote**
  sem vazamento entre treino e avaliação.

A execução informa `training_ready=false` em todos os candidatos. Esta
etapa não altera a decisão operacional nem o startup gate.

## Qualificação visual assistida — próxima etapa implementada

Use, depois da preparação e atualização do repositório:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.ui.faltando_neural_review
```

Para escolher uma preparação específica:
`python -m src.ui.faltando_neural_review --manifest "C:\visionx-neural-main\reports\faltando_neural\run_XXXXX\manifest.json"`.

A interface offline lista casos individuais e trincas candidatas. Gabarito e
teste são exibidos inteiros, sem crop oculto. O usuário confirma cada par
como `CONFIRMED_OK` (componente presente), `CONFIRMED_NG` (componente
ausente) ou `REJECTED` (recorte impróprio); requer checkbox humano, e
registra eventual divergência contra o rótulo original, sem alterar a fonte.

Uma trinca SIDE/TOP/MID só é validada manualmente depois de todos os três
pares terem rótulo confirmado equivalente ao rótulo arquivado; gera
`human_group_id`, nunca falsifica `event_id` original da AOI.
As sugestões de semelhança dHash consideram ambos gabarito/teste na mesma
iluminação e **não** removem imagens ou alteram rótulos.

Cada decisão é salva imediatamente em
`reports/faltando_neural/run_*/qualification.json`, com confirmação
SHA-256 da origem, em escrita atômica. O estado é restaurado ao reabrir.
Os originais em `public`, o dataset e o `manifest.json` não mudam.

Esta revisão é preparatória: `training_ready=False` continua intacto,
não treina CNN, não cria splits e não interfere no ODIN em Produção.

## Treinamento experimental aprovado pelo operador (08/10/2026)

O operador confirmou que os rótulos do acervo são válidos e **autorizou
treinamento imediato sem revisão manual obrigatória**. O painel visual
anterior continua disponível para auditoria opcional. O treinamento acessa
os recortes **já extraídos**; não renomeia ou move imagens originais.

O modelo `src/core/neural/faltando_cnn.py` é uma CNN comparativa
(gabarito versus teste) com extrator compartilhado e seleção por
`SIDE/TOP/MID`. Eventos monoimagem legados usam máscara SIDE; cada
trinca de três luzes OK com OCR `board/parts/value` coerente
é agrupada provisoriamente como um evento. As imagens da trinca
não são três eventos independentes. A saída é um único logit da
presença/ausência; o maior score NG das iluminações disponíveis
determina o score do evento. KNN e motores físicos não são consultados.

O treino é executado na máquina nova, no **mesmo ambiente Python**:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -c "import torch; print(torch.__version__)"
python -m src.scripts.train_faltando_cnn --epochs 25 --batch-size 4 --size 160 --device cpu
```

Caso `import torch` falhe, é necessário disponibilizar o pacote PyTorch
compatível nesse ambiente (sem instalar software com privilégio de
administrador). Evite alterar o ambiente Python do Windows XP.
O treinamento pode levar vários minutos na CPU corporativa.

**Saídas do treino** (nenhum arquivo será enviado ao GitHub):

```text
reports/faltando_neural/models/experiment_<timestamp>/
    faltando_cnn_candidate.pt
    training_report.json
    training_summary.txt
```

A avaliação utiliza holdout com separação por board/parts e similaridade
de imagem, evitando que uma trinca apareça parcialmente no treino e no
holdout. O relatório inclui `FN_NG_as_OK`: NG verdadeiro liberado
erroneamente como OK. Se não for possível formar holdout com OK/NG
independentes, o treino falha sem salvar modelo.

**Limites e estado:**
- Apenas 10 NG SIDE históricos e nenhum NG TOP/MID: métrica pequena, sem
  cobertura de falhas reais multilight.
- Rede pequena treinada do zero: demonstrador/linha de base; não alegar
  transfer learning nem generalização comprovada.
- Checkpoint `experimental=True`, `production_approved=False`.
  Nenhuma alteração em `main.py`, no roteador KNN, no modo Produção ou
  na regressão de startup. Substituição de motores e automação de decisão
  serão etapas posteriores após avaliar os resultados reais.


## Resultado real da v1 — 08/10/2026

O operador treinou a v1 com `python -m src.scripts.train_faltando_cnn
--epochs 25 --batch-size 4 --size 160 --device cpu` e enviou
`training_report.json` e `training_summary.txt`. A execução local
resultou em:

| Medida | Resultado |
|---|---:|
| Capturas preparadas | 117 |
| Eventos (25 trincas OK + 42 monoimagem) | 67 (57 OK / 10 NG) |
| Treinamento | 54 (46 OK / 8 NG) |
| Desenvolvimento (holdout) | 13 (11 OK / 2 NG) |
| NG detectados (TP) | 0 |
| NG erroneamente considerados OK (FN) | **2** |
| OK corretamente considerados OK (TN) | 11 |
| OK erroneamente considerados NG (FP) | 0 |
| Acurácia | 84,62% |
| Recall NG | **0%** |
| Perda treino época 1 → 25 | 1,337919 → 0,000206 |
| Perda no holdout época 25 | 0,637349 |

A rede respondeu **OK para todos os 13 eventos** e errou os dois NG
do holdout: `2026-10-01_07-53-17-716_FALTANDO.png` e
`2026-10-01_09-53-18-089_FALTANDO.png`. Ambos possuem componente
OCR aproximado R475; esse detalhe sugere dificuldade de generalizar
com poucos NG, sem provar um único defeito de mecanismo universal.
Todas as 25 épocas obtiveram **TP_NG=0** na validação.
Os 84,62% expressam prevalência OK, não detecção confiável.
O declínio quase completo de perda apenas no treinamento
é compatível com sobreajuste.

**v1: REPROVADA para julgamento automático de FALTANDO.**
Nenhum motor de Produção foi substituído.

## CNN FALTANDO v2 — mudanças e comando

A v2 implementa uma CNN comparativa de duas escalas:
imagem integral e recorte central de 70% redimensionado,
preservando mais detalhes da região de inspeção. Cada uma recebe
RGB de gabarito, RGB de teste e diferença absoluta dos dois.
Mantém SIDE/TOP/MID agrupados, respeitando máscara de luz
para imagens legadas SIDE. O modelo utiliza mapas espaciais 2×2
antes da cabeça de classificação, sampler OK/NG balanceado,
perda auxiliar por iluminação, dropout e early stopping.
Não introduz regras físicas de decisão nem usa a memória KNN.

A v2 registra **probabilidades por evento e por iluminação**,
erros FP/FN e melhor época. O mesmo split (seed 42) permite
comparação de desenvolvimento com a v1; **não é teste cego**,
pois já vimos as falhas da v1 nesse conjunto. Para validar
generalização serão necessários eventos inéditos, especialmente
NG reais em TOP/MID, antes de qualquer substituição operacional.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.train_faltando_cnn_v2 --epochs 25 --batch-size 4 --size 160 --device cpu
```

Salva localmente em `reports/faltando_neural/models/experiment_v2_*/`:

- `faltando_cnn_v2_candidate.pt` — pesos **candidatos**;
- `training_report_v2.json` — treino, split e resultados;
- `holdout_predictions_v2.json` — cada caso, verdade,
  probabilidade NG, iluminação e erro;
- `training_summary_v2.txt` — resumo para leitura humana.

**Ainda não executado nos dados reais nesta etapa.**
A v2 permanece desligada no ODIN, com
`production_approved=False`. Enviar os três relatórios
para avaliação antes de discutir roteador KNN/CNN.

## Resultados reais da v2 — 08/10/2026

Arquivos recebidos: `training_report_v2.json`,
`holdout_predictions_v2.json` e `training_summary_v2.txt`.
Foram usados os **117 frames / 67 eventos** já existentes, com
54 eventos no treino (46 OK / 8 NG) e 13 na validação de
desenvolvimento (11 OK / 2 NG); 25 épocas, CPU, batch 4,
imagem 160×160, recorte de foco 70%.

| Indicador de desenvolvimento | CNN v1 | CNN v2 |
|---|---:|---:|
| NG detectados (TP) | 0/2 | **2/2** |
| NG liberados como OK (FN) | 2 | **0** |
| OK corretos (TN) | 11/11 | **11/11** |
| Falsos NG em OK (FP) | 0 | 0 |
| Acurácia | 84,62% | **100%** |
| Recall NG | 0% | **100%** |

Os dois NG SIDE recuperados: `2026-10-01_07-53-17-716_FALTANDO.png`
(score NG 0,999785) e
`2026-10-01_09-53-18-089_FALTANDO.png` (score NG 0,999793).
Scores NG dos 11 OK ficaram entre 0,000016 e 0,000243.
**Scores sigmoid não são probabilidades calibradas de precisão operacional.**

**Achado da seleção de checkpoint:** o relatório original selecionou
época 18 (loss 0,000099), embora a menor perda na curva
seja na época 23 (0,000070). Causa: um limiar absoluto
`0.0001` afetava tanto a escolha dos pesos quanto
a paciência do early stopping. Correção publicada em
`src/scripts/train_faltando_cnn_v2.py`: pesos agora
seguem a menor perda registrada, mantendo tolerância
somente para `patience`. O checkpoint local original
continua sendo o da época 18; requer novo treino para
reproduzir a seleção corrigida.

**Atenção à evidência:** esses 13 eventos são o mesmo
conjunto examinado durante a v1. O resultado é promissor,
mas é **validação de desenvolvimento, não teste cego**.
Apenas 2 NG SIDE foram avaliados e não há NG reais TOP/MID.
Não habilitar CNN ou roteador KNN→CNN automaticamente por
este resultado. Próxima etapa: teste independente com novos
OK/NG reais, particularmente NG TOP/MID, idealmente shadow
inference sem comandar produção.


## Replay CNN v2 do acervo FALTANDO completo (08/10/2026)

**Objetivo:** julgar com os pesos v2 existentes os casos OK/NG
de `public/ok_archive` e `public/ng_archive`:
esperado no último inventário: 10 NG SIDE legado,
32 OK SIDE legado, 25 SIDE OK explícitos, 25 TOP OK e
25 MID OK; **117 PNGs / 67 eventos** (25 trincas
por nome+OCR). Este número não é fixo: o replay compara
o acervo atual com todos os itens do `manifest.json`.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.replay_faltando_cnn_v2
```

**Saídas do replay**:

```text
reports/faltando_neural/replays/archive_v2_<timestamp>/
    archive_replay_v2.json
    archive_replay_v2.txt
```

O script:

1. Carrega a rede treinada em modo inferência puro; **não usa KNN**,
   nem ensina a CNN, nem altera imagens ou memória da produção.
2. Valida que checkpoint v2 e manifesto possuem o mesmo SHA-256
   da preparação; verifica os PNGs originais por hash e a contagem
   total de arquivos atuais. Se o arquivo mudou, falha em vez de
   omitir casos. Se houver outra preparação, informe
   `--manifest "caminho\manifest.json"` e, se necessário,
   `--checkpoint "caminho\faltando_cnn_v2_candidate.pt"`.
3. Avalia SIDE legado como monoimagem. Avalia cada luz da trinca
   individualmente, mas a decisão multilight final é **por evento**:
   máximo logit NG entre as luzes, consistente com o treino.
4. Registra por evento e por PNG scores de NG, rótulo, decisão
   e erros, mais matriz de confusão por modo e total.
5. Não ativa automaticamente o modelo no ODIN mesmo com 100%.

**Como interpretar:**

- `passed_known_archive_regression = true`:
  nenhum dos exemplos conhecidos falhou no replay. É útil para
  identificar se o checkpoint reconhece seu acervo.
- `passed_known_archive_regression = false`:
  alguma imagem/evento falhou. O JSON mostra exatamente qual.
- Nenhum resultado desse comando sozinho comprova
  generalização da CNN em NG de novas peças. Grande parte
  das imagens já esteve no treinamento ou no desenvolvimento.
- Ainda não existem NG reais em TOP/MID; a CNN não teve como
  demonstrar essa classe sob as duas luzes.

**Integração KNN→CNN FALTANDO no ODIN normal:** condicionada
a analisar o relatório real. O replay em si não altera os
motores da produção ou o gate de inicialização.

## Replay completo aprovado e integração no ODIN — 08/10/2026

O operador executou `python -m src.scripts.replay_faltando_cnn_v2`
com os pesos v2 de **época 18**, hash SHA-256
`6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599`.

Resultados: **117/117 PNGs, 67/67 eventos**, com
**10/10 NG e 107/107 OK**, zero FP e zero FN.
SIDE legado: 32 OK + 10 NG; SIDE atual, TOP e MID:
25 OK cada. Não existe NG real TOP/MID. São **casos
conhecidos usados no desenvolvimento/treino**, portanto
não é medida independente de segurança em produção.

**Integração técnica publicada:**

- `main.py` instala `install_faltando_cnn_live(MoEOrchestrator)`
  depois de todos os wrappers físicos, substituindo a decisão
  da categoria `FALTANDO/MISSING` pelo `FaltandoCNNV2`
  diretamente. Nenhum KNN, MissingExpert ou SSIM é consultado
  para esta categoria. Demais categorias continuam normais.
- O checkpoint é lido somente quando necessário e tem caminho
  e SHA fixados ao artefato verificado no replay; se estiver ausente,
  incompatível ou danificado, a rede não libera peça: `REVISÃO OBRIGATÓRIA`.
- A rede recebe referência, teste e foco central, como no treino.
  SIDE legado e SIDE/TOP/MID passam pela máscara de iluminação.
  A fusão multilight final conserva o mecanismo geral da AOI.
- Telemetria registra score NG **não calibrado**, hash dos pesos,
  iluminação e motor CNN no `detail`/`decision_trace`.
- O `production_decision_policy` **bloqueia AUTO-OK** para decisões
  desta CNN experimental até validação independente. O Modo Produção
  permite revisão humana 0=OK/1=NG, sem alterar a automatização
  das demais categorias. Respostas NG já exigiam operador.
- A interface de inspeção mostra o veredito CNN mesmo quando o
  motor não produz retângulos físicos de defeito.

Para atualizar e abrir:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python main.py
```

**Importante:** não criar `models/` no GitHub nem enviar pesos:
o checkpoint reside na pasta local já existente. Confirmar no
debug `cnn_v2_status=INFERENCE_OK` e
`cnn_v2_checkpoint_verified=True`. Nenhum teste físico
de AOI foi reportado após ativar esta integração.


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



---

## 09/10/2026 — Hipótese de motor visual compartilhado: FALTANDO + memória KNN (auditoria offline)

**Solicitação:** a AOI XP pode usar nomes distintos (__EMBORCADO__, __INVERTIDO__,
__DESLOCADO__ etc.) para imagens cujo defeito visual aparente é um
**componente ausente**. O rótulo original não é verdade-terreno da
natureza física do defeito. A pasta __public/ok_archive__ contém
gabarito à esquerda e teste à direita; pequenas diferenças de marcação,
brilho e captura devem ser toleradas por um detector de normalidade visual,
sem transformar automaticamente uma variação natural em NG.

**Cuidado com a premissa:** nem todo componente EMBORCADO/INVERTIDO/DESLOCADO
tem o mesmo mecanismo visual de FALTANDO; a categoria MUITO ADESIVO tem
especialista próprio. Não substituir essas categorias nem liberar uma
peça apenas por essa generalização antes de avaliá-la em NG reais.
O rótulo da pasta OK/NG, isoladamente, também não comprova origem humana
ou independência em relação ao treinamento.

**Implementação desta etapa (somente avaliação, sem mudança de julgamento):**
- __src/services/faltando_cross_category_audit.py__ e
  __src/scripts/audit_faltando_cross_category.py__.
- Inventaria **todos os PNGs** de __public/ok_archive__ e
  __public/ng_archive__ por meio de __inventory_archives__. Separa
  __category_hint__ (nome original da AOI), categoria OCR observada e
  classe do arquivo (OK/NG). Não rebatiza nem move nenhum arquivo.
- Para PNG válido sem conflito OK↔NG de mesmo conteúdo, extrai
  automaticamente gabarito/teste via __AOIPairExtractor__ e executa
  __FaltandoCNNLive.inspect__ em SIDE/TOP/MID, incluindo outras categorias.
  Não chama memória KNN nem realiza treino. Os resultados mostram a
  classificação experimental, pontuação NG não calibrada, revisões,
  falhas de extração e divergências entre pasta e modelo.
- Agrupa resultados por categoria e luz; conta explicitamente
  __archived_ng_called_ok__ e __archived_ok_called_ng__, sem esconder
  modelos indisponíveis, revisões, duplicatas conflitantes ou PNGs
  corrompidos. **Não** interpreta resultados por frame como resultados
  independentes por evento SIDE/TOP/MID.
- Preserva o bloqueio atual de Produção para __FALTANDO CNN experimental__:
  nenhum novo auto-OK nem auto-NG, nenhum pacote/comando 0/1 para a AOI XP.
  O __VerifiedKNNMemory__ permanece isolado por
  placa/componente/categoria/iluminação e só reconhece par exato
  com rótulo humano. Sem autorização para usar similaridade aproximada
  de KNN para liberar automaticamente pequenas variações.

**Comando no PC Windows, no ambiente Python atual:**

~~~powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.audit_faltando_cross_category
~~~

**Relatórios para avaliação:** __reports/faltando_neural/cross_category_audit/audit_*/cross_category_audit.json__
e __cross_category_audit.txt__. Avaliar quantos NG confirmados de cada
categoria foram chamados OK, quais categorias/iluminações não foram
cobertas, mudanças de inscrição e casos multilight. Mesmo 100% do
arquivo histórico **não prova generalização** nem valida auto-OK por CNN.

**Próxima decisão só depois dos resultados e revisão independente:**
caso a análise mostre que as categorias AOI são proxies confiáveis de
ausência física, considerar um roteador **visual unificado** com
rastreabilidade da categoria original e casos suspeitos em revisão;
exigir NG reais independentes da CNN para cada família/iluminação
antes de autorizar liberação 0/1 sem operador. **Modo Produção
100% automático por CNN não foi ativado nesta etapa.**



---

## 09/10/2026 — Telemetria visual CNN FALTANDO v2 + KNN e debug XP

**Objetivo:** eliminar painéis vazios quando o ODIN decide com
__faltando_cnn_v2.py__ e/ou recuperação exata __KNOWN_KNN__.
**Escopo estritamente visual e de diagnóstico**: não muda motor, memória,
treinamento, consenso SIDE/TOP/MID, limiares produtivos ou comandos 0/1.

**Análise dos especialistas:**
- Widget CNN FALTANDO v2 na visão normal e nos três lanes SIDE, TOP e MID.
  Mostra categoria AOI original, rota, score NG não calibrado, complemento
  OK (não probabilidade de acerto), status de inferência, iluminação,
  verificação e digest abreviado do checkpoint.
- Widget KNN exato quando essa foi a rota efetiva, com rótulo humano;
  não declara que executou uma CNN que não participou.
- Nos eventos multilight, o motor atual de cada luz aparece sem os
  cards físicos antigos vazios. As telas de adesivo permanecem intactas.

**Decisão e confiança:**
- CNN: visualiza score NG **não calibrado**; não apresenta os percentuais
  como acurácia/certidão de que a peça está OK.
- Multilight: lista os votos SIDE/TOP/MID, o estado do consenso e a
  elegibilidade supervisionada, que continuam calculados pelo motor real.
- KNN: rótulo humano recuperado de PNGs exatos, sem percentuais
  fictícios de similaridade não calculada.

**Influência dos motores:**
- Mostra uma linha CNN por luz realmente inferida, com score NG original.
  Em fusão multilight, são **votos independentes**, não uma soma ponderada
  de porcentagens inventadas.
- Caso KNOWN_KNN registra que a decisão se originou do registro humano
  exato; se uma luz não usou CNN, não recebe score CNN.
- Motores físicos legados mantêm sua visualização já existente.

**Memória de anomalias KNN:**
- NEW_CNN = KNN pesquisada antes da CNN, **nenhum par exato humano**;
  não significa que similaridade da KNN seja 0%.
- KNOWN_KNN = histórico humano exato, identifica OK/NG e origem.
- MULTILIGHT_MIXED = identifica quais luzes usaram KNN e quais usaram CNN.
  Não desenhar barras 0% em ausência de medição.

**Debug copiável do XP:**
- Resumo visível da CNN, checkpoint SHA-256, status, score NG local,
  pontuações e rotas SIDE/TOP/MID, consenso, motivo e status de
  supervisão, além de bloco KNN (match exato e rótulo humano).
- Preserva a estrutura JSON e as análises SIDE/TOP/MID detalhadas;
  os antigos campos de ausência física/INVERTIDO sem cálculos deixam
  de poluir o bloco inicial quando só a CNN foi usada.
- A fusão mantém **os scores reais de cada luz** em
  __cnn_v2_light_diagnostics__; não reaproveita os valores apenas SIDE.

**Teste de regressão:** __tests/test_neural_telemetry_panels.py__, com
fixture INVERTIDO SIDE/TOP/MID de scores distintos baseado em captura
real de 09/10/2026. Os testes comprovam renderização e mapeamento
dos dados, **não avaliam sensibilidade da CNN a novos NG físicos**.

