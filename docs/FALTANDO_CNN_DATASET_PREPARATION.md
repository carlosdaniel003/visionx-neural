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



## 09/10/2026 — Estado da memória KNN e influência visual

- O painel **Memória de Anomalias • KNN** agora mostra estado de busca
  independentemente de um rastreamento de pesos de outros especialistas.
- **JÁ VISTO • MEMÓRIA KNN** só aparece para par gabarito/teste humano
  exato. **CASO NOVO • SEM MATCH EXATO** significa que aquele par não foi
  recuperado, não que o tipo físico de defeito jamais tenha ocorrido.
- A mensagem flutuante permanece até o julgamento e desaparece com a
  mesma animação sincronizada do 0/1; fontes com rota ausente não
  recebem a etiqueta de "primeira vez".
- **Influência dos motores** apresenta por luz SIDE/TOP/MID cards
  explicativos com verde OK, vermelho NG, amarelo revisão e score
  NG bruto da CNN. A proporção da barra não é confiança calibrada,
  percentual de acerto nem peso real de fusão aritmética.
- KNN exata apresenta rótulo humano sem inventar score de similaridade.
- Não foram alterados treino, classificadores, prioridades de memória,
  julgamento 0/1 ou política de produção.


## 09/10/2026 — Memória KNN na mesma mensagem flutuante do veredito

**Correção do layout:** a antiga mensagem flutuante independente
"MEMÓRIA MISTA / JÁ VISTO / CASO NOVO" sobrepunha a mensagem
"ILUMINAÇÃO ATUAL". O ODIN não instala mais o overlay KNN separado.
O cartão existente de veredito ("FALHA FALSA",
"DEFEITO REAL" ou "REVISÃO OBRIGATÓRIA") mostra abaixo,
**dentro da mesma borda amarela**:

- **JÁ VISTO • KNN EXATO:** par gabarito/teste confirmado por humano,
  idêntico a registro de memória, não similaridade aproximada.
- **CASO NOVO • SEM MATCH EXATO:** não há par exato humano KNN para
  aquele evento; não se afirma que o defeito físico é novo.
- **MEMÓRIA MISTA • 3 LUZES:** SIDE/TOP/MID tiveram rotas diferentes.
- **MEMÓRIA CONFLITANTE • REVISÃO:** recuperação contraditória.
- Sem rota registrada: segunda linha oculta, sem inventar consulta KNN.

O cartão unificado preserva **dimensões, posição, aparência, tempo
de animação, sincronização de entrada e fade-out do comando 0/1**.
A informação KNN compartilha o mesmo efeito de opacidade do veredito,
não tem animação ou tempo de vida próprios. O painel de
"ILUMINAÇÃO ATUAL" permanece abaixo, sem sobreposição.

Os métodos antigos do módulo de memória independente foram
preservados para testes/compatibilidade, mas não são instalados por
__main.py__. A correção é **somente de interface**; sem alterações
nos modelos, memória, treinamento, julgamento ou produção.


## 09/10/2026 — Status visual binário JÁ VI / NUNCA VI e painel KNN por iluminação

**Necessidade observada na AOI:** uma inspeção DESLOCADO teve
SIDE = \`KNOWN_KNN\` (OK), TOP = \`KNOWN_KNN\` (OK) e
MID = \`NEW_CNN\` (OK). O aviso "MEMÓRIA MISTA" era tecnicamente
descritivo, mas não respondia à pergunta operacional "já vi esse caso?".
Para este evento, a resposta correta na tela é **JÁ VI**.

**Novo contrato de apresentação (não altera decisão):**
- **JÁ VI**: ao menos uma das luzes SIDE/TOP/MID corresponde exatamente
  a um par gabarito/teste já confirmado por humano no KNN.
- **NUNCA VI**: existe consulta válida ao KNN, mas nenhuma luz consultada
  encontrou par humano exato. Isto NÃO prova que a classe física de defeito
  seja inédita; apenas que o par exato não foi recuperado.
- Dados ausentes / rota inválida / conflito sem match verificado: não
  inventar "NUNCA VI"; o cartão de memória mantém explicação técnica,
  e o veredito não exibe subtítulo KNN incerto.
- Nunca mostrar "MEMÓRIA MISTA" ou outra **terceira alternativa** na
  mensagem flutuante. O debug continua registrando as três rotas individuais.

**Memória de anomalias KNN:**
- Cabeçalho visual amplo com JÁ VI ou NUNCA VI, preenchimento colorido.
- Cartões SIDE/TOP/MID, com **verde** para KNN par exato confirmado,
  **amarelo** para imagem nova direcionada à CNN, **cinza** para dado
  indisponível; status por luz e rótulos humanos OK/NG preservados.
- Exemplo DESLOCADO observado: "JÁ VI • 2/3 ILUMINAÇÕES
  RECONHECIDAS", SIDE verde OK, TOP verde OK, MID amarelo CNN.
- Porcentagens da CNN não são interpretadas como probabilidades de
  reconhecimento pela memória. Sem simular similaridade KNN.

**Teste de regressão:** \`tests/test_binary_knn_memory_dashboard.py\`.
A mudança só afeta os rótulos da interface e o desenho do painel.
Não altera KNN, CNN, treinamento, decisões em Teste/Sombra/Produção,
checkpoint, nem o envio de 0/1 à AOI XP.


## 09/10/2026 — Painel Neural Explicável por Iluminação (SIDE/TOP/MID)

**Natureza:** visualização diagnóstica e auditável dos pixels, sem alterar
classificação, treinamento, KNN, checkpoint, limiares ou comandos 0/1.

**Fonte dos dados:** utiliza os recortes de teste **e gabarito** dos dois
epicentros calculados por \`build_lighting_context\` e já utilizados por
\`build_adhesive_view_payload\`. Não reexecuta a extração da AOI,
não desenha bounding boxes manualmente e não altera a imagem de origem.

**Por iluminação** (SIDE / TOP / MID) o painel apresenta:
- **Epicentro maior (contexto):** (1) cinza da imagem teste;
  (2) mapa de **diferenças visuais absolutas** entre teste e gabarito
  sobreposto à imagem real, sem falsos destaques quando a diferença é zero;
  (3) aproximação local em blocos obtida por média de pixels.
- **Epicentro menor (foco AOI):** as mesmas três representações,
  recortadas do quadrado menor extraído originalmente pela AOI.
- Métricas apenas **descritivas**: dimensões do recorte em pixels,
  diferença média absoluta na faixa 0..255 e desvio padrão de contraste.
  Pixels indisponíveis mostram mensagem explícita e não score zero.

**Precisão terminológica obrigatória:**
- A CNN FALTANDO v2 *não recebe imagens em escala de cinza*:
  usa quatro entradas RGB com letterbox: referência e teste completos,
  e seus crops centrais (\`focus_fraction\` especificado no checkpoint).
- As caixas AOI **maior e menor** são recortes de diagnóstico.
  Não são necessariamente iguais ao crop central da entrada real da CNN.
- **Mapa de diferenças não é Grad-CAM, saliência treinada,
  atenção da CNN, evidência de causalidade nem reconstrução por decoder.**
  Reconstrução por blocos aqui é somente pixelização/agrupamento visual.
  O score NG da CNN permanece no card original do especialista.

**Layout:** uma fila horizontal de painéis de especialistas
compatível com a barra horizontal já existente; por dentro, cada
epicentro reorganiza as três miniaturas em 3, 2 ou 1 colunas conforme
a largura, com rolagem vertical para notebooks e imagens inteiras
escaladas sem zoom infinito. Nenhum ROI é armazenado no arquivo de
memória; os dados da UI são descartados quando o ciclo é limpo.

**KNN:** quando uma iluminação é KNOWN_KNN, o painel continua exibindo
os pixels diagnósticos, mas informa **CNN não executada** nessa luz.
Os resultados OK/NG e a memória KNN não são afetados.

**Regressões:** \`tests/test_neural_evidence_board.py\`
junto com testes existentes de inspeção e especialistas SIDE/TOP/MID.


## 09/10/2026 — Revisão visual do Painel Neural Explicável (horizontal)

Motivação: o painel de SIDE mostrava card textual "JÁ VI • KNN EXATO"
à esquerda e o card azul/acinzentado de evidências à direita, com
três imagens empilhadas verticalmente e barra de rolagem vertical
*interna*. Isso desperdiçava altura e destoava do tema do ODIN.

**Novo contrato da interface (modo multilight):**
- Remover o card de **texto KNN/CNN** do setor de especialistas sempre
  que os recortes visuais do evento estiverem presentes. Os detalhes
  completos continuam disponíveis nos painéis Memória de Anomalias,
  Decisão e Confiança, e Copiar debug. Sem alteração de memória.
- Um **único painel preto com borda amarela** para os recortes AOI.
  Descartados fundos azul-acinzentados e textos explicativos longos.
- Uma faixa horizontal contínua com **seis cards** de mesmo tamanho:
  **MAIOR — Cinza, Diferenças, Blocos**; **MENOR — Cinza,
  Diferenças, Blocos**; separador sutil entre os epicentros.
- Uma **barra de rolagem horizontal interna**, com setas ◀/▶,
  desliza para os lados quando as seis imagens não couberem no espaço.
  Jamais reorganizar esses cards em colunas verticais. O contêiner
  neural e o scroll de especialistas não possuem scrollbar vertical.
- Altura fixa do painel neural, miniaturas completas (aspect ratio
  preservado) em todos os tamanhos de viewport. Em monitor grande a
  faixa cabe integralmente; em notebook, deslocamento horizontal.
- SIDE/TOP/MID permanecem regiões independentes. O controle de
  navegação horizontal mestre legado continua disponível para
  especialistas determinísticos; o painel neural tem controle próprio.
- Quando ROI maior ou menor está ausente, mostrar SEM RECORTE em seus
  respectivos cards, nunca inventar pixels/score.
- Grad-CAM e mapas de atenção não são inferidos desses mapas de
  diferença; as visualizações continuam diagnóstico de pixels
  externos ao checkpoint.

Mudança estritamente de **apresentação**, não toca CNN FALTANDO v2, KNN,
training, inferência, julgamento SIDE/TOP/MID ou comandos 0/1 da XP.
Teste: \`tests/test_neural_evidence_board.py\`.


## 09/10/2026 — v2 explicabilidade neural REAL (sem filtros AOI)

**Mudança:** substituir a tríade antiga de cinza/diferenças/blocos (operações
OpenCV sobre pixels AOI) por mapas computados exclusivamente com as
atividades internas do checkpoint **CNN FALTANDO v2** validado por SHA-256.

**Modelo e limitação importante:** \`FaltandoCNNV2\` é classificador sem decoder.
Seu encoder recebe **9 canais** (gabarito, teste e módulo da diferença)
para cada escala, não encoders RGB independentes. A arquitetura usa
5 blocos Conv/GroupNorm/SiLU, pooling 2×2 e head linear. O código usa a
saída espacial de **\`encoder.4\`** antes do pooling.

**Três cartas neurais reais, em cada um dos dois epicentros AOI:**

1. **DIF. LATENTE** — comparar, com os MESMOS pesos \`encoder[:5]\`,
   as features do par \`(gabarito, teste)\` contra o contrafactual
   \`(gabarito, gabarito)\`; reduzir por média \`abs(Fpar - Fbase)\`.
   Diferença zero produz azul, azul→vermelho corresponde à magnitude
   normalizada RELATIVA de cada mapa, não um score operacional.
2. **GRAD-CAM** — forward real pela CNN (os quatro tensores de entrada,
   máscara SIDE e duas escalas), gradiente do logit da classe local
   (NG se logit≥0, caso contrário OK representado por -logit)
   sobre a saída de \`encoder.4\` do ramo completo; média dos gradientes
   sobre o espaço, soma dos canais ponderados, ReLU e normalização.
3. **ATIVAÇÃO CNN** — energia RMS espacial dos canais de \`encoder.4\`
   gerados pelo par real de referência/teste, não transformação da
   imagem original. **NÃO** é decoder nem reconstrução RGB literal.

**Geometria:** cada epicentro é submetido à mesma rotina
\`_letterbox_rgb\` e \`_focus_crop\` que alimenta o classificador.
A visualização desfaz o padding para mostrar o mapa apenas no recorte.
**As imagens são sondagens da CNN sobre as regiões AOI**, não o Grad-CAM
da inferência operacional original no quadro integral: não deduzem
o score final nem demonstram causalidade completa. O score local
é isolado e identificado como não calibrado.

**Roteamento e segurança:**
- Em \`KNOWN_KNN\`, o resultado 0/1 continua vindo **exclusivamente da
  memória humana**; a mesma CNN FALTANDO v2 pode rodar após o julgamento
  para EXPLICAR visualmente o par, sem consultar/modificar a memória.
- Em \`NEW_CNN\`, a explicação ocorre **após o julgamento CNN original**,
  em worker Qt auxiliar sem bloquear a thread de interface.
- ADESIVO e categorias fora de \`uses_faltando_v2\` não iniciam worker CNN.
- A instância auxiliar usa \`FaltandoCNNLive._load()\`, com a verificação
  atual de SHA, esquema/versão e ponteiro online, sem treinar ou trocar pesos.
  A captura do hook de forward/grad e dos contrafactuais é serializada por
  lock para não cruzar as iluminações; mapas antigos são descartados por
  ID do ciclo ao receber uma nova inspeção.
- Em checkpoint ausente, inválido, categoria não suportada, recorte
  indisponível ou exceção, a interface escreve **CNN INDISPONÍVEL**
  e **não** substitui imagens por filtros OpenCV, simulados ou scores zero.

**Visual:** seis cards preto/amarelo em uma linha e rolagem horizontal,
três para o epicentro maior e três para o menor; sem scroll vertical
interno. O checkpoint continua sendo classificativo; o termo
"reconstrução de ativação" se refere somente a projeção RMS, não a
reconstrução em pixels aprendida por decoder.

**Testes:** \`tests/test_faltando_explainability.py\` faz forward real,
autograd, mapas sobre pares iguais/diferentes, não alteração de pesos,
verificação de checkpoint e faltas de ROI;
\`tests/test_neural_evidence_board.py\` cobre worker isolado,
sem pixel fallback, evento antigo descartado, modo KNN e layout.


## 09/10/2026 — Hotfix: fechamento súbito do ODIN após executar CNN Grad-CAM

**Sintoma informado na fábrica:** ao terminar uma inspeção, o aplicativo
fechou sem diagnóstico. A regressão mais recente colocava uma instância
PyTorch em cada QRunnable da thread pool GLOBAL da interface Qt. Isso
pode resultar em falha **nativa**, exaustão de memória ou múltiplas
execuções convolucionais simultâneas. Sem o log da máquina não é
possível afirmar a causa exata.

**Mitigação de isolamento e segurança:**
- Sondas explicativas passam a ser executadas como processo Python
  **filho descartável** com \`python -m
  src.core.neural.faltando_explainability_runner --child\`; nenhuma
  chamada a Torch/Grad-CAM ocorre no processo Qt de visualização.
- Limite **uma** execução por vez para todas as iluminações (sem
  três modelos Torch paralelos). Filhos forçam
  \`OMP_NUM_THREADS=1\`, \`MKL_NUM_THREADS=1\`.
- Timeout de 45 s para filhos; erros, checkpoint inválido,
  encerramento por acesso inválido à memória/segfault, falta de
  RAM e outros códigos não-zero aparecem como **CNN INDISPONÍVEL**
  em vez de encerrar o ODIN.
- Entradas de ROIs são \`npz\` temporário com tipo e dimensões
  validadas, saída também \`npz\` sem pickle e somente mapas
  explicitamente marcados \`neural=true\`, reduzidos a até 640 px
  de lado, nunca filtros pixelados como fallback.
- Tarefas Qt não usam mais \`QRunnable.setAutoDelete(True)\` enquanto
  o slot aguarda retorno; objetos são retidos até processamento
  do sinal para evitar descarte prematuro.
- Id de ciclo ainda impede mostrar mapas antigos após mudança de
  inspeção, e nenhuma exceção modifica dados de decisão, KNN,
  treinamento ou comandos 0/1.
- **Desativação de emergência** sem código/merge: iniciar o ODIN
  com \`$env:VISIONX_DISABLE_NEURAL_MAPS="1"\` na sessão PowerShell.
  As inspeções continuam; os seis cards mostram "CNN INDISPONÍVEL".
  Remover a variável da sessão e reiniciar para reativar mapas.

**Limitação:** testes automáticos podem simular acesso inválido e
timeout, mas não reproduzem necessariamente drivers, DLLs ou
OpenMP da estação Windows XP/Windows da fábrica. Repetir uma inspeção
em modo Teste e coletar traceback/event log antes de reativar
produção automática.

**Regressões:** \`tests/test_faltando_explainability_crash_isolation.py\`
+ suíte existente da CNN v2/Qt.


## 09/10/2026 — SSIM Debugger: TESTE real + sobreposição de ativações CNN

**Revisão do operador:** os mapas latentes/Grad-CAM/RMS isolados pareciam
gradientes coloridos semelhantes entre si, sem mostrar em que região do
componente a rede respondia. O painel deve permitir comparar o componente
VISÍVEL com a projeção CNN, à maneira de um SSIM Debugger, sem substituir
a explicabilidade por uma simples diferença OpenCV.

**Nova composição visual** \`src/core/neural/faltando_activation_overlay.py\`:

- **Fundo de cada cartão = recorte real da imagem TESTE**, correspondente
  ao epicentro MAIOR ou MENOR retornado pelo AOI.
- **Sobreposição = somente mapa 2D efetivamente extraído da CNN**:
  diferença latente contrafactual, Grad-CAM local ou projeção RMS
  das ativações da camada encoder.4.
- A intensidade da cor/opacidade depende do sinal CNN normalizado,
  com áreas pouco ativas preservadas como TESTE ORIGINAL (não se
  pinta todo o componente com uma camada azul artificial).
- **GAB = miniatura do GABARITO** dentro do canto superior esquerdo,
  com borda amarela e legenda breve. O gabarito é apenas referência
  visual: não produz peso, score, bbox inferida ou julgamento novo.
- Diferença latente usa Jet (azul → vermelho); Grad-CAM usa Hot
  (escuro → amarelo/branco); ativação interna usa Bone (neutros).
  O formato dos mapas é escala relativa **local** a cada epicentro;
  jamais tratar a coloração como confiança probabilística da CNN.
- As miniaturas de todos os modos usam a mesma imagem de TESTE original,
  facilitando identificar no componente as regiões com ativação.
  Mantêm-se os seis cards lado a lado, tema preto/amarelo e apenas
  scroll horizontal interno responsivo.
- Os 3 resultados continuam derivados da **CNN v2 real**, com
  checkpoint validado por SHA e execução no subprocesso isolado.
  O modelo continua SEM decoder; “ativação CNN” é projeção RMS,
  **não** reconstrução pixel a pixel aprendida.
- Se a CNN estiver indisponível, **não** gerar overlays falsos com
  filtros clássicos de pixels. O painel mostra erro como antes.

**Testes:** \`tests/test_faltando_activation_overlay.py\` verifica que
sem sinal neural o TESTE original é preservado, uma área com resposta
alta muda somente onde deveria, o gabarito está na miniatura
identificada e nenhum mapa inválido vira filtro substituto.
\`tests/test_faltando_explainability.py\` valida a nova composição após
forward/autograd real, e \`tests/test_neural_evidence_board.py\`
valida os rótulos e a entrega ao widget PyQt.
Nenhuma modificação em inferência operacional, memória KNN, classificação
NG/OK, teclas 0/1 ou modo Produção.


## Modo Sombra de baixa latencia - 09/10/2026

- Somente no Modo Sombra: a primeira imagem SIDE continua com OCR. As iluminacoes TOP e MID reutilizam board, parts, value e category da SIDE da mesma peca; nao repetem OCR.
- Gravar os PNGs de depuracao das barras, renderizar seis visoes neurais e executar as reconstrucoes Grad-CAM nao fazem parte do caminho critico de Sombra. As capturas originais e os dados de aprendizado continuam sendo preservados.
- Para categorias da CNN FALTANDO v2, a inferencia recebe os pares completos (gabarito/teste) e seu crop interno. Detecao SSIM e extracao geometrica externa sao dispensadas apenas nesta rota de Sombra. Outras categorias mantem sua logica original.
- Em TOP, o comando de iluminacao MID pode ser enviado apos a captura TOP, antes da analise TOP, sobrepondo rede e CPU. Sao apenas LEFT/RIGHT/DOWN de iluminacao. O ODIN nunca deve enviar PRESS_0 ou PRESS_1 no Sombra.
- Tecla 0/1 da operadora XP encerra imediatamente a coleta da peca e grava em background somente as luzes efetivamente capturadas ANTES do comando. Isto evita atribuir o rotulo humano aos frames tardios da peca seguinte. Trincas completas SIDE/TOP/MID continuam gerando tres registros com um rotulo; pares parciais sao registrados sem treinamento incremental automatico multilight.
- Os registros de mesmo evento mantem event_id e lighting_mode por foto. Imagens sao salvas mesmo quando a IA concorda; a deduplicacao de arquivo existente permanece ativa. A persistencia no disco e a recarga KNN ocorrem fora da thread da interface.
- Medir tempo real no notebook com frames XP e hardware de fabrica. Meta 2-3 segundos e objetivo, nao benchmark aprovado. O envio pela rede, duas imagens estaveis, OCR inicial e latencia fisica das trocas de luz continuam influenciando o tempo.
- Testes: test_shadow_fast_capture.py, test_shadow_partial_persistence.py, test_adhesive_multilight_automation.py, test_multilight_learning.py e test_fast_xp_decision_cycle.py.
