## 16/10/2026 — Ativação local reversível do checkpoint 97,70% (ENSAIO ASSISTIDO)

O operador solicitou **substituir a CNN atualmente usada** pelo primeiro
checkpoint que obteve 85/87 (97,70%) no holdout experimental. A CNN
original **reprovou a regressão histórica de 117 OK**, acertando
somente 98/117 (19 falsos NG). Por isso a implementação permite
**substituição efetiva dos pesos de INFERÊNCIA CNN**, mas SOMENTE com
**revisão humana obrigatória**, sem decisões automáticas 0/1 do XP
por esse checkpoint. Isso é um ensaio supervisionado em estação, NÃO
aprovação de Produção. Não promete 97,70% de acurácia operacional:
no mesmo teste de 106 imagens, teve 93/106, detectou 63/63 NG mas
rejeitou 13/43 OK. No teste antigo 87, 29 imagens reaparecem no
teste de 106. Validação independente permanece pendente.

**Implementação:** `src.scripts.deploy_faltando_97p70_trial`. O
instalador autentica a identidade SHA-256 do checkpoint original
(`c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c`)
contra relatório original e benchmark local, valida o checkpoint
original atual antes de instalar, copia os pesos experimentais
sem alterá-los, executa **warmup no carregador real**, depois
escreve atomicamente `reports/neural_online/live_active.json`
no contrato `visionx.specialist_active_pointer.v1`.

**Atenção:** a existência de um `live_active.json` qualquer causa
bloqueio da instalação. Não é permitido sobrescrever ponteiro de
treinamento online/implantações anteriores. Não renomear ou editar
manualmente nenhum ponteiro. A transição exige **ODIN fechado** e
reinício posterior, para impedir mistura de checkpoints no cache da
sessão ou captura da mesma peça.

**Comandos PowerShell (ODIN fechado, venv ativo):**

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

# Estado atual (também detecta ponteiro inválido)
python -X utf8 -m src.scripts.deploy_faltando_97p70_trial status --root .

# INSTALA / SUBSTITUI CNN na próxima inicialização, SEM AUTOMATISMO 0/1
python -X utf8 -m src.scripts.deploy_faltando_97p70_trial install --root . --ack CONFIRMO_19_FALSOS_NG_E_REVISAO_MANUAL

# Recomeçar o ODIN e conferir modo revisável, sem ativação automática
python -X utf8 -m src.scripts.deploy_faltando_97p70_trial status --root .
```

Para reverter **com ODIN fechado**:

```powershell
python -X utf8 -m src.scripts.deploy_faltando_97p70_trial rollback --root .
```

Depois **reiniciar** o ODIN. O rollback move o ponteiro para
`reports/neural_online/manual_trial_audits` e a CNN original fixada
volta a ser escolhida por `FaltandoCNNLive`; os dois arquivos de peso
são preservados para auditoria. Se o ponteiro tiver sido alterado
ou for de outra implantação, o comando falha sem destruí-lo.

**Trava produtiva implementada no nível final:** a política
`production_decision_policy` bloqueia **auto-OK E auto-NG**
para resultados gerados pelo hash desse ensaio, inclusive quando a
fusão SIDE/TOP/MID sobrescreve `operator_review_required` da luz
individual. A decisão só pode ser confirmada pelo operador, 0/OK
ou 1/NG. `FaltandoCNNLive` também identifica cada inferência
individual com `cnn_v2_trial_manual_only=true` e força
`production_review_required=true` quando este checkpoint é carregado
via ponteiro validado.

**Não afetado:** memória KNN, OCR, XP, configuração de inspeção,
máscaras, thresholds, modelos DESLOCADO/ADESIVO, pesos originais,
dados brutos, aprendizado automático. Na falta de SHA, ponteiro
inválido ou carregamento incompleto: falha para `REVISÃO
OBRIGATÓRIA`, jamais OK silencioso.

**Limite:** este modo ASSISTIDO ainda não prova que o modelo
é mais seguro que a CNN base na linha. Após observar novas peças
e uma regressão independente incluindo OK e NG de todas as
categorias, decidir se é adequado promover o uso produtivo normal.
Até lá, nenhuma automatização poderá se basear nesse candidato.

---

## 16/10/2026 — Benchmark realizado: preservar o candidato de 97,70%

**Decisão operacional de desenvolvimento:** parar a série de fine-tunings
que vinham reduzindo o recall de NG, e **congelar como REFERÊNCIA
EXPERIMENTAL** o primeiro candidato CNN V2. Checkpoint SHA-256 fixo:
`c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c`.
Sem alterar runtime, pesos, KNN, limiares, fusão, XP ou ponteiro ativo.

**Relatório real do usuário:**
`reports/aoi_universal_quality/benchmarks/20261016T181313_610125Z_e527c1/head_to_head_report.json`.
O benchmark autenticou os dois checkpoints e reutilizou as mesmas
**106** imagens do teste universal (sem retreinamento), mas **29
das 106** sobrepõem imagens já vistas no holdout antigo. Portanto, não
é avaliação cega e não permite reivindicar 97,70% em produção.

| Indicador no MESMO conjunto de 106 | Original 97,70% | Universal |
|---|---:|---:|
| Acertos gerais | **93/106 (87,74%)** | **93/106 (87,74%)** |
| NG corretamente reconhecidos | **63/63 (100%)** | 53/63 (84,13%) |
| **NG→OK** (grave: defeito liberado) | **0** | **10** |
| OK corretamente reconhecidos | 30/43 | **40/43** |
| **OK→NG** (falso alarme) | **13** | **3** |

O modelo original **não ganhou em acurácia total**, mas fez **0 NG→OK**
nessas 106 imagens. Não se deve usar a CNN universal sozinha para
substituí-lo: ela deixou passar dez NG. Também não se deve afirmar que
a CNN original está validada universalmente, porque neste teste
**DESLOCADO, EMBORCADO e INVERTIDO têm apenas OK**; os únicos NG nas
categorias AOI são 62 em FALTANDO e 1 em MUITO ADESIVO. A categoria
AOI pode estar errada sobre a falha física. O usuário já havia
identificado INVERTIDO físico em casos arquivados como FALTANDO, e
a CNN original os detectou neste ensaio (por exemplo os MID 0851 e 1437).

**Distribuição dos 13 OK→NG originais:** EMBORCADO 5; INVERTIDO 3;
DESLOCADO 2; FALTANDO 2; MUITO ADESIVO 1. Portanto, **11/13** estão
fora da categoria AOI `FALTANDO`, mas continuam sendo casos de qualidade
geral que não podem ser simplesmente ignorados. Na série anterior de
117 imagens históricas do FALTANDO, a CNN original teve apenas
**98/117** acertos: **19 OK→NG**. Os resultados antigos 85/87=97,70%
e 68/68 NG pertencem a OUTRO holdout e não substituem a medição 93/106.

**Diagnóstico em limiares operacionais preexistentes (0,10/0,90)
SEM alterar limiares:** a CNN original apresentou 63 NG conclusivos,
zero NG liberados ou em REVIEW; dos 43 OK, 22 OK conclusivos,
13 REVIEW e **8 falsos NG conclusivos**. Isso reforça que a melhora
necessária é no falso alarme OK→NG, sem degradar a detecção NG.

**Discordância como hipótese SOMENTE offline, não regra produtiva:**
dos 106, os dois concordam em **83 acertos** e **3 erros (ambos marcam
NG em OK)**. Discordam em **20**: **10 NG** que a CNN original detectou
e a universal classificou OK, e **10 OK** que a original marcou NG e
a universal classificou OK. Se uma análise hipotética marcasse toda
discordância como **REVIEW, nunca auto-OK**, haveria 20 REVIEW, 53 NG
conclusivos corretos, 30 OK conclusivos corretos e 3 falsos NG restantes
nas 86 decisões concluídas; **não** satisfaz 97% nem demonstra
segurança operacional. Essa simulação não autoriza substituir o
especialista nem introduzir um novo filtro na linha.

**Direção aprovada para investigação futura:** preservar o checkpoint
original e seu recall como referência; avaliar reduções de falso
alarme **sem liberar automaticamente qualquer NG** e sem usar o
holdout já reutilizado para escolher novo threshold, época ou regra.
Se houver necessidade de um segundo estágio, somente um protocolo
offline e reversível que encaminhe casos incertos para REVISÃO,
após definir a semântica operacional de REVIEW. Precisamos de
exemplos físicos independentes e de regressão de todas as categorias
antes de qualquer promoção. O usuário **já revisou** as pranchas dos
14 exemplos difíceis; não solicitar repetição dessa revisão.

---

## 16/10/2026 — Voltar a focar no candidato de 97,70% (SEM treino)

**Diretriz do operador:** cessar a sequência de retreinamentos e focar
no **primeiro checkpoint** de FALTANDO V2 que atingiu **85/87
(97,70%)** no holdout original (68/68 NG e 17/19 OK). SHA-256
`c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c`.
É necessário manter no relatório **o problema que o reprovou**:
no conjunto **histórico original**, acertou apenas **98/117**,
classificando **19 imagens OK como NG**. A porcentagem 97,70% foi
obtida em **outro conjunto** que o da CNN universal, e não autoriza
Produção.

A CNN universal recente alcançou 93/106 (87,74%) em suas **106
imagens**, com 10 NG→OK e 3 OK→NG, mas não é possível
interpretar a queda de 97,70% para 87,74% como piora diretamente
comparável por causa das diferentes partições.

Para comparar **os mesmos 106 pares** sem treinamento, alterar
checkpoint, fusão ou limiares, foi criado o CLI de benchmark somente
leitura `src.scripts.benchmark_original_97p70_offline`.
Ele:
1. Encontra e autentica o relatório e checkpoint originais com
   SHA-256 **fixado no código**, modo experimental e métricas
   85/87 e 98/117.
2. Autentica o relatório/checkpoint universal por hash,
   schema e proveniência, recompõe a **mesma partição** das 559
   amostras pós deduplicação e confirma a igualdade dos **IDs
   de treino, validação e teste** do relatório publicado.
3. Reavalia **apenas a CNN 97,70%** nos mesmos 106 pares
   originais do teste universal. Reusa os scores da CNN universal
   publicados no `training_report.json`, sem voltar a treiná-la.
4. Produz tabela de acurácia, NG→OK, OK→NG por categoria,
   erros exclusivos de cada CNN, discordâncias, marcadores
   conhecidos da triagem humana, e diagnóstico **separado** com
   limiares já existentes 0,10/0,90 (a métrica primária .50
   é apenas do laboratório).
5. Preserva a medição histórica 98/117, explicita exemplos
   vistos em holdouts anteriores, nunca altera nenhum peso,
   dataset, `live_active.json`, calibração nem roteamento.
   Não seleciona automaticamente modelo vencedor, não promove
   candidato e não reivindica avaliação cega.

**Comando local (após integrar esta PR à `central`):**

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

python -X utf8 -m src.scripts.benchmark_original_97p70_offline --root . --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json" --original-report "reports/faltando_neural/archive_experiments/20261016T125315_996834Z_c3e70a/training_report.json" --universal-report "reports/aoi_universal_quality/experiments/20261016T174911_634701Z_347cc2/training_report.json"
```

Saída: `reports/aoi_universal_quality/benchmarks/<run>/head_to_head_report.json`.
A comparação **não exige o modelo universal ser carregado para inferência**,
pois reutiliza suas predições originais assinadas por SHA do checkpoint
e da fonte. Se qualquer fonte mudou ou não existe, falha sem escrever.

**Próximo passo, somente após o benchmark:** identificar qual arquitetura
tem melhor cobertura NG **nesse mesmo conjunto** e quantos OK históricos
a CNN original deixa passar; só elaborar eventual correção com
método e amostras *independentes*, sem repetir otimização ad hoc
do holdout que já foi visto. O checkpoint de 97,70% permanece
**referência experimental fixada**, nunca `live_active.json`.

---

# CNN universal de qualidade AOI — candidato offline de laboratório

**Data:** 16/10/2026. **Status:** experimento de desenvolvimento; sem aprovação de Produção.

## Motivação e proveniência

O operador confirmou que a AOI frequentemente encaminha **FALTANDO físico em outras
categorias**. O modelo universal compara o par **gabarito/teste** e produz uma
decisão experimental binária de **qualidade OK/NG** independente da categoria
atribuída pela AOI. A informação `category_hint` permanece como **metadado
de auditoria**, não é uma entrada da CNN e não se transforma em rótulo de
tipo físico. A arquitetura interna reutiliza a CNN multiescala V2, com
gabarito, teste e diferença; o checkpoint universal tem **schema distinto e
incompatível com o carregador produtivo**.

O manifesto local:
`reports/faltando_neural/run_20261016T154441_358349Z/manifest.json`.
Possui **569** PNGs e pares extraídos: 356 NG, 213 OK **provisórios**.
Cobertura por categoria AOI: FALTANDO 420, INVERTIDO 67, DESLOCADO 43,
EMBORCADO 29, MUITO ADESIVO 10. Iluminações: SIDE 269, MID 150, TOP
150. Trincas por nome 150; eventos físicos multilight confirmados **0**.

A revisão humana de 14 imagens já registrada no arquivo de triagem
`docs/faltando_v2_operator_triage_20261016.json` vale para os **mesmos PNGs
e SHA-256**, independentemente de sua categoria AOI. Desses 14, **13 NG
gerais** (9 FALTANDO + 4 INVERTIDO) permanecem NG no ensaio universal,
e 1 `IGNORAR` é omitido. Não é permitido inventar confirmação para as
demais 555 imagens. Duplicatas exatas podem ser eliminadas com total
registrado. O modelo universal **não classifica o tipo físico**, só OK/NG
provisório de qualidade.

## Comando para treinamento local, após integrar esta PR

Com o ODIN fechado, no ambiente virtual do repositório
`C:\visionx-neural-main`:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
python -X utf8 -m src.scripts.train_aoi_universal_quality_offline --root . --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json" --epochs 5 --seed 42 --validation 0.20 --test 0.20 --image-size 128 --allow-provisional-archive-labels
```

A inicialização padrão é aleatória e reproduzível. Opcionalmente, pode-se
acrescentar `--initialize-from-pinned-champion` para estudar
**transfer learning** com pesos iniciais da CNN V2 experimental já existente
(pesos são apenas **lidos**, e o relatório guarda seu hash). Essa inicialização
tem viés favorável aos exemplos de FALTANDO já vistos e **não** é um
benchmark cego. Recomendação inicial: testar primeiro a opção padrão.

A implementação cria o diretório
`reports/aoi_universal_quality/experiments/<run>/` contendo
`aoi_general_quality_offline_candidate.pt` e `training_report.json`.
Não altera `public/`, `qualification.json`, KNN, OCR, fusão, thresholds,
rotas AOI, ou `reports/neural_online/live_active.json`.

**Guarda de comando:** o parâmetro
`--allow-provisional-archive-labels` é obrigatório. Sem esse opt-in,
nenhuma atualização de pesos ocorre.

## Regras do split e da seleção

- Três conjuntos distintos: ~60% treino, ~20% validação e ~20% teste.
  Não existe garantia de 60/20/20 exatos porque grupos relacionados
  devem permanecer juntos.
- Bloqueio de imagens no mesmo minuto **entre diferentes categorias da AOI**,
  mesmo com rótulos diferentes; isso é cerca de segurança, **não**
  confirmação de que é a mesma peça física.
- Contatos/ref-test exatamente duplicados, ou visualmente quase
  duplicados na mesma iluminação, formam uma única unidade de split.
  Se os dados permitirem uma separação válida, os IDs das fontes em
  cada fold e contagens por categoria/qualidade são salvos no relatório.
- O treino usa amostragem balanceada por categoria×qualidade, com
  limite de peso para classes raras, mas SEM simular rótulos físicos,
  eventos triplos ou situações de Produção.
- A **validação** controla a seleção da época (inclusive época 0);
  a métrica prioriza acurácia balanceada macro por categoria quando
  existe NG e OK naquela categoria. O **teste** só é avaliado uma vez,
  após selecionar o checkpoint.
- Métricas discriminam **falsos OK entre NG** (grave), falsos NG
  entre OK, sensibilidade/especificidade, acurácia, suporte por
  categoria AOI e categoria sem classe representada. Uma categoria
  com somente NG ou OK no teste recebe métricas ausentes para a
  classe inexistente, sem inventar 100%.
- Meta exploratória: 98% desejado / **97% mínimo**, mas qualquer número
  neste arquivo deve ser descrito como **experimental**. As imagens
  de FALTANDO se sobrepõem aos experimentos anteriores, e a
  independência por evento/peça física **não foi confirmada**.
- O script nunca define `production_approval_eligible=true`.
  Gates `independent_physical_events_confirmed`,
  `all_labels_individually_verified` e
  `baseline_other_defects_replay_qualified` seguem falsos. Mesmo
  que algum indicador atinja 97–98%, é **proibido promover
  automaticamente** o checkpoint.
- Para uso operacional futuro: rotas de todas as categorias, incluindo
  MUITO ADESIVO e categorias desconhecidas, deverão ter
  testes independentes por peça, avaliação de segurança,
  regressões históricas de FALTANDO/INVERTIDO/EMBORCADO/DESLOCADO e
  avaliação dos motores já existentes, especialmente DESLOCADO.
  Essa PR não altera nenhum roteador de Produção.

## Resultado esperado no terminal

O script imprime a loss e o acerto por época, a **melhor época
selecionada apenas por validação**, e um resumo do teste final.
Nenhuma acurácia real é antecipada; a execução na estação com os 569
PNG originais é necessária. Em caso de separação insuficiente por
grupos, conflito de hash/rotulagem ou falta de classes, falha sem
treinar ou promover.

**Próximo passo de avaliação real:** coletar eventos novos de várias
categorias, confirmar OK/NG por operador, registrar IDs de peça/evento
físico quando disponíveis e criar um teste completamente independente.
