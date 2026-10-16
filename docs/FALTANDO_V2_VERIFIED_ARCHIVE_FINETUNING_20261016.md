## 16/10/2026 — Correção experimental: preservar OK enquanto a CNN aprende NG

**Treino real da PR #48 na estação, 5 épocas, 420 fontes (411 pares únicos: 294 NG,
117 OK, nove repetições exatas removidas).** Split agrupado por semelhança visual
(110 agrupamentos): 324 imagens no treino e 87 no holdout, todas com **rótulos
provisórios das pastas**, sem revisão individual. O champion original (SHA
`6e4a31e8…`) acertou no holdout 38/87 (49 NG→OK, nenhum OK→NG).
O candidato `c13ca329…` acertou **85/87 = 97,70% binários**, corrigiu 49
NG→OK, mas produziu **2 novos OK→NG binários no holdout**, e caiu no
histórico de 117/117 para **98/117** imagens corretas. No histórico, houve 19
erros novos de iluminação OK, nove deles NG conclusivos pelos limiares
operacionais 0,10/0,90. Gates **REPROVADOS** e pesos não promovidos.
O jornal online passou de 112/357 para 345/357 acertos por iluminação, mas
perdeu um OK. A redução da loss até 0,095419 **não garante generalização**.

**Diagnóstico do código:** o replay forçado tinha 158 pares incorretos e todos
eram NG; após o sampler balanceado, 158 passos de gradiente só para NG foram
executados a **cada época**, potencialmente deslocando o limite decisório.

**Correção da etapa atual:**
- Somente com `--archive-labels-experimental`, limita o replay de erros
  por época e faz lotes de dois exemplos, um NG difícil e um OK-âncora.
  Rotaciona os erros NG para cobrir defeitos diferentes nas cinco épocas.
  Prioriza os OK com maior score NG individual por luz do modelo vigente.
  As âncoras incluem OK do replay histórico e online já disponível, além dos
  OK do subconjunto de treino, **nunca o holdout**.
- Mede após cada época `historical`, `online` e `train`; escolhe a
  melhor época por regressão do histórico (principalmente OK) e recuperação
  NG do online/treino. O holdout é avaliado **uma única vez ao final** sobre
  o checkpoint escolhido (além da medição inicial do champion), evitando
  escolher época pelo próprio holdout.
- Meta de desenvolvimento: **98%**; piso aceitável do experimento:
  **97%** de acerto binário por imagem no holdout, usando limiar 0,50
  apenas como métrica de treinamento. O piso adicional não substitui os
  gates já existentes de não piorar OK/NG, preservar todo histórico e
  evitar novos falsos OK. Uma acurácia de 97% com regressão crítica segue
  **REPROVADA**.
- Relatório separa por SIDE/TOP/MID os diagnósticos com limiares operacionais
  `score<=0,10 => OK`, `score>=0,90 => NG`, `0,10<score<0,90 => revisão`;
  estas métricas **não** alteram a inferência nem o comando XP.
- A rota estrita de qualificação individual fica com o comportamento de
  replay original; a alteração de gradiente e seleção de época fica restrita
  aos candidatos experimentais em lote. Nenhuma promoção automática de pesos
  e nenhuma edição de imagens, KNN, OCR, fusão ou checkpoint ativo.

**Após integrar a PR**, para repetir **um novo** treino offline com os
mesmos dados históricos sem recapturar nem revisar manualmente as 420 fontes:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
python -X utf8 -m src.scripts.train_faltando_cnn_v2_verified_archive --root . --manifest "reports/faltando_neural/run_20261016T114234_996180Z/manifest.json" --epochs 5 --holdout 0.25 --seed 42 --archive-labels-experimental
```

A comparação atual é apenas offline com rótulos presumidos. Atingir 97–98%
no holdout não garante 97–98% em Produção, não certifica independência
física da peça nem cobertura de categorias DESLOCADO/INVERTIDO/EMBORCADO.
**Não promover automaticamente nenhum checkpoint.**

---

## Experimento solicitado: aceitar rótulos do arquivo em lote (16/10/2026)

O operador confirmou que não fará 420 revisões individuais e autorizou usar
os rótulos históricos de pasta para iniciar **um experimento de treinamento**.
Isso **não** constitui revisão visual individual de gabarito/teste, nem prova
de identidade física entre as 126 trincas propostas pelo nome. A rota rigorosa
de revisão individual acima permanece disponível e inalterada.

Após incorporar a PR desta funcionalidade à `central`, na estação que
contém os recortes extraídos, execute **com o ODIN fechado**:

```powershell
cd C:\\visionx-neural-main
git switch central
git pull origin central

python -X utf8 -m src.scripts.train_faltando_cnn_v2_verified_archive --root . --manifest "reports/faltando_neural/run_20261016T114234_996180Z/manifest.json" --epochs 5 --holdout 0.25 --seed 42 --archive-labels-experimental
```

A opção `--archive-labels-experimental` é o aceite explícito desta hipótese.
Sem ela, o treinador **continua exigindo** `qualification.json` individual.

Controles preservados:

- Confere os hashes SHA-256 das **origens** com o manifesto, a presença e a
  decodificação de cada recorte gabarito/teste. Arquivos incompletos, PNG alterado,
  symlink de recorte, origem em pasta contrária, classe diferente de FALTANDO e
  revisão individual `REJECTED` bloqueiam a execução. Revisões individuais
  OK/NG já registradas, se houver, têm prioridade sobre o nome da pasta.
- Detecta pares exatamente iguais em rótulos opostos e bloqueia o treino.
  Duplicatas exatas do mesmo rótulo são descartadas. Mantém o agrupamento para
  **split**, por nome (mesmo rótulo) ou event_id indicado, sem afirmar que sejam
  um evento físico nem fundir três luzes na perda. Comparações visuais próximas
  também não podem atravessar o treino/holdout. O mesmo Board/Parts não é,
  sozinho, prova de que duas capturas OK e NG são da mesma peça.
- Cada registro da rota em lote é um evento **monoluz provisório**, não uma
  trinca humana confirmada. Atestado em lote fica somente como procedência do
  experimento, com `individual_pair_review_complete=false`. O programa não
  inventa `pair_inspected=true`, não cria `qualification.json` e não escreve
  novos rótulos no acervo.
- Mantém o checkpoint original, o journal KNN, os limiares, o XP e o apontador
  `reports/neural_online/live_active.json` intocados. Bloqueia se esse
  apontador já existe; a situação exige auditoria antes de experimentar.
- Otimizador real AdamW; amostra OK/NG balanceadamente; força os erros de treino;
  compara os pesos novos ao champion no histórico, no online e num holdout
  separado do fine-tuning atual. Reprovação dos gates é documentada.
- Grava somente em
  `reports/faltando_neural/archive_experiments/<timestamp>/` os arquivos
  `faltando_v2_archive_experimental_candidate.pt` e `training_report.json`.
  Relatório inclui `qualification_mode=ARCHIVE_FOLDER_BULK_EXPERIMENT`,
  `production_approval_eligible=false`, `candidate_only=true`,
  `promoted=false` e `automatic_ok_authorized=false`.

**Limites:** amostras sem conferência individual podem ter classe incorreta,
recortes equivocados ou repetições da mesma peça física. O holdout resiste a
imagens idênticas/quase idênticas e nomes de captura, mas não certifica
independência física nem substitui inspeção com placas inéditas. Mesmo gates
verdes do experimento **não** autorizam implantar o candidato. A CNN atende
também DESLOCADO/INVERTIDO/EMBORCADO, que exigem testes independentes de
regressão antes de qualquer promoção futura. Nenhum checkpoint com estes
rótulos presumidos pode ser promovido diretamente por este fluxo.

Arquivos desta rota:
`src/services/faltando_neural_archive_experiment.py`,
`src/scripts/train_faltando_cnn_v2_verified_archive.py`,
`tests/test_faltando_archive_label_experiment.py`.

---

# CNN FALTANDO V2 — Fine-tuning REAL do acervo histórico qualificado

## Por que o ODIN ainda vota OK em ausências conhecidas

Os milhares de exemplos visuais importados em `public/ng_archive`/ `public/ok_archive` alimentam assinaturas da **memória KNN**, sem atualizar os pesos CNN. A CNN V2 pode continuar no checkpoint `6e4a31e8…`, mesmo após várias decisões NG similares. Regras posteriores de fusão recuperaram parte dos NG, mas o modelo CNN não generalizou adequadamente.

Este trabalho acrescenta uma **rota de treinamento efetivo de pesos**, reutilizando a arquitetura `FaltandoCNNV2` e o checkpoint ativo, separada do ciclo XP. A saída é `faltando_v2_verified_candidate.pt` gerado com gradientes AdamW reais em CPU, mais relatório com scores NG por evento/luz. Não existe publicação automática em `live_active.json`.

## Corrigido: manifesto antigo com IDs de trinca repetidos (16/10/2026)

Uma execução real extraiu **420 pares** (295 NG e 125 OK), sem PNGs inválidos, mas ao abrir o painel recebeu `ValueError: IDs de trinca repetidos`. O motivo era o mesmo nome/timestamp de trinca em `ok_archive` e `ng_archive`, embora sejam **rótulos e associações independentes**.

- A preparação nova usa `OK::<nome>` / `NG::<nome>` para os identificadores de grupo, preservando `source_name_hint`.
- A qualificação aceita o **manifesto legado já criado**, normalizando IDs repetidos entre OK e NG **apenas em memória**, sem escrever no manifesto e sem solicitar nova extração de imagens. IDs duplicados do **mesmo rótulo** continuam rejeitados como ambíguos.
- A revisão de associação SIDE/TOP/MID **continua manual**, mesmo quando nomes são iguais. Os rótulos de cada par também exigem confirmação individual.

Com o ODIN fechado e a `central` atualizada para incluir esta correção, retome **o mesmo manifesto**:

```powershell
python -X utf8 -m src.ui.faltando_neural_review --manifest "reports/faltando_neural/run_20261016T114234_996180Z/manifest.json"
```

Não apague `reports/faltando_neural/run_20261016T114234_996180Z` nem reexecute a preparação para resolver apenas essa colisão. O painel salvará a qualificação em `qualification.json` na mesma pasta. Os 126 candidatos a trincas seguem **não confirmados**, e a CNN **ainda não foi treinada** com esses 420 pares até haver rótulos humanos suficientes e executar o treinador offline.

## Fontes de verdade e rótulos

O nome das pastas `ok_archive` e `ng_archive` é uma etiqueta inicial, **não constitui confirmação humana individual**. O pipeline existente cria os recortes AOI sem adulterar os originais, e sua GUI `src.ui.faltando_neural_review` permite ao operador confirmar OK, NG ou rejeitar cada par. Guarda `qualification.json` com a identificação SHA-256 da fonte e `pair_inspected=true`.

O novo treinador utiliza **somente** casos `CONFIRMED_OK`/`CONFIRMED_NG` com SHA ainda válido, par gabarito/teste legível, revisão humana documentada; trincas SIDE/TOP/MID só são agrupadas quando o operador confirmou **cada par** e o vínculo visual com `CONFIRMED_VISUAL_ASSOCIATION`. Sem qualificação, **treinamento recusado**. Rótulo humano retificado pode divergir da pasta: o modelo aprende a etiqueta humana, não a pasta. Não há rótulos automáticos gerados a partir da classificação da CNN, KNN, geometria ou do nome do arquivo.

## Executar na estação — ODIN fechado, fora do ciclo de inspeção

Após atualização da branch que contém este script, executar no PowerShell na raiz real do projeto:

```powershell
cd C:\visionx-neural-main

# 1. Extrair pares gabarito/teste de OK e NG históricos, somente leitura das fontes
python -X utf8 -m src.services.faltando_neural_dataset

# 2. Qualificar visualmente os pares de cada caso usando a interface existente
python -X utf8 -m src.ui.faltando_neural_review
```

O manifest de preparação gerado fica em `reports/faltando_neural/run_<data>/manifest.json`. Identifique esse caminho na saída do primeiro comando. **Se já existir qualificação humana anterior**, não é necessário reaprovar sem motivo: pode reutilizar o manifesto original que gerou o `qualification.json`, desde que as imagens e hashes continuem íntegros.

Somente depois de conferir visualmente exemplos **NG de ausência real** e **OK de componente presente**, confirme também vínculos SIDE/TOP/MID quando forem efetivamente da mesma peça:

```powershell
# Substitua run_XXXX pelo diretório efetivamente gerado na estação
python -X utf8 -m src.scripts.train_faltando_cnn_v2_verified_archive --root . --manifest "reports/faltando_neural/run_XXXX/manifest.json" --epochs 5 --holdout 0.25 --seed 42
```

O comando não funciona sem PNGs da estação e não deve ser executado durante inspeção de produção. Para simplificar a revisão inicial, comece pelos NG recorrentes da classe FALTANDO (como ausência SO-8 e furo) e inclua diversidade real de OK: não basta repetir a mesma foto para aumentar contagens.

## Como o modelo aprende

1. Carrega checkpoint ativo `_base_model` com SHA-256, schema, `experimental` e `production_approved=False` verificados. Não usa candidato rejeitado. Se existir ponteiro CNN online ativo, esta primeira versão interrompe o treino histórico até que sua origem seja auditada.
2. Confere rótulos humanos, integridade SHA-256 das fontes, evita arquivos conflitantes NG/OK, e monta grupos multilight exclusivamente verificados.
3. Realiza split determinístico por **grupo visual e OCR board/parts**, mantendo trincas indivisíveis. Imagens quase duplicadas gabarito/teste não podem cair em treino e holdout ao mesmo tempo. Exige ambos os rótulos NG/OK com pelo menos 2 eventos de cada classe **em ambos os subconjuntos**.
4. Usa a base histórica pinada e o journal online humano (quando disponível) como âncoras de regressão, além dos novos exemplos verificados do treino. A amostragem é balanceada por rótulo, perda `BCE_event + 0.15 * BCE_per_light` do treinador online, otimizador AdamW, CPU em poucas threads.
5. Antes de otimizar, mede erros da CNN no treino. Reforça **todo evento de treino incorreto** com passos extras em cada época; segura o holdout fora desse gradiente. O checkpoint de saída contém `state_dict` NOVO e marca `production_approved=False`.
6. Compara **antes/depois** no holdout qualificado por evento e iluminação e no replay histórico/online. Bloqueia gates se houver aumento dos erros NG/OK, sobretudo falso OK `NG→OK` ou novo falso NG `OK→NG`.

### Saídas e preservação do champion

Apenas arquivos de um diretório novo são criados:

```text
reports/faltando_neural/verified_candidates/<timestamp>/
  faltando_v2_verified_candidate.pt
  training_report.json
```

O relatório guarda hashes do checkpoint anterior/candidato, manifesto/qualificação, IDs de treino e holdout, histórico de perda, scores e erros por iluminação, métricas de baseline e modelo treinado, gates, e `candidate_only=true`, `promoted=false`. **Mesmo se os gates passarem**, o modelo permanece um candidato: deve ser comparado ao champion no acervo completo, revisado e submetido a novo processo de aprovação antes de qualquer ativação. Nenhum arquivo `live_active.json` é criado ou modificado.

## Limites industriais e próximo passo

- **Não** há os arquivos locais do arquivo AOI no repositório GitHub. O PR implementa o treinador e testes de treinamento CPU com imagens sintéticas, não é um checkpoint já retreinado com as suas peças.
- `holdout` é independente do *fine-tuning atual*, mas pode conter imagens similares às originalmente utilizadas na formação do champion. Não tratá-lo como um estudo cego industrial nem prometer melhoria estatística com poucas amostras.
- A qualificação humana pode conter rótulos incorretos; conflitos de par duplicado são bloqueados. Verificar casos OK e NG semelhantes, peças distintas e iluminação desbalanceada.
- A CNN ativa usa uma única classificação binária para categorias relacionadas (`FALTANDO`, `INVERTIDO`, `EMBORCADO`, `DESLOCADO`). O script histórico só faz fine-tuning do arquivo visual FALTANDO qualificado: **é obrigatório testar regressões nas outras categorias** antes de promover pesos.
- Os cinco/seis casos recentes aprovados pela fusão devem ser repetidos para comparar resultados CNN puros; não usar pontuação final KNN para alegar que a CNN aprendeu.

## Arquivos

`src/scripts/train_faltando_cnn_v2_verified_archive.py`, `tests/test_faltando_verified_archive_finetuning.py`, workflow `faltando-cnn-v2-training-tests.yml`.

Critério de conclusão desta PR: o código consegue, localmente e fora da linha, criar um candidato com pesos realmente diferentes e relatório verificável usando entradas já qualificadas. A execução com todos os PNGs da estação e validação cruzada de categorias vem depois, com resultados anexados para revisão.
