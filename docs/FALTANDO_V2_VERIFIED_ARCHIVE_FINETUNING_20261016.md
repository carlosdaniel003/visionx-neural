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
