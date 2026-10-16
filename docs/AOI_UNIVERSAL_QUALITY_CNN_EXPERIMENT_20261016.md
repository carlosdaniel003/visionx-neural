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
