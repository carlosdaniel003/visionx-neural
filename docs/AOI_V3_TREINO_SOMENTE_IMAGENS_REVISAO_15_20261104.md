## 04/11/2026 — Diagnóstico do aprendizado em CADA época (V3 offline)

A execução com as 14 correções do operador apresentou:
`selected_epoch=0`, `previous_v3_14_correct_in_training_replay=0`
e `operator_verified_14_correct_in_training_replay=0`. O checkpoint
selecionado foi a V3 ANTERIOR, pois a seleção conservadora rejeitou
todas as seis épocas. Isso **não demonstra** que nenhuma das seis
épocas aprendeu: o relatório antigo só avaliava as 14 imagens
após carregar o checkpoint selecionado.

O treinador V3 agora mede todos os 14 casos humanos (9 NG e
5 OK) **antes e depois de cada época** e registra:
- `operator_training_replay.metrics`: total correto, NG→OK, OK→NG;
- `mean_true_label_score` e `mean_true_label_log_loss`:
  mudança gradual, mesmo quando o limiar de 0,5 ainda não é cruzado;
- `comparison_with_epoch_zero`: quais imagens passaram a OK/NG
  correto ou pioraram, e quantos erros independentes de validação
  apareceram em relação à V3 inicial;
- `selection_reason`: explícito quando a época foi rejeitada por
  regressão OK→NG ou NG→OK na validação.

O **checkpoint conservador permanece selecionado SOMENTE pela
validação**, nunca pelos 14 exemplos do treino. Se nenhuma época
respeitar o gate de não regressão, continua na época 0.

Ao mesmo tempo, após verificar os hashes e o ponteiro produtivo,
o treinador preserva **cada época treinada** em
`epoch_diagnostics/epoch_NN_experimental_only.pt`, com SHA-256
individual no relatório. São pesos com schema `diagnostic_only`,
sem campo `state_dict` de checkpoint produtivo, sem avaliação pelo
teste independente, sem permissão para produção ou XP.
`human_learning_diagnosis` indica se alguma época melhorou o
replay humano e qual foi a melhor época EXPLORATÓRIA para estudo.
Esse indicador é *diagnóstico in-sample*, não critério para promoção.

**Como executar novamente:** manter o comando de treinamento abaixo,
com `--initialize-from-v3-report` apontando para o checkpoint
**original de 17:35**, SHA
`781d5d4d6cb80461a990e9a325560b75688f24f4647846ef12023e3898e15e1c`,
não para o candidato de 18:38 que apenas reproduziu a época 0
e tem outro SHA de arquivo.

**Após a execução**, enviar `training_report.json` novo. A seção
`epoch_history` finalmente permitirá distinguir:
(1) falha em aprender as 14 imagens em todas as épocas;
(2) aprendizagem de imagens humanas com regressão de OK/NG
na validação;
(3) aprendizagem aprovada pelo gate de validação.
Sem prova de generalização em peças inéditas, nenhum desses
candidatos será promovido automaticamente.

---

## Complemento — como interpretar o replay das 14 correções

O relatório/terminal da nova execução mostram
`previous_v3_14_correct_in_training_replay` e
`operator_verified_14_correct_in_training_replay`.
São contagens calculadas nos 14 exemplos agora **incluídos no
treino**. Servem para verificar se a CNN passou a reproduzir os
julgamentos humanos, mas **não** comprovam generalização nem
substituem `test` e `verified_previous_v3_baseline`,
que comparam ambos os modelos sobre a mesma nova partição.
Se os 14 forem acertados, não declarar "100%" de acerto industrial.

---

# ODIN AOI — CNN V3: aprender somente pelas imagens e pela qualidade OK/NG

**Data:** 04/11/2026. **Escopo:** experimento offline CNN V3, sem mudança
no motor produtivo, memória KNN ou envio 0/1 ao XP.

## Decisão do operador

A categoria textual exibida pela AOI **não é confiável** como identificação
do defeito físico. Um arquivo nomeado `FALTANDO` pode representar
**adesivo nos terminais** ou **componente invertido**, e não ausência.
Por isso a V3 não deve derivar dela o julgamento de qualidade.

A CNN V3 agora usa somente:
- pixels dos recortes **gabarito e teste** (full e focus);
- a identificação da iluminação real (SIDE, TOP, MID);
- a verdade binária **OK/NG**, com proveniência humana ou rótulo
  provisório do arquivo.

O nome do PNG e a categoria AOI são apenas **localizadores e metadados
de origem**, não entradas de inferência, rótulos físicos, critérios de
balanceamento, divisão do dataset, seleção de épocas ou métricas V3.
O roteador da interface/estação antiga ainda pode usar a categoria;
essa mudança não o substitui.

## 15 erros conferidos pelo operador

A revisão está em `docs/aoi_v3_operator_image_quality_20261104.json`:
- **9 NG fisicamente confirmados:** seis com adesivo nas pontas ou nos
  terminais de bronze; três com componente de cabeça para baixo;
- **5 OK fisicamente confirmados**, apesar de a V3 ter indicado NG;
- **1 excluído:** `2026-10-21_1458_FALTANDO_3.png`, caso intermediário,
  removido manualmente do `ng_archive` pelo operador.

Os 14 rótulos confirmados são fixados pelo **SHA-256 do PNG original**,
conferido contra o manifesto e o arquivo real. Uma fonte alterada
faz o treino falhar. A exclusão explícita funciona mesmo se o PNG
não existir mais na pasta, sem alterar ou reescrever o manifesto
original. Nenhuma categoria AOI é usada para interpretar o erro.

Essas 14 capturas eram parte do teste anterior (200/215) e agora
passam **obrigatoriamente para treino**, acompanhadas de todos os
grupos visualmente semelhantes e do mesmo minuto. Não aparecem na
nova validação nem no novo teste. **Acertá-las depois de treinar
não poderá ser divulgado como acurácia independente.**

## Implementação

- `src/services/aoi_v3_operator_image_reviews.py`: lê a revisão
  humana de 15 casos e compara os SHA com as fontes atuais.
- `src/services/aoi_universal_quality_archive.py`: usa as revisões
  **somente quando a V3 solicita**, preservando a lógica histórica
  de outros treinadores; ignora o caso incerto, mesmo apagado.
- `split_universal_events(balance_by_aoi_category=False,
  pinned_training_sources=...)`: une quase duplicatas e minutos;
  força para treino os grupos vinculados às 14 imagens; balanceia
  classes OK e NG **sem referência a qualquer categoria**.
- `src/services/aoi_v3_binary_quality_metrics.py`: avalia apenas
  matriz de confusão OK/NG e cobertura por iluminação.
- `src/services/aoi_v3_verified_candidate.py`: recupera o
  checkpoint anterior por relatório, SHA-256 e schema, **somente
  leitura**. Sem o hash correto aborta.
- `src/scripts/train_aoi_quality_cnn_v3_offline.py`: inicia
  opcionalmente com os pesos V3 anteriores (93% no teste anterior),
  amostra com peso inverso por qualidade e peso 2x nos 14 humanos.
  Épocas só podem substituir o ponto de partida se **nenhum dos dois
  erros de validação** piorar em relação ao início. Mantém V2
  pinada como comparação opcional. Exporta relatório/candidato novos,
  sem promoção automática e sem tocar no XP.

## Próxima execução no Windows (somente OFFLINE)

Encerrar o ODIN e a inspeção XP antes de executar. No PowerShell,
com `.venv` ativo e na raiz `C:\visionx-neural-main`:

```powershell
git switch central
git pull origin central

python -X utf8 -m src.scripts.train_aoi_quality_cnn_v3_offline --root . --manifest "C:\visionx-neural-main\reports\faltando_neural\run_20261104T163621_087850Z\manifest.json" --epochs 6 --allow-provisional-archive-labels --allow-verified-trial-pointer-readonly --initialize-from-v3-report "reports\aoi_quality_cnn_v3\candidates\20261104T173525_801668Z_90bed0\training_report.json"
```

O treinador lê o checkpoint anterior em
`reports/aoi_quality_cnn_v3/candidates/20261104T173525_801668Z_90bed0/aoi_quality_cnn_v3_candidate.pt`
cujo SHA esperado é
`781d5d4d6cb80461a990e9a325560b75688f24f4647846ef12023e3898e15e1c`.
Se o relatório/peso ou uma imagem humana tiver mudado, **não**
desligar a validação; guardar o erro para diagnóstico.

O comando gera uma **nova pasta**, `training_report.json` e
`aoi_quality_cnn_v3_candidate.pt`. O relatório contém
`verified_previous_v3_baseline`, `verified_pinned_v2_baseline`,
métricas OK→NG e NG→OK, contagem de exemplos humanos no treino,
evidência de isolamento do teste e seleção de época sem degradar
um dos erros da validação.

**Não comparar diretamente o novo total do teste com 200/215:**
os 14 exemplos humanos saíram do teste e as partições foram refeitas
por qualidade, não pela categoria. A comparação válida na rodada
é entre o modelo original V3 e o novo V3 avaliados **no mesmo novo
conjunto**, com ressalva de que o acervo já foi utilizado em
experimentos anteriores. O campo `blind_holdout` continuará falso.

## Condições para avançar

- Sem aumento de NG→OK ou OK→NG na validação, em relação à V3 antiga.
- Verificar o teste separado, especialmente os 5 OK e 9 NG
  aprendidos, sem contá-los no teste.
- Conferir rótulos duvidosos e validar em **novas peças reais
  não vistas no treinamento**, incluindo variações de luz.
- Só após evidência nova considerar uma etapa separada de
  integração em modo sombra e aprovação humana de produção.

Nenhuma promessa de 100% nem decisão automática é feita nesta etapa.
