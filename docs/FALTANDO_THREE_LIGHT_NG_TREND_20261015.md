# Recuperação do NG pelo conjunto SIDE/TOP/MID — evento 63c48e2b (15/10/2026)

## Evidência real (Modo Teste)

Imagem AOI: na referência há estrutura metálica ocupando parte do furo; na inspeção o furo aparece livre em SIDE/TOP/MID. O operador classifica como NG. O debug reportou `REVISÃO OBRIGATÓRIA`, `multilight_local_review`, duração **6,653 s**, CNN ativa original SHA `6e4a31e8...`:

| Luz | CNN score NG bruto | CNN local | Melhor NG KNN | Melhor OK KNN | Diferença NG−OK | NG suporte |
|---|---:|---|---:|---:|---:|---:|
| SIDE | 0,0000436467 | OK | 0,951926 | 0,908387 | +0,043539 | 29 |
| TOP | 0,6390351653 | REVIEW | 0,947828 | 0,903074 | +0,044754 | 17 |
| MID | 0,0000331602 | OK | 0,915403 | 0,850404 | +0,064999 | 7 |

**Diagnóstico:** o KNN exigia vantagem local >=0,06, best NG >=0,93 e ao menos duas luzes `NG_SIMILAR`. Cada luz ficou individualmente inelegível. TOP CNN saiu REVIEW e interrompeu o consenso em `INVALID_LIGHT_EVIDENCE`, impedindo a contraprova física, que só era executada para CNN OK/OK/OK. Não havia justificativa para considerar essas três comparações como probabilidade de NG nem para habilitar comando automático.

## Regra adicional de convergência NG tricolor

Não altera limites da rota `NG_CORROBORATED` atual, pesos CNN, KNN exata, filtros de origem, nem a decisão do XP.

Para a classe AOI FALTANDO em rota `NEW_CNN`, um sinal `NG_TREND_THREE_LIGHTS` pode existir se **todas as três** iluminações tiverem:

- KNN comparou vizinhos NG e OK na mesma categoria e iluminação (não são casos de categoria cruzada);
- pelo menos **3 assinaturas NG distintas** com similaridade >=0,90;
- melhor NG >=0,91 e vantagem melhor NG−melhor OK >=0,04;
- média das vantagens nas três luzes >=0,05.

O NG como **classificação, não como comando**, só é apresentado se a CNN inteira estiver verificada com o mesmo SHA, três inferências `INFERENCE_OK`, 2 scores OK <=0,10 e uma luz de revisão `0,60 <= score NG < 0,90`, identificada como `REVISÃO OBRIGATÓRIA` em seu diagnóstico bruto. A função `trusted_three_light_ng_trend` reconfere todos esses campos antes de aceitar o registro.

Concluídas essas condições, o veredito exibido passa a `DEFEITO REAL`, `is_defect=True`, regra `knn_visual_three_light_ng_with_cnn_review_v1`. Continua `production_review_required=True`, `eligible_for_final_decision=False`, **nenhum 0/1 automático**. O operador pode corrigir o NG se a peça real estiver boa.

Se faltar uma das três luzes, modelo não estiver verificado, houver vizinho OK competitivo, rótulos cruzados, scores fora do padrão ou suporte insuficiente, nenhuma promoção visual ocorre: mantém a fusão anterior, frequentemente `REVISÃO OBRIGATÓRIA`.

### Análise física no caso REVIEW

Em CNN com os mesmos três diagnósticos verificados (2 OK + 1 REVIEW favorável a NG), a contraprova física independente agora executa as duas ROIs por luz e registra resultados, inclusive `SUSPECT_ABSENCE`, `GEOMETRY_REVIEW` ou `UNAVAILABLE`. Não altera o veredito sozinha no ramo REVIEW. A KNN é executada depois desse diagnóstico e a classificação NG por convergência pode também mostrar a suspeita física se identificada.

## Validação

Os testes novos usam exatamente as similaridades e os scores CNN registrados no evento para testar a **regra de fusão**, não substituem um replay com os PNGs reais nem a avaliação de holdout separado por peça. Os 5 eventos anteriormente corretos e casos OK de furo vazio/pad intacto devem ser novamente inspecionados antes de considerar automatizar NG. Medir latência adicional da sonda física em revisão (baseline 6,653 s).

Este PR deliberadamente não reclassifica todo `REVIEW` como NG e não relaxa os critérios de três votos para autorização de produção.

Arquivos: `src/core/knn_visual_corroboration.py`, `src/core/physical_missing_veto.py`, `src/core/corroborated_ng_verdict.py`, telemetria/UI, `tests/test_faltando_three_light_ng_trend.py`, workflow compartilhado.
