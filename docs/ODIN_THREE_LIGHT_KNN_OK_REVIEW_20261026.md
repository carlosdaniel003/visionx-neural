# Arbitragem CNN NG × KNN OK multilight — 26/10/2026

Evento **b7f173d1f00b430daba2a75df2d57bbe**, inspeção 14:42:06,
categoria AOI FALTANDO, **OK confirmado pelo operador**, ODIN declarou NG.

## Diagnóstico comprovado pelo debug

- A correção de SIDE operou: foi coletado SIDE/TOP/MID e o tempo subiu
  para 7,67 s, mas a fusão final continuou NG.
- CNN trial SHA `c13ca329...` (não são os candidatos 15/24 offline);
  scores NG 0,9991896/0,9991896/0,9999464, não calibrados.
- `cnn_v2_consensus=REVIEW`, motivo `INVALID_LIGHT_EVIDENCE`;
  portanto os três scores altos não representam consenso confiável.
- KNN C A/B votou **OK** nas três luzes, mas `status=REVISAO`,
  `reason=SEM_SUGESTAO_CONFIAVEL`, similaridade insuficiente.
  Isso NÃO significa `EXACT_PAIR` nem recuperação de rótulo humano:
  `recognition_match=NOT_FOUND`.
- KNN visual independente nas três luzes encontrou vizinhos OK
  mais similares que NG:
  - SIDE 0,951006 OK vs 0,944863 NG;
  - TOP 0,951006 OK vs 0,938456 NG;
  - MID 0,954742 OK vs 0,922532 NG.
  `knn_visual.status=CONFLICT`, não `OK` categórico.
- Evidência física forte de ausência: **não demonstrada**
  (`hard_missing_evidence=false`, `physical_score=0.0`).

## Mudança no ODIN

`src/core/multilight_fusion.py` agora preserva os **três** resultados
KNN C (antes o dict herdado do MID escondia SIDE/TOP).

`src/core/knn_visual_corroboration.py` usa estes votos em conjunto
com a busca visual independente para arbitrar o caso estreito:
**FALTANDO + CNN NG com consenso inválido + 3x KNN C A/B OK em revisão
+ 3x similaridade OK > NG com melhor OK >=0,95 + nenhuma evidência
física de ausência**.

Resultado final: `REVISÃO OBRIGATÓRIA` e metadado
`knn_ok_counterevidence`. Exibe a recomendação visual favorável a OK,
preserva o NG bruto para diagnóstico, mas não classifica como `FALHA
FALSA` nem autoriza comando 0/1 automático.

Esta regra **não modifica** os critérios de detecção de NG da CNN,
os pesos, KNN, acervo ou XP. Casos com NG físico forte, sem 3 votos KNN
OK, vizinhos NG favorecidos, outras categorias e verdadeiro consenso
CNN NG continuam seguindo a política anterior.

## Aplicação e teste

Com ODIN fechado: `git pull origin central`, iniciar novamente
e repetir com acompanhamento do operador. O evento deve aparecer
como `REVISÃO OBRIGATÓRIA`, não `FALHA FALSA`, e nenhum comando
XP 0/1 é liberado automaticamente pela contraprova.

Teste automatizado em
`tests/test_knn_three_light_ok_counterevidence.py`.
Como os sinais são por similaridade aproximada, ainda são exigidos
ensaios industriais com NG reais e OK para justificar qualquer
promoção futura a decisão automática.
