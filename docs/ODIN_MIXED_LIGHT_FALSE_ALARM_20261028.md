# ODIN — 28/10/2026, segundo OK conhecido sob iluminação variável

Evento `4f7da09d175846fdb468f57f084c6e9e`, 09:58:35 (Manaus).
O operador confirmou falha falsa; o ODIN apresentou REVISÃO OBRIGATÓRIA.

**Diagnóstico:** fusão multilight `multilight_local_review`.
CNN: SIDE 0,25600529 NG; TOP 0,25600529; MID 0,85979676.
Consenso CNN `REVIEW/INVALID_LIGHT_EVIDENCE`; físico 0.0.
KNN C SIDE A/B = NG/NG, **em revisão** por similaridade fraca
(A=0,85431065; B=0,80228127), não um NG corroborado.
TOP e MID A/B = OK/OK, ambos em revisão.
KNN TOP: 5/5 OK em A e 4/5 OK em B; MID: 4/5 OK em A e 4/5 OK em B.
KNN visual independente:
- SIDE: OK .944720 vs NG .930544; margem NG-OK -.014176
- TOP: OK .972946 vs NG .935611; margem -.037334
- MID: OK .960225 vs NG .932796; margem -.027429
Nenhuma luz com NG forte ou tendência NG corroborada; nenhum par exato
foi encontrado na memória (a existência de vizinhos OK não é prova
de identidade de pixels).

## Ajuste

Novo reconhecimento supervisionado
`_mixed_light_ok_candidate` em
`src/core/knn_visual_corroboration.py` executa **antes** dos
detectores anteriores para um evento cuja fusão é **revisão**, não NG.
Exige simultaneamente: três luzes revisadas sem NG físico; CNN
inconclusiva e SIDE/TOP orientadas a OK; KNN-C READY nas três,
TOP/MID A+B votando OK e 4+ OK entre cinco por descritor, SIDE
votando NG fraco; e busca visual independente favorecendo OK nas
três com margens mínimas e sem evidência NG corroborada.

Quando satisfeitas, a **classificação apresentada ao operador** será
`FALHA FALSA`, com razão
`mixed_light_knn_ok_cnn_review_manual_v1`.
Ela representa uma **sugestão para este padrão**, NÃO comprovação
de par exato ou de aprovação física da peça atual.

A política de produção recompõe o contrato de evidência e
retorna `auto_allowed=False`, `proposed_decision=""`,
`operator_review_required=True`. É obrigatório **0=OK** ou
**1=NG** pelo operador. Nenhum comando XP é gerado automaticamente
pela alteração. Pesos CNN, acervo e memória KNN intactos.

Teste de regressão usa os scores reais desse evento e avalia veto
de ausência física, score NG forte da CNN, voto KNN-C adverso em
TOP/MID, NG forte SIDE, duas luzes apenas, ausência de maioria de
vizinhos OK, contraprova visual desfavorável e adulteração do
relatório de fusão.

**Limite:** a regra é baseada em indicadores visuais e memória
aproximada. Antes de qualquer liberação autônoma futura, validar
o conjunto em NG verdadeiros de geometrias semelhantes e com
operador independentemente.
