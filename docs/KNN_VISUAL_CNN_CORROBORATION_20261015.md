# Corroboração KNN visual + CNN V2 (15/10/2026)

## Problema observado
Evento AOI `da7d0b7cb9cd4334a8862d7cf6c246fc`: CNN FALTANDO V2 original `6e4a31e8…` votou OK em SIDE/TOP/MID; a contraprova física detectou ausência e encerrou como REVISÃO. O índice KNN tinha **1.715 registros**, mas o reconhecimento operacional só procurava pares pixel-idênticos. O painel de influência renderizava `1-score NG` quase integralmente verde, sem refletir a contradição física.

## Implementação
- `src/core/knn_visual_corroboration.py`: calcula assinaturas de anomalia por luz diretamente dos pares gabarito/teste da peça atual, consulta os registros KNN em RAM, separa OK e NG e exclui duplicatas de assinaturas. Requer evidência NG de **ao menos duas imagens distintas por luz**, similaridade local >=0,90 (melhor >=0,93), margem >=0,06 sobre vizinhos OK e acordo NG em >=2 luzes. Não considera simples repetição do mesmo PNG dez vezes como dez votos. Sem NG local, permite explorar NG rotulado de outra categoria coberta pela CNN V2, com limiar 0,97 e três imagens distintas, pois a categoria AOI pode diferir do tipo visual de defeito.
- Fluxo multilight: após a contraprova física, uma contradição NG consistente da KNN pode revisar o OK da CNN. **Somente quando também houver `SUSPECT_ABSENCE` físico**, o ODIN apresenta `DEFEITO REAL`; em ambos os cenários mantém `production_review_required=true`, `eligible_for_final_decision=false`, sem liberar 0/1 automático. A decisão KNN exata permanece prioritária e intacta.
- `production_decision_policy` verifica novamente a discordância, ainda que uma camada visual sobrescreva o resultado.
- O painel mostra a CNN como **voto OK contradito** em amarelo e não como uma barra verde validada; linhas KNN exibem similaridade (não probabilidades). Não modifica scores nem pesos.
- Testes sintéticos para luz/categoria, empate NG/OK, repetição exata, rótulos, contradição física, gate de produção e painel.

## Limites e validação
Não há imagens do acervo local nem checkpoint novo no GitHub. Não treinamos os pesos da CNN e não declaramos que os casos NG observados terão similaridade suficiente antes do replay real. Os limiares de corroboração são conservadores e **não certificados por holdout**. Esta mudança altera a apresentação de NG/revisão, mas nunca emite decisão autônoma por vizinhos aproximados. O tempo deve ser medido na estação e comparado ao baseline de 8,31 s; a consulta usa memória em RAM e não reconstrói índice.
