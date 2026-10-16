# Memória visual reconhecida — evento 6a62411f (16/10/2026)

## Diagnóstico real

No evento de inspeção `6a62411f956c4ca39711ebb566dba28f` (Modo Teste, **5,086 s**), a fusão multilight corretamente resultou em **DEFEITO REAL (NG)**:

- CNN FALTANDO V2: SIDE NG 99,809438% bruto, TOP NG 99,679935%, MID OK 0,446203% NG. Consenso CNN `REVIEW` por discordância, checkpoint verificado **`6e4a31e8...`** sem promoção de novos pesos demonstrada.
- KNN EXATA: `NEW_CNN`, `NOT_FOUND`, nenhum par gabarito/teste pixel-idêntico recuperado.
- KNN VISUAL, pela memória de assinaturas já carregada: `NG_CORROBORATED` em SIDE/TOP/MID; SIDE **31** exemplares NG de suporte, melhor similaridade 0,999124; TOP **12**, 0,999605; MID **18**, 0,999427. Comparação de vizinhos OK também foi realizada.
- O cartão combinado exibia **`NUNCA VI`** apesar dessa corroboração porque o status `memory_seen_state` continuava `NUNCA_VI` sempre que não havia par exato. O `memory_feedback_state` só retornava `JÁ VI` ou `NUNCA VI`, e o cartão só mapeava esses dois. O painel KNN também descrevia individualmente cada luz com `NEW_CNN` como `NUNCA VI`.

## Contrato corrigido

O novo estado **`VI_SEMELHANTE`** é uma terceira opção **somente de apresentação**, sem alterar rotas decisórias: só é exibido quando não houve match exato e há evidência consistente de `NG_CORROBORATED` em pelo menos duas luzes concretamente marcadas `NG_SIMILAR`, ou `NG_TREND_THREE_LIGHTS` nas três luzes. Status declarado sem luzes verificáveis não é suficiente para dizer que o sistema recuperou semelhantes.

Ordem de preferência:
1. `JA_VI`: pelo menos uma iluminação tem par humano **exato** verificado, mesmo se as demais forem `NEW_CNN`.
2. `VI_SEMELHANTE`: sem par exato, mas a busca KNN visual local recuperou exemplos NG corroborados. Mostra luzes, melhores similaridades e número de assinaturas NG que sustentaram o diagnóstico.
3. `NUNCA_VI`: consulta exata realmente ocorreu mas nenhum par foi recuperado e não há corroboração visual suficiente. A explicação diferencia falta de par exato de nunca ter visto o tipo de defeito.
4. Sem consulta: nenhum rótulo inventado; conflito exato segue a política atual de revisão.

O cartão unificado mostra **DEFEITO REAL** e o subtítulo azul **VI SEMELHANTE**, sem overlay adicional nem redimensionamento de janelas. O painel de influência KNN passa a destacar os cards SIDE/TOP/MID com esse status apenas nas luzes corroboradas. A memória visual não é uma probabilidade calibrada, e não afirma que a CNN aprendeu os exemplos; o checkpoint CNN permanece **independente** da memória KNN.

## Não alterado

- **Decisão OK/NG** já produzida pela CNN, KNN e fusão multilight.
- Peso/checkpoint/treinamento da CNN; nenhuma promoção automática de modelo.
- Memórias e imagens em `ok_archive`/`ng_archive`; nenhuma gravação nem nova importação.
- Limiar KNN, pesquisa, OCR, rotação de luzes, comunicação XP, comando `0/1`, confirmação humana, etapa de inspeção ou tempos de inferência.

## Validação

Os testes unitários usam os **números do log real**, com casos negativos (sem assinatura visual suficiente, status indevido, KNN exata prioritária) e exercitam renderização real do cartão Qt e do painel KNN. Além disso, reexecutar na estação este NG e um OK conhecido após integrar o PR. Comparar o tempo ao baseline 5,086 s. Testes sintéticos não demonstram novos pesos treinados nem maior acurácia; mostram que a interface respeita a memória já recuperada.
