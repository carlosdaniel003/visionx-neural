# 16/10/2026 — Corrigir falso OK da CNN contra memória KNN NG em três luzes

## Evento observado
Evento `0652ff8f309143958dc9d8d0a2ce2a5f`, Modo Teste, ciclo 7,197 segundos. O operador informa NG real (componente SO8 presente na referência, ausente no teste):

| Luz | CNN NG bruto não calibrado | KNN melhor NG | KNN melhor OK | Suporte NG | Status KNN |
|---|---:|---:|---:|---:|---|
| SIDE | 0,002014 | 0,998912 | 0,957091 | 8 | NG_LEANING |
| TOP | 0,001187 | 0,998856 | 0,902414 | 8 | NG_SIMILAR |
| MID | 0,0000158 | 0,999230 | 0,897332 | 16 | NG_SIMILAR |

O resultado KNN `NG_CORROBORATED` indica memória VISUAL NG recuperada em pelo menos TOP/MID; `trend_ng_lights` inclui SIDE/TOP/MID, com vantagem média NG−OK de `0.080053`. A rota EXATA `NEW_CNN/NOT_FOUND` não impede o reconhecimento aproximado. A CNN FALTANDO V2 continua usando o checkpoint SHA `6e4a31e8…`, melhor época 18, e votou OK nas três luzes — erro de generalização para este NG. **Importar imagens para `ng_archive` não atualiza automaticamente seus pesos.**

A sonda física calculou métricas, mas devolveu `GEOMETRY_REVIEW` nas três luzes porque os pares de recortes precisaram de aparagem. Nenhuma luz confirmou `SUSPECT_ABSENCE`. A regra anterior exigia essa confirmação física para transformar `NG_CORROBORATED` em NG, por isso mostrou `REVISÃO OBRIGATÓRIA`.

## Nova regra de classificação visual supervisionada

`knn_three_light_strong_ng_recall_vs_cnn_ok_v1` corrige somente a **classe visual**, não a decisão automática no Windows XP. Requisitos cumulativos:

- AOI categoria `FALTANDO`, rota `NEW_CNN`, fusão multilight, **três votos CNN OK** com `INFERENCE_OK`, checkpoint SHA-256 de 64 hexadecimais igual/verificado nas três luzes, cada score NG <=0,10 e categoria local `FALTANDO`.
- KNN visual `NG_CORROBORATED`, mesma categoria, `trend_ng_lights` abrangendo SIDE/TOP/MID e ao menos duas luzes `NG_SIMILAR`. As assinaturas NG foram deduplicadas pelo mecanismo KNN existente.
- **Todas as três** iluminações: melhor similaridade NG >=0,995, vantagem NG−melhor OK >=0,04, suporte de pelo menos cinco assinaturas NG distintas, vizinhos OK presentes, sem correspondência NG de categoria cruzada.
- Duas luzes ou mais com vantagem NG−OK >=0,09, e vantagem média das três >=0,075.
- Sonda física de três luzes efetivamente executada, cada iluminação `GEOMETRY_REVIEW`, sem `UNAVAILABLE` e sem `SUSPECT_ABSENCE`. Esse estado é uma limitação da medição, **não conta como voto físico NG**. O conhecimento NG é exclusivamente da memória KNN verificada.

Quando cumpridos: `verdict=DEFEITO REAL`, `is_defect=true`, mas `operator_review_required=true`, `eligible_for_final_decision=false`, `auto_allowed=false`. A interface exibirá NG e `VI SEMELHANTE`, sem confundir reconhecimento por memória KNN com treino dos pesos CNN.

Sem dados suficientes, com vizinhos OK competitivos, menos de três luzes, registros contraditórios, categoria cruzada, falha CNN ou sonda indisponível, preserva a classificação anterior (normalmente `REVISÃO OBRIGATÓRIA`). **Nunca libera 0/1 automático por essa rota.**

### Não alterado
- Checkpoint CNN ativo, treinamento, limiares de classificação CNN, autoaprendizado e promoção.
- Memória gravada, `ng_archive`, `ok_archive`, roteador de par exato, cache RAM.
- Comandos industriais do XP, atalhos, filtros AOI e motor de aquisição.
- Nenhum modelo adicional na inferência; somente verificações lógicas sobre scores já calculados.

### Validação e limites
Os testes reproduzem **números do log**, os status das três luzes, os estados de verificação e o veto de emissão XP. O novo limiar foi escolhido para a evidência observada e ainda **não foi calibrado com conjunto independente de OK/NG**. Testar os seis casos anteriores, exemplos de componente presente e outros formatos de ausência no PC de produção antes de considerar qualquer automação.

O arquivo do evento `0652ff8f...` fornecido é um debug textual e uma captura composta; **não é** um journal supervisionado de imagens originais. Portanto não é possível retreinar/prometer nova acurácia CNN com ele. Para treinar pesos de verdade, usar os pares originais íntegros de `ng_archive` e `ok_archive` com rótulos verificados, gerando candidato e holdout por evento, sem promover o checkpoint atual automaticamente.
