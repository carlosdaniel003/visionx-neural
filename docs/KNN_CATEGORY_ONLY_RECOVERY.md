# Etapa 1 — Recuperação KNN com OCR somente de Categoria (15/10/2026)

## Contexto e evidência

No evento AOI `f39c34cd51bb48abb7538afd6eec550e`, o operador confirmou **NG real** após uma resposta CNN FALTANDO V2 erroneamente OK em SIDE/TOP/MID. O log da consulta KNN exata informou `NEW_CNN`, `NOT_FOUND`, `OCR/categoria ou par incompleto`, embora o pré-aquecimento tivesse 1.516 registros (1.132 legados e 384 persistentes). Esses totais **não provam** que o par desta peça estava armazenado; a captura atual não tem Board, Parts ou Value.

Após PR #33, `ScreenMonitor` mantém `board=""`, `parts=""`, `value=""`, apenas a categoria é capturada. O arquivo durável `PersistentArchiveKNN` já usa identidade `categoria + iluminação + dois fingerprints da imagem`. O índice **legado** `VerifiedKNNMemory._key` continuava rejeitando essas capturas pela ausência de Board/Parts, impedindo acesso a pares antigos mesmo que fossem idênticos.

## Mudança de recuperação EXATA

- Continuar verificando os **registros históricos** por `_entry` com JSON original, rótulo e fonte humana, metadados completos, caminho seguro, imagens PNG e hashes auditáveis. Nada é reclassificado ou migrado.
- Para **captura atual com Board/Parts/Value todos vazios**, indexar e consultar registros legados íntegros também por `categoria + iluminação + fingerprint gabarito + fingerprint teste`. A correspondência continua **pixel-idêntica de ambas as imagens**; não relaxar tolerância nem aceitar imagem parecida como `KNOWN_KNN`.
- Para captura que ainda fornece metadados, continuar usando a chave antiga estrita: diferenças de Board, Parts ou Value deixam o registro `NEW`. Dados parcialmente preenchidos não ativam a rota categoria-only.
- Qualquer rótulo OK/NG contraditório nos mesmos dois fingerprints impede o atalho e gera `MEMORY_CONFLICT`/revisão. Rótulos automáticos, ausência de PNG, erro de integridade e categoria/iluminação diferentes não criam memória humana.
- Reaproveitar o índice legado pré-aquecido. Após invalidação, a reconstrução dirigida da captura categoria-only filtra `categoria/luz` (sem usar Parts indisponível), preservando a verificação individual dos arquivos. O status `verified_memory_timing.legacy_lookup_mode` distingue `exact_visual_category_light`, `legacy_metadata` e `unavailable`.

## Vizinhos semelhantes: diagnóstico isolado, sem efeito na operação

`src/services/startup_regression/knn_visual_neighbors.py` calcula a assinatura de divergência `build_anomaly_signature` de um par recortado e compara com assinaturas KNN históricas e persistentes já carregadas usando `compare_anomaly_signatures`. Restringe **categoria e luz** e mostra até `top_k` vizinhos **por classe**, com score, grupos, origem e status de proveniência.

Os scores de similaridade **não** são probabilidades calibradas, nem prova de par idêntico, nem classificação autorizada. Registros legados no diagnóstico recebem `LEGACY_NOT_EXACT_VERIFIED`. O relatório declara `diagnostic_only=true`, `decision_authorized=false`, `recognition_route=null`.

Para executar **com o ODIN fechado** e com PNGs de gabarito/teste já extraídos do mesmo evento, no PC que possui as memórias:

```powershell
python -X utf8 -m src.services.startup_regression.knn_visual_neighbors --reference "caminho\gabarito_SIDE.png" --test "caminho\teste_SIDE.png" --category FALTANDO --lighting SIDE --top-k 5
```

Repetir para TOP e MID com os **próprios pares daquela iluminação**, sem combinar luzes diferentes. O JSON é gravado somente em `reports/startup_regression/knn_neighbors_*.json` (ou caminho explícito em `--output`). A ferramenta não altera `memory.json`, dados históricos, pesos CNN, interfaces, OCR, fusão, gate XP ou emissão 0/1.

## Validação exigida

- Testes: identidade exata com OCR reduzido, não reconhecimento de pixels diferentes, isolamento por categoria/luz, conflito de rótulos, exclusão de fonte automática, metadados parciais, consulta persistente e diagnóstico comparativo separado.
- Observar na estação: quantos `KNOWN_KNN` são recuperados com pares idênticos; quantos continuam `NEW_CNN`; simetria do diagnóstico OK/NG e custo do índice após warmup. Não declarar acurácia de generalização sem avaliação holdout humana por peça/evento.
- **Escopo:** somente Etapa 1. Não retreinar CNN FALTANDO V2, não promover checkpoints, não alterar thresholds, não permitir auto-OK por similaridade, não corrigir automaticamente o rótulo de casos antigos. O fato de o operador ter confirmado NG no evento citado **não prova** que o journal CNN online tenha concluído treino/promoção; investigar somente na Etapa 2 autorizada.
