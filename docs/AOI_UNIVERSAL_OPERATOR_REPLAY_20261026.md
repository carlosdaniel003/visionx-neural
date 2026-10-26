# 26/10/2026 — Diagnóstico do candidato universal (91,51%)

Relatório analisado:
`reports/aoi_universal_quality/experiments/20261026T145405_571315Z_a90255/training_report.json`.

- 106 no teste: 97 corretos; 8 NG→OK, 1 OK→NG.
- Os 8 NG→OK do teste são todos da pasta NG com hint AOI FALTANDO.
- Na validação os 6 casos NG INVERTIDO foram classificados OK, incluindo
  dois PNGs cuja inversão o operador confirmou.
- De 15 capturas com qualidade revisada em 26/10, apenas 6 pertencem
  ao manifesto de 569 capturas: 4 no split de treino, 2 em validação;
  **nenhum no teste**. As outras 9 (incluindo caso 7 OK e caso 8 NG)
  não fizeram parte do treinamento. O rótulo corrigido do caso 7 não
  pode ser contado como exemplo usado nesse treino.
- Teste sem NG em DESLOCADO, EMBORCADO e INVERTIDO: a acurácia de
  91,51% não qualifica essas categorias.

## Replay sem retreinamento das 15 confirmações humanas

Para verificar de fato o candidato treinado nas 15 imagens, inclusive as
9 não incluídas no manifesto, executar com o ODIN fechado:

```powershell
python -m src.scripts.audit_aoi_universal_operator_confirmed --root . --training-report "reports/aoi_universal_quality/experiments/20261026T145405_571315Z_a90255/training_report.json" --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json"
```

O script valida SHA-256 das 15 imagens humanas e do checkpoint, reutiliza
6 pares do manifesto e extrai em diretório diagnóstico **isolado** os
9 pares gabarito/teste ausentes, usando o mesmo extrator offline AOI.
Se um par não for recuperável, aborta sem inferir acertos artificiais.
Saída: `reports/aoi_universal_quality/operator_replay/<run>/operator_confirmed_replay.json`.
O JSON enumera predições, scores, origem dos pares e conjunto original
(TREINO, VALIDAÇÃO, TESTE ou FORA DO MANIFESTO).

Não toca checkpoints ativos, `live_active.json`, KNN, acervo ou XP.
Os 15 casos são um replay de desenvolvimento, **não teste cego** nem
amostras físicas independentes. Nenhum resultado libera auto-OK.

A evolução dos pesos deve priorizar NG→OK e a cobertura por
iluminação/defeito físico, sem ensinar os rótulos errados da AOI.
