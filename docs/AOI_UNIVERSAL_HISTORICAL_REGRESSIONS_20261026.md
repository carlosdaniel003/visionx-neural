# ODIN — auditoria de regressões no teste histórico (26/10/2026)

Segundo reparo offline concluído em `20261026T170334_000420Z_32ad0d`:
- 24/24 decisões do operador no treino: 15 NG, 9 OK.
- 59/59 no recorte de validação restante, com qualidade **provisória**.
- **104/106** no antigo teste que já foi examinado: **2 NG→OK**.
- O candidato offline anterior acertava os 106 do mesmo teste.
- A validação restante não tem NG `INVERTIDO`, e somente
  EMBORCADO/FALTANDO têm as duas classes representadas.
- A generalização em novas peças não foi demonstrada; produção não autorizada.

O script `src/scripts/audit_aoi_universal_historical_regressions.py`
reabre os dois candidatos por SHA-256, reconstrói o mesmo conjunto
histórico e **compara cada captura**. Ele interrompe a execução se
a comparação não reproduzir exatamente os dois NG que deixaram
de ser detectados. Apenas para investigação, não é teste cego.

No PowerShell, com ODIN fechado:

```powershell
git pull origin central
python -m src.scripts.audit_aoi_universal_historical_regressions --root . --new-report "reports/aoi_universal_quality/validation_repairs/20261026T170334_000420Z_32ad0d/24_operator_repair_report.json" --previous-report "reports/aoi_universal_quality/hard_case_repairs/20261026T161704_538044Z_b17706/hard_case_repair_report.json" --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json"
```

Saída: `reports/aoi_universal_quality/historical_regressions/<execucao>/odin_two_historical_ng_regressions.zip`
com original, gabarito, teste, identificadores, hashes e scores
dos dois modelos. Os dois rótulos históricos **são provisórios**;
o operador deverá confirmar fisicamente OK/NG. Não modificar a
classificação com base apenas na pasta `ng_archive`.

Nenhum treinamento, movimentação de acervo, alteração de
`live_active.json`, intervenção KNN ou envio XP é realizado.
