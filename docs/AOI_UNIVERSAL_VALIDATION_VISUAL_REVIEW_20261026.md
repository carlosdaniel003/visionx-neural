# Revisão visual do reparo CNN — 26/10/2026

Análise do `hard_case_repair_report.json` da execução
`20261026T161704_538044Z_b17706`:

- Melhoria observada no próprio TREINO supervisionado: 7/15 -> 15/15 (6 INVERTIDO recuperados).
- Validação relacionada: 88/99 -> 90/99 (88,89% -> 90,91%).
- NG->OK na validação: 7 -> 3; OK->NG: 4 -> 6.
- Teste histórico: 106/106, mas NÃO CEGO, já avaliado anteriormente e com rótulos herdados da AOI.
- Melhor época: 6. Alvo de produção NÃO atingido.

A próxima verificação não deve repetir treino nem assumir rótulos provisórios corretos.
O comando abaixo produz ZIP de 9 divergências, com **captura AOI original,
gabarito e teste** e os respectivos SHA-256. Cada uma exige classificação
de qualidade física individual, independentemente da categoria fornecida pela AOI.

Com o ODIN fechado, em `C:\visionx-neural-main`:

```powershell
git pull origin central
python -m src.scripts.audit_aoi_universal_validation_visual --root . --repair-report "reports/aoi_universal_quality/hard_case_repairs/20261026T161704_538044Z_b17706/hard_case_repair_report.json" --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json"
```

Enviar o ZIP `odin_validation_visual_review.zip` gerado em
`reports\aoi_universal_quality\validation_visual_reviews\<execucao>\`.

Este script:
- Reavalia exatamente o checkpoint reparado e os 99 exemplos de validação.
- Compara as contagens de divergências com o relatório anterior; falha se divergirem.
- Verifica as imagens originais por SHA-256 e protege a integridade do checkpoint.
- Não escreve qualificação, não altera imagens do acervo, nem pesos ou ponteiro ativo.
- Não envia comando XP. O ZIP se limita a triagem humana do acervo anterior.

**Não há autorização de produção nem garantia de performance em peças físicas novas.**
