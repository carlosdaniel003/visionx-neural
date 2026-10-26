# ODIN — nove vereditos do operador, correção offline adicional (26/10/2026)

Os nove casos do ZIP `odin_validation_visual_review.zip` receberam julgamento
físico humano **sem alterar a categoria AOI**:

- Casos **1,2,3,4,8,9: OK** (CNN havia rejeitado incorretamente).
- Casos **5,6,7: NG** (CNN havia aprovado incorretamente).
- As etiquetas existentes das pastas coincidiam com esses vereditos.
- Não mover imagens do acervo; marcar **somente estes nove SHA-256**
  como confirmados, incluindo SHA-256 dos recortes gabarito/teste.

Registro: `docs/odin_operator_validation_review_20261026.json`.
Verificador: `src.services.odin_operator_validation_labels`.
Carregador experimental: `src.services.aoi_universal_quality_archive`.

Os nove casos passam a somar **24 capturas humanas (15 NG, 9 OK)**,
incluindo os seis INVERTIDO que já foram ensinados no primeiro reparo.

## Segundo reparo CNN exclusivamente offline

O reparo parte do checkpoint experimental da **época 6**:
`reports/aoi_universal_quality/hard_case_repairs/20261026T161704_538044Z_b17706/aoi_operator_hard_case_candidate.pt`,
com SHA-256 pinado. A CNN em produção não é carregada nem modificada.

O script retira da validação usada para seleção os nove casos antes
revisados, os pares/horários associados e vizinhos visualmente similares
ao material supervisionado. Os exemplos antigos do acervo continuam em replay.
O veto de regressões não permite que a validação remanescente aumente
NG→OK **nem** OK→NG em relação ao checkpoint de partida.

No PowerShell do computador da AOI, com ODIN fechado:

```powershell
git pull origin central
python -m src.scripts.train_aoi_universal_24_human_repair --root . --previous-report "reports/aoi_universal_quality/hard_case_repairs/20261026T161704_538044Z_b17706/hard_case_repair_report.json" --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json" --epochs 6
```

Relatório gerado:
`reports/aoi_universal_quality/validation_repairs/<execucao>/24_operator_repair_report.json`.

**Proibições**: não usar o `live_active.json` como modelo inicial;
não editar os pesos ativos nem a pasta de acervo; não chamar XP;
não promover checkpoint de esquema experimental; não interpretar
24/24 acertos em imagens já treinadas como teste cego.

Se nenhum novo checkpoint superar o controle sem regressões, será
selecionada a **época zero** e o diagnóstico registrará ausência
de melhora suficientemente demonstrada. A validação restante e o
teste histórico ainda possuem rótulos AOI provisórios. Necessária
validação independente com peças novas antes de produção.
