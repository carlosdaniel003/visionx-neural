# Correção supervisionada OFFLINE — CNN universal (26/10/2026)

**Diagnóstico verdadeiro após replay das 15 capturas confirmadas:**
12 NG reais, 3 OK reais. Candidato universal acertou somente 4/12 NG
e 3/3 OK. Todos os 6 INVERTIDO confirmados foram erroneamente OK;
4 pertenciam ao TREINO do candidato anterior, 2 à VALIDAÇÃO.
A falha não se resolve aprovando o checkpoint por acurácia agregada.

Script: `src/scripts/train_aoi_universal_hard_case_repair.py`.
Faz fine-tuning partindo SOMENTE do candidato universal experimental
SHA verificado; 6 pares preparados + 9 pares extraídos em diretório
isolado; 15 fontes validadas com SHA da revisão humana. Repete os
casos humanos dentro de cada época e intercala replay do antigo TREINO.
Registros do mesmo minuto antes na validação são movidos ao TREINO; o
antigo TESTE fica intocado e serve APENAS de regressão retrospectiva,
pois já foi analisado e não é cego. Se houver coincidência de minuto
com o teste o script ABORTA.

Os candidatos ainda dependem parcialmente de rótulos provisórios.
Modelo não é aprovado para produção. Ponteiro atual `live_active.json`,
checkpoints originais, KNN, arquivo PNG e comandos XP são preservados.

## Execução de manutenção

Com ODIN fechado:

```powershell
git pull origin central
python -m src.scripts.train_aoi_universal_hard_case_repair --root . --training-report "reports/aoi_universal_quality/experiments/20261026T145405_571315Z_a90255/training_report.json" --manifest "reports/faltando_neural/run_20261016T154441_358349Z/manifest.json" --epochs 8
```

Saída: `reports/aoi_universal_quality/hard_case_repairs/<run>/hard_case_repair_report.json`,
com métricas por época (12 NG físicos, 3 OK), veto de regressões na
validação e retrospecto do antigo teste. Um score alto nos 15 casos
não é evidência de generalização, pois participaram da otimização.
Se o melhor for época zero, o modelo NÃO melhorou com segurança sob
os critérios limitados e o candidato permanece apenas experimental.

**Próxima condição antes de produção:** validar novas peças/eventos físicos
independentes e contraprovas OK/NG nas cinco categorias AOI e diferentes
iluminações, incluindo os erros anteriormente observados. Nenhum score
suficiente sobre o acervo antigo autoriza envio automático 0/1.
