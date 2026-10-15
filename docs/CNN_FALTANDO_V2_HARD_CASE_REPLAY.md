# Etapa 2 — CNN FALTANDO V2: reforço supervisionado e auditoria do evento

**Data:** 15/10/2026. **Escopo:** somente CNN FALTANDO V2. A Etapa 1 de KNN está integrada na `central`; a Etapa 3 de evidência física independente ainda não começou.

## Diagnóstico observado e limites da evidência

No evento `f39c34cd51bb48abb7538afd6eec550e`, a CNN FALTANDO V2 classificou como OK três imagens (SIDE/TOP/MID); o operador informou ter confirmado **1/NG real** posteriormente. O debug capturado **antes do julgamento humano** não contém prova de que `DecisionPersistenceQueue`, `OnlineLearningQueue`, o treinador subprocessado e os gates tenham concluído. Os diretórios `reports/neural_online/` e `data/knn_archive_memory/` são locais e ignorados pelo Git; não aparecem no checkout do CI.

O pipeline pré-existente já grava confirmações humanas `NEW_CNN` (não grava `production_auto`), executa replay histórico, faz treinamento com AdamW em CPU e só troca o ponteiro `live_active.json` após gates. Entretanto, em cada época forçava explicitamente **somente o evento mais recente**; falhas confirmadas anteriores dependiam de seleção probabilística pelo sampler e podiam não receber gradiente na época. Um candidato reprovado aparecia como `COMPLETED` no arquivo `statuses/<id>.json` embora `latest_event.json` dissesse `REJECTED`.

## Mudanças (sem pesos ativados ou thresholds novos)

- O treinador mede **antes de treinar** o comportamento do checkpoint anterior em todos os eventos humanos online. Seleciona para reforço por época cada evento incorreto (incluindo uma falha individual de MID quando o evento agregado está correto) e sempre o caso novo, sem duplicatas de ID. Os demais exemplos permanecem no replay ponderado normal.
- Relatórios anteriores sem lista de casos mantêm o comportamento compatível de somente o evento novo. A lista `forced_events_per_epoch` e `forced_previous_error_count` aparece no relatório de treino. Divergência evento/rótulo/luz no replay interrompe o treino.
- Experimentos especiais `candidate_only` com `recovery_correction_id` preservam o conjunto fixo retificado, comparabilidade e isolamento de checkpoints. Não altera a arquitetura, loss 0,15 padrão, número padrão de épocas, taxa, limiares, conteúdo do acervo ou gates.
- `statuses/<id>.json` registra agora `PROMOTED`, `REJECTED` ou `FAILED` como estados distintos; saída 0 **sem outcome válido** não vira sucesso. Mantém `exit_code` e a compatibilidade do arquivo JSON v1.
- Serviço offline `src/services/startup_regression/cnn_confirmed_event_audit.py` encontra journals pelo **event_id AOI real**, verifica hashes SHA-256 dos pares por iluminação, consulta status, resultado de validação e ponteiro do checkpoint, tudo **somente leitura**.

## Verificar o NG real na estação (com o ODIN fechado)

Atualize a branch somente após revisão/merge do PR. O comando abaixo deve ser executado no **mesmo PC** que salvou o julgamento:

```powershell
cd C:\visionx-neural-main
python -X utf8 -m src.services.startup_regression.cnn_confirmed_event_audit --aoi-event-id f39c34cd51bb48abb7538afd6eec550e
```

O JSON fica em `reports/startup_regression/cnn_confirmed_audit_*.json`. Interpretação:
- `NO_JOURNAL`: não foi encontrado journal que vincule a decisão ao evento (não equivale automaticamente a ausência da memória KNN).
- `JOURNAL_FOUND` com `label=NG`, `views=SIDE/TOP/MID`, `assets_verified=true`: capturas supervisionadas presentes no journal.
- `job_status=PENDING/REJECTED/FAILED/PROMOTED`: a tarefa não terminou, candidato não passou gate, falhou, ou promoveu novo checkpoint. `JOURNAL_FOUND` **não** é prova de pesos treinados. Em versões anteriores, `COMPLETED` não prova promoção; consultar `outcomes`.
- `checkpoint_active_for_event=false` não significa que pesos posteriores não tenham incorporado o evento; o ponteiro aponta somente o evento da última promoção.

**Limite operacional:** o comando não reexecuta inferência nem avalia acurácia com pares reais. Validar na estação o NG e os exemplos OK/NG de placas independentes antes de considerar melhorado o reconhecimento. Backups de `reports/neural_online/` e `reports/faltando_neural/` continuam indispensáveis.

## Segurança e próximas validações

Nenhum checkpoint treinado foi promovido pelo PR; nenhum comando ao XP, roteamento KNN, detecção física, UI ou auto-OK foi alterado. Os **gates existentes** de replay de todos os exemplos históricos e online continuam em vigor, mas não substituem holdout independente. A lógica operacional anterior continua podendo promover **futuros** candidatos quando seus gates internos forem satisfeitos, sem que este PR mude essa política. Não declarar recuperação de NG real sem diagnóstico local e avaliação em imagens inéditas.

Testes no CI verificam seleção de hard cases (evento NG e MID), isolamento de luzes/rótulos, estados do journal, integrity checks, gates e preservação do checkpoint original. Concluir esta etapa somente após CI e validação local; **não avançar à Etapa 3 sem OK**.
