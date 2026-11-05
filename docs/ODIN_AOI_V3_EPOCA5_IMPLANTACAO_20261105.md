# ODIN AOI / VisionX — ativação V3 época 5 no Teste, Sombra e Produção

**Data:** 05/11/2026. **Decisão:** operador autorizou usar a época 5
no lugar da V3 época 3 em todas as rotas da V3, mesmo com regressão
OK→NG em validação. Isso é uma **exceção operacional deliberada**,
não homologação estatística ou aprovação do checkpoint offline.

## Evidência e identidade FIXA

Fonte local (não versionada no GitHub):
`reports/aoi_quality_cnn_v3/candidates/20261105T171228_989626Z_44072c/training_report.json`.
SHA do relatório: `47d5d7c3bfe6863f887cf747f70e623ab69e299626ecee2bdf107e1593b92c32`.

Snapshot escolhido, não o `aoi_quality_cnn_v3_candidate.pt`:

`reports/aoi_quality_cnn_v3/candidates/20261105T171228_989626Z_44072c/epoch_diagnostics/epoch_05_experimental_only.pt`.

SHA-256 dos pesos: `eff1d3bf5fd05e2763792e22f634ebd40f432ebab4712055c9441f2fff61fc6d`.

A rede acertou **24/24 pares NG no próprio TREINO** (25 fontes
humanas, uma deduplicada), e **14/14 exemplos humanos anteriores**.
Na validação de 219 exemplos: **0 NG→OK**, mas **2 OK→NG**.
A época 0 marcou 0/0 erros nessa validação, logo o gate conservador
rejeitou a época 5. **A época 5 não foi avaliada no teste pós-seleção
nem num holdout físico cego.** O `training_report.json` conserva
`selected_epoch_using_validation_only=0`, `production_eligible=false`.
Nenhuma destas informações é alterada pela ativação.

## Implementação

- `src/core/neural/aoi_v3_epoch5_contract.py` verifica o SHA do
  relatório real e do snapshot real, origem época 3, identidade
  do manifesto, 25 revisões, métricas e flags de experimental.
  Não libera um arquivo `.pt` genérico.
- `src/scripts/activate_aoi_quality_v3_epoch5.py` faz preflight de
  pesos/3 iluminações e escreve atomicamente somente o ponteiro
  `reports/aoi_quality_cnn_v3/production_v3_epoch5.json`.
- `src/core/neural/aoi_quality_cnn_v3_production_trial.py` seleciona
  **época 5 antes da época 3** quando o novo ponteiro estiver válido.
  Emprega a mesma CNN V3 compartilhada em SIDE, TOP e MID.
- `src/core/neural/aoi_quality_cnn_v3_primary.py`: Teste/MSS,
  Sombra e Produção mostram época e hash realmente usados.
- `src/core/neural/aoi_quality_cnn_v3_trial_consensus.py`: exige
  que as três iluminações pertençam à MESMA época e SHA.
  Vote/ponteiro falsificado ou misto retorna erro em vez de inferir OK.
- `src/ui/production_confidence_gate.py`: preserva a política
  de veto NG e a exigência de evidência independente para pressionar 0.
  Um score V3 não equivale a liberação física autorizada.
- Não se instanciam motores V2 nos casos anteriormente cobertos por
  V2; memória humana KNN e especialista MUITO ADESIVO conservam
  suas responsabilidades. F2/F3, que pertencem a outro sistema/repo,
  não são abrangidos pela atualização da AOI/VisionX.
- Não adiciona regras por nome de PNG, classes da AOI, thresholds
  oportunistas nem treinamento online V3.
- A época 3 permanece no disco e no ponteiro antigo para **rollback**.
  O arquivo V2 legado continua para auditoria, não para decisão principal.

## Passos na estação (Windows/PowerShell)

1. Fechar completamente ODIN e manter XP desconectado do envio
   de teclas automático enquanto valida o novo modelo.
2. Atualizar código:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
```

3. Ativar a época 5 (é obrigatório ter o relatório e o `.pt`
   original no local e com os SHAs acima):

```powershell
python -X utf8 -m src.scripts.activate_aoi_quality_v3_epoch5 --root . --activate
```

A saída deve apresentar `EPOCH5_ACTIVE`, `epoch: 5`,
`checkpoint_sha256: eff1d3bf...`, `validation_OK_false_NG: 2`
e `automatic_OK_from_V3_only: false`.
Qualquer erro de arquivo/SHAs **impede** a nova ativação; nunca
desligar esta verificação nem alterar os metadados do checkpoint.

4. Com XP sem envio, iniciar o aplicativo:

```powershell
python -X utf8 main.py
```

5. No Modo Teste, inspecionar ao menos um PNG NG e um OK real,
   depois repetir SIDE/TOP/MID. O debug deve informar
   `cnn_v3_checkpoint_epoch=5`,
   `cnn_v3_checkpoint_sha256=eff1d3bf...` e
   `aoi_v3_trial.epoch=5` / `status=READY`.
   A classificação da CNN em modo Sombra e Produção deve carregar
   **o mesmo SHA**, não um misto de épocas. Avaliar os falsos NG
   conhecidos antes de conectar o XP.

6. Se ocorrer `ERROR` ou uma decisão incorreta na linha,
   interromper a operação automática e reverter.

### Reversão limpa

Fechar ODIN, desconectar XP, executar:

```powershell
python -X utf8 -m src.scripts.activate_aoi_quality_v3_epoch5 --root . --deactivate
```

Isso remove SOMENTE `production_v3_epoch5.json`.
Se o ponteiro histórico época 3 for válido, a V3 anterior
volta a carregar após reiniciar o ODIN; **não há fallback automático
para a V2**. Se o ponteiro antigo não existir, os casos novos
ficam bloqueados, sem produzir 0/XP a partir de CNN ausente.

## Limites e aferição posterior

24/24 no treino NÃO demonstra 100% em novas ocorrências.
Os 2 falsos NG em validação são conhecidos, e ainda falta a
comprovação independente em lotes físicos futuros. Avaliar
falsos NG, NG liberados, persistência humana e variação visual
por iluminação, sem chamar treino visto de teste cego.

A ativação por código GitHub NÃO escreve automaticamente os arquivos
locais do computador do operador. O comando de ativação é
necessário para que os pesos locais passem a ser usados.
