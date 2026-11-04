# ODIN AOI — integração controlada da CNN V3 época 3 na Produção (04/11/2026)

**Pedido do operador:** incorporar o modelo V3 da época 3 ao fluxo real
de Produção. **Estado do código:** integrado, mas a ativação depende de
um comando local explícito, porque o arquivo `.pt` reside somente na
estação Windows. Nenhum commit no GitHub ativa a estação remotamente.

## Decisão e limite obrigatório da primeira implantação

A época 3 do treinamento
`20261104T192753_047124Z_6b733a` aprendeu **13/14** imagens
humanas (**8/9 NG**, **5/5 OK**). A imagem
`2026-10-21_1432_FALTANDO_SIDE.png` continuou sendo um **NG
físico classificado como OK**. A validação desta época teve
**3 NG→OK**, **0 OK→NG** (216/219), empatando nas contagens
com a época 0. O teste independente não foi executado para a
época 3, e os rótulos da validação continuam amplamente provisórios.

Por isso, **não é tecnicamente justificável autorizar PRESS_0/OK
exclusivamente pela V3**. A implantação explícita atual tem
escopo **NG VETO ONLY**:

- Os pares gabarito/teste são avaliados em SIDE/TOP/MID pela V3,
  sem consultar categoria AOI na sua decisão.
- Para inspeções **novas**, uma iluminação V3 com score NG ≥ 0,5
  impede um OK automático do fluxo legado. Em captura de Produção
  completa e íntegra, a política binária rejeita com `1/NG`,
  atribuindo a proveniência
  `aoi_quality_v3_epoch3_ng_veto_trial_v1`. Este NG é um voto
  operacional **não verificado fisicamente** e não deve virar
  automaticamente verdade para treino.
- Se a V3 votar **OK**, esse voto sozinho nunca autoriza
  `0/OK`; a autorização continua sendo das políticas legadas
  de evidência física e memória humana.
- O atalho `KNOWN_KNN` continua prioritário para a memória
  confirmada, sem ser sobrescrito pela CNN experimental.
- Ausência do ponteiro mantém o comportamento anterior.
- Falha de hash, schema ou iluminação incompleta/contraditória
  detectada enquanto a V3 está ativa **interrompe** uma emissão
  automática em vez de inventar OK/NG.
- A política de Modo Teste/Sombra permanece sem novos comandos XP.
  Apenas a política binária do Modo Produção consome o veto V3.
- Não altera nem reescreve `reports/neural_online/live_active.json`,
  pesos da V2, memória KNN ou `agente_industrial_xp.py`.

Isso **não é substituição integral da CNN V2**. É integração
controlada da V3 como contraprova de NG. Mudar para a V3
como autoridade exclusiva de `0/1` exigiria avaliação de
NG efetivamente liberados e um holdout de peças novas.

## Integridade

Origem fixa: `training_report.json` da rodada de 19:27 UTC.
O snapshot de época 3 fica em:

```text
reports/aoi_quality_cnn_v3/candidates/20261104T192753_047124Z_6b733a/epoch_diagnostics/epoch_03_experimental_only.pt
```

SHA-256 esperado:

```text
fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce
```

O carregador e ativador verificam hash, manifesto original, hash
da V3 prévia, arquitetura, três luzes, resolução 96, fração
de foco 70%, época, 13 acertos humanos e 3/0 erros de validação.
Rejeitam arquivos ausentes ou alterados. O checkpoint continua
declarado experimental e incompatível com o carregador produtivo
V2; um contrato próprio limita a autoridade do voto ao veto NG.

## Como ativar NA ESTAÇÃO, com reversão

Fechar o ODIN e colocar o XP fora do ciclo de inspeção.
Fazer backup dos arquivos locais `reports/` e revisar o comando.

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

python -X utf8 -m src.scripts.activate_aoi_quality_v3_production_trial --root . --activate
```

O comando só terá êxito se o relatório e o snapshot EXISTIREM
no local indicado e seus hashes corresponderem. Ele grava
apenas:

```text
reports/aoi_quality_cnn_v3/production_v3_trial.json
```

Reiniciar o ODIN para iniciar ensaio supervisionado.
Antes de reconectar o XP, testar com envio físico desconectado
e confrontar a classificação em OKs verdadeiros, NGs reais,
a imagem SIDE de 14:32 ainda errada, novas peças e variações
de iluminação. Validar com a TI/manutenção a direção 0/1
e a parada segura. A aprovação do operador autoriza o
procedimento local, não prova ausência de regressões.

### Como desativar/voltar imediatamente à política anterior

Parar a inspeção, encerrar o ODIN e executar:

```powershell
python -X utf8 -m src.scripts.activate_aoi_quality_v3_production_trial --root . --deactivate
```

Reiniciar o ODIN. O script remove **somente o ponteiro da V3**,
depois de conferir seu conteúdo. O checkpoint V2, sua memória
e o ponteiro `live_active.json` não são modificados.
Se o ponteiro estiver corrompido, o comando falha em vez de
sobrescrever um arquivo desconhecido; diagnosticar localmente.

## Testes de regressão

- `tests/test_aoi_quality_v3_production_trial.py`:
  hash e schema, retomada, desativação, inferência real com tensor,
  prioridade de memória, categoria AOI irrelevante e votos SIDE/TOP/MID.
- `tests/test_aoi_quality_v3_production_policy.py`:
  veto em TOP mesmo com SIDE OK; ausência de autorização OK
  da V3; parada em hash/snapshot incompleto.
- `tests/test_aoi_quality_cnn_v3_offline.py` e
  `tests/test_aoi_v3_epoch_diagnostics.py`: treinamento/replay,
  checkpoints de época e isolamento de produção.
- As suítes legadas de fusão multilight, roteamento da memória,
  controle de Produção e V2 permanecem executadas pelo CI.

**Importante:** sucesso de CI é verificação do código, não
medição do modelo na máquina. Nenhuma decisão ou tecla XP
foi enviada daqui.
