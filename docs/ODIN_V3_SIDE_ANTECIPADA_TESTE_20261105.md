# ODIN AOI — SIDE antecipada pela CNN V3 época 5 no Modo Teste

## Problema

Em evento XP de 05/11/2026 com categoria AOI EMBORCADO, a SIDE mostrou
CNN V3 época 5 OK extremamente forte, KNN C A e B OK com cinco
vizinhos OK nos dois descritores, mas ainda houve TOP/MID. MID votou NG,
gerando falso alarme de fusão. Esses scores não são probabilidades calibradas.

## Contrato implementado

`build_side_early_decision(..., allow_supervised_v3=True)` admite uma nova
rota **somente se a interface estiver no Modo Teste**. Mantém a exigência
geral de confiança >=97%, direção coerente com score NG, `SIDE`,
categoria roteada pela antiga V2 (FALTANDO, EMBORCADO, INVERTIDO ou
DESLOCADO), CNN V3 `PRIMARY` ativa, identidade SHA e época 5 verificadas
pelo contrato fixado e consenso `KNN C A/B` com recomendação seletiva,
sem conflito, com top-5 da mesma classe em ambos os descritores.

Se a CNN e a KNN divergirem, o checkpoint estiver ausente/alterado,
o score não atingir 97%, faltar vizinho ou houver veto físico, a
aquisição continua para TOP/MID. A rota não usa nome de PNG, data,
categoria como sinônimo da falha física nem sobreposição de regras V2.

O marcador `side_early_exit` registra `scope=SUPERVISED_TEST_ONLY`,
`model_epoch=5`, hash, luzes examinadas e dispensadas, mantendo
`production_review_required=True` e `operator_review_required=True`.
**A nova rota não autoriza PRESS_0/PRESS_1**, não remove revisão KNN C,
não modifica pesos/treino, memória, fonte XP ou política de Produção.
A função `side_automatic_decision` e o gate XP mantêm o bloqueio de V3
para OK sem prova independente. Na Produção e no Modo Sombra, a
rota supervisionada V3 não é solicitada; o caminho convencional
continua buscando TOP/MID quando nenhuma rota preexistente autoriza SIDE.

## Teste e limite

```powershell
python -X utf8 -m unittest tests.test_side_early_decision tests.test_adhesive_multilight_inspection -v
```

Confirmar no Modo Teste com eventos OK e NG físicos distintos:
uma SIDE V3+KNN concordante >=97% deve mostrar `FALHA FALSA` ou
`DEFEITO REAL` e dispensar TOP/MID; discordâncias, incompletude e
scores abaixo do limiar devem coletar as demais luzes.
Repetir na Produção apenas com XP sem envio automático para confirmar
que a nova rota V3 não alterou comandos ou autorização operacional.

O resultado é uma **otimização de captura diagnóstica**, não prova de
que SIDE isolada evita escapes NG em Produção. Uma adoção automática
futura exigiria regressão física independente incluindo NG difíceis,
rótulos humanos e falhas OK.
