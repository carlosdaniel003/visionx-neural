# ODIN: 28/10/2026 — recuperação OK recorrente TOP/MID

Evento supervisionado `0077f20173f343e6b1ea3427e4a3d92b`,
inspecionado em 28/10/2026 às 09:43:43 (Manaus).
Operador: **FALHA FALSA**. ODIN: **DEFEITO REAL**.

## Evidência do erro

- A CNN trial permaneceu NG nas três luzes, mas seu próprio
  consenso era `REVIEW/INVALID_LIGHT_EVIDENCE`.
- Ausência física não corroborada: `physical_score=0.0` e
  `hard_missing_evidence=false`.
- KNN C TOP: melhor vizinho OK em A 0,99245913 e B 0,98483213,
  `public/ok_archive/2026-10-27_1318_FALTANDO_TOP.png`,
  SHA pixel `046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d`.
- KNN C MID: melhor vizinho OK em A 0,97646753 e B 0,94535756,
  `public/ok_archive/2026-10-27_1318_FALTANDO_MID.png`,
  SHA pixel `338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29`.
- MID KNN C: `SUGERE_OK`, 5 vizinhos OK em A e B. SIDE e TOP
  KNN C: votos A/B OK sem sugestão confiável.
- Busca de assinaturas independente: SIDE 0,949029 OK / 0,949006 NG
  (empate), TOP 0,995616 OK / 0,942122 NG,
  MID 0,996298 OK / 0,924989 NG.
  SIDЕ e TOP `CONFLICT`, MID `NO_CONCLUSION`.
- Índice de pares exatamente idênticos: `matched_count=0`.
  Logo, presença no `ok_archive` NÃO prova par idêntico atual.
  Há correspondência **semelhante** de evento físico OK, cuja
  qualidade foi confirmada pelo operador.

## Causa

O guard anterior `_three_light_ok_counterevidence` exigia
`REVISAO` e status `CONFLICT` em **todas** as três luzes;
neste caso, MID estava `SUGERE_OK` e `NO_CONCLUSION`,
e SIDE tinha melhor assinatura OK apenas 0,949029.
Por isso o caminho `FALHA FALSA` nunca era executado.

## Novo caminho bem delimitado

`_recognized_two_light_operator_ok` usa dois hashes de fontes
historicamente OK do **mesmo evento** (TOP/MID) e os dois
descritores KNN A/B, com limiares mínimos distintos por luz.
Exige banco READY em três luzes, votos OK A/B, TOP/SIDE em
revisão, MID `SUGERE_OK`, TOP/MID com assinaturas OK
bem separadas de NG, SIDE não conclusivo sem tendência NG forte,
CNN em consenso inválido e ausência física não corroborada.

Se todos os critérios forem atendidos, **classificação exibida
`FALHA FALSA`** e **revisão do operador necessária**.
A política de produção recomputa o contrato de evidência do
evento, sem confiar apenas no campo final.

**Nunca envia 0 ou 1 automaticamente**, não altera pesos CNN,
memória KNN, ponteiro live, arquivo histórico, nem XP.
`auto_allowed=False`, `proposed_decision=""`.
O operador deve verificar a peça real e pressionar
`0` (OK) ou `1` (NG).

Teste de regressão com os scores reais do evento inclui veto de
NG físico, votos NG em qualquer luz, hash adulterado, evidência
visual NG e descarte da classificação se o contrato for falsificado.
