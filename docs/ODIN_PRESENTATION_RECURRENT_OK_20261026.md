# ODIN — apresentar FALHA FALSA sem ocultar confirmação XP (26/10/2026)

Evento `c71c622892164d1484f701fa5a050b54`, inspecionado
em 26/10/2026 15:43:15 no Modo Produção.

## Diagnóstico reproduzido pelo operador

O **veredito final do julgamento já estava correto**:

- `Veredito: FALHA FALSA`, `Defeito: False`;
- `Regra de fusão: operator_confirmed_recurrent_ok_three_light_review_v1`;
- `Revisão obrigatória: True`, porque o comando XP `0` continua
  exclusivamente humano nessa rota aproximada (não há par exato);
- SIDE KNN A=0,99325113, B=0,97756581, ambos usando a referência OK
  `2026-10-21_1412_FALTANDO.png`;
- TOP/MID também apresentam contraprova KNN OK; CNN continua sem
  consenso válido e ausência física não foi corroborada.

A interface escondia a classificação certa e exibia
**REVISÃO OBRIGATÓRIA** por três bugs independentes:

1. `verdict_feedback_state` testava a pendência XP *antes* da
   classificação `FALHA FALSA`;
2. `ProductionAutonomyController._finish_presentation` descartava
   intencionalmente a opção `FALHA FALSA` ao escolher o texto do overlay
   de intervenção;
3. `ProductionInterventionFeedbackOverlay.show_reason` não aceitava
   `FALHA FALSA` e convertia textos desconhecidos em revisão.

## Correção somente na apresentação

- O overlay do julgamento aceita **FALHA FALSA** verde **somente**
  para memória OK recorrente com o contrato inteiramente validado
  pela política de produção (SHA/referência, três luzes, veto físico,
  revisão humana exigida). Casos não validados continuam amarelo/vermelho
  como REVISÃO OBRIGATÓRIA.
- O controlador de produção mantém o texto `FALHA FALSA` da política
  para o overlay de intervenção, em vez de substituí-lo por revisão.
- O overlay de intervenção mostra **CONFIRMAÇÃO HUMANA NECESSÁRIA**,
  **FALHA FALSA** em verde, e **0 = OK | 1 = NG**.
  A pendência humana NÃO é ocultada.
- Teste `tests/test_recurrent_ok_production_presentation.py` comprova
  o veredito, apresentação, bloqueio de XP automático e fallback de
  evidência adulterada.

Nenhuma alteração na CNN, KNN, acervo, confiança, julgamento, política
de autorização, ponteiro live ou transmissão 0/1 do XP.

## Aplicação

Com ODIN fechado, `git pull origin central`, reiniciar.
Na reprodução do caso: exibir FALHA FALSA + confirmação humana necessária.
O botão/tecla **0** continua sendo a confirmação manual necessária.
