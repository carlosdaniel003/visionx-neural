# ODIN — reconhecimento da falha falsa de 29/10/2026 (TOP/MID OK histórico)

## Reprodução real

Evento `8525be296415464a83fcd0764ce7e9d7`, estação XP, 29/10 às 07:29
(Manaus): o operador informou **FALHA FALSA**, porém a fusão exibiu
`DEFEITO REAL` e o modo Produção contínua podia converter resultados
não elegíveis em `NG PREVENTIVO`. A captura foi válida, tempo 5,313s.

A CNN experimental (`c13ca329...`) votou NG em todas as iluminações:
SIDE/TOP 0,9987717; MID 0,9998896. Entretanto, **seu próprio consenso
era `REVIEW/INVALID_LIGHT_EVIDENCE`**, sem evidência física de ausência
(`hard_missing_evidence=false`, `physical_score=0.0`).

A KNN C READY reconheceu como melhor vizinho A/B em **TOP** a fonte
`public/ok_archive/2026-10-27_1318_FALTANDO_TOP.png`:
0,98621705 / 0,97805809. Na **MID**, a fonte
`2026-10-27_1318_FALTANDO_MID.png` foi nº 1 em A/B:
0,9825117 / 0,9710344. São duas iluminações do mesmo
padrão histórico OK confirmado pelo operador.

A KNN visual independente favoreceu OK em TODAS:
SIDE 0,991922 vs NG 0,947968; TOP 0,991922 vs 0,942076;
MID 0,988892 vs 0,924092, sem luz NG corroborada.

## Causa

O detector legado `_recognized_two_light_operator_ok` tinha critérios
para um evento anterior, exigindo TOP A >=0,99/B >=0,98 e MID
`SUGERE_OK` de motivo `A_E_B_CONCORDAM` com dois votos de
sugestão. A nova TOP ficou pouco abaixo destes limiares; a MID passou
pela rota seletiva `B_SELETIVA_E_VOTOS_A_B_CONCORDAM`, com
sugestão B `SUGERE_OK` e A `REVISAO`. A função rejeitou a
referência correta e a fusão manteve os votos fortes da CNN.

## Correção

Uma segunda regra independente
`operator_known_top_mid_ok_review_20261029_v1`, em
`knn_visual_corroboration.py`, exige cumulativamente:

- Categoria FALTANDO, três capturas, CNN experimental de mesmo SHA
  com consenso inválido, relatório e rotas íntegros.
- Três votos KNN C A/B OK, banco READY, SIDE/TOP em revisão e
  MID na rota seletiva acima.
- TOP/MID melhor vizinho A/B = a mesma fonte histórica de 27/10,
  com **caminho e SHA de pixels pinados**; TOP A>=.985, B>=.975;
  MID A>=.98, B>=.97. Quatro entre cinco vizinhos OK por descrição.
- Nenhum vizinho NG nessas listas >=.93, vantagem de pelo menos .06
  sobre o NG mais próximo; SIDE mínimo três de cinco OK e nenhum NG
  >=.93.
- KNN visual independente favorável a OK nas três luzes, margens
  SIDE >=.04, TOP >=.045, MID >=.06 e sem NG corroborado;
  integridade dos scores e ausência de evidência física NG.

A classificação exibida é **`FALHA FALSA`** quando a regra inteira
coincide. A política do XP recomputa todos os critérios a partir dos
votos originais, e o modo Produção contínua **não converte** essa
recuperação validada em `NG PREVENTIVO`.

**Importante:** isso ainda é identificação visual aproximada,
não prova de par pixel-idêntico nem certificação física da captura atual.
O resultado mantém `operator_review_required=true`,
`eligible_for_final_decision=false`, `auto_allowed=false`,
`proposed_decision=""`. **O operador confirma 0 ou 1 manualmente.**
O histórico de falhas reais visualmente parecidas deve continuar
fazendo parte da proteção antes de qualquer uso autônomo.

Não altera arquivos PNG, rótulos, checkpoint, rotas de par exato,
regras antigas nem protocolo 0/1 do XP.

## Validação

`tests/test_oct29_operator_recurring_ok.py` reproduz os valores de
diagnóstico do log e testa adulteração de referência, NG concorrente,
ausência física, CNN válida em NG, votos contrários, redução de
semelhança, perda de luz e bloqueio do comando XP. Usa dados
estruturados do log; não substitui replay do PNG original da estação.

Ao atualizar, executar teste na estação com XP isolado e registrar
veredito `FALHA FALSA`, nome da regra e resultado do gate.
Apenas após testar contra OK e NG reais da mesma geometria será
possível concluir se o reconhecimento generaliza.


## Autorização posterior do operador — Produção contínua (29/10/2026)

Após o reconhecimento acima, o operador autorizou explicitamente decisão
automática **0=OK** para esse caso, no Modo Produção supervisionado,
sem confirmação manual de cada placa. A rota continua confinada ao
contrato `operator_known_top_mid_ok_review_20261029_v1`; o marcador de
memória não é autorização suficiente por si só.

O gate `production_decision_policy` recalcula a correspondência TOP/MID
pelas evidências originais e somente libera
`supervised_operator_known_top_mid_ok_20261029_auto_v1` quando:
- `production_continuous_auto=True` (armado exclusivamente no controlador de Produção);
- há três iluminações completas e rotas `NEW_CNN` válidas para FALTANDO;
- TOP/MID se relacionam com as referências históricas OK e hashes pinados,
  sem vizinhos NG fortes, com KNN C READY e contraprova independente OK;
- consenso CNN é inválido, sem ausência física comprovada, discordância
  física ou marca de geometria que exige revisão (`GEOMETRY_REVIEW`);
- a peça e a geração de análise ainda são as mesmas, não há pausa/intervenção
  e a política continua válida no instante de transmitir.

O controlador preserva **5 segundos de janela para barra de espaço**, envia
uma única vez via fluxo normal `save_label("OK", source="production_auto")`
e exige confirmação de êxito do comando XP. Se XP falhar, bloqueia a
continuidade e solicita intervenção, sem transmitir outro comando presumido.
Teste e Sombra continuam sem autorização automática.

A decisão não comprova que os pixels da captura atual sejam idênticos
ao arquivo histórico, nem equivale a rótulo humano de treinamento. O evento
automático fica em `reports/production_auto_unverified/` e não entra na
KNN/CNN. Nenhum arquivo do acervo nem peso de rede foi alterado.
A pausa é um controle do software, não intertravamento de segurança.


## Diagnóstico e apresentação — evento 08:22 de 29/10/2026

O evento `a614cc6e056746d88b41f417a978c6a5` teve, segundo
o debug, **veredito final FALHA FALSA** pela regra
`operator_known_top_mid_ok_review_20261029_v1`. A CNN emitiu NG
bruto em SIDE 0,9998211, TOP 0,9987717 e MID 0,9998896; isso não
constitui consenso CNN validado. A KNN C recuperou novamente TOP
0,986217/0,978058 e MID 0,982512/0,971034 da fonte OK histórica.
A evidência independente favoreceu OK em SIDE 0,995509 contra
0,951698 NG, TOP 0,991922 contra 0,942076 e MID 0,988892
contra 0,924092. Não houve corroboração física forte NG.

**Correção de visualização:** a seção `INFLUÊNCIA DOS MOTORES` mantém
as barras e escores CNN originais, mas marca os três votos como
**CNN NG CONTRARIADA PELA MEMÓRIA OK**, e a KNN como reconhecimento
histórico visual **não exato**. O cabeçalho informa `FALHA FALSA`
como veredito **consolidado**; a apresentação não reescreve resultados
internos da CNN nem cria correspondência exata falsa.

**Correção de debug:** `Revisão da evidência (antes do gate XP)` não é
equivalente a solicitação humana. O relatório multilight passa a expor
separadamente o controlador, estado de armamento, regra da política,
permissão automática, comando proposto e confirmação real do envio.
O controlador arma `production_continuous_auto` após a conclusão da
análise e antes da apresentação da decisão; um snapshot tomado antes
disso não pode ser lido como envio proibido nem revisão exigida.

Para esta regra, a política **já existente na central** autoriza
`0/OK` automático quando as evidências originais são novamente
validadas, aguardando a janela de 5 s para pausa por espaço e
revalidando antes de enviar. A alteração atual é de **observabilidade
e testes**, não muda limiares, treinamento, comandos XP ou a pausa.
Capturas inválidas, falhas de comunicação, pausa ou contrato não
verificado não podem ser convertidos em liberação 0/OK por conveniência.
Uma parada por anomalia operacional não deve ser confundida com revisão
de rotina da classificação.

A validação inclui replay estruturado de 08:22, verificações da
apresentação CNN/KNN, política armada e desarmada, rejeição de hashes
adulterados e visibilidade do comando XP (proposto ≠ enviado).
O teste na estação física continua necessário.
