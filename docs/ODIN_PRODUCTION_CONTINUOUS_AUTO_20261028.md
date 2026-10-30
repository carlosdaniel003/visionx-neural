# ODIN — Modo Produção contínuo 0/1 (28/10/2026)

Solicitação do operador: sem revisão obrigatória por peça, com pause
pela barra de espaço. A política separa julgamento físico de comando XP.

## Decisão automática

- A lógica existente de SIDE, KNN C supervisionada e CNN com evidência
  verificada permanece em vigor.
- As **duas** rotas previamente confirmadas como falha falsa pelo
  operador em 28/10 ganham **0/OK automático** apenas quando o contrato
  de evidência é novamente verificado: `operator_recurrent_ok_top_mid_review_v1`
  e `mixed_light_knn_ok_cnn_review_manual_v1`. Os nomes legados
  continuam nos logs, mas não exigem toque por peça no Modo Produção.
- Uma inspeção **completa, válida nas 3 iluminações e inconclusiva**
  sem rota OK autorizada é **1/NG preventivo automático** (fail closed).
  O evento não é proclamado como defeito físico comprovado:
  `verdict=NG PREVENTIVO`,
  `reason=production_uncertain_auto_ng_quarantine_v1`.
- Captura incompleta, falta de uma iluminação/rota identificada, falha
  de transporte, troca de evento, modo Teste/Sombra, pausa e erro XP
  **não são resolvidos com um comando adivinhado**. São falhas
  operacionais que exigem parada/recuperação; não são revisão rotineira.

## Janela de pausa

As duas novas rotas OK e o NG preventivo têm 5 segundos de espera
após a apresentação para o operador pausar com espaço.
Na emissão a política é **revalidada** e a geração/peça/mode/pausa
são conferidos. A pausa não desfaz comando já transmitido.

## Proteção de treinamento

As decisões automáticas do novo modo **não equivalem a etiquetas
físicas confirmadas** e não podem alimentar nem KNN nem CNN.
`DecisionPersistenceQueue` as arquiva separadamente sob
`reports/production_auto_unverified/YYYYMMDD/`, inclusive recortes
disponíveis, hashes e o motivo da política. A pasta de treinamento
OK/NG não é contaminada por NG preventivo ou OK aproximado.

## Limitação explícita

Regras OK por similaridade aproximada ainda podem errar em peças
diferentes. A validação automática e o operador com acesso ao pause
diminuem o risco, mas não são um sistema independente de intertravamento
do equipamento. A instalação deve ser testada com o XP desconectado
antes de usar a saída física e é necessário verificar o significado
e retorno de comandos 0/1 com a TI/manutenção da estação.


## Extensão autorizada de 29/10/2026

O operador autorizou também o reconhecimento de falha falsa
`operator_known_top_mid_ok_review_20261029_v1` para **0=OK automático**
somente no Modo Produção contínuo. A revalidação preserva as provas
TOP/MID, as três iluminações, os vetos físicos e a janela de **5 segundos
para espaço**. A nova razão auditável é
`supervised_operator_known_top_mid_ok_20261029_auto_v1`.
As decisões desta rota são isoladas do treinamento da mesma forma que as
outras decisões OK aproximadas. Em Teste e Sombra, a regra é só leitura e
continua sujeita a operador.

A autorização não se estende ao atalho SIDE
`side_verified_archive_visual_review_v1`, que permanece manual: essa
outra rota tem somente uma iluminação e evidência de geometria que pode
exigir revisão. Alterá-la exige avaliação separada.

## Métricas da sessão diária — apenas julgamentos 0/1 (29/10/2026)

Problema identificado em Produção: o cartão mostrava **27 ANÁLISES**
porque contava todo resultado renderizado, enquanto só **17 inspeções**
foram efetivamente julgadas (2 OK AUTO e 15 NG AUTO; nenhuma MANUAL).
Dez análises sem comando também entravam indevidamente no tempo médio.

**Contrato novo:** ANÁLISES = OK AUTO + NG AUTO + MANUAL. O contador
e o tempo são atualizados **somente após o julgamento 0/1 confirmado**;
uma captura analisada e descartada, parada, pausa ou falha de envio XP
não conta e não compõe a média. A duração continua sendo o tempo da
captura ao resultado renderizado, sem espera do operador nem scroll.
A gravação de cada decisão, contagem e tempo é atômica no JSON diário,
sem duplicar julgamento da mesma geração/evento.

O rótulo legado `PRECISÃO` significa **taxa de decisões automáticas
entre decisões 0/1 concluídas**, não acurácia do classificador contra
verdade física independente. Análises sem julgamento não fazem parte
do denominador; decisão manual entra no denominador, não no numerador.

**Migração de sessões v1:** o arquivo legado preserva o histórico
de OK AUTO/NG AUTO/MANUAL. O contador ANÁLISES é recomposto a partir
desses registros; por exemplo, 27 vira 17 neste caso. Quando o
arquivo antigo contém tempos de análises sem decisão, **não há como
subtrair precisamente esses tempos de um somatório agregado**.
O programa reinicia só o acumulador da média (exibe `—` até novo
0/1 com tempo válido), sem inventar média histórica nem apagar os
arquivos de dias anteriores. Se todas as análises antigas já tinham
0/1, preserva a média histórica válida. Formato novo: schema v2.

A mudança não altera evidências, julgamentos, memória IA, parâmetros
KNN/CNN, comandos 0/1, pausa nem política de produção.


## 30/10/2026 — Produção binária automática, sem terceira classe ou revisão de qualidade

Contrato operacional atualizado a pedido do operador para o ambiente de
simulação/supervisão:

- **Inspeção concluída e elegível:** apenas dois comandos:
  `0/OK = FALHA FALSA`, `1/NG = DEFEITO REAL`. O veredito visível
  acompanha o comando definido pela política de Produção, sem
  `NG PREVENTIVO` ou `REVISÃO OBRIGATÓRIA` como terceira classe.
- Mantém a prioridade das rotas de OK/NG com provas revalidadas.
  Para uma **trinca completa** sem autorização comprovada de OK,
  seleciona `1/NG`, registra `production_binary_unverified_ng_v1`
  e **preserva a classificação original** em `source_verdict` e
  `source_reason`. Isso é uma decisão binária de rejeição, **não
  confirmação física de um defeito**.
- Um atalho SIDE validado com veto físico inconclusivo nunca recebe
  `0/OK` por suposição; a política binária poderá rejeitar com 1 se
  o contrato de conclusão estiver íntegro. Não se abre revisão de
  qualidade por peça no controlador.
- Captura incompleta/roteamento ausente, falha do agente XP ou comando
  TCP não confirmado ficam como **erro operacional sem 0/1**. Não
  converter uma falha técnica em falsa decisão de qualidade.
  O controlador interrompe a sessão e fornece diagnóstico, sem criar
  tela de revisão ou registrar julgamento inexistente.
- A pausa por Espaço, espera de apresentação (5s quando exigida),
  proteção de geração/evento e envio uma única vez continuam.
- Decisões NG sem confirmação física não alimentam KNN/CNN:
  evidência arquivada em `reports/production_auto_unverified/`.
  Métricas só contabilizam transmissão 0/1 confirmada.
- Teste e Sombra mantêm seu fluxo anterior. Não alterou o
  `agente_industrial_xp.py`.

**Atenção:** o comando 1 para uma captura inconclusiva continua podendo
ser um falso NG, mesmo que o rótulo público seja binário. O comando 0
permanece condicionado às contraprovas para não liberar NG como OK.
Uma parada de software por falha operacional não é uma terceira classe
de defeito. Validar isoladamente o protocolo 0/1, a interrupção de XP
e as peças OK/NG adversárias antes de qualquer uso físico.
