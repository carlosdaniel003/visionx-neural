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


## 03/11/2026 — Correção humana durante a pausa do Modo Produção

Fluxo: ODIN apresenta um NG e está aguardando a emissão automática;
o operador aperta **Space** e, enquanto a peça ainda é a captura atual,
aperta **0/OK**. O atalho da janela agora abre o mesmo gate de revisão
humana que existia nos modos anteriores, salva o novo rótulo na memória
KNN durável e cancela qualquer callback 1/NG pendente.

- O botão pode permanecer oculto: o atalho chama o controlador de
  Produção somente quando há **inspeção concluída, mesma geração,
  tela atual, modo Produção e pausa ativa**, antes de qualquer
  transmissão automática. Teclas físicas XP recebidas no modo pausado
  seguem o mesmo gate, mas não retransmitem ao XP.
- Se a tecla for pressionada no próprio ODIN, **PRESS_0** ou
  **PRESS_1** deve ser transmitido com sucesso antes da persistência
  humana. Falha na conexão mantém a peça na revisão, sem atualização
  de memória, sem contador manual e sem avanço fictício.
- A atualização usa o mesmo `OPERATOR_CONFIRMED` e
  `confirmed_at_ns` da correção feita em Modo Teste. Uma decisão
  `production_auto` nunca equivale a julgamento humano.
- O contador de Produção registra uma única decisão **MANUAL**
  concluída por evento. Depois do 0/1 manual, o sistema permanece
  pausado até novo Space do operador.
- O agente XP não precisa ser alterado: só a aplicação moderna foi
  modificada. Testar com a saída física isolada antes da operação real.

### Reutilização da mesma tela RGB após nova correção

Além do reconhecimento anterior por **ambos os recortes** gabarito/
teste, um julgamento humano novo salva também o hash exato da
**tela completa da AOI**, por luz. Uma repetição pode passar por
`operator_latest_exact_frame_triplet_v1` mesmo que OCR e recortes
variem, somente se as três telas SIDE/TOP/MID forem pixel-idênticas
às do **mesmo evento humano** mais recente, na mesma categoria e
iluminação.

A autorização de 0/OK ainda exige três pares disponíveis, sondas
independentes de ausência física **NO_STRONG_EVIDENCE** por luz,
ausência de hard missing/erro de captura, confirmação humana cronológica
persistente íntegra e nova validação no gate antes da tecla XP.
Marcações adulteradas e memória divergente bloqueiam o comando,
sem reaproveitar simplesmente um score da CNN. Isso não modifica
nem retreina a CNN nem considera 98% de similaridade como identidade.

Frames antigos importados do `ok_archive`, sem confirmação humana
do frame completo no novo formato, fornecem apenas diagnóstico exato
visual, não autorização automática.


### Exceção confirmada apenas na SIDE

Se o ODIN encerrou a inspeção antecipadamente na SIDE e o operador
corrigiu 1/NG para 0/OK durante a pausa, o julgamento humano pode ser
recuperado na **próxima ocorrência da mesma tela SIDE pixel-idêntica**.
A rota `operator_latest_exact_frame_side_v1` exige a mesma categoria,
iluminação e hash RGB integral, fonte `OPERATOR_CONFIRMED` cronológica,
captura SIDE final íntegra, sem hard missing, e sonda independente de
ausência física `NO_STRONG_EVIDENCE`. O gate revalida a identidade
humana antes do XP, sem aguardar TOP/MID nem depender do ID de evento.

Capturas não idênticas, conflito de humanos, falha de sonda, ausência
física ou captura incompleta continuam sem autorização de 0/OK por
essa exceção. O atalho de SIDE não modifica as regras visuais genéricas
de semelhança (KNN A/B) nem confunde 98% com igualdade exata.

## 03/11/2026 — Julgar imediatamente após a rolagem em Produção

Solicitação do operador: retirar a espera de 5 segundos **após** a rolagem
até a parte inferior da interface. A partir desta alteração, a conclusão da
rolagem agenda a política binária 0/OK ou 1/NG nos próximos ciclos do Qt,
sem pausa visual adicional de 900 ms e sem os antigos atrasos de 2/5 s.
Os atrasos históricos mencionados acima estão substituídos por esta regra.

- Mantêm-se a apresentação e as velocidades da rolagem vertical/horizontal,
  a espera de renderização e o posicionamento inicial no topo.
- A barra de espaço continua pausando a apresentação e bloqueando comandos
  ainda não enviados. Como **não existe mais janela extra após a rolagem**,
  correções Space+0/1 devem acontecer durante a apresentação antes do envio.
- A política, os vetos físicos, a geração/evento, o estado de pausa,
  o envio único e a confirmação XP são revalidados antes de qualquer 0/1.
  Sem captura válida ou transporte confirmado, permanece erro operacional,
  não julgamento fictício.
- Se não houver área rolável, mantém-se a apresentação mínima existente.
  Modo Teste/Sombra, CNN/KNN, memória, pesos e agente XP não mudam.

Validar primeiro com XP isolado, especialmente interrupção via Space durante
a rolagem, captura trocada, OK/NG difíceis e comando enviado uma única vez.

## 03/11/2026 — Corrigir deadlock SIDE de revisão sem TOP/MID

Evidência de estação: `event_id=76df84c3eaaa4cb4a0a1cf14d16a285c`,
categoria FALTANDO, CNN SIDE `FALHA FALSA`, score NG bruto
`0.030528292059898376` e KNN C com A/B apenas aproximados.
O atalho `side_verified_archive_visual_review_v1` dispensava TOP/MID
**apenas para solicitar revisão**, marcava
`eligible_for_final_decision=False` e passava a análise para o gate
binário. A política rejeitou corretamente a captura incompleta com
`production_binary_operational_capture_fault_v1`; o controlador entrou
em `operational_fault` e Space+0/1 ficou bloqueado. Ocorreu impasse
entre dois contratos, sem comando de qualidade ao XP.

**Novo contrato:** em Modo Produção, a memória visual aproximada da
referência 14:16 não encerra a SIDE para revisão. Quando nenhuma rota
SIDE autônoma revalidada estiver disponível, o ODIN continua a aquisição
TOP/MID da **mesma inspeção** e só publica o julgamento após a fusão
completa. Uma trinca válida permite OK quando comprovado pela política
ou NG binário de rejeição com motivo preservado; se for inconclusiva,
o NG não vira verdade humana para treinamento. Space+0/1 pode corrigir
uma inspeção concluída ainda não transmitida, nunca uma captura
incompleta ou um evento antigo.

A política binária é armada antes da primeira renderização da fusão
final, eliminando o flash de `REVISÃO OBRIGATÓRIA` da CNN como
terceiro resultado produtivo. O estado interno de revisão é mantido no
debug/auditoria. Falha de captura, evento errado, erro XP e captura
incompleta permanecem falhas **operacionais sem comando inventado**;
não são automaticamente convertidas em OK nem NG.

As rotas SIDE legitimamente habilitadas (CNN/última memória humana exata
com validação específica) não mudam. Modo Teste/Sombra, checkpoints,
acervos, thresholds KNN/CNN e agente XP não mudam. Validar com o XP
fisicamente isolado, incluindo pausa durante rolagem e repetição de
0/1 manual, antes da liberação real.

## 03/11/2026 — MUITO ADESIVO: proveniência multilight e gate XP

Incidente real `4bd0f318cccc4915b3156a741429c5b9`:
a entrada `valid_epicenter` e SIDE/TOP/MID estavam completas,
`fusion_rule=adhesive_multilight_strong_auxiliary`, `DEFEITO REAL`,
TOP/MID com evidência física forte. Porém a fusão especializada
`fuse_multilight` retornava sem registrar
`detail.recognition_light_routes`, que
`_completed_continuous_inspection` exige para distinguir conclusão
de captura inválida. Resultado incorreto: `CAPTURA INVÁLIDA`,
`production_binary_operational_capture_fault_v1`, sem 1/NG ao XP.

A fusão especializada agora usa o **mesmo coletor de proveniência**
por luz da fusão genérica. Registra apenas as rotas reais de
SIDE/TOP/MID, o agregado (`NEW_EXPERTS`, `MULTILIGHT_MIXED` ou
vazio), diagnósticos e rótulo humano autenticável de `KNOWN_KNN`,
no detalhe e na trilha de decisão; não copia a rota TOP para todas.
A confirmação da fusão de adesivo e os limiares físicos não mudam.

Uma trinca íntegra e conhecida passa à política binária de Produção.
O resultado NG confirmado pelos especialistas pode finalmente propor
`1/NG` após a rolagem, ainda sujeito a pausa, revalidação e ACK XP.
OK só segue sua política usual, sem converter semelhança KNN em
memória exata nem em rótulo de treinamento. Ausência de luz,
rota vazia/desconhecida, `KNOWN_KNN` sem prova humana ou
`capture_error` continuam bloqueando comandos por falha operacional.
Modo Teste/Sombra não passam a emitir teclas 0/1.

### Contraprova NG humana exata em iluminação sem especialista de adesivo

O roteador `KNOWN_KNN` pode reconhecer um par RGB com julgamento
humano NG sem executar o especialista físico de adesivo. Isso gera
scores físicos zerados; não significa que a luz esteja OK. A fusão
poderia inferir OK com base nesses zeros e liberar indevidamente 0.

A autorização binária de adesivo agora exige para cada luz KNOWN
`recognition_memory_verified=True` e `recognition_match=EXACT_PAIR`,
além do rótulo humano e da rota. Se a fusão concluir OK, porém qualquer
luz KNOWN registrar rótulo NG, há **parada operacional com motivo
`production_adhesive_known_ng_conflict_v1`**; nenhum 0/OK ou NG
físico é inventado. Resultado físico NG completo sem contradição
segue a política normal. A correção de memorizações humanas mais
recentes por frame exato continua sujeita ao contrato independente.


## 03/11/2026 — Memória operacional das decisões AUTO com prioridade SIDE

A confirmação do comando XP 0/OK ou 1/NG em Produção agora cria também
um **registro operacional durável**, com categoria AOI, hashes de pixels
dos recortes gabarito/teste **da SIDE**, event_id, razão da política,
veredito e quantidade de iluminações originalmente coletadas.
Banco SQLite independente: `data/production_auto_side/decisions.sqlite3`.
O commit acontece **após** a confirmação de transmissão ao XP.
Falha, descarte sem 0/1 ou comando pendente não geram registro.
Cada evento é salvo uma vez. Uma decisão baseada na própria memória
AUTO não é novamente aprendida (impede reforço circular).

Em nova ocorrência, o ODIN consulta a memória **após** a memória humana
exata/qualificada e **antes** de pedir TOP/MID. A consulta exige mesma
categoria e **os dois recortes AOI pixel-idênticos** (SHA-256 exato),
não aceita similaridade 97/98% nem nomes/IDs de evento como identidade.
Se houver registro coerente, encerra pela SIDE e utiliza a rolagem
normal; o gate de Produção relê SQLite, reconfere consistência, veto
físico e correções humanas antes do envio:
- 1/NG pode ser reexecutado como rejeição operacional, **mesmo quando
  a origem era NG preventivo/inconclusivo**: nunca declarar que isso
  confirmou um defeito físico.
- 0/OK só pode ser reutilizado quando a SIDE **atual também julga OK**,
  a SIDE de origem não era NG e a sonda física atual retorna
  `NO_STRONG_EVIDENCE`. Se a SIDE indica NG, geometria inconclusiva,
  indisponibilidade da sonda ou evidência física de ausência: TOP/MID
  continuam necessários e não se infere OK com base no histórico.
- Qualquer correção humana para o mesmo par ou frame exato prevalece.
  Registro físico NG contraditório bloqueia auto-OK. Julgamentos
  AUTO opostos para o mesmo par deixam a memória AUTO em conflito
  até revisão humana. Não há atalho se a fonte SQLite estiver
  indisponível/adulterada; uma falha durante a revalidação interrompe
  o XP sem inventar decisão.

**Proveniência:** `PRODUCTION_AUTO` não é
`OPERATOR_CONFIRMED`. Essa memória fornece o mesmo
benefício imediato de relembrar um caso **idêntico**, mas **não**
transforma automaticamente uma previsão em rótulo humano, treino
supervisionado KNN/CNN nem retreinamento de pesos. O log não verificado
e a quarentena `reports/production_auto_unverified/`
permanecem; as novas rotas `production_auto_side_recall_ok_v1`
e `production_auto_side_recall_ng_v1` entram nessa quarentena.
Modo Teste/Sombra, julgamentos humanos, pausa por Space, decisão única
e ACK XP continuam iguais. A performance depende de repetição **exata**
do par; se houve mudança de pixels, a aquisição normal segue.

**Validação física obrigatória:** antes de habilitar em linha, XP
isolado, replay OK e NG não vistos, NG visualmente semelhante a OK,
erros de comunicação, novo julgamento humano corretivo, alteração de
um pixel e falha da sonda física. Não presumir que um teste in-sample
comprove segurança operacional.


## 04/11/2026 — Resultado visível por 1.000 ms; sem rolagem na Produção

**Regra vigente, que substitui a rolagem e os atrasos históricos
das seções anteriores.**

Quando a inspeção produtiva conclui uma decisão visual SIDE ou
SIDE/TOP/MID e o resultado final é renderizado, o controlador:

1. atualiza os painéis e o aviso flutuante do veredito
   `FALHA FALSA` ou `DEFEITO REAL`;
2. sem alterar a barra vertical nem horizontal, nem voltar ao topo,
   agenda somente **1.000 ms** de apresentação com `QTimer`;
3. após o segundo, revalida `production_decision_policy()`, a
   identificação do evento/captura, a pausa por espaço, armamentos
   supervisionados e transmissão única ao XP; se aprovada, envia
   **0/OK ou 1/NG** no próximo ciclo Qt;
4. diante de ausência de autorização, falha ou evento inválido,
   conserva as travas/parada operacional existentes, sem forçar 0/1.

O temporizador é pausável: `Espaço` interrompe a contagem, retomar
conserva o tempo restante. `Espaço + 0/1` corrige a captura pausada
antes de qualquer transmissão automática; como a janela é de apenas
1 segundo, ela **não oferece o tempo de revisão da antiga rolagem**.
O comando já transmitido não pode ser desfeito pela pausa.
Troca de modo, substituição de evento ou descarte cancelam comandos
pendentes e callbacks atrasados. Não existe `sleep`, animação ou
espera de rolagem após a análise.

**Escopo:** somente a apresentação pós-análise no `Modo Produção`.
O módulo do agente XP permanece intacto, assim como
SIDE/TOP/MID durante a **aquisição**, fusão, validação física,
memória automática e humana, CNN/KNN, proveniência, métricas,
quarentena, decisão binária e confirmação de envio XP. O tempo de
análise já calculado segue separado do segundo de apresentação.

**Validação operacional:** testar primeiro sem acionamento físico,
com XP isolado. Confirmar OK e NG, pausa e correção humana durante o
segundo, troca de evento e modo, ausência de rolagem, nenhuma liberação
com captura incompleta, falha de ACK e **apenas um PRESS_0/PRESS_1**
por evento. A janela de 1 segundo diminui significativamente a
oportunidade de intervenção.

## 04/11/2026 — Nova decisão em reinspeção XP solicitada

A operação `Reanalisar XP` substitui a antiga exceção diagnóstica
somente leitura. Quando o XP retransmite a mesma anomalia após um 0/1,
o ODIN cria novo evento, executa a inspeção normal, apresenta o veredito
por 1 segundo e permite outro 0/1 mediante os mesmos gates de modo,
pausa, identidade, validação e sucesso de envio TCP. A nova decisão
conta como julgamento independente; callbacks de evento anterior
continuam incapazes de transmitir novamente.

Se o operador optar por Teste e corrigir para 0/OK, essa confirmação
humana entra imediatamente na memória KNN e prevalece contra a decisão
AUTO anterior nas correspondências validadas. O dataset grava outro
registro humano; CNN online elegível recebe novo caso supervisionado.
A decisão AUTO original continua no banco operacional SQLite e não é
reclassificada retroativamente como verdade humana. Não existe
aprendizado de rótulo na reanálise descartada ou no envio manual falho.

**Segurança:** a repetição exige novo frame XP SIDE estável. Não é
replay de arquivo e assinatura visual aproximada não prova que a
mesma peça física esteja parada. Validar com acionamento físico
isolado antes de permitir um segundo 0/1 na máquina industrial.

## 06/11/2026 — Memória humana SIDE validada prevalece sobre bloqueio genérico V3

No Modo Produção binário, a CNN V3 época 5 continua sem autoridade
para autorizar 0/OK isoladamente. Entretanto, quando a SIDE já terminou
com memória humana corroborada, sonda física verificada e contrato
`memory_first_side_decision()` revalidado, o gate não bloqueia o
comando 0 apenas porque `cnn_v3_primary=True`.

A prioridade é: integridade da V3, veto NG verificado da V3, depois
a autorização independente de memória SIDE para OK/NG, preservando
o restante das regras binárias existentes. Falha de SHA/proveniência,
evidência adulterada ou resultado incompleto continuam bloqueando XP.
Não se converte uma simples mensagem visual ou score V3 em permissão
de hardware.

O controlador já aguarda 1.000 ms após pintar o veredito, revalida
o gate imediatamente antes de emitir 0/1, respeita a barra de espaço,
a identidade da peça, o envio único e o ACK do XP. O ajuste não muda
a duração, o agente Windows XP, os pesos CNN nem a memória.

## 06/11/2026 — Diferenciar captura inválida de falta de autorização V3

No evento XP `8543febfba2f463a816eb412a7399e63`, o diagnóstico
encerrou SIDE com `FALHA FALSA` e imagem válida, mas a V3 época 5
não homologada não possuía autorização independente para `0/XP`.
A política retornou `cnn_v3_primary_ok_requires_independent_confirmation`
e o controller apresentou incorretamente `captura ou rota incompleta`.

Agora uma exigência explícita de confirmação humana para um veredito
visual concluído é mantida como `operator_review`, sem falsear uma
falha técnica. O comando XP permanece bloqueado até julgamento humano
desse evento. SHA inválido, captura técnica incompleta e problemas
de transporte continuam como parada operacional, sem transmitir
comando conjectural. O tempo de apresentação de 1.000 ms e o atalho
Espaço são mantidos.
