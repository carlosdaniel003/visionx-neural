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
