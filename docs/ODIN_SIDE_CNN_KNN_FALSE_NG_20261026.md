# ODIN — NG indevido SIDE com KNN C em desacordo (26/10/2026)

Evento real: `e70363365eaa4681b822b95557977e63`, FALTANDO,
SIDE, 14:30:39 em 26/10/2026. Operador classificou a peça **OK**
apesar do julgamento visual `DEFEITO REAL`.

A captura mostra o componente e terminais presentes no par da AOI,
com diferenças de iluminação e posição. Log:
- CNN FALTANDO V2 **ativa de produção/supervisão**, SHA `c13ca329...`,
  score NG não calibrado `0.999189555644989`.
- KNN C READY, votos A/B `OK`, mas sugestões A/B `REVISAO`;
  `status=REVISAO`, `reason=SEM_SUGESTAO_CONFIAVEL`.
- Evidência física `hard_missing_evidence=false`; o score físico
  não constitui prova independente de componente faltando.
- Política antiga `side_trial_ng_human_review_v1` encerrava SIDE só
  pelo alto score CNN e dispensava TOP/MID.

## Mudança na proteção do julgamento

`src/core/side_early_decision.py` agora impede SOMENTE o
encerramento antecipado da CNN experimental em `FALTANDO`
quando coexistem:
- NG da CNN, sem evidência física forte de ausência;
- diagnóstico KNN C READY, esquema e integridade conhecidos,
  somente diagnóstico e nenhuma autorização automática;
- A e B votam OK, mas permanecem em revisão, sem sugestão confiável.

Nesse conflito, o ODIN deve concluir TOP e MID antes do único
julgamento multilight. **Não muda o resultado da CNN para OK,
não libera 0/XP e não reduz o limiar para detecção de NG**.
Se as análises adicionais também concluírem NG, continua em revisão
humana; não há promessa de que toda falsa reprovação seja corrigida.

Ficam intactos o caso de ausência física corroborada, NG legítimos
sem dois votos KNN OK e todos os outros tipos AOI.

Para aplicar na estação: `git pull origin central`, reiniciar ODIN,
e observar a próxima ocorrência. Antes do pull a alteração não atua
em produção. Testes de regressão do gate cobrem a combinação real,
complementação de luzes e proteção contra decisões XP prematuras.

Os checkpoints offline 24/24 continuam separados e **não são**
os pesos carregados pelo julgamento de produção deste evento.
