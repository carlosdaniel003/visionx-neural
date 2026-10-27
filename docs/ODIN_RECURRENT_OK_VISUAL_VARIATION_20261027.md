# ODIN — 27/10/2026: OK recorrente com variação visual

Evento confirmado pelo operador: `c3af7000a5b147bca4c03a695abc6675`.
A estação produziu `REVISÃO OBRIGATÓRIA` mesmo com componente presente.
Não foi erro de renderização: a regra da fusão realmente não alcançou
a rota `FALHA FALSA`.

## Por que a implementação anterior falhou

A rota anterior exigia que o arquivo OK confirmado fosse o vizinho
**número 1**, com similaridades A >= 0,99, B >= 0,975. O novo evento,
sob iluminação/posicionamento diferentes, encontrou o mesmo SHA de OK
no **2º vizinho A** (0,88399081) e no **3º vizinho B** (0,82105397).
Nas listas SIDE os vizinhos NG ficaram pouco à frente:
A NG=0,88679835, B NG=0,83597105.

Nas três luzes, uma *outra* comparação independente de assinaturas
favoreceu OK:
SIDE 0,953651 OK vs 0,948180 NG;
TOP 0,953651 OK vs 0,942180 NG;
MID 0,952869 OK vs 0,927543 NG.
CNN retornou `REVIEW/INVALID_LIGHT_EVIDENCE`; `physical_score=0`;
KNN C A/B votaram OK nas três luzes, mas seguem em REVISAO.
`recognition_match=NOT_FOUND` continua correto: par exato não foi encontrado.

## Correção específica

Uma segunda modalidade de reconhecimento compara a fonte SHA-pinned
**já confirmada pelo operador**:
`public/ok_archive/2026-10-21_1412_FALTANDO.png`
(hash de pixels `c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9`).

Condições da rota:
- KNN C READY e 2 votos A/B OK (ainda diagnóstico) em **SIDE, TOP, MID**;
- comparação de assinaturas nas três luzes: vizinho OK >= 0,95
  e margem NG-OK <= -0,005 em cada luz, média de vantagens >= 0,012;
- a fonte OK SHA-confirmada aparece entre as **3 primeiras posições
  de A e B na SIDE**, similaridade A >= 0,88, B >= 0,82;
- vizinhos NG SIDE < 0,90 e não superam a fonte OK em mais de 0,02;
- nenhum indício físico corroborado de componente ausente e
  consenso CNN inconclusivo.

Se essas condições coexistirem, o resultado do sistema **para
conferência humana** é `FALHA FALSA`, com regra
`operator_confirmed_ok_pattern_three_light_review_v1`.
Uma segunda verificação no gate de produção protege a apresentação
contra alterações do registro da fonte e votos.

**Não autorizar aprovação automática**: `auto_allowed=False`,
`proposed_decision=""`, `operator_review_required=True`.
A legenda deve mostrar **FALHA FALSA** e **CONFIRMAÇÃO HUMANA NECESSÁRIA**.
O operador deve pressionar **0** se confirmar OK ou **1** caso NG.
A origem OK tem rótulo confirmado, a captura atual só tem similaridade
e não ganhou classificação física humana pelo algoritmo.

As regras antigas (par forte, revisão multilight, NG físico, outras
categorias) permanecem intactas. Sem os sinais de contraevidência
ou com vizinhos NG fortes, não usar esta exceção.

Implementação:
`src/core/knn_visual_corroboration.py`,
`src/ui/production_confidence_gate.py` e
`src/ui/decision_verdict_feedback.py`.
Teste de regressão:
`tests/test_knn_three_light_ok_counterevidence.py`
reproduz os valores do log e verifica o bloqueio XP.
