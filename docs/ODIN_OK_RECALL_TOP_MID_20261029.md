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
