# ODIN — 28/10/2026 — memória conhecida na SIDE sem TOP/MID (revisão)

## Evento

`809af820e68a4ec28714bd9e7bed9f58` em Modo Produção: o
julgamento final, agora correto, foi **FALHA FALSA**, mas o tempo do
ciclo ficou em 15,7358s porque precisou SIDE/TOP/MID.

A SIDE havia julgado **FALHA FALSA**, com score NG CNN experimental
`0,034367572516`, equivalente a confiança local `0,9656324`.
Isso fica abaixo do limiar de encerramento antecipado CNN de 0,97.
A memória estritamente exata encontrou `NOT_FOUND`, pois não há prova
de que o par gabarito/teste tenha pixels idênticos a uma captura humana
anterior. Portanto, a rota automática de memória exata não foi acionada.

A KNN C READY na SIDE trouxe como vizinho nº 1 dos descritores A/B
`public/ok_archive/2026-10-21_1416_FALTANDO.png` com semelhanças
`0,86618201` e `0,83100362`, respectivamente. A identidade e o
rótulo físico OK dessa **fonte histórica** já são confirmados pelo
operador e protegidos por SHA de arquivo/pixels. A sugestão A ainda
votou NG, B OK e ambas exigiram revisão. A busca de assinaturas
independente da SIDE teve melhor NG `0,988808` contra OK `0,977865`.
**Não existe certificação de que toda nova captura semelhante seja
OK; vizinhos NG próximos impedem repetir automaticamente o rótulo.**

## Fluxo adicionado

`src/core/side_recurrent_ok_review.py` qualifica **apenas** a fonte
histórica 14:16 e somente no Modo Produção, quando a SIDE traz:
- CNN FALTANDO V2 com checkpoint correto e score NG <=0,04;
- arquivo canônico OK válido com prova humana/SHAs, ausência de origem
  NG aposentada;
- KNN C READY e primeiro vizinho A/B da SIDE = arquivo canônico OK
  com limiares 0,85 e 0,82, margem >=0,04 contra NG nos cinco
  primeiros e nenhum NG >=0,90;
- ausência de conflito na memória e de evidência física explícita NG;
- sonda de ausência física independente executada no **par SIDE atual**
  em **duas escalas**, sem suspeita de ausência, sem erro de captura,
  admitindo aparagem geométrica apenas para **revisão humana**.

Se todos os filtros passarem, o ODIN deixa de pedir TOP/MID e apresenta
`FALHA FALSA` como **hipótese de memória OK recorrente**, com o operador
obrigatoriamente escolhendo **0=OK** ou **1=NG**. A regra
`side_verified_archive_visual_review_v1` deve indicar as luzes
dispensadas. A política de produção valida novamente a fonte antes
de publicar a recomendação. **Não há envio automático de 0 ou 1.**
Se qualquer prova falhar, o ciclo anterior de três iluminações
permanece inalterado.

Isso permite economizar tempo de aquisição TOP/MID em uma ocorrência
historicamente reconhecível, mas ainda não implementa a solicitação
de repetição autônoma do julgamento por imagem aproximada. Para tanto,
é preciso validar com pares físicos OK/NG inéditos e muito semelhantes,
inclusive contraprovas de componentes ausentes, antes de aceitar a
equivalência visual como identidade de inspeção.

## Arquivos e testes

- `src/core/side_recurrent_ok_review.py`
- `src/ui/adhesive_multilight_inspection.py`
- `src/ui/production_confidence_gate.py`
- `tests/test_side_recurrent_ok_review.py`

A nova rota não modifica pesos da CNN, rótulos do acervo,
regras de memória exata, automação geral de três luzes,
outros tipos de defeito ou protocolo XP.
Testar com saída do XP isolada e conferir os logs da SIDE antes
de qualquer uso em produção real.
