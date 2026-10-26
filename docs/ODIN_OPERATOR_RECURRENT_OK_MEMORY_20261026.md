# ODIN — memória OK recorrente validada (26/10/2026)

## O problema

O evento `c69b71764ba34ab6add7be8fde7a5dae` (14:56:58, FALTANDO)
é **falha falsa/peça OK**, confirmado novamente pelo operador.
A correção de 3 luzes já funcionou e apresentou REVISÃO OBRIGATÓRIA,
mas não reconheceu a memória recorrente como OK.

Na SIDE, o banco KNN C recuperou em primeiro lugar exatamente a
**mesma origem histórica** nos dois descritores (não a mesma captura):
- `public/ok_archive/2026-10-21_1412_FALTANDO.png`
- `pixel_sha256=c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9`
- similaridade visual A=0,99325113; geometria B=0,97756581.
- NG mais próximo da SIDE muito inferior: A=0,85977105; B=0,82187452.

Independente disso, KNN visual preferiu OK nas três iluminações
(SIDE 0,995819/0,950001; TOP 0,951006/0,938456;
MID 0,954742/0,922532), sem hard-missing físico.
CNN trial permaneceu NG ~99,99% (score não calibrado), mas
seu consenso válido foi `REVIEW/INVALID_LIGHT_EVIDENCE`.
A memória exata do mesmo **par** não foi encontrada: `NOT_FOUND`.

## Implementação

`src/core/multilight_fusion.py` preserva os cinco vizinhos A/B por luz.

`src/core/knn_visual_corroboration.py` adiciona uma referência OK
**explicitamente confirmada pelo operador** com identidade de pixels
do arquivo do acervo e verifica:

1. Evidências multilight OK em três luzes pelas duas KNN, com as
   três margens OK > NG e nenhuma indicação física de ausência;
2. Identidade de fonte OK fixa, com mesmo SHA de pixels, como melhor
   vizinho em A e B da SIDE, com similaridade A >= 0,99, B >= 0,975;
3. Sem vizinho NG com similaridade >=0,90 entre os cinco primeiros
   de cada descritor; preserva caso NG ambíguo para revisão;
4. Banco READY e sugestões/regras de revisão íntegros;
5. A saída é **FALHA FALSA** com `production_review_required=True`,
   `eligible_for_final_decision=False` e XP automático desautorizado.

`src/ui/production_confidence_gate.py` confirma a identidade e
exibe FALHA FALSA ao operador, mantendo `auto_allowed=False`,
`proposed_decision=""` e 0/1 manual. O operador decide
0 para aprovar ou 1 para rejeitar. Não gera novos pesos ou
salvamentos e não promove todo o `ok_archive` a rótulo humano.

O registro de referência é restrito a esse caso explicitamente
confirmado. Novos padrões recorrentes precisam ser qualificados
por humanos com identificação de fonte e SHA; não serão aceitos
automaticamente por estarem no diretório OK.

## Execução

Fechar ODIN, `git pull origin central`, iniciar novamente e
reproduzir a peça sob supervisão. Esperado: resultado apresentado
**FALHA FALSA**, mas intervenção humana obrigatória para envio XP.
Para permitir 0 automático no futuro, será necessário validar
negativos difíceis de mesmo componente e registrar um contrato
específico, não simplesmente ampliar limiares de similaridade.
