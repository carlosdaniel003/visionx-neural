# 30/10/2026 — Prioridade da memória humana sobre CNN na SIDE

## Caso que motivou a alteração

Evento `88584d2737a84b6da0fe8f5887e2e037`, Produção, 07:58.
Julgamento físico do operador: FALHA FALSA. A CNN FALTANDO v2
classificou NG (`ng_score_uncalibrated=0.999807894`) e a memória
exata retornou `NOT_FOUND`, apesar de existir um julgamento humano
OK anterior muito semelhante.

A memória humana do snapshot `memory.json` encontrou OK = 0,99798172
e NG = 0,95079253. Os dois descritores da KNN C, que consultam os PNGs,
encontraram o MESMO `public/ok_archive/2026-10-21_1412_FALTANDO.png`:
A=0,98236720, B=0,96262484, contra os vizinhos NG A=0,86601348,
B=0,82652949. Não há prova de par RGB pixel-idêntico. Um score
visual de 96% NÃO é uma probabilidade industrial de 96% de correção.

## Nova rota de decisão

`src/core/side_memory_first.py` implementa a estratégia genérica
`side_operator_memory_first_96pct_v1`. Em Modo Produção:

1. Antes de decidir pela CNN, consultar a memória humana persistida
   que já existia, sem transformar automaticamente PNG em prova humana.
   A nova rota exige `OPERATOR_CONFIRMED`, registro íntegro e
   classificação OK ou NG com similaridade >= 0,975 e vantagem
   >=0,04 sobre a memória humana da classe oposta.
2. Consultar KNN C READY da iluminação SIDE. Os primeiros vizinhos
   de A e B devem concordar com a memória humana, apontar para o
   **mesmo PNG, classe e SHA**, e cada um possuir similaridade >=0,96.
   Nenhuma referência NG/OK oposta no top-5 pode estar a menos de
   0,04 do vizinho preferido. É uma regra por evidência, não por
   nome do componente, hora do evento ou PNG fixo.
3. Para `OK`, exigir sonda física da captura atual em duas escalas,
   sem suspeita de ausência. Para diferenças geométricas da AOI,
   permitir APENAS aparagem central sem interpolação <=3 px e <=1%
   por eixo, se a sonda registrou duas escalas sem ausência. Falha,
   sonda inconclusiva, ausência, ou distorções maiores bloqueiam
   o novo caminho e preservam o fallback existente.
4. Quando os testes passam, o novo veredito é
   `FALHA FALSA` (OK) ou `DEFEITO REAL` (NG), com
   `eligible_for_final_decision=True` e somente SIDE; TOP/MID
   dispensadas. A CNN não prevalece em evento reconhecido.
   **Sem treinamento ou troca de checkpoint da CNN.**
5. O `production_decision_policy` recalcula a qualificação a partir
   da evidência original antes de autorizar `0/OK` ou `1/NG`.
   O controlador conserva a pausa por espaço e o atraso de cinco
   segundos antes de transmitir. Decisões automáticas permanecem
   separadas da memória de julgamentos humanos e do treinamento.
6. A seção **PROVENIÊNCIA** informa a fonte decisiva, SHA do PNG,
   ID do julgamento humano, semelhanças A/B e resultado físico.

A rota é limitada à categoria de busca FALTANDO porque a contraprova
física atual só verifica **ausência**. A categoria AOI não é considerada
verdade sobre o tipo real do defeito. Para outras categorias, as
regras existentes continuam em vigor. CONFLICT / NOT_READY / falha
de consulta nunca são tratados como reconhecimento positivo.

## Testes e limitações

`tests/test_side_memory_first.py` verifica números reais do evento
fornecidos pelo log, rótulos OK/NG, margem contrária, discordância de
PNG, fonte não humana, geometria mínima/fora de alcance, ausência e
marcadores adulterados. Os testes automatizados NÃO substituem uma
avaliação com imagens reais não vistas de componentes OK e NG muito
parecidos. Os PNGs da estação não estão versionados no GitHub.

**Validar primeiro com XP isolado e sem liberação física de peças**:
conferir acertos em eventos OK recorrentes, NG fisicamente semelhantes,
desvio geométrico real, a seção PROVENIÊNCIA e a janela de pausa.
O novo roteador tem risco de aprovar NG como OK se houver parecidos
não distinguíveis pelos descritores; meta de 97% de taxa de autojulgamento
não equivale a precisão industrial medida por ground truth.
