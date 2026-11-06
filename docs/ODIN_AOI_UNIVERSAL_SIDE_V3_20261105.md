## Atualização operacional de 06/11/2026 — Produção supervisionada

As restrições de autorização XP descritas abaixo referem-se à
conclusão **visual** isolada da V3. A rota SIDE V3 época 5
**corroborada** passou a poder emitir 0/OK ou 1/NG em Modo Produção
binário supervisionado, com revalidação do mesmo contrato visual
(`_eligible_v3_side_for_review`), apresentação de 1.000 ms,
pausa Espaço e controle de evento/ACK. Esse direito não se estende
a V3 isolada, captura incompleta, prova adulterada, nem Teste/Sombra.
O modelo segue experimental, sem estimativa de precisão 100%
comprovada em holdout físico cego.

# ODIN AOI — V3 universal e prioridade SIDE (05/11/2026)

A CNN V3, quando ativada por checkpoint SHA verificado, classifica todos
os pares de pixels sem escolher o motor por categoria. UNKNOWN e
MUITO ADESIVO percorrem a mesma inspeção; SIDE é examinada primeiro.

## Encerramento visual SIDE em todos os modos

V3 época 5 íntegra com score direcional >=97% e KNN C A/B concordante
no top-5 pode encerrar a aquisição sem TOP/MID em Teste, Sombra ou
Produção. O marcador é apenas classificação visual; a similaridade de
vizinhos não prova identidade nem autoriza 0/OK no XP.

Se a inspeção chegar às três luzes e a V3 verificar OK nas três,
com mesma época, SHA e ausência de contradição física/memória,
o veredito é FALHA FALSA. A exigência de confirmação, se houver,
é preservada em campos próprios, sem converter o score em 0,5.

A categoria lida por OCR pode permanecer nos logs históricos, mas
não determina mais o motor nem a fusão. Os pesos CNN não foram
modificados. Como a época 5 não passou em teste físico cego,
a política XP mantém a trava de liberação de OK baseada só na V3.

## Acervo sem categoria

Novos arquivos OK/NG usam nomes neutros `AAAA-MM-DD_HHMM_AOI_SIDE.png`
(ou TOP/MID), com leitura compatível de PNGs históricos. Capturas UNKNOWN
ou sem OCR podem ser arquivadas se o frame do evento existir. O importador
KNN valida PNG, pixels e pasta OK/NG sem rejeitar divergências de OCR.

SIDE NG tem a mesma proteção contra contraprovas humanas OK aproximadas ou
KNN A/B contraditórias, independentemente da categoria. A fusão V3 OK
unânime só preserva FALHA FALSA na ausência de evidência física ou conflito.

## 06/11/2026 — revisão diagnóstica não é revisão obrigatória

A estratégia C atua como observador. Quando CNN e KNN C A/B concordam,
nenhuma das duas ganha autoridade XP por isso; a concordância não cria
`operator_review_required`. Somente contradições reais entre o resultado
visual e a KNN C (ou vetos independentes) exigem investigação.

O atalho SIDE V3 época 5 exige o mesmo contrato SHA, scores e concordância
A/B com cinco vizinhos da mesma classe; não usa o nome da categoria nem
IDs específicos. Quando o veredito é FALHA FALSA e não há conflito, os
campos visual `production_review_required`, `operator_review_required`
e `decision_trace.operator_review_required` são `false`. A política XP,
**independente do veredito visual**, continua sem transmitir `0/OK` apenas
com CNN V3 e KNN C. Regressões cobrem Teste, Sombra, Produção e categoria
UNKNOWN, além de divergências e tentativa de bypass do gate.

## 06/11/2026 — SIDE OK com KNN C não seletiva e memória humana

A KNN C pode retornar `REVISAO / SEM_SUGESTAO_CONFIAVEL` porque
as distâncias A/B não atingiram os critérios próprios de sugestão, mesmo
quando seus cinco vizinhos por descritor pertencem a OK. A CNN V3 época 5
pode apresentar confiança OK >=97% e a memória persistente humana recuperar
`HUMAN_OK_VISUAL_CANDIDATE` da SIDE, com assinatura >=97%, vantagem >=4pp
sobre NG, rótulo `OPERATOR_CONFIRMED` e proveniência verificável.

O encerramento visual SIDE agora aceita essa **corroboração conjunta**,
independentemente da categoria física, sem reclassificar a KNN C de
`REVISAO` para `SUGERE_OK`. Mantém o contrato original de epoch/SHA,
confiança direcional, ausência de veto físico/conflito e cinco vizinhos
OK com SHA de fonte em ambos descritores. O marcador distingue
`HUMAN_OK_KNN_C_A_B_TOP5_UNANIMOUS` da via seletiva antiga.

Se faltar memória humana qualificada, a KNN divergir, a CNN for NG,
a fonte estiver inconsistente ou o veto físico existir, TOP/MID são
coletadas normalmente. O atalho só reduz a aquisição e apresenta
`FALHA FALSA`, **sem conceder comando 0/OK ao XP**. A similaridade
humana não é identidade exata e o score CNN não é probabilidade calibrada.

Regressões em `tests/test_side_early_decision.py` e
`tests/test_adhesive_multilight_inspection.py` verificam os três modos,
categorias diferentes, bloqueios de NG e falhas de proveniência.

## 06/11/2026 — KNN C não seletiva, A/B com vizinho NG no top-5

A KNN C é observadora. A exigência anterior de cinco vizinhos
unânimes podia forçar TOP/MID quando os votos A/B eram ambos OK, mas
um vizinho NG distante aparecia entre os cinco. Para CNN V3 época 5
com checkpoint validado, classificação **OK na SIDE >=97%**, sem veto
físico, memória conflitante ou revisão operacional, existe uma rota
visual adicional **não seletiva**:

- KNN C READY, diagnóstico `REVISAO / SEM_SUGESTAO_CONFIAVEL`, votos
  A e B OK, motivo de insuficiência de similaridade e sem divergência;
- o vizinho mais próximo em **ambos** os descritores é OK, com todas as
  cinco fontes de cada descritor presentes, ordenadas e com SHA válido;
- memória humana pesquisada com `NO_MATCH`, sem NG humano mais próximo
  que OK; não promove memória aproximada a reconhecimento exato;
- sem provas físicas ou conflitos, o fluxo encerra a inspeção **visual**
  como FALHA FALSA na SIDE e registra
  `CNN_V3_KNN_C_A_B_OK_NONSELECTIVE` nos diagnósticos.

Nenhum limiar de similaridade KNN foi reduzido, nem a recomendação
`REVISAO` foi falsificada. O desempate usa a direção diagnóstica do
modelo principal e o par de votos A/B, sem identidade do evento, categoria
ou arquivo específico. Se o voto A/B divergir, o primeiro vizinho for
NG, a memória humana for NG/ambígua ou faltar proveniência, TOP/MID
continuam. **Não autoriza 0/OK automático ao XP**: essa rota é uma
classificação visual e os escores de CNN não estão calibrados. Esta
política pode perder defeitos só visíveis em TOP/MID; ainda requer
replay e inspeção física independentes antes de autorizar produção.

Testes para categorias variadas, três modos e contraprovas ficam em
`tests/test_side_early_decision.py` e
`tests/test_adhesive_multilight_inspection.py`.

## 06/11/2026 — SIDE com memória humana OK qualificada e KNN C mista

Uma SIDE pode receber voto OK da V3 época 5 com confiança direcional
**abaixo de 97%**, enquanto a memória persistente confirma uma
correspondência *visual aproximada* a um OK humano qualificado
(`HUMAN_OK_VISUAL_CANDIDATE`) e os votos KNN A/B apontam OK.
O top-5 de A/B pode conter NG distantes e a KNN sinaliza
`REVISAO / SEM_SUGESTAO_CONFIAVEL / VIZINHOS_VISUAIS_DIVERGENTES`.

Há uma alternativa **estritamente visual** à confiança CNN >=97%:
exige checkpoint V3 época 5 verificado, decisão CNN OK, memória humana
qualificada (OK >=97%, vantagem >=4 pontos sobre NG, origem
`OPERATOR_CONFIRMED` na SIDE e categoria correspondente), KNN C READY
com ambos os votos OK, vizinho top-1 OK em A/B superior ao NG mais
próximo por pelo menos a margem já definida de 4 pontos, e cinco
fontes ordenadas, identificadas por SHA em cada descritor.
Não se altera o limiar de 97% da rota usual, nem os limiares KNN,
nem a saída original `REVISAO` da estratégia C.

Se qualquer fonte for inválida, houver KNN NG concorrente próxima,
memória humana ambígua ou NG, veto físico ou score CNN contraditório,
a aquisição continua para TOP/MID. A marca
`HUMAN_OK_KNN_C_A_B_NEAREST_OK_MIXED` explica por que o atalho foi
usado apesar do score CNN inferior a 97%; o campo
`cnn_97pct_threshold_met` torna a decisão auditável.

Esta conclusão **não autoriza 0/OK automático ao XP** nem prova
identidade RGB exata. A classificação da CNN, os atributos de
similaridade e os julgamentos humanos continuam distintos.
A liberação automática requer validação física independente com
eventos NG difíceis e casos OK, inclusive NG somente visíveis em TOP/MID.

## 06/11/2026 — SIDE NG corroborada por KNN C não seletiva e humano/acervo

Mesmo quando a V3 época 5 verificada marca DEFEITO REAL na SIDE com
score NG >=97%, a KNN C pode retornar `REVISAO` por
`SIMILARIDADE_VISUAL_INSUFICIENTE` nos descritores A/B. Os votos
A/B podem continuar NG; o top-5 de B pode conter OK em posições
distantes, sem contradição real.

A rota visual `HUMAN_NG_KNN_C_A_B_ARCHIVE_RECONCILED` encerra SIDE
somente quando TODAS as evidências coincidem:
- voto CNN V3 época 5 NG validado por SHA, confiança direcional >=97%;
- memória persistente `HUMAN_NG_VISUAL_CANDIDATE` da mesma SIDE,
  `OPERATOR_CONFIRMED`, similaridade NG >=97%, margem >=4pp contra OK;
- KNN C READY não seletiva por distância, ambos os votos A/B NG, com
  top-3 NG em cada descritor e top-5 ordenado com SHA íntegro;
- `RECONCILED_NG` no arquivo persistente com correspondência do
  record_id ao operador NG, mesma categoria/luz, similaridade
  arquivo NG >=97%, margem >=4pp contra OK e proveniência verificável;
- sem conflito de memória, revisão operacional, veto de captura/físico
  ou desacordo entre CNN e KNN.

Uma referência OK distante na cauda do top-5 não impede a corroboração.
A KNN C mantém o status diagnóstico original `REVISAO` e a
comparação por assinatura **não** vira correspondência RGB exata.
Nenhum limiar KNN é reduzido, nenhum arquivo/evento é tratado por nome
e nenhuma regra é condicionada à categoria.

A conclusão `DEFEITO REAL` dispensa TOP/MID em Teste, Sombra e
Produção como classificação visual, mas `production_auto_allowed=false`
e o gate XP **não envia 1/NG automaticamente apenas por essa rota**.
Requer validação independente de falsos NG antes de autorização
operacional. Os testes também garantem que memória OK concorrente,
proveniência alterada e votos discordantes mantenham TOP/MID.

## 06/11/2026 — SIDE NG sem match humano qualificado, top-5 A/B NG

A V3 época 5 pode identificar NG com score direcional >=97% em uma
SIDE inédita, enquanto a KNN C retorna `REVISAO` unicamente por
distância insuficiente. A memória humana confirma que a consulta
foi realizada, mas pode retornar `NO_MATCH` (sem assinatura >=97%).
Isso difere da rota `HUMAN_NG_KNN_C_A_B_ARCHIVE_RECONCILED`, que
continua exigindo uma âncora humana e de acervo qualificada.

A rota visual `CNN_V3_KNN_C_A_B_NG_NONSELECTIVE` se aplica sem depender
da categoria AOI nem de imagens nomeadas:
- modelo V3 época 5 com voto e checkpoint SHA íntegros, SIDE NG >=97%;
- KNN C READY, `REVISAO / SEM_SUGESTAO_CONFIAVEL`, `A/B=NG`,
  apenas motivo de similaridade insuficiente;
- cinco vizinhos NG em cada descritor A/B, ordenados, distintos por SHA,
  com >=3 referências SHA idênticas entre ambos os top-5;
- memória persistente consultada, `NO_MATCH` com NG humano mais
  semelhante que OK (sem inventar confirmação de par, sem converter
  proximidade em aprovação humana);
- consulta ao arquivo persistente também `NO_MATCH`, sem prova
  reconciliada oposta;
- nenhum veto físico, inconsistência de captura, conflito ou revisão
  operacional.

A KNN mantém seu veredito diagnóstico `REVISAO`. O atalho é um resultado
**visual** `DEFEITO REAL` para Teste/Sombra/Produção, encerrando apenas
a aquisição (sem TOP/MID). Como os scores da V3 não são calibrados e
os rótulos históricos são provisórios, não há autorização de `1/NG`
automático por essa rota. A validação física independente de falsos NG
continua necessária antes de qualquer liberação automática. Regressões
cobrem NG, contraprovas OK, falsificação da origem, incompletude,
limiar CNN, múltiplas categorias e todos os modos.

## 06/11/2026 — resolução única da SIDE e memória humana exata

A decisão antecipada da SIDE é selecionada **uma única vez por evento**
e reaproveitada na finalização, sem reabrir a consulta à memória entre
o veredito SIDE e o encerramento do ciclo. Teste, Sombra e Produção
podem recuperar um frame XP completo pixel-idêntico já confirmado por
humano. A prova é validada por categoria, iluminação e última decisão,
e para OK exige a sonda física prevista em seu contrato; não há
promoção de semelhança KNN a identidade exata.

As rotas de memória aproximada com contraprova física e de replay
automático preservam escopo de Produção. A CNN V3 época 5, os
limiares KNN, a política XP 0/1 e a fusão de três iluminações **não**
foram alterados. Se a SIDE não for reconhecida com prova adequada,
o fluxo ainda captura TOP/MID — inclusive quando MID detecta NG
apesar do voto SIDE OK.

O índice de PNGs históricos da consulta visual faz a busca
**diagnóstica** por iluminação e pode reconhecer o mesmo frame apesar
de mudança de categoria OCR. Isto não prova julgamento humano,
nem concede autorização XP. O teste de replay de arquivos legados
agora expressa esse contrato preexistente.
