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
