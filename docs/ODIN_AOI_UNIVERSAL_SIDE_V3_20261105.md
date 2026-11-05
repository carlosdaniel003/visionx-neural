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
