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

**Atualização de 30/10/2026 — extensão para todas as categorias oficiais:**
a regra foi ampliada para `FALTANDO`, `DESLOCADO`, `INVERTIDO`,
`EMBORCADO` e `MUITO ADESIVO` (canônica `MUITOADESIVO`).
Aliases OCR normalizados usam o mesmo escopo de memória; categoria
desconhecida/conflitante, memória ausente, KNN C LOADING e divergência
OK/NG continuam em fallback, sem 0/OK automático.

**Limite do veto físico:** `physical_absence_probe` continua sendo
somente uma verificação independente de **ausência de estrutura esperada**
em duas escalas. Ela pode barrar uma aprovação em qualquer categoria,
mas **não verifica autonomamente inversão, tombstone, deslocamento,
polaridade ou excesso de adesivo**. A correspondência humana + KNN A/B
e margem de 4pp é a base da precedência visual nessas categorias,
com risco residual de defeitos fisicamente semelhantes. A telemetria
traz `physical_probe_scope=CROSS_CATEGORY_ABSENCE_VETO_ONLY`.
O código não cria sonda física universal nem aumenta artificialmente
a confiança. Necessário validar com NGs difíceis de cada categoria
e saída XP isolada antes de liberar produção.

A categoria AOI continua sendo apenas escopo de busca auditável,
não verdade física do defeito. CONFLICT / NOT_READY / falha de consulta
nunca viram reconhecimento positivo.

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


## 30/10/2026 — KNN C com alinhamento de memória e novo limiar 97%

O limiar de autorização dos descritores originais **A e B** foi elevado
de `0.96` para **`0.97`** (`side_operator_memory_first_97pct_v2`).
O score da memória humana continua com mínimo `0.975`, margem >=`0.04`.
Elevar o limiar isoladamente NÃO resolve a captura de 08:47, pois a KNN C
encontrou A/B na faixa 0.80–0.84, enquanto a memória humana encontrou
OK 0.9774185 / NG 0.93243586.

### Correção da divergência de representação, sem manipular porcentagens

As medidas anteriores não representam o mesmo fenômeno:
- **KNN C A/B:** descritores de foco/forma e geometria extraídos do PNG AOI;
- **Memória humana:** vetor persistido `anomaly_signature` de diferenças
  entre gabarito/teste (JSON `memory.json`).

`src/core/knn_c_archive_alignment.py` agora cria uma pista adicional
**na mesma métrica da memória humana**: percorre os registros persistentes
validados de origem `ARCHIVE_PNG`, organiza vetores por categoria,
associa cada hipótese humana a sua fonte `OPERATOR_CONFIRMED` e compara:
1. captura atual versus assinatura de um registro de archive;
2. assinatura humana selecionada versus a mesma assinatura do archive;
3. melhor fonte da classe contrária versus fonte preferida.

Cada correspondência deve ser >=**97%** nos dois vínculos e conservar
margem >=**4pp** sobre o archive da classe oposta. Trata-se de
**semelhança de assinatura, não de par RGB exato**.
O índice é atualizado por geração do snapshot persistente, não lê
cada PNG a cada inspeção e não reatribui rótulos por algoritmo.

O resultado é exibido em `detail.knn_c.aligned_archive`:
`RECONCILED_OK`, `RECONCILED_NG`, `AMBIGUOUS`, `NO_MATCH`
ou `UNAVAILABLE`, com IDs de registro, caminho da fonte,
SHA de proveniência, scores comparáveis e origem da hipótese humana.

### Encerramento SIDE

A rota original continua disponível quando A/B >=97% concordam
**no mesmo PNG/SHA**. Caso contrário, a nova pista alinhada pode
corroborar o julgamento humano previamente confirmado, mas só se
a KNN C estiver `READY` e nenhum dos cinco vizinhos A/B de classe
contrária tiver score >=90%. A contraprova física independente,
revalidação na política XP, janela de pausa e não-aprendizado de
decisões automáticas permanecem ativos.

A pista alinhada não fabrica `source_pixel_sha256` nem substitui
os números A/B exibidos, que podem continuar em 80–84%.
Uma memória humana forte sem um registro de archive correspondente
continua insuficiente para `0/OK` automático. Se os vetores persistidos
do arquivo não confirmarem o humano, o sistema segue para CNN/fusão.

**Limite técnico:** a escala compacta de `anomaly_signature` pode
confundir diferenças físicas distintas com mapas parecidos. Todos os
testes de CI trabalham com pares sintéticos e regras de regressão;
**não há comprovação de que o evento 08:47 tenha atingido 97% nessa
pista adicional sem os PNGs e o JSON locais reais**. Validar com
XP isolado e NG reais difíceis das cinco categorias antes da linha.
