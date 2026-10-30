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

## 30/10/2026 — Correção de ordem: conciliação KNN C pós-diagnóstico

O roteador de memória (`verified_memory_router.py`) é interno ao
wrapper da KNN C. Portanto, ao procurar `detail.knn_c` durante a
recuperação da memória humana, o diagnóstico ainda não havia sido
adicionado. A conciliação `aligned_archive` agora é executada
em `knn_c_live.py`, imediatamente após `bank.diagnose()`, quando
o índice está READY e existe `confirmed_visual_recall`.
A chamada anterior, prematura e redundante foi retirada.

A mudança não dispensa os limiares atuais de 97%/4pp nem a
confirmação pela fonte archive, não altera classificação CNN,
veto físico, regras XP, nem a janela de pausa por espaço.
Há regressão de integração em `test_knn_c_archive_alignment.py`
para garantir que o diagnóstico READY exponha `aligned_archive`.
O caso real 08:47 depende da reavaliação com PNGs e JSON locais.


## 30/10/2026 — Prioridade do OK humano na TOP após as três luzes

Evento real `8dee2f1d331149eeb14718ff44386db3` de 10:07:57:
a consulta de memória humana na TOP recuperou um OK confirmado com
similaridade `0.9774185`, contra NG `0.93243586`. O arquivo de
TOP associado pelo índice de assinaturas retornou `RECONCILED_OK`,
`archive_similarity=0.9774185`, `human_archive_similarity=1.0`.
O classificador local marcou SIDE e TOP como `FALHA FALSA`; MID
permaneceu em `REVISÃO OBRIGATÓRIA`. Na SIDE, a memória humana tinha
NG `0.98651746` e OK `0.97774325`, portanto status `AMBIGUOUS`.
A regra de memória anterior só executava encerramento antecipado em SIDE;
o julgamento TOP não participava da prioridade após a fusão.

A rota geral `operator_human_multilight_97pct_v1` avalia a memória
persistida `OPERATOR_CONFIRMED` nas três iluminações, sem codificar
fonte, peça ou categoria específica. Exige: categoria reconhecida;
capturas SIDE/TOP/MID completas; pelo menos duas classificações
locais `FALHA FALSA`; nenhuma classificação local `DEFEITO REAL`
ou ausência física forte; e pelo menos uma iluminação com memória humana
OK >=97% e vantagem >=4 pontos percentuais sobre NG humano
da **mesma iluminação**. Um candidato `HUMAN_NG_VISUAL_CANDIDATE`
inequívoco em qualquer iluminação bloqueia a rota. Uma memória
`AMBIGUOUS` de outra iluminação não apaga automaticamente
o OK qualificado, mas fica preservada na auditoria.

O resultado visual é `FALHA FALSA`, mesmo quando o veto físico
atual impede uma decisão automática; nesse caso, solicita operador,
**sem converter o OK reconhecido em NG PREVENTIVO**. O envio de
`0/OK` em Produção contínua exige também revalidação no gate XP
e verificação em duas escalas das capturas atuais SIDE/TOP/MID;
`SUSPECT_ABSENCE`, `UNAVAILABLE`, geometria não admitida,
captura incompleta, dados adulterados e pausa mantêm a liberação
bloqueada. A tolerância de aparagem do comparador de memória é
até 3 pixels e até 1% por eixo, sem resize. Preserva 5 segundos
para a barra de espaço e grava as decisões automáticas sem
transformá-las em julgamentos humanos de treinamento.

**Risco e limite:** a similaridade de 97% é uma métrica de assinatura,
não 97% de certeza de qualidade. A SIDE desse evento contém um NG
humano ainda mais semelhante que o OK da SIDE, embora ambíguo.
O reconhecimento TOP não prova por si só a qualidade física de uma
nova peça, nem a sonda de ausência detecta inversão/deslocamento/
adesivo. Antes de usar o comando real no XP, validar com XP isolado
e com controles NG e OK difíceis, em especial pares conflitantes
entre iluminações. Os testes usam fixtures de scores do log, não
os PNGs originais da estação.


## Contrato permanente — vizinho NG aproximado versus memória OK humana

A regra de 30/10/2026 distingue **duas fontes que não são intercambiáveis**:

1. Vizinho visual NG da `KNN C`: uma correspondência por semelhança
   (por exemplo, na SIDE do evento `8dee2f1d331149eeb14718ff44386db3`,
   `public/ng_archive/2026-10-21_1442_FALTANDO.png` com
   similaridade visual A `0.81875214`); não é prova de par exato,
   nem comprova que a peça atual contenha o mesmo defeito.
2. Memória humana `OPERATOR_CONFIRMED`: evidência de decisão anterior
   sobre outra observação, qualificada pelo score e margem na mesma luz,
   com proveniência preservada. A TOP do evento teve OK `0.9774185`,
   contra NG `0.93243586`. A SIDE registrou um NG humano
   `0.98651746` com status `AMBIGUOUS`; **esse registro JSON não
   foi identificado como o PNG NG de 21/10 às 14:42**.

**Invariante de julgamento:** um PNG NG apenas semelhante não
substitui, por si só, um OK humano historicamente confirmado e
qualificado. O índice KNN pode continuar recuperando esse PNG para
diagnóstico, mas não pode promovê-lo indevidamente a identidade exata,
rótulo humano atual ou voto NG físico definitivo.

**Invariante de segurança:** nenhuma evidência conflitante é apagada
da auditoria. Um candidato humano NG inequívoco em outra luz, uma
corroboração independente de NG, ausência física ou captura inválida
continua sujeito aos bloqueios pertinentes. Uma memória humana OK
aproximada não significa 97% de probabilidade de peça boa. Sem
contraprova física válida, pode-se exibir o reconhecimento visual,
mas não transmitir `0/OK` automático.

`tests/test_human_multilight_priority.py` protege o contrato:
(1) injetar o vizinho NG aproximado conhecido no diagnóstico KNN C
não muda o julgamento OK humano qualificado; (2) o NG e os scores
originais persistem no debug/registro; (3) corroboração NG independente
bloqueia a prioridade; (4) os testes existentes mantêm a exigência
de operador para NG humano inequívoco e para veto físico.
A regra é genérica, não depende da data ou do nome do arquivo.


## 30/10/2026 — OK humano muito forte e PNG corroborado (12:39)

Evento de referência: `c0f029555d004dd4954867532edf1862`.
O operador confirmou que a referência SIDE
`public/ok_archive/2026-10-21_1424_FALTANDO.png` representa a
mesma aparência de **falha falsa**. Essa confirmação orientou a regra;
não converte a semelhança em identidade RGB comprovada.

Evidências registradas pela captura:
- `OPERATOR_CONFIRMED` SIDE: OK `0.99019399`, NG `0.95938641`;
  vantagem de OK **3,080758 pontos percentuais**. A consulta humana
  devolvia `AMBIGUOUS`, porque o limiar geral exige margem >=4pp.
- KNN C A: OK `0.98084619`, melhor NG `0.89033113`.
- KNN C B: OK `0.95345213`, melhor NG `0.81259430`.
- O **mesmo PNG OK** foi vizinho principal em A e B, com mesmo SHA
  de pixels. Nenhum par RGB exato foi encontrado pelo roteador.
- CNN FALTANDO V2 em SIDE indicava NG `0.98113567`. A regra de
  rejeição preventiva da Produção teria proposto `1/NG` após uma
  fusão inconclusiva SIDE/TOP/MID. Naquele debug, envio XP
  confirmado = `None`, não há comprovação de acionamento de tecla.

Nova alternativa **genérica e estrita** dentro de
`src/core/side_memory_first.py`, sem nomes/datas/arquivos fixos:

1. A memória humana deve vir de `OPERATOR_CONFIRMED` válido,
   com categoria canônica conhecida e **mesma luz SIDE**,
   `status=AMBIGUOUS`, OK >=99% e vantagem >=3pp sobre NG.
2. O banco KNN C deve estar READY, com A/B votando OK e os primeiros
   vizinhos A/B apontando para o **mesmo arquivo OK e SHA**.
   Exige similaridade A >=98% e B >=95%.
3. Exige também que **todo vizinho NG** dos top-5 A/B tenha
   similaridade <90% e fique pelo menos 5pp abaixo do melhor
   vizinho OK do respectivo descritor.
4. Antes de qualquer `0/OK`, exige o **veto físico atual**
   de duas escalas sem suspeita de ausência, com geometria
   admissível. Erros, ausência física, categoria inválida,
   fonte não humana, banco incompleto ou contradição forte
   fazem o caso cair no fluxo conservador; não aprovam peças.
5. Quando as evidências se confirmam, a **SIDE encerra como
   FALHA FALSA**, dispensa TOP/MID e registra
   `knn_agreement_mode=KNN_AB_STRONG_HUMAN_OK`,
   a referência PNG, SHA, scores e contraprova física.
   O gate de Produção **revalida** os marcadores antes de propor
   `0/OK`, preservando a pausa por espaço e a espera normal.
   O score publicado é similaridade, não probabilidade real.

A regra anterior para NG humano e a rota padrão de margem >=4pp
permanecem inalteradas. Nem o checkpoint CNN, nem os dados de
treinamento, nem a política de teclas XP foram modificados.
A validação é baseada em fixtures dos scores do log, não é uma
confirmação industrial independente. Validar com fotos de NG
semelhantes e com liberação XP desabilitada antes da operação autônoma.


## 30/10/2026 — SIDE visual OK com sonda física inconclusiva (13:06)

Evento `6d57176272ce49bf8eb0302cfed8e6ea`, executado na estação
com commit `ae5814ec22e735c0a4c93ec10cb995e5fe24f504`.
A memória SIDE retornou **OK humano 0.99019399 / NG 0.95938641**,
KNN C **A OK 0.98084619 / B OK 0.95345213**, ambos reconhecendo o
mesmo PNG e SHA em `public/ok_archive/`. O operador confirmou a
condição visual como falha falsa. O evento terminou, porém, com
`side_early_exit={}` e TOP/MID processadas; a política de produção
sugeriu NG preventivo. O código daquele commit retornava `None`
silenciosamente de `build_side_memory_first()` quando a sonda física
não satisfazia `_verified_probe()`; portanto, o resultado da sonda
não permaneceu no debug. Não há PNG original da captura disponível
neste repositório para determinar qual critério físico foi rejeitado.

**Mudança na fronteira entre classificação visual e saída XP:**

- O candidato fortemente qualificado `KNN_AB_STRONG_HUMAN_OK`
  permanece `FALHA FALSA` com `analyzed_lights=["SIDE"]` e
  `skipped_lights=["TOP","MID"]` **mesmo se a sonda retornar
  `SUSPECT_ABSENCE`, `UNAVAILABLE` ou falhar**.
- Se a sonda em duas escalas for válida, o contrato pré-existente
  continua permitindo o 0/OK via gate de produção, com revalidação.
- Se o veto físico for inconclusivo ou suspeito, o caso é
  **FALHA FALSA visual, aguardando decisão humana**. O sistema não
  autoriza 0/OK automático, não envia 1/NG preventivo e não cria
  treinamento automático. A inspeção não avança para TOP/MID.
- O debug inclui `side_memory_first.physical_probe` completo:
  status, reason, escalas, métricas e geometria, além de
  `physical_gate=OPERATOR_REQUIRED` ou `VERIFIED`.
  O gate XP revalida separadamente os marcadores antes de qualquer
  comando automático; alterar o texto do veredito não autoriza 0.
- O reconhecimento visual não prova integridade física. Preserva-se
  a proteção para um componente realmente ausente, especialmente
  contra a regressão FALTANDO de 07/10. As rotas NG e a memória
  comum com limiares menores não tiveram o veto alterado.

Reproduzir a situação física com o XP em modo supervisionado
e coletar o novo debug da sonda. As regressões com `SUSPECT_ABSENCE`,
`UNAVAILABLE`, exceção, marcadores adulterados, pausa e saída 0/1
estão em `tests/test_side_memory_first.py`.


## 30/10/2026 — Liberação supervisionada opt-in de OK SIDE com conflito físico

Evento `37a68aae712646b9a758dd8ff3be823b`, Modo Produção:
SIDE **FALHA FALSA**, memória humana OK `0.99019399`, NG
`0.95938641`, KNN A OK `0.98084619`, KNN B OK `0.95345213`.
O operador confirmou pessoalmente o falso positivo da AOI e solicitou
liberação automática sob supervisão. A física registrou
`SUSPECT_ABSENCE` **somente na escala 0.58**:
score `0.95587`, coverage `0.71882`, background `0.84268`;
escala 0.76 sem suspeita (`background=0.28904`), geometria
`[318,570]` x `[317,567]`. **Isto é uma contradição física real
no algoritmo e NÃO pode ser tratado como detector de presença validado**.

### Uso restrito

- `Ctrl+Shift+O`: **arma/desarma** um experimento de liberação
  supervisionada quando estiver em `Modo Produção` **entre peças**
  (sem análise pendente). Inicia **desarmado**; não persiste.
- Se a memória SIDE reconhecer OK humano >=99% + margem >=3pp,
  KNN A>=98% e B>=95% no mesmo PNG/SHA, sem NG concorrente forte,
  e o veto físico for `SUSPECT_ABSENCE` em **exatamente uma**
  das duas escalas verificadas, com geometria exata ou aparagem
  máxima de 3px/1%, a política poderá propor `0/OK` mesmo com a
  contradição física, **somente enquanto explicitamente armada**.
- Propostas da nova rota `side_strong_human_ok_supervised_probe_override_auto_ok`
  aguardam a apresentação e mais **5 segundos** antes do envio XP.
  A emissão revalida o armamento, a captura atual, a política e o
  estado de pausa, garantindo no máximo uma tentativa por geração.
- **Espaço pausa e desarma imediatamente** tanto essa rota quanto
  a KNN C supervisionada. Retomar não rearma; rearmar exige uma nova
  ação entre peças. Mudança de modo desarma. A UI exibe armamento
  e que se trata de autorização apesar de alerta físico.
- `SUSPECT_ABSENCE` nas **duas** escalas, geometria fora do contrato,
  `UNAVAILABLE`, ausência física forte, conflito de memória, categoria
  inválida, fonte adulterada ou inspeção pendente **não autorizam** a
  nova rota. Não se cria exceção por nome de arquivo, categoria OCR
  ou ID do evento.
- O julgamento automático nesta rota **não é dado de verdade para
  treinamento**. Vai para `reports/production_auto_unverified/`, com
  a única luz SIDE realmente observada quando disponível; não inventa
  imagens TOP/MID. O histórico da política registra que foi override
  supervisionado de um veto físico.
- Sem armamento, mantém `FALHA FALSA` visual na SIDE, interrompe TOP/MID,
  **aguarda 0/1 humano**; não emite NG preventivo nem 0 automático.
- Esse modo aumenta explicitamente o risco de liberar um NG físico.
  Requer bancada/ambiente isolado, operador olhando as imagens reais,
  validação com NG difícil e parada imediata se houver divergência.
  O temporizador de 5s não substitui julgamento humano nem garante
  interrupção a tempo.

Implementação: `src/core/side_memory_first.py` valida elegibilidade;
`src/ui/production_confidence_gate.py` propõe/revalida 0/OK;
`src/ui/production_autonomy_controller.py` guarda armamento efêmero,
executa temporizador e desarma; `src/services/decision_persistence.py`
impede o treinamento de aprovações ainda não confirmadas.
Regressões: `tests/test_side_supervised_release.py`.
