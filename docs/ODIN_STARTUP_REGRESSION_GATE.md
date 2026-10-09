## 09/10/2026 — Ajuste de escopo CNN e duas falhas falsas confirmadas pelo operador

**Decisão humana:** MUITO ADESIVO permanece no arquivo, mas fica **fora do
denominador da meta de 98% da CNN atual**. Uma CNN própria para adesivo
será considerada no futuro, quando houver dados OK/NG suficientes.
Não apagar PNGs, não substituir a inspeção adesivo existente, não treinar
CNN de adesivo agora. Os relatórios preservam também o placar do
**acervo inteiro**, para não confundir exclusão com sucesso.

**Replay real 09/10, antes desta alteração:** 212 PNGs, 202 do escopo
CNN FALTANDO V2; 200/202 decisões conclusivas corretas (99,01%), com
15/15 NG reconhecidos e 0 NG classificados OK. Os outros 10 são adesivo
(8 OK e 2 NG), sem CNN. Zero manifestos explícitos multilight.

**Duas imagens OK confirmadas pelo operador:**
- `2026-10-02_1542_FALTANDO.png`: OCR identificou categoria FALTANDO,
  mas faltou ao menos um campo de identidade. **Replay somente leitura**
  agora registra quais campos `board/parts/value` estão ausentes e
  **executa a CNN se o OCR da categoria coincidir**. O replay não usa
  identidade para decidir visualmente; a proteção operacional de
  identificação, KNN/XP e produção permanece inalterada. Resultado
  CNN efetivo só será conhecido com nova execução na estação.
- `2026-10-09_0857_DESLOCADO_SIDE.png`: OK confirmado; score CNN NG
  `0.10031582415103912` e predição binária OK, mas decisão operacional
  foi **REVISÃO OBRIGATÓRIA** porque o corte conservador para OK é 0,10.
  **Não forçar FALHA FALSA nem modificar limiar com base em um único OK**:
  não há NG DESLOCADO real disponível para validar segurança. A confirmação
  humana do rótulo não altera o veredito obtido da CNN nem o histórico.
  O desenvolvimento especializado DESLOCADO continua arquivado.

Novos indicadores: `requested_cnn_scope.target_met` (meta das
categorias cobertas), `overall.requested_cnn_scope_98pct_target_met`,
e `overall.historical_98pct_target_met` (meta de TODO arquivo,
incluindo categorias sem CNN; permanece separado). Revisões e casos
inválidos na CNN continuam falhas. Nenhum checkpoint, treinamento,
limiar produtivo ou gate foi modificado. Validar novamente na estação.

---

# ODIN — Gate de regressão visual na inicialização

## 09/10/2026 — Meta principal de regressão CNN em todo o histórico visual (98%)

**Critério do operador:** todas as imagens históricas com rótulo OK/NG
devem passar novamente pelas **CNNs**, usando o próprio PNG:
- OK original continua OK; NG original continua NG;
- tolerância inicial de 1–2% de discordância, isto é, >=98% de
  concordância histórica **sobre o acervo inteiro**, não somente
  exemplos fáceis;
- revisar erros individuais, com resultados separados por
  categoria, iluminação e principalmente NG; revisão obrigatória
  NÃO é classificada como um acerto;
- a KNN NÃO substitui o resultado CNN, nem converte erro em sucesso.

**Limitação física do acervo, comprovada pelos diagnósticos anteriores:**
`public/ok_archive` / `ng_archive` tinham 212 PNGs na última
varredura; os 952 JSONs de memória incluem 815 registros legados v2
sem par de imagens. **Uma CNN não pode reexecutar 815 inspeções
visuais a partir de assinaturas vetoriais não invertíveis.** Para
análise CNN desses casos, recuperar novas capturas verdadeiras.
Não inventar pixels nem declarar 952/952 com KNN.

**O que já existe:** `FaltandoCNNLive` usa checkpoint FALTANDO V2
para FALTANDO, EMBORCADO, INVERTIDO e DESLOCADO (escopo efetivo
da CNN compartilhada). Resultado anterior de 201/202 para esse
escopo (~99,50%) supera a tolerância de 98%, mas precisa ser
reconfirmado na estação e inclui uma revisão SIDE.
A categoria MUITO ADESIVO ainda usa motor físico especializado
e **não possui CNN treinada pronta**. Na última contagem havia
10 PNGs de adesivo: sem CNN, 98% do total de 212 não pode
ser comprovado. Não treinar uma CNN adesivo com poucos exemplos
sem holdout NG/OK apropriado; primeiro qualificar dados.

**Implementação somente leitura:**
- `src/services/startup_regression/cnn_full_history_replay.py`
  reanalisa **todos** os PNGs do inventário validado, obtém
  gabarito/teste pelo extrator AOI real, verifica categoria pelo
  OCR e avalia CNN FALTANDO V2 **sem chamada KNN**. Checkpoints
  precisam passar verificação SHA-256 e metadados.
- Para cada PNG registra esperado, categoria, iluminação,
  hash, CNN, score NG bruto não calibrado, predição binária
  informativa, veredito operacional, revisão/erro, checkpoint.
  `PASSOU` exige decisão operacional OK/NG correspondente;
  revisão, OCR inválido, arquivo corrompido, categoria sem CNN
  ou checkpoint ausente NÃO passam.
- Métricas por categoria, iluminação e rótulo: OK→OK, OK→NG,
  OK→revisão, NG→NG, NG→OK, NG→revisão. Também separa
  acerto binário do score bruto de aprovação operacional.
- Eventos TOP/MID/SIDE só são agregados com manifesto
  de três imagens explicitamente vinculado. Não inventar
  evento a partir de nomes semelhantes ou minutos próximos.
  O replay percorre **todos** os `manifest_links` de cada PNG:
  um arquivo compartilhado por dois eventos não apaga a auditoria
  de nenhum deles. Se o papel SIDE/TOP/MID do manifesto não coincide
  com a iluminação realmente inferida pela CNN, o evento falha
  em vez de fingir que houve três inferências válidas.
  Manifestos inválidos e identificadores de evento duplicados
  impedem a aprovação histórica.
- Além dos >=98% gerais, **OK e NG separadamente** precisam
  alcançar >=98% entre seus respectivos PNGs (quando a classe
  existe no acervo). Isso impede que predominância de OK esconda
  NG classificados como OK ou enviados à revisão.
- O TXT apresenta resultados por modelo, classe, categoria e
  iluminação, além dos eventos explícitos reprovados.
- Retenção total é acertos / **total de PNGs**, inclusive
  categorias sem CNN. `historical_98pct_target_met` exige
  >=98% do arquivo todo e **cobertura completa por CNN**.
  Mesmo se atingido, o relatório declara
  `production_approved=false`: reexecutar imagens usadas
  no treino não mede desempenho em defeitos inéditos.
- `cnn_full_history_replay_cli.py`: gera
  `reports/startup_regression/cnn_full_history_*.json`
  e TXT, sem gravar dataset, pesos, decisões XP ou ativar gate.
- `tests/test_cnn_full_history_replay.py` e o workflow
  `startup-regression-cnns.yml` validam os contratos.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.startup_regression.cnn_full_history_replay_cli
```

**Próximos passos:** analisar o novo relatório real da estação.
Se categoria CNN coberta cair abaixo de 98%, identificar casos,
checar dados/rótulos e treinar candidato CNN com conjunto
independente sem substituir checkpoint atual. Em adesivo,
primeiro qualificar quantidade e variedade de OK e NG para
justificar CNN especializada. Sem KNN e sem lançamento
operacional só pelo indicador histórico.

---


## 09/10/2026 — Avaliação seletiva KNN com teste por dia separado (somente leitura)

**Ponto de partida real:** auditoria com 952 registros (891 OK e 61 NG).
O baseline KNN Top 5 classificou 26 NG como OK; a política experimental
balanceada com revisão deixou 2 NG incorretamente liberados como OK, porém
gerou 248 revisões e só 23 NG detectados automaticamente. Essa diferença
veio sobretudo de abstenções, **não** de uma classificação NG de 100%.
Os dois NG restantes eram de FALTANDO:
`memory_NG_20260804_082605_918.json` e
`memory_NG_20261007_123710_374.json`.
Não subir limiares usando esses mesmos registros para anunciar 100%.

**Código autorizado (somente diagnóstico):**

- `src/services/startup_regression/knn_selective_holdout.py`: separa
  a avaliação em três conjuntos por **dia completo inferido do nome JSON**:
  `memory_train`, `calibration`, `heldout_test`. Datas inteiras não
  atravessam as partições; eventos declarados e assinaturas idênticas
  encontradas em mais de uma partição são retirados da validação.
- A distribuição por data é determinística e estratificada por dias
  com NG para tentar oferecer NG em todos os três grupos. **Não é
  necessariamente cronológica**; sem ID físico da placa/lote,
  independência real ainda NÃO está comprovada. Não chamar
  `heldout_test` de validação em placas inéditas.
- A assinatura da consulta é comparada somente contra os vetores
  guardados em `memory_train`, usando
  `compare_anomaly_signatures` e voto KNN existente de referência.
  O rótulo humano da consulta serve apenas para métricas APÓS a
  decisão, não para alterar a inferência. Ausência de vizinhos
  de uma das classes vira `REVISAO_OBRIGATORIA`.
- A calibração examina um **grid fixo antes do teste** de tetos
  de voto NG para liberar OK (0, .10, .20, .25, .30, .35, .40)
  e semelhança mínima (.80, .85, .90, .95). Uma política só é
  candidata se não liberar nenhum NG como OK NA CALIBRAÇÃO
  e mantiver pelo menos 20% dos OK automáticos na calibração.
  Dentre as elegíveis, seleciona maior liberação OK e
  congela os parâmetros antes do teste.
- Se não houver ≥3 dias com NG, se faltarem classes em alguma
  partição ou se nenhuma política calibrada for útil, o
  diagnóstico relata BLOCKED e não finge aprovação.
- Mede **separadamente** resultado no teste reservado:
  NG liberados como OK, NG classificados NG, NG revisados,
  OK classificados OK, OK classificados NG, OK revisados.
  Disponibiliza referência TOP5 no **mesmo conjunto reservado**.
- Mesmo teste com zero NG liberados indevidamente informa
  `EXPLORATORY_NO_NG_MISSED_NOT_PRODUCTION_VALIDATED`,
  `production_approved=false`: amostras correlacionadas
  ou conjunto NG pequeno não comprovam zero risco em produção.
  Limite binomial 95% quando zero falhas é meramente
  ilustrativo e pressupõe independência não demonstrada.

**Execução na estação:**

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.startup_regression.knn_selective_holdout_cli
```

**Arquivos gerados:**
`reports/startup_regression/knn_selective_holdout_*.json` e
`knn_selective_holdout_*.txt`. Enviar esses dois arquivos para
decidir próximos testes. `tests/test_knn_selective_holdout.py`
foi integrado ao workflow Windows de regressão.

**Estado inalterado:** produção KNN/CNN, dataset e
`main.py` não são modificados; gate de startup desativado.
Faltam validação em peças/lotes físicos separados, revisão
CNN FALTANDO V2 201/202, OCR e prova das três iluminações
antes de considerar o encerramento operacional.

---


## 09/10/2026 — Auditoria experimental da discriminação NG da KNN

**Base observada em `memory_levels_20261009_173758_183510`**:
952 assinaturas humanas elegíveis; 925 `CONCORDA`, 27 `DIVERGE`
no teste leave-one-record-out (top-5). Entre 61 NG, 26 foram
erroneamente classificados como OK; entre 891 OK, apenas 1
foi incorretamente NG. A concordância total de 97,16% NÃO
representa segurança NG (sensibilidade NG = 35/61 = 57,38%).
Há 59 registros pertencentes a grupos de assinatura duplicada,
sem conflito exato de rótulos detectado no relatório original.

**Implementação autorizada: comparação offline, sem mudança operacional.**

`src/services/startup_regression/knn_ng_discrimination_audit.py`:
reutiliza `compare_anomaly_signatures` e
`KNNExpert._weighted_vote` no baseline. É proibido consultar o
próprio registro, sair da categoria/iluminação, aceitar evento
igual quando `event_id` é conhecido ou instanciar modelos.
A distância entre duas assinaturas é calculada uma vez e
reutilizada entre os cinco métodos diagnósticos:

- `BASELINE_TOP5`: voto inverso por distância existente;
- `SEM_DUPLICATAS_TOP5`: remove todas as assinaturas
  idênticas à consulta, reduz cópias dos vizinhos por hash;
- `BALANCEADO_3_POR_CLASSE`: até três vizinhos de cada classe,
  com média do peso inverso de distância por classe;
- `BALANCEADO_SEM_DUPLICATAS`: balanceamento após exclusão
  de cópias, empates e assinaturas contraditórias;
- `BALANCEADO_COM_REVISAO`: mesmo balanceamento e abstinência
  explícita quando similaridade máxima < 0,80, margem de
  voto NG em torno de 0,5 menor que 0,10, falta uma classe
  ou há conflito/empate. **Os limiares são hipóteses
  exploratórias não calibradas com conjunto independente.**

**Contrato de métricas:** cada modo contabiliza separadamente
`correct_NG`, `missed_NG_as_OK`, `review_NG`,
`correct_OK`, `false_NG_on_OK`, `review_OK`.
Revisão não é contada como detecção NG automática correta.
Acurácia de decisões automáticas divulga explicitamente
o denominador após abstenções; a taxa de revisão é informada.
Resultados por categoria são apresentados separadamente.

`src/services/startup_regression/knn_ng_discrimination_cli.py`
gera `reports/startup_regression/knn_ng_audit_*.json` e TXT.
`tests/test_knn_ng_discrimination_audit.py` exercita
voto baseline, isolamento estrito categoria/luz, exclusão de
cópias e eventos, balanceamento, revisão, rótulos conflitantes,
ausência de NG, proibição de writes e de inicialização de modelo.
Workflow Windows atualizado.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.startup_regression.knn_ng_discrimination_cli
```

**Não usar esta comparação como ganho confirmado em imagens novas.**
Amostras repetidas podem pertencer à mesma placa ou lote sem
`event_id` declarado; deduplicar vetores não substitui um
teste segregado por placa/lote/data. Antes de adotar qualquer
política em produção, calibrar exclusivamente em dados de treino,
validar em testes independentes (especialmente defeitos NG) e
manter revisão obrigatória em caso incerto. Nenhuma alteração em
`main.py`, `KNNExpert`, `public/dataset`, CNN FALTANDO V2
ou startup gate foi feita.

---


## 09/10/2026 — Dois níveis de memória: avaliação independente e sem migração

Último diagnóstico de evidência (`evidence_recovery_20261009_172032_693807`):
578 PNGs indexados, 952 JSONs: 92 v3, 45 v2 auditáveis,
815 v2 sem imagens de gabarito/teste ou referências visuais
nos campos existentes. Nenhuma reconstrução nova. A assinatura
de 224 atributos não permite recuperar os pixels originais.

**Nível A — memória de assinaturas legadas.**
`legacy_knn_signature_audit.py` usa o comparador
`compare_anomaly_signatures` e o peso inverso de distância
`KNNExpert._weighted_vote` da produção. O histórico é
avaliado em teste leave-one-record-out: não consultar o
próprio JSON, somente pares da mesma categoria e luz, excluir
`event_id` igual quando disponível. Relatar vizinhos,
predições, voto NG, similaridade, discordâncias, empates e
assinaturas iguais com classes conflitantes. Não usar
rótulo arquivado como predição. **Essa precisão não mede
cobertura da memória visual dos 212 screenshots nem
generalização fora da base histórica.**

**Nível B — propostas para índice visual exato.**
`archive_exact_memory_plan.py` valida PNGs por inventário
CRC/SHA e extrai gabarito/teste com o OCR real. Calcula a
chave do mesmo `VerifiedKNNMemory._key` usado na produção;
cruza com v3 verificadas e v2 auditáveis simuladas e marca
`JA_VERIFICADO_EXATO_V3`, `PAR_LEGADO_EXATO_PENDENTE`,
`PENDENTE_ORIGEM_HUMANA`, `PENDENTE_ORIGEM_HUMANA_E_LUZ`,
`OCR_INVALIDO`, `CATEGORIA_OCR_DIVERGENTE`,
`CONFLITO_COM_MEMORIA_EXISTENTE` e outras falhas.
O rótulo extraído da pasta visual não constitui confirmação
humana. Apenas a procedência legítima e o contexto físico
podem qualificar um registro v3 no futuro. `ready_for_import=0`
até haver qualificação real.

**Ferramenta:** `two_level_memory_diagnostic_cli.py`.
Produz em `reports/startup_regression`
`memory_levels_*.json` (ambos relatórios estruturados e
separados) + `memory_levels_*.txt`.
`tests/test_two_level_memory_diagnostic.py` e workflow
Windows verificam casos OK/NG, OCR, hashes, escopo, conflitos
e nenhum efeito colateral na memória.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.startup_regression.two_level_memory_diagnostic_cli
```

**Permissões desta etapa:** READ-ONLY diagnóstico.
É proibido migrar registros, criar pares gabarito/teste
por inferência de nome, reclassificar o acervo, ajustar limiar
só para obter 100%, retreinar CNN ou habilitar o bloqueio
de inicialização. `main.py` e motores operacionais ficam
inalterados. Após relatórios reais, decidir separadamente
sobre pendências do arquivo e sobre o caso DESLOCADO SIDE
revisado pela CNN FALTANDO V2.

---


## 09/10/2026 — Recuperação de evidência visual histórica

**Estado anterior confirmado nos relatórios da estação:**
860 JSONs `visionx.memory.v2`, 45 pares de imagens legados
auditáveis (somente simulação), 815 sem par PNG; três pares
tinham correspondência com os screenshots históricos, sem
migração. A etapa seguinte, autorizada pelo operador, é **apenas
buscar evidência visual e diagnosticar o vínculo dos registros**.

Novo código: `src/services/startup_regression/historical_evidence_recovery.py`,
`historical_evidence_recovery_cli.py` e testes dedicados, mais
extensão do workflow Windows de reconciliação.

Busca de forma local e read-only nos PNGs dentro de
`public/dataset`, `public/ok_archive` e `public/ng_archive`.
Cria índice por hash de **pixels decodificados**, não por apenas
nome, tempo, categoria ou hash de bytes comprimidos. Índices são
efêmeros; nenhuma memória é escrita. Caminhos simbólicos externos,
traversal e caminhos não localizados são ignorados.

Para cada JSON:
- verifica schema, confirmação de operador, assinatura e
  referências declaradas pelo próprio registro;
- procura as imagens em locais antigos por basename e hash,
  reportando o **grau de confiança** de cada vínculo;
- quando `storage.source_image_fingerprint` comprova a origem,
  opcionalmente reextrai gabarito e teste via
  `AOIPairExtractor` e confronta o OCR/identidade da placa,
  componente, categoria e valor com o JSON;
- a simulação `PAR_RECONSTRUIDO_PARA_REVISAO` exige também
  hashes declarados do gabarito **e** do teste, idênticos aos
  recortes reextraídos. Se houver referência explícita ao PNG
  fonte no mesmo diretório do JSON, porém sem hash da origem,
  pode executar OCR diagnóstico, mas o caso permanece marcado
  `ORIGEM_JSON_LOCAL_SEM_HASH_PARA_REVISAO` e NÃO vira KNOWN;
  a reconstrução com hashes não equivale a proveniência auditada;
- versões antigas que jamais registraram os hashes ou PNGs podem
  permanecer sem evidência suficiente. Esse é um resultado
  válido de diagnóstico, não uma regressão da CNN.

Exemplos de estados que **não autorizam migração**:
`ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH`,
`TESTE_POR_HASH_SEM_GABARITO`,
`DOIS_HASHES_DE_PARES_LOCALIZADOS_PARA_REVISAO`,
`RECONSTRUCAO_OCR_DIVERGENTE`,
`SEM_EVIDENCIA_VISUAL_LOCALIZAVEL`.
Relatório inclui totais por categoria de recuperação, OK/NG
separados e inventário de arquivos de evidência encontrados.

Comando:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.startup_regression.historical_evidence_recovery_cli
```

`reports/startup_regression/evidence_recovery_*.json` e
`evidence_recovery_*.txt` são a saída solicitada.
**Sem migração, sem alteração de CNN/KNN, sem gate bloqueante**.
Só com essa evidência será possível definir a próxima etapa
de recuperação auditável sem fabricar um resultado 100%.

---


## 09/10/2026 — Adaptador legado somente leitura (fase de simulação)

**Referência da estação:** em 212 PNGs e 952 JSONs da memória,
o diagnóstico anterior registrou 860 `SCHEMA_NAO_SUPORTADO`,
92 `VERIFICADO_KNN`, 99 PNGs com contexto legado porém
registro inelegível, 20 sem correspondência de contexto e 1 OCR inválido.
O objetivo não é mudar o rótulo de um registro, mas verificar se
um par antigo pode ser reconhecido com evidência íntegra.

**Novo módulo:** `src/services/startup_regression/legacy_memory_compat.py`.
Ele lê todos os JSONs de memória não-v3, identifica seu schema
(inclusive `SEM_SCHEMA`), e testa **sem persistência**:

1. Confirmação explícita do operador e mesmo rótulo no JSON e
   na subpasta OK/NG, sem aprovar fonte automática;
2. Assinatura de anomalia válida (embedding antigo isolado não
   atende ao critério de atalho por par exato da KNN);
3. PNGs legíveis de referência/teste no diretório do JSON, sem
   travessia de caminho, symlink ou uso de pares inferidos;
4. Identidade visual exata pelo `image_fingerprint`, coerente
   com hash declarado quando existir;
5. Board, Parts, Value, categoria e iluminação normalizados pela
   `VerifiedKNNMemory._key`, **sem** copiar metadados do nome do PNG.

Se e somente se todas as verificações acima passarem, um registro
pode virar `LEGADO_PAR_AUDITAVEL_SIMULADO`: ele é indexado
**temporariamente no reconciliador**, nunca injetado no KNN
operacional. Em cada screenshot o relatório registra:
`LEGADO_PAR_SIMULADO_CONCORDA`,
`LEGADO_PAR_SIMULADO_DIVERGE`,
`LEGADO_CONFLITO_EXATO` ou `SEM_PAR_LEGADO_AUDITAVEL`.
Conflitos exatos são sinalizados de modo conservador.

Relatórios incluem a taxonomia completa dos esquemas e das razões
de inelegibilidade de cada registro, exemplos de caminhos para
diagnóstico e quantidade de compatibilidades simuladas. Nenhum caso
entra em `PAR_VERIFICADO` apenas pela simulação. A KNN de produção,
os rótulos, a CNN FALTANDO V2 e `main.py` não mudaram.

**Próxima ação:** executar novamente
`python -m src.services.startup_regression.archive_reconciler_cli`
na estação e enviar `reconciliation_*.json` + `reconciliation_*.txt`.
Após analisar os schemas/referências reais, propor uma migração
auditável somente se for tecnicamente possível; nenhuma migração
ou startup gate está autorizado nesta fase.

---


## 09/10/2026 — Reconciliação automatizada sem alterações de memória

O operador autorizou iniciar pelo diagnóstico **somente leitura**:
a importação/migração histórica e o bloqueio no `main.py`
**NÃO** foram autorizados nesta fase. O plano antigo com MoE
e o rascunho de "CNN MEMÓRIA" seguem como histórico; a regra
atual é **CNN FALTANDO V2 + MEMÓRIA KNN existente**.

**Base real reportada:** 212 PNGs, CNN FALTANDO V2 201/202
(1 revisão em `2026-10-09_0857_DESLOCADO_SIDE.png`);
MEMÓRIA KNN 89/212, 122 sem cobertura, 1 caso com OCR
incompleto. Nenhum dos 17 NG possui match KNN verificável.
A validação de CNN é independente da recuperação KNN.

Implementação nova:
`src/services/startup_regression/archive_reconciler.py` e
`archive_reconciler_cli.py`, workflow/teste Windows dedicado.
O reconciliador varre os registros `public/dataset/nao_anomalia` e
`public/dataset/anomalia` **mesmo se não forem elegíveis para o
carregador KNN**, sem copiar imagens. Para registros verificáveis,
chama a validação existente `VerifiedKNNMemory._entry`; para o
screenshot, usa `VerifiedKNNMemory._key` calculada de OCR
real e hashes exatos dos recortes AOI, sem utilizar o nome
do arquivo como substituto do OCR.

Registra a causa de rejeição de cada JSON (schema, rótulo,
proveniência humana, assinatura, imagens PNG faltantes,
dados incompletos ou hash inconsistente) e identifica por
screenshot os possíveis candidatos de correlação por:
par exato; teste exato mas gabarito diferente; screenshot
de origem igual; metadados equivalentes mas registro inelegível;
contexto compatível com imagem diferente; categoria/luz sem
registros. **Esses diagnósticos não promovem registros à
memória e não contam como aprovação**.

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.archive_reconciler_cli
```

Gerar `reports/startup_regression/reconciliation_*.json` e
`reconciliation_*.txt`, enviar para análise e só então
escolher migração auditável ou correção de OCR. Proibido
inscrever um rótulo deduzido da pasta como confirmação humana,
alterar limiares da CNN para forçar 100% ou validar por vizinhos
aproximados no lugar do par exato. **Startup bloqueante
permanece desativado.**

---


## Correção de requisito — 09/10/2026 — CNN FALTANDO V2 + MEMÓRIA KNN

**Esclarecimento do operador:** `CNN MEMÓRIA` significava
**MEMÓRIA KNN existente**, não um segundo classificador CNN.
Esta seção prevalece sobre o contrato anterior de duas CNNs.

1. A CNN FALTANDO V2 reanalisa sem consulta KNN todas as imagens
   das categorias FALTANDO, EMBORCADO, INVERTIDO e DESLOCADO,
   excluindo MUITO ADESIVO.
2. A **memória KNN atual**, para todas as categorias sem exceção,
   usa `VerifiedKNNMemory.lookup` (mesma regra da Produção):
   apenas registro humano exato com gabarito/teste, Board,
   Parts, Value, categoria e iluminação compatíveis.
3. A memória KNN recupera **o rótulo do dataset**, nunca o rótulo
   de `ok_archive`/`ng_archive`. O rótulo do arquivo visual é
   usado somente depois, para comparar a resposta e contabilizar
   `PASSOU` ou `REGRESSAO`.
4. Se o par visual não estiver comprovadamente na memória,
   marcar `SEM_COBERTURA`, não inventar OK/NG. Exemplos
   antigos podem não ter seus dois PNGs na memória KNN.
   Registros autoaprendidos, semelhantes mas não idênticos
   e contraditórios não autorizam aprovação.
5. Falha de OCR, imagem inválida, registro incoerente,
   ausência do checkpoint da CNN ou erro em qualquer motor
   impedem a aprovação. Não dispensar exemplos por categoria
   e não compor três luzes sem `event_id` verdadeiro.
6. A validação KNN é um teste de **integridade da memória e
   recuperação de histórico conhecido**. Não mede detecção
   independente de falhas novas. A CNN avaliada nos exemplos
   de treinamento também não é teste cego.
7. O código de diagnóstico foi conectado na CLI existente
   `cnn_archive_validation_cli.py`; **não ativar** gate em
   `main.py` antes dos relatórios da estação e do aceite
   expresso do operador.

O contrato antigo que exigia uma segunda CNN e seus pesos não
se aplica após esse esclarecimento.

---


## Revisão de escopo — 09/10/2026 — substituir replay MoE por duas CNNs

**Esta seção substitui as premissas anteriores de validação por
especialistas físicos/KNN neste documento.** As seções abaixo
documentam o histórico do plano anterior e não são autorização
para ligá-lo como gate operacional.

**Contrato novo:**

1. **CNN FALTANDO V2**, apenas `FALTANDO`, `EMBORCADO`,
   `INVERTIDO`, `DESLOCADO` (sem adesivo). Executar inferência
   pura no gabarito/teste completos de todos os arquivos elegíveis OK
   e NG, em SIDE/TOP/MID conforme imagem.
2. **CNN MEMÓRIA**, modelo separado, em **todas as categorias** sem
   exceção, inclusive `MUITO ADESIVO`. Não confundir com memória KNN
   de imagem conhecida nem usar o rótulo do arquivo como resposta.
3. Em cada modelo, pasta OK exige `FALHA FALSA`; pasta NG exige
   `DEFEITO REAL`. Resultado de revisão, score não finito,
   checkpoint não verificado, OCR/recorte inválido ou ausência de
   qualquer imagem elegível reprova o modelo. Nenhum rótulo humano
   é alterado para passar. A aprovação é **100% em cada CNN**, sem
   média ou compensação entre modelos.
4. Um arquivo histórico sem luz explícita é SIDE. Novos PNGs
   SIDE/TOP/MID devem ser avaliados sob sua luz própria; sem manifesto
   verdadeiro não declarar que três imagens pertencem à mesma peça.
   Validar por arquivo preserva todo o acervo e não cria eventos
   sintéticos. Extensão futura a fusão por evento exige `event_id`
   auditável e não mascara erro individual desta política.
5. **Fail closed**: modelo ausente, pesos incompatíveis, arquivo
   inválido, inferência interrompida ou ausência de cobertura nunca
   pode ser contabilizado como OK. Relatórios JSON/TXT precisam
   registrar cada erro e os subtotais separados.
6. **Em toda abertura:** inventariar integralmente e inferir novamente
   sobre os arquivos; sem cache de rótulo previsto, sem retreinar no
   startup, sem KNN e sem rede. A janela operacional/servidor AOI só
   poderão ser criados após ambos os testes passarem.

**Implementação disponível para integração segura:**

```text
src/services/startup_regression/cnn_archive_validation.py
src/services/startup_regression/cnn_archive_validation_cli.py
tests/test_startup_regression_cnns.py
.github/workflows/startup-regression-cnns.yml
```

A `FaltandoCNNLive` usa checkpoint verificável e sua inferência CNN
pura. O segundo modelo **ainda não tem classe/checkpoint identificado
na branch central**; portanto `CNN_MEMORIA` fica com status
`MODEL_UNAVAILABLE`. `verified_memory_router.py` é KNN por par
exato; `neural_judge.py` possui encoder CNN + KNN, mas nenhum
dos dois satisfaz uma segunda CNN classificada de forma autônoma.
**Não substituir por esses motores e não instalar gate no `main.py`
antes da conexão real do segundo modelo.**

**Diagnóstico sem bloqueio:**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.cnn_archive_validation_cli
```

Os arquivos `reports/startup_regression/cnn_validation_*.json` e
`cnn_validation_*.txt` expõem as métricas e casos individuais;
é esperado que a avaliação conjunta reprove por ausência do segundo
modelo. Falha de teste **não ativa a produção**, mas também não impede
a inicialização atual do ODIN enquanto a integração não estiver pronta.

**Próxima dependência:** operador informar onde está a CNN MEMÓRIA
(módulo, checkpoint e método de inferência), para implementar o
adaptador e só então a tela/trava de startup, com testes de falha e
sucesso em ambiente local.

---


**Status:** Etapa 1 concluída pelo inventário real; **Etapa 2 implementada para diagnóstico SIDE sem memória**, aguardando resultado no computador da fábrica. Etapas 3–5 e gate bloqueante não implementados.  
**Registro:** 08/10/2026.  
**Escopo:** aplicação do computador novo, branch `central`; arquivos visuais `public/ok_archive/` e `public/ng_archive/`.  
**Documento-mestre do histórico:** [`estrutura_projeto.md`](../estrutura_projeto.md).

## 1. Necessidade industrial e objetivo

As pastas `C:\visionx-neural-main\public\ok_archive` e
`C:\visionx-neural-main\public\ng_archive` possuem evidências reais já
rotuladas por operador. A maioria das evidências históricas corresponde à
iluminação **SIDE**; novos eventos podem preservar **SIDE, TOP e MID** da
mesma anomalia da AOI.

**Objetivo:** em **toda inicialização**, antes de disponibilizar o painel
operacional, reanalisar o acervo completo com o **mesmo código e especialistas
físicos e modelos visuais, excluindo integralmente KNN e memória episódica**. O ODIN só pode entrar em
operação após aprovação integral da regressão. Se houver erro, divergência,
arquivo inválido ou cobertura incompleta, a entrada em operação é bloqueada,
com diagnóstico acessível em uma janela restrita de validação.

É um **teste de não regressão contra casos conhecidos**, não prova de
generalização da IA nem licença para memorizar cegamente as respostas. O
aprendizado continua vindo de rótulos humanos e do dataset separado; o gate
não treina, não retreina e não altera a verdade dos exemplos para conseguir
100%.

## 2. Contrato de aprovação

### 2.1 Acervo antigo — imagem única (legado SIDE)

Quando um screenshot histórico não possui marcador explícito de iluminação nem
manifesto multilight, classificá-lo para **replay monoimagem em SIDE**.

| Origem | Rótulo humano esperado | Único resultado aceitável |
|---|---|---|
| `ok_archive/` | OK | `FALHA FALSA`, sem revisão |
| `ng_archive/` | NG | `DEFEITO REAL`, sem revisão |

`REVISÃO OBRIGATÓRIA` não equivale a acerto, inclusive se a memória
individual for de NG ou OK. Assim, o acervo legado serve como contrato de
regressão estrito da decisão individual sob SIDE.

**Observação de implantação:** exemplos NG antigos podem ter sido arquivados
precisamente porque o ODIN de então os classificou erradamente. Esses casos
podem reprovar no primeiro diagnóstico, e isso **não autoriza inverter rótulos
automaticamente**, alterar limiares só para passar ou ignorar o arquivo.
Precisam ser corrigidos na lógica e novamente aprovados pelo teste.

### 2.2 Acervo novo — evento multilight

O evento completo compartilha:

```text
event_id
Board / Parts / Category / Value (mesmo OCR)
expected_label = OK ou NG
SIDE.png, TOP.png, MID.png
```

O replay deve executar **uma análise local por iluminação**, sob
`lighting_mode` correto, e depois usar a **mesma fusão operacional** aplicada
na AOI real:

- `MUITO ADESIVO`: fusão especializada já existente.
- Outras categorias válidas: `src/core/multilight_fusion.py`.

**A aprovação é do julgamento final do evento, não de cada PNG isolado.**

| Rótulo humano | Resultado final obrigatório |
|---|---|
| OK | `FALHA FALSA`, sem revisão |
| NG | `DEFEITO REAL`, sem revisão |

Caso NG válido:

```text
SIDE = FALHA FALSA
TOP  = DEFEITO REAL
MID  = FALHA FALSA
             ↓
fusão final = DEFEITO REAL → PASSOU
```

Exigir NG individual para SIDE, TOP e MID seria **um erro**: destruiria a
finalidade do multilight, no qual o defeito pode aparecer apenas sob uma luz.

Uma sessão incompleta que se declara multilight **não pode** ser tratada
silenciosamente como SIDE para conseguir passar; ela recebe
`INCOMPLETO/INVÁLIDO` e bloqueia a operação. O legado monoimagem explícito
continua suportado.

### 2.3 Resultado global

**Abrir o painel operacional somente quando:**

```text
total de arquivos descobertos = total de arquivos processados ou vinculados
nenhum arquivo sem rótulo/identidade válida
nenhum arquivo danificado ou ilegível
nenhum evento multilight incompleto
nenhuma classificação divergente
nenhuma revisão obrigatória em caso com rótulo OK/NG
todos os eventos e exemplos históricos aprovados
```

Estados por exemplo/evento:

- `PASSOU`: resultado final coincide com o rótulo esperado.
- `REGRESSÃO`: resultado difere do rótulo, inclusive revisão obrigatória.
- `INVÁLIDO`: screenshot corrompido, layout/recorte/OCR indispensável
  inválido, categoria não identificável, dados ausentes ou exceção de análise.
- `INCOMPLETO`: manifesto/evento declara três iluminações, mas falta alguma.
- `CONFLITO`: arquivo/identidade aparece rotulado de maneiras incompatíveis
  ou há associação contraditória comprovada.
- `SEM COBERTURA`: não há exemplos qualificados suficientes para executar o
  gate (não anunciar 100% de aprovação sobre zero casos).

Qualquer estado diferente de `PASSOU` bloqueia a **operação normal** e exige
diagnóstico/correção. A tela restrita de manutenção não é o painel do ODIN e
**não** recebe nem envia comandos à AOI.

## 3. Inventário e interpretação dos arquivos

### 3.1 Fontes e compatibilidade de nomes

Diretórios configurados em `src/config/settings.py`:

```text
settings.OK_ARCHIVE_DIR = public/ok_archive/
settings.NG_ARCHIVE_DIR = public/ng_archive/
```

Usar caminhos derivados de `settings`, **não** o caminho literal do PC do
operador; o projeto pode estar instalado em outro disco/pasta.

Inventariar recursivamente todos os PNGs, incluindo nomes antigos como:

```text
2026-10-01_07-36-23-677_FALTANDO.png
2026-10-06_1042_MUITO_ADESIVO.png
```

e novos nomes com iluminação:

```text
..._FALTANDO_SIDE.png
..._FALTANDO_TOP.png
..._FALTANDO_MID.png
```

Os nomes históricos podem variar. O parser deve tolerar os padrões conhecidos
sem inferir informação que não existe. `_SIDE/_TOP/_MID` no final identifica
iluminação **quando explícito**; ausência de sufixo no legado indica SIDE.

A **pasta** determina o rótulo esperado OK/NG. O nome e o OCR podem contribuir
para resolver categoria, mas categoria/Board/Parts/Value usados na análise
devem ser extraídos e validados com as regras atuais ou recuperados de
metadados humanos confiáveis; não inventar OCR.

Arquivos de metadados/manifesto devem ser reconhecidos explicitamente;
arquivos desconhecidos, PNG ilegível ou extensão não suportada não podem ser
ignorados silenciosamente. Não mover, renomear nem apagar arquivos no gate.

### 3.2 Manifesto de regressão por peça (novo contrato de arquivamento)

**Antes de liberar o modo multilight completo no gate**, evoluir a gravação
das evidências para guardar uma associação explícita, transacional e auditável:

```json
{
  "schema": "visionx.archive_regression.v1",
  "event_id": "id-real-da-aoi",
  "expected_label": "NG",
  "aoi_info": {
    "board": "valor-do-ocr",
    "parts": "referencia-do-ocr",
    "category": "FALTANDO",
    "value": "texto-do-ocr"
  },
  "source": "network",
  "frames": {
    "SIDE": {"path": "arquivo_SIDE.png", "sha256": "hash-dos-bytes"},
    "TOP":  {"path": "arquivo_TOP.png", "sha256": "hash-dos-bytes"},
    "MID":  {"path": "arquivo_MID.png", "sha256": "hash-dos-bytes"}
  }
}
```

O JSON acima **é especificação proposta**, não formato já implementado.
Registrar hash, origem e caminhos efetivamente persistidos (inclusive quando
a deduplicação visual reutilizar um PNG pré-existente). Gravar o manifesto
**somente quando o conjunto referenciado estiver consistente e disponível**,
com gravação atômica. Não criar manifesto que aponte para arquivos que a fila
ainda não concluiu.

Para fotos antigas, não fabricar `event_id` original ou OCR que não existe.
Criar metadado de qualificação/legado auditável apenas se necessário, sem
alterar os PNGs.

**Não agrupar SIDE/TOP/MID só por horário, sequência de nomes ou categoria.**
Isso poderia juntar peças distintas. Imagens novas sem manifesto confiável
devem ser sinalizadas para qualificação, não falsamente fundidas.

### 3.3 Conflitos entre as duas pastas

Comparar fingerprints e identidades antes do replay. A mesma evidência visual
em `ok_archive` e `ng_archive` pode refletir erro de registro,
reclassificação legítima ou contexto diferente. Registrar conflito para
investigação; **não escolher o rótulo automaticamente** e não apagar o exemplo.

## 4. Motor de replay — mesma análise, sem AOI física

```text
PNG original da AOI
    ↓
decodificação + verificação de integridade
    ↓
localização das barras azul/vermelha
    ↓
extração integral de GABARITO e TESTE
    ↓
OCR Board/Parts/Value + normalização da categoria
    ↓
detect_anomalies / EpicenterExtractor
    ↓
MoEOrchestrator.inspect + especialistas físicos SEM KNN
    ↓
resultado local SIDE/TOP/MID
    ↓
se multilight: fuse_multilight(analyses, category)
    ↓
resultado final comparado ao rótulo do arquivo/evento
```

Reutilizar/extrair a lógica de
`src/services/screen_monitor.py` (barras, extração, OCR),
`src/core/adhesive_multilight_analysis.py` (análise por luz),
`src/core/multilight_fusion.py`, módulos físicos/semânticos e memória
instalados no `main.py`. **Não manter uma segunda implementação
simplificada** de inspeção específica do startup.

O replay recebe imagens **do disco**, nunca manda
`PRESS_LEFT/PRESS_RIGHT/PRESS_DOWN` ao Windows XP, não ativa captura
MSS, não abre servidor/receptor da AOI, não envia `PRESS_0/PRESS_1`, não
incrementa métricas de produção e não usa estados de botões do operador.

**Atualização obrigatória aprovada pelo operador — replay cego à memória:**
o replay NÃO deve instanciar nem consultar KNN, dataset de exemplos,
protótipos, vizinhos, similaridades persistidas ou memórias episódicas.
A mesma imagem pode estar no dataset: sua presença não pode tornar a
aprovação mais fácil. A decisão deve se apoiar exclusivamente nos detectores,
na geometria, nas regras físicas e nos modelos visuais não baseados na busca de
amostras. O score da memória é zero; não há `memory_veto`, `best_match` ou
`memory_priority`. A regra vale para SIDE histórico e para a futura fusão
multilight, inclusive `INVERTIDO`, cuja extensão de análise consultava
novamente o KNN.

A CNN ou outro modelo de parâmetros previamente treinados continua elegível
se não procurar a imagem atual em uma base de exemplos durante a inferência;
não se permite consulta/reuso de correspondência exata com o acervo.
O replay jamais chama `save_label`, `DatasetManager.save_sample`, recarga
da memória ou arquivamento. Qualquer tentativa de acesso KNN é erro de
isolamento e invalida a execução. A operação normal mantém sua memória
habilitada; o bypass é exclusivo do modo diagnóstico offline.

**Cuidado do quadrado menor:** usar sempre o gabarito/teste **completos** da
área da AOI, com análise física local/contextual **e da área de inspeção completa**.
A terceira escala de *memória* full-frame NÃO é consultada nesse replay;
a exigência de imagem completa diz respeito aos detectores visuais,
não a uma comparação com assinaturas antigas. Não validar só a ROI pequena:
o defeito pode estar fora dela.

**Não há autocorrespondência com o dataset neste replay:** a consulta
KNN/memória é impedida por construção. Ainda assim, usar as imagens
históricas para calibrar repetidamente os mesmos limiares pode superajustar
o resultado ao acervo; manter também uma validação independente/holdout
quando houver dados qualificados.

## 5. Sequência de inicialização — bloqueio antes da produção

```text
python main.py
    ↓
QApplication / tela mínima de validação (sem produção)
    ↓
carrega configurações e especialistas físicos SEM carregar KNN/memória
    ↓
inventaria 100% dos arquivos OK/NG e manifestos
    ↓
replay completo com progresso real
    ↓
compara todos os resultados ao histórico humano
    ↓
gera relatório final
    ↓
100% PASSOU?
    ├── SIM → instancia/libera ControlPanel e serviços operacionais
    └── NÃO → mantém SOMENTE tela de diagnóstico / manutenção restrita
```

A janela `ControlPanel` e seus receptores/hook operacionais não devem ser
criadas/ativadas antes da aprovação, porque a construção do painel atual
pode preparar serviços de rede e captura. A tela de validação deve ser
independente, responsiva e mostrar `N/T`, etapa, arquivo/evento atual e
contagem de falhas. Cancelar o teste **não** libera a produção.

O gate deve ser executado **a cada partida**, mesmo sem alterações de código
ou arquivos. Hash/cache pode servir para inventário/integridade, **não
substituir a reanálise completa**. Não permitir `skip`, `force pass` ou
atualização automática do rótulo em ambiente produtivo.

**Diagnóstico/qualificação inicial:** antes de ativar o bloqueio definitivo,
executar uma fase de levantamento apenas para obter o conjunto real de falhas
do acervo antigo. Essa fase não é considerada gate aprovado nem autoriza
suprimir casos; depois da qualificação, **o modo exigido é fail-closed**.
Correções devem ser testadas e validadas antes de liberar produção. Qualquer
modo de manutenção deve manter a AOI desacoplada.

Para acervo grande, processar incrementalmente em worker separado da thread
da UI, com uso controlado de RAM e progresso por arquivo/evento. Nenhum caso
deve ser descartado por timeout silencioso.

## 6. Relatório obrigatório

Relatório no diretório próprio de diagnósticos, separado dos archives/dataset,
em **JSON estruturado e relatório legível**. Pode registrar horário, hash do
código/configuração, versão do contrato, total de arquivos, total de eventos,
contagem de rótulos e distribuição por categoria/iluminação.

Cada caso deve expor, no mínimo:

```text
status = PASSOU / REGRESSÃO / INVÁLIDO / INCOMPLETO / CONFLITO
source_path
expected_label = OK / NG
detected_category
lighting_mode(s)
event_id (quando existe)
verdict por iluminação
verdict final
operator_review_required
score, confidence, physical_score, fusion_rule
diagnóstico de OCR/recorte (quando relevante)
erro/stack trace resumido (quando houver)
```

Exemplo ilustrativo (não é resultado de execução):

```text
GATE DE INICIALIZAÇÃO ODIN

OK: 14/14 casos aprovados
NG: 17/18 casos aprovados
TOTAL: 31/32 casos aprovados

STATUS: BLOQUEADO
FALHA: ng_archive/2026-10-01_..._FALTANDO.png
TIPO: REGRESSÃO
ESPERADO: DEFEITO REAL
OBTIDO: FALHA FALSA
ILUMINAÇÃO: SIDE
```

Falha de leitura, divergência, conflito e ausência de cobertura precisam estar
visíveis. A origem dos dados sensíveis de produção não deve ser enviada
automaticamente para internet/terceiros. O relatório nunca deve modificar
as imagens e rótulos humanos.

## 7. Arquitetura proposta (separação de responsabilidades)

Sugestão de módulos para a **futura implementação**, sujeitos a refinamento
durante a primeira etapa:

```text
src/services/startup_regression/
    archive_inventory.py       # enumera arquivos, integridade, classificação
    archive_manifest.py        # escreve/lê vínculos de peças multilight
    replay_preprocessor.py     # reutiliza extração AOI + OCR existente
    inspection_runner.py       # chama motor real sem UI/rede
    verdict_contract.py        # regras de aprovação mono/multilight
    regression_runner.py       # orquestra replay e progresso
    regression_report.py       # relatório JSON/humano

src/ui/startup_regression_window.py
    # janela restrita de progresso, aprovação e bloqueio

tests/test_startup_regression_*.py
    # unitários, integração e arquivos antigos/novos
```

`main.py` deve apenas coordenar a validação antes da construção do painel,
sem absorver parser, pré-processamento ou regras de negócio. O arquivo de
evidência OK/NG continua separado de
`public/dataset/{anomalia,nao_anomalia}`.

Nenhuma alteração do `agente_industrial_xp.py` é necessária para replay
offline de imagens arquivadas.

## 8. Plano incremental de implementação

**Regra de execução:** trabalhar em **uma etapa por vez**, apresentar testes,
evidências e mudanças; avançar somente após a aprovação explícita do usuário.

### Etapa 1 — Inventário e diagnóstico do acervo existente

- Confirmar nomes, formatos, dimensões e legibilidade de todos os PNGs
  reais de OK/NG (sem excluir arquivos antigos).
- Identificar quais são LEGADO SIDE, quais já são SIDE/TOP/MID e quais possuem
  metadados confiáveis para associação por evento.
- Medir quantas imagens há por pasta/categoria/iluminação.
- Identificar conflitos, arquivos corrompidos, OCR indecifrável e possíveis
  pares não relacionáveis sem inventar vínculo.
- Gerar apenas relatório de inventário; sem abrir gate e sem alterar PNGs.

**Aceite:** relatório reprodutível do acervo real, com pendências explícitas.

#### Implementação da ferramenta de Etapa 1 — 08/10/2026

**Implementado no código:** inventariador independente, somente leitura.
**Pendente para aceitar a Etapa 1:** rodá-lo com as duas pastas reais do
computador da fábrica e examinar o relatório. Nenhuma contagem de arquivos
reais foi presumida com base no print.

Arquivos criados:

```text
src/services/startup_regression/archive_inventory.py
src/services/startup_regression/archive_inventory_report.py
src/services/startup_regression/__main__.py
tests/test_startup_regression_inventory.py
.github/workflows/startup-regression-inventory.yml
```

**Comando no PowerShell do computador com o código atualizado:**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression
```

A raiz do projeto é detectada automaticamente. Caso a execução seja iniciada
fora da pasta do projeto, usar:

```powershell
python -m src.services.startup_regression --root "C:\visionx-neural-main"
```

O comando lê `public/ok_archive/` e `public/ng_archive/`, inclusive arquivos
em subpastas; **não altera nenhum PNG, não renomeia, não cria arquivos nas
pastas e não toca no dataset**. A saída fica em:

```text
reports/startup_regression/inventory_<data>.json
reports/startup_regression/inventory_<data>.txt
```

A pasta de relatórios é ignorada pelo Git via `.gitignore`. O relatório JSON
contém o inventário completo e os fingerprints; o TXT traz resumo e pendências
para leitura rápida. Pedir esses relatórios para concluir o diagnóstico real.

Nesta etapa são verificados: arquivos descobertos, assinatura e estrutura
PNG/CRC, decodificação OpenCV, dimensões, hash do arquivo, fingerprint dos
pixels, categoria **sugerida apenas pelo nome**, iluminação explícita ou
SIDE legada, repetição exata de imagens, conflitos entre OK e NG, além de
manifestos reais quando existirem.

**Uma imagem sem manifesto não é agrupada com outras por horário/nome.**
Um único PNG deduplicado pode ser citado por dois eventos ou até mais de uma
iluminação no manifesto; a ferramenta mantém vínculos separados. Manifesto
corrompido ou incompleto nunca ganha vínculo parcial válido.

**Limite explícito:** o inventário não executa OCR, não confirma categoria
por leitura da tela, não chama o MoE/KNN nem produz qualquer veredito da IA.
Itens cuja categoria só possa ser recuperada pelo OCR ficam pendentes para
a Etapa 2. A ferramenta **não bloqueia a inicialização**, não altera
`main.py`, não treina a memória e não aciona o Windows XP.

**Próxima ação:** executar no PC real, trazer os arquivos JSON/TXT e revisar
as pendências. Somente com aceite do inventário avançar para a Etapa 2.

### Etapa 2 — Runner de replay monoimagem SIDE

- Separar pré-processamento de screenshot AOI da interface Qt.
- Reutilizar recorte completo, OCR, pipeline físico/semântico e políticas
  da operação real **sem qualquer consulta KNN ou memória episódica**.
- Executar casos antigos `OK` e `NG` em leitura e registrar veredito,
  revisão e divergências.
- Garantir que o quadrado menor não limita o campo de evidências.
- Qualificar as regressões pré-existentes sem mascará-las.

**Aceite:** todos os PNGs legados são processados ou falham com erro explícito;
replay não altera dataset/arquivos nem aciona hardware.

#### Diagnóstico real e telemetria adicional — 08/10/2026

Primeira execução industrial do `side_replay` (119 SIDE históricos):

- `OK`: 0/102 aprovados; 101 classificados como `DEFEITO REAL` e
  1 inválido por OCR ausente (imagem de 1920×1080).
- `NG`: 17/17 aprovados; todos `DEFEITO REAL`.
- `KNN`: explicitamente desabilitado em todas as observações.
- **Conclusão operacional:** replay físico ainda NÃO distingue
  adequadamente OK e NG; não ativar o gate de inicialização. Não ajustar
  limiares para fazer o histórico passar sem investigar os especialistas.

Foi adicionada a telemetria de inspeção **somente ao executor de replay**:
`src/services/startup_regression/replay_telemetry.py`.

O relatório agora traz para **cada caso**:

```text
telemetry.schema = visionx.side_replay_telemetry.v1
telemetry.reason / fusion_rule / dominant_engine / final_score / cutoff
telemetry.engines[]:
   id, label, active, triggered, selected
   raw_score, effective_score, threshold, final_influence, summary
telemetry.physical_readings:
   leituras escalares presentes em detail (sem inventar campos ausentes)
telemetry.geometry:
   dimensões completas da imagem de referência e teste
   global_box (AOI)
   old_epicenters / selected_epicenters / raw_anomaly_boxes
   final_bounding_box / specialist_boxes
   fonte coordenadas = AOI_EXTRACTED_IMAGE_XYWH
diagnostics:
   por categoria + rótulo
   por regra de fusão + status
   por motor dominante + status
   motores disparados nas regressões (não mutuamente exclusivos)
```

O `build_lighting_context` do motor existente é executado uma vez e
compartilhado com a inferência; coletar telemetria **não** executa
inferência duas vezes e **não** modifica a decisão. O registro serializa
somente campos seguros, evitando máscaras binárias, imagens e assinaturas
de memória. O modo sem KNN continua obrigatório; dados ausentes são `N/D`.

Com o repositório atualizado, executar novamente:

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

Coletar os novos `side_replay_*.json` e `side_replay_*.txt` para comparar
pontuação dos especialistas e a seleção de região em **OK e NG** da mesma
categoria. **Etapa 2 ainda aberta; Etapa 3 e gate bloqueante não iniciados.**

#### Implementação da Etapa 2 — motor físico sem memória

**Regra irrevogável no gate de regressão:** replay cego aos exemplos antigos.
O KNN não é instanciado; a extensão de fusão física e a rota específica
`INVERTIDO` recebem `_replay_without_memory=True` e não computam nem
consultam assinaturas armazenadas. A análise física continua reutilizando o
`detect_anomalies → EpicenterExtractor → MoEOrchestrator`.

```text
src/services/startup_regression/inspection_runner.py
src/services/startup_regression/side_replay.py
tests/test_startup_regression_replay.py
.github/workflows/startup-regression-side-replay.yml
```

**Comando no PowerShell com o código `central` atualizado:**

```powershell
cd "C:\visionx-neural-main"
python -m src.services.startup_regression.side_replay
```

**Saída:** `reports/startup_regression/side_replay_<data>.json` e
`side_replay_<data>.txt`. O comando avalia **somente os 119 screenshots
históricos sem sufixo**; as 90 imagens com `SIDE/TOP/MID` explícitos
aguardam o replay por evento na Etapa 3.

A leitura utiliza o mesmo `ScreenMonitor.process_external_image` que a
produção para encontrar barras, recortar gabarito/teste completos e recuperar
OCR. No replay, `_replay_no_debug=True` impede a gravação de PNGs temporários
em `public/debug_crop`. Um OCR indisponível/incompleto, divergência de
categoria ou erro físico retorna `INVALIDO`, jamais aprovação.

Para garantir o isolamento, `SideInspectionRunner` rejeita resultados que
declarem motor KNN ativo, memória consultada ou peso KNN diferente de zero.
A CPU executa os mesmos especialistas físicos, sem carregar os exemplos do
dataset. A avaliação não abre o painel, não captura MSS, não envia comandos XP,
não ensina a IA e **ainda não bloqueia** a inicialização operacional.

O resultado real do PC deve ser revisado antes de declarar esta etapa
aprovada. Falhas de Tesseract ou dependências também devem aparecer no
relatório e ser corrigidas, não ignoradas.

**Validação técnica:** workflow `Startup Regression SIDE Replay`,
execução `37780008623` em GitHub Actions Windows,
**22 testes executados, todos aprovados**, inclusive MoE físico real em
`FALTANDO` e `INVERTIDO` com KNN bloqueado por teste. Não equivale a
119/119 imagens reais aprovadas: essa medição depende dos PNGs locais.

**Observação operacional:** quando houver divergência ou falha, o comando
termina com código de saída 1 **depois de salvar os relatórios**. Isso é
correto para scripts de diagnóstico e **não** bloqueia o `main.py` nem
altera o ODIN em operação.

### Etapa 3 — Manifesto e replay por evento multilight

- Evoluir persistência futura de OK/NG para associar
  `event_id + OCR + expected_label + SIDE/TOP/MID`, com hashes/caminhos
  realmente persistidos e commit atômico do manifesto.
- Suportar screenshots com sufixos `SIDE/TOP/MID` e reconhecer
  arquivos históricos sem sufixo como SIDE.
- Reprocessar separadamente as três iluminações e passar pela fusão real,
  preservando política especializada de adesivo.
- Recusar sessão incompleta/ambígua sem tentar fusão artificial.

**Aceite:** caso NG `SIDE OK / TOP NG / MID OK` passa se a fusão resultar NG;
um evento de três imagens vale uma única decisão.

### Etapa 4 — Gate bloqueante e tela de inicialização

- Carregar modelos/memória **antes** de autorizar o painel.
- Exibir somente progresso/diagnóstico durante replay.
- Liberar `ControlPanel` apenas após 100% de aprovação.
- Impedir abertura operacional se houver regressão, revisão, arquivo inválido,
  manifesto incompleto, conflito ou ausência de cobertura.
- Persistir relatório mesmo em falha/cancelamento.

**Aceite:** testes instrumentados comprovam que nenhum receptor, captura ou
comando XP fica ativo quando o gate falha.

### Etapa 5 — Regressões automatizadas e validação na fábrica

- Unitários de parser legados, classificação, deduplicação/identidade e
  comparação de vereditos.
- Integração com screenshots completos sintéticos/anonimizados e memória real
  instalada, sem hardware XP.
- Cenários OK/NG, revisão, imagem corrompida, OCR ausente, categoria errada,
  arquivo duplicado, rótulos conflitantes, três luzes completas/incompletas.
- Testes de inicialização aprovada/bloqueada e idempotência do relatório.
- Medir tempo e RAM em acervo grande; comprovar que a UI não congela.
- Fazer diagnóstico inicial do acervo operacional e corrigir regressões,
  **sem remover a obrigatoriedade dos casos conhecidos**.

**Aceite final:** 100% do acervo qualificado passa no ambiente alvo e o
bloqueio opera corretamente diante de regressões injetadas.

## 9. Itens não abrangidos e regras de segurança

- Não instalar nova IA, retreinar CNN/KNN ou alterar a fusão só para passar
  o gate; mudanças no motor exigem tarefa e validação próprias.
- Não reclassificar automaticamente imagens humanas ao observar divergência.
- Não converter casos de revisão em OK ou NG por conveniência.
- Não supor que três imagens NG precisam ter três vereditos NG isolados.
- Não comparar apenas o quadrado menor nem reutilizar captura visual de outra
  peça.
- Não tratar 100% no acervo conhecido como garantia estatística de precisão
  em peças novas.
- Não mover/apagar arquivos e não interferir no processo físico da AOI.
- Não permitir que falha da tela de validação leve à abertura normal por
  fallback.

## 10. Estado e próximo passo

- **Etapa 1 concluída e aprovada:** inventário real em 08/10/2026,
  209 PNG válidos (119 SIDE históricos; 90 multilight explícitos sem manifesto).
  192 OK e 17 NG; nenhuma duplicata pixel a pixel. Os vínculos das 90
  iluminações multilight ficam para qualificação na Etapa 3.
- **Etapa 2: primeiro replay real diagnosticado** (17/119 aprovados,
  101 regressões e 1 OCR inválido); telemetria física detalhada adicionada
  e aguardando novo relatório industrial. KNN permanece desabilitado.
- **Etapas 3 a 5 e gate bloqueante: NÃO implementados**.
- O `main.py`, decisão em produção, dataset de aprendizado e
  `agente_industrial_xp.py` permanecem inalterados.
- **Próxima ação:** executar `python -m src.services.startup_regression.side_replay`
  no PC da fábrica, enviar os relatórios `side_replay_*.json` e
  `side_replay_*.txt` e qualificar cada divergência, sem alterar rótulos.
  Não iniciar a Etapa 3 antes do aceite explícito.


### 08/10/2026 — Verificador CNN FALTANDO v2 do acervo conhecido (isolado)

Foi criado `src/scripts/replay_faltando_cnn_v2.py` para julgar offline
todos os `FALTANDO` do `ok_archive` e `ng_archive` usando pesos
CNN v2 já treinados, com SIDE legado e SIDE/TOP/MID. Os resultados
por foto e evento ficam em
`reports/faltando_neural/replays/archive_v2_*/`.
**Esse verificador NÃO implementa o gate bloqueante de inicialização**,
não usa KNN, não altera modelo e não reconfigura Produção.
Seu sucesso integral significa somente não regressão nos casos
conhecidos, porque o próprio acervo também alimentou o treino v2.
A avaliação independente continua necessária para liberação
automática de novas peças.

### 08/10/2026 — CNN FALTANDO v2 na análise normal, gate geral independente

O replay separado `replay_faltando_cnn_v2.py` processou com
sucesso todas as **117 imagens FALTANDO / 67 eventos** conhecidos
(10 NG, 107 OK; zero falhas). A CNN v2 foi incorporada ao fluxo
normal da categoria FALTANDO, mas **não equivale à implementação do
gate geral de regressão bloqueante na inicialização**.
A regressão histórica passa, porém casos inéditos não foram
testados. No Modo Produção, AUTO-OK por CNN experimental
permanece bloqueado e requer operador. Nenhum comportamento
de replay isolado foi alterado.
