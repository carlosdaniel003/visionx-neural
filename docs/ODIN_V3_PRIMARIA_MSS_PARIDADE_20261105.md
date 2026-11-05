# ODIN AOI — CNN V3 principal e reconhecimento MSS do acervo (05/11/2026)

## Decisão e limites

Por autorização explícita do operador, a CNN AOI V3 (época 3) passa a ser
**o classificador visual principal nas categorias que anteriormente usavam
a CNN FALTANDO V2**: FALTANDO, EMBORCADO, INVERTIDO e DESLOCADO.
Aplica-se aos modos Teste (inclusive MSS), Sombra e Produção.

A substituição é feita na construção do `MoEOrchestrator` em
`main.py`: o slot antes ocupado pelo `FaltandoCNNLive` recebe
`AOIQualityV3Primary`. Assim, os casos NOVOS usam V3 sem executar a
inferência V2. O roteador KNN de par humano exato continua antes do
classificador e preserva a última decisão física confirmada.
A CNN V3 experimental também continua atuando como contraprova NG nas
rotas de especialistas em categorias fora do escopo da antiga V2.
MUITO ADESIVO preserva o especialista físico próprio, que nunca foi
substituído pela V2.

Os scripts, checkpoints e testes V2 não são apagados: são históricos/
diagnósticos e não são usados como inferência normal do painel. A
auditoria histórica `Validar acervo OK/NG` ainda verifica a CNN V2
antiga e a KNN; NÃO mede o checkpoint V3 principal.

### Ativação e falha

O checkpoint permanece o snapshot época 3 SHA-verificado pelo
`AOIQualityV3ProductionTrial`, com o ponteiro local
`reports/aoi_quality_cnn_v3/production_v3_trial.json`.
A ativação já foi confirmada na estação pelo operador em 05/11.
O mesmo ponteiro, criado originalmente como `ng_veto_only`, continua
servindo de **autorização de carregamento do checkpoint experimental**
para a nova rota visual principal. A primazia está definida no código
do painel, não no nome histórico do ponteiro.

Se o ponteiro, modelo, SHA ou voto estiverem ausentes/adulterados,
a nova rota retorna `REVISÃO OBRIGATÓRIA` e status `ERROR`;
**não** faz fallback silencioso para a V2. A política binária
interrompe o envio se o contrato V3 ficar inválido.

### Segurança de decisões XP

**A V3 ainda não está homologada para liberar OK automaticamente.**
O checkpoint época 3 perdeu 1 NG real entre 14 exemplos humanos
revisados e 3 NG entre 219 exemplos na validação, sem holdout
independente por peça. Portanto:

- Um veredito visual V3 `FALHA FALSA` em Teste ou MSS é apresentado
  como inferência, não como confirmação física do rótulo.
- Em Produção, um **OK isolado da V3 não autoriza 0/OK**. A política
  exige contraprova humana verificada conforme os contratos existentes.
- Um NG V3 pode gerar `1/NG` no ciclo de produção completo, por
  política binária; isso é rejeição operacional de segurança,
  **não diagnóstico físico garantido**.
- A rota antiga de Produção não pode contornar esse bloqueio.
- A lógica de pausa, identidade do evento, envio único e ACK XP não foi
  alterada. Não conectar o XP antes de verificar numa estação isolada.

O objetivo continua ser 0/1 binário, mas o estágio atual **não
certifica produção autônoma 0/1 sem risco de liberar NG físico**.

## MSS e acervo OK/NG

O scanner `ArchiveVisualPresenceIndex` já usado pelo XP
é agora chamado após cada captura MSS válida, em worker próprio.
O frame MSS original completo é consultado nas luzes SIDE/TOP/MID,
pois uma imagem mostrada na tela pode ter metadados de luz
diferentes dos do arquivo. A busca mostra:

- `EXACT_FRAME`: SHA-256 dos **pixels da tela completa** idêntico
  ao PNG do acervo, mesma categoria/iluminação do índice;
- `SIMILAR`: semelhança de descritores do frame, **não igualdade**;
- `AMBIGUOUS` ou conflito de OK/NG: não inferir verdade física;
- `NO_SIMILAR`: nenhum candidato na busca atual.

**FRAME EXATO ≠ PAR AOI EXATO.** O `KNOWN_KNN/EXACT_PAIR` exige
igualdade RGB dos dois recortes gabarito/teste e evidência humana.
O `ok_archive` e o `ng_archive` podem conter rótulos de pasta
provisórios. Reconhecer uma imagem OK do acervo **não deve reescrever
o veredito V3 nem liberar 0/OK** sem conferência física.

A abertura de um PNG dentro de um visualizador pode acrescentar bordas,
mudar escala/resolução ou modificar pixels do frame que o MSS capturou:
não prometer `FRAME EXATO` nesse caso. Uma repetição real da mesma
tela, pixel-idêntica, é reconhecível. O debug MSS passa a registrar
`cnn_v3`, `archive_visual_presence`, rótulo, luzes e hash da tela,
sem ocultar a distinção entre quadro completo e par AOI.

## Aprendizado e regressões

A memória humana KNN mantém suas correções persistentes.
O treino incremental automático **V2** não recebe novos casos
do painel operando com V3: a fila recebe a identidade explícita
`v3_primary_runtime` do orquestrador, inclusive em julgamentos
`KNOWN_KNN`. O marcador de análise `cnn_v3_primary` também
bloqueia o treino V2. A elegibilidade não depende de um ponteiro
global de outra execução, preservando os testes e ferramentas
históricas V2 isoladas. Não há treinamento online V3 homologado
nesta alteração. O retreinamento de V3 continua offline, com
validação explícita antes da troca de pesos.

Os testes do CI exercitam o roteamento visual V3, erros de checkpoint,
ausência de fallback V2, painel V3, integridade do XP, consulta do mesmo
scanner em MSS, conflito e exatidão do frame. Testes de contrato não
substituem o replay das imagens reais e uma avaliação física cega.

## Atualização e teste supervisionado local

Fechar ODIN e manter o XP desconectado do envio automático:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
python -X utf8 main.py
```

Verificar no Modo Teste/MSS uma imagem OK e uma NG conhecidas, e
comparar o frame MSS salvo com o PNG original (mesmos pixels/escala).
Verificar o status V3, a rota KNN, `EXACT_FRAME` versus
`EXACT_PAIR` e a ausência de comandos XP durante o teste.
Depois testar SIDE/TOP/MID com peças **fisicamente confirmadas** e
as duas classes de erro. **Não habilitar operação autônoma de
liberação 0/OK com base apenas no percentual V3**.

Para suspender a V3, parar o ODIN e usar o comando de desativação
documentado em `AOI_V3_PRODUCTION_NG_VETO_TRIAL_20261104.md`.
Após desativar, esta versão nova **não retorna automaticamente à V2**;
os casos novos ficam bloqueados em revisão por falta de checkpoint.


## 05/11/2026 — Correção de proveniência MSS e rótulos V3

O evento MSS `8dba014902844f52be97ad1c69105d84` demonstrou
**CNN V3 primária ativa** e score NG bruto de 1,0 sobre uma imagem
reexibida pelo operador a partir do `ok_archive`. O KNN C apontou
`2026-10-02_1029_FALTANDO.png` como vizinho B de 0,7449; esse
valor NÃO demonstra igualdade dos pixels nem generalização da V3.

Foram corrigidos problemas sistêmicos, sem cadastrar nomes de PNG como
exceções e sem alterar limiar ou pesos:

- A captura MSS agora consulta a memória com o **frame imutável
  associado ao event_id**, antes de serializar o relatório. O JSON
  passa a informar `PENDING`, `UNAVAILABLE` ou o resultado obtido,
  sem permanecer silenciosamente vazio. A conclusão posterior do
  worker atualiza o relatório do mesmo evento.
- O mesmo acervo usado pelo XP continua comparando SHA de todos os
  pixels do frame. Quando um PNG do acervo está **redimensionado e
  reexibido** em uma janela, `archive_rendered_source_match` pode
  registrar sua origem por correspondências ORB, alinhamento afim
  robusto e fidelidade de pixels em tiles, usando candidatos KNN
  efetivamente recuperados, não lista fixa de arquivos. Alteração
  focal, origem fora das pastas, categoria incorreta e conflito de
  rótulos impedem afirmação de correspondência.
- Fonte visual reexibida recebe `JÁ VI • ARQUIVO REEXIBIDO`, com
  caminho e rótulo da pasta; NÃO recebe falsamente
  `EXACT_PAIR` de KNN nem `EXACT_FRAME` de pixels originais.
- Painéis de especialistas, influência, KNN e rede neural recebem o
  motor efetivamente utilizado. A seção legada `cnn_v2` deixa de
  ser serializada nos eventos classificados apenas pela V3.
  Sondas Grad-CAM V2 não são exibidas como explicação da V3.

A identificação visual continua **diagnóstica**: não altera
automática e silenciosamente um NG da V3 para OK porque o rótulo
da pasta não equivale a inspeção física independente. Assim, o
modelo ainda pode errar em um arquivo conhecido: reparar o
classificador exige novo conjunto validado e treino dos pesos.

### Identificar exatamente onde ocorre a divergência

No computador onde estão o checkpoint e o PNG histórico:

```powershell
python -X utf8 -m src.scripts.audit_v3_archive_image --root . --archive "public/ok_archive/2026-10-02_1029_FALTANDO.png"
```

O auditor **somente lê** o PNG original e o checkpoint V3 já ativo,
executando o extrator AOI e a CNN da estação. Reporta `original`,
`original_model_vs_folder`, os dois hashes dos recortes e
`v3_ng_score_uncalibrated`. Não treina, troca pesos nem
rotula de novo a imagem.

Opcional: exportar a imagem MSS copiada do ODIN para PNG e adicionar
`--mss caminho/captura.png` para comparar extração, geometria
e predição nas duas imagens. O auditor não trata o nome do arquivo
como uma correção humana.

- Se o original também retorna NG, o problema é **do modelo V3
  nessa amostra**, não apenas do MSS. Treinar outro candidato e
  demonstrar NG→OK/OK→NG em validação independente ainda será
  necessário; esta alteração não declara o falso NG resolvido.
- Se o original retorna OK, mas a captura MSS retorna NG, a
  reexibição/extração introduziu mudança visual relevante.
  Corrigir a normalização de entrada requer as duas imagens e
  conferência da geometria capturada.

**Critério de sucesso:** não confundir correção de texto, memória de
arquivo, inferência do modelo e autorização de 0/XP. A melhoria
de precisão real ainda depende de repetição e verificação local.


## 05/11/2026 — Diagnóstico resolvido: divergência de geometria MSS

O auditor somente leitura do PNG de arquivo
`public/ok_archive/2026-10-02_1029_FALTANDO.png` retornou:

- `original_model_vs_folder=AGREES`;
- `verdict=FALHA FALSA`, CNN V3 `ng_score=0.0022295827511698008`;
- `reference_shape=(459,521,3)`, `test_shape=(459,522,3)`;
- checkpoint `fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce`.

O debug MSS anterior, com o mesmo checkpoint V3 e decisão NG=1.0,
mostrou `sample_crop=(600,521,3)` e `test_crop=(459,522,3)`.
O extrator antigo `ScreenMonitor._extract_photo` às vezes devolvia
o **strip inteiro**, com altura limitada a 600 px, quando não achava
borda cinza sustentada. A falta da borda NÃO prova que a foto
real tenha 600 px. Isso pode acrescentar fundo externo ao gabarito,
gerando outra entrada à CNN e quebrando identidade dos recortes KNN.

### Correção estrutural (sem regra de classe / arquivo)

- O novo módulo puro `src/services/aoi_paired_photo_geometry.py`
  centraliza o alinhamento de duas fotos da mesma interface AOI.
- `ScreenMonitor` registra limites geométricos verificáveis das
  fotos extraídas em cada barra azul/vermelha.
- Apenas uma borda inferior **cinza sustentada** é considerada
  âncora confiável. Um limite calculado pela última linha não-cinza
  ou pelo fim do strip não serve como prova.
- Se uma das duas fotos não tiver borda e a sua irmã tiver, o
  sistema reextrai somente a foto incerta a partir do alinhamento
  relativo às duas barras da mesma interface (topo e base medidos).
  Não utiliza nome de arquivo, rótulo OK/NG, limiar CNN, OCR da
  categoria nem a altura fixa 459 px.
- Quando nenhuma das duas fotos tem borda confiável, o sistema
  **não inventa** uma correção; informa `BOTH_UNBOUNDED`.
- O mesmo método é aplicado na captura MSS e em
  `process_external_image`, que produz os pares de treino/auditoria.
  Não há dois algoritmos divergentes.
- O debug MSS preserva `transport.crop_geometry` com status e
  dimensões antes/depois do alinhamento. A ferramenta de auditoria
  V3 também inclui esse bloco.

### Resultado e limite de comprovação

A regressão sintética reproduz o gabarito 600px e teste 459px
e verifica que o algoritmo devolve recortes de 459px **a partir
da borda da foto irmã**. Isso demonstra correção de extração,
não a predição final do screenshot real: a imagem MSS do evento
ainda não foi submetida ao replay após essa mudança.

Após `git pull origin central`, repetir a mesma imagem no MSS
com XP desconectado, copiar debug e verificar
`Alinhamento geométrico AOI: ALIGNED_FROM_SIBLING_BOUNDARY`
quando houver a borda irmã válida. Comparar hashes/tamanhos com o
auditor do PNG e confirmar `FALHA FALSA` no resultado real.
Se o MSS estiver reescalado, o hash ainda pode mudar;
`ARQUIVO REEXIBIDO` não é `PAR AOI EXATO`.
Não promover auto-OK apenas com base nesse replay.
