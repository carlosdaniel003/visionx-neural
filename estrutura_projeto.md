# Estrutura do Projeto: VisionX Neural

**Módulos Existentes:**
- `src/config/settings.py`: Centralização de todas as variáveis de ambiente, caminhos e constantes mágicas.
- `src/services/ng_image_archive.py`: Arquivo visual opcional de decisões finais NG em fila de background, independente do dataset e da memória KNN.
- `src/services/ok_image_archive.py`: Arquivo visual opcional de decisões humanas OK em fila de background, usando a mesma evidência de `Copiar imagem`.
- `src/services/image_archive_naming.py`: Formato compartilhado de nomes dos arquivos visuais OK/NG.

**Fluxos Principais (Planejados):**
1. **Pilar 1 (Extrator Visual):** Monitoramento contínuo da tela usando `mss` para detectar a janela da AOI.
2. **Pilar 2 (Cérebro Comparativo):** Rede siamesa avaliando propostas de defeitos.
3. **Pilar 3 (Display HUD):** Janela transparente sobreposta sinalizando as anomalias detectadas.
4. **Pilar 4 (Active Learning):** Salvamento local de recortes aprovados/rejeitados em `public/dataset/`.

**Dependências Base:**
- PyTorch (Redes Neurais)
- OpenCV (Visão Clássica / Tratamento de Imagem)
- mss (Captura de tela ultrarrápida)
- PyQt6 (Criação do HUD transparente)

**Arquivo visual NG opcional:**
- Toggle **ativado por padrão** em toda inicialização do ODIN. O operador pode desativá-lo manualmente durante a sessão.
- Ativado: cada evento válido do Windows XP pode gerar **no máximo uma** evidência final `NG` em `public/ng_archive/`, usando exatamente o mesmo frame completo disponibilizado pelo botão `Copiar imagem`.
- Nome: `YYYY-MM-DD_HHmm_CATEGORIA.png`, por exemplo `2026-10-02_0811_FALTANDO.png`. O formato usa ano, mês, dia, hora e minuto, seguido da categoria normalizada.
- Para o arquivamento XP, a fonte continua sendo `src/services/network_xp_frame.py`, que valida que o `event_id` do frame preservado é o mesmo do diagnóstico atual. O botão genérico `Copiar imagem` usa essa mesma evidência quando a origem é XP.
- Não existe fallback para `current_ng` ou outro recorte. Se o frame XP do evento atual não estiver disponível, nenhuma imagem substituta é arquivada.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.
- Deduplicação obrigatória por `event_id`: o mesmo evento XP nunca pode gerar duas imagens de arquivo, mesmo se o `PRESS_1` enviado pelo VisionX reaparecer pelo hook global do XP como `CMD_NG`.
- O arquivamento só é permitido enquanto existe uma captura de rede ativa, com análise ativa e categoria AOI não vazia.
- `SEM_CATEGORIA` não é um nome de arquivo válido para o fluxo automático de evidências NG. Se a categoria já tiver sido limpa, o evento não deve ser salvo novamente.


**Arquivo visual OK opcional:**
- Existe um segundo toggle **`Salvar imagens OK`**, exibido imediatamente abaixo de **`Salvar imagens NG`**.
- O toggle inicia **ATIVADO por padrão** em toda abertura do ODIN e o operador pode desativá-lo durante a sessão.
- Visualmente, o bloco OK deve manter o mesmo layout, dimensões, tipografia, hover, focus e estado checked do bloco NG.
- Quando ativado, cada julgamento humano final `OK` pode gerar **no máximo uma** evidência em `public/ok_archive/`.
- Julgamentos humanos aceitos: botão/atalho do ODIN (`source="button"`) e teclado físico do XP (`source="xp_keyboard"`).
- Decisão automática de Produção (`source="auto"`) **não** gera arquivo OK.
- A imagem salva deve ser **exatamente a mesma evidência completa resolvida por `Copiar imagem`** para aquele evento.
- O contrato compartilhado de evidência fica em `src/services/capture_evidence.py`, por meio de `current_copy_image_snapshot()` e `current_copy_image_event_id()`.
- O arquivo OK aceita tanto captura recebida do **Windows XP** quanto captura local **MSS**, desde que exista análise ativa, `event_id` válido e categoria AOI válida.
- Uma captura local MSS nunca pode usar como fallback um frame XP anterior.
- Não usar `current_ng`, ROI, foco ou outro recorte como imagem substituta.
- O formato do nome é o mesmo do arquivo NG: `YYYY-MM-DD_HHmm_CATEGORIA.png`.
- A implementação de nome compartilhada fica em `src/services/image_archive_naming.py`.
- `SEM_CATEGORIA` não é permitido no arquivamento automático OK.
- A gravação é assíncrona em fila daemon e não pode bloquear julgamento, envio de tecla, limpeza da interface ou recepção da próxima captura.
- Deduplicação obrigatória por `event_id`: um mesmo evento não pode ser salvo duas vezes caso o julgamento retorne pelo hook do XP.
- **Deduplicação persistente por conteúdo visual somente para OK:** se uma imagem pixel a pixel idêntica já existir em `public/ok_archive/`, um novo julgamento OK dessa mesma imagem não deve criar outro PNG, mesmo que apareça muitos eventos depois ou após reiniciar o ODIN.
- A verificação é feita pelo conteúdo da imagem, não pelo nome do arquivo nem pelo `event_id`. Portanto arquivos antigos com o padrão de nome legado também contam como duplicatas se contiverem exatamente os mesmos pixels.
- A fila OK indexa os PNGs já existentes em background para não bloquear o julgamento. Novas imagens realmente diferentes continuam sendo salvas normalmente.
- Essa deduplicação por conteúdo **não se aplica ao arquivo NG**; o fluxo NG permanece com sua regra atual.
- O arquivo é somente evidência visual/auditoria e não participa do dataset, KNN, protótipos, score, confiança ou decisão.

Fluxo:

```text
captura XP ou MSS analisada
        ↓
operador julga OK
        ↓
Salvar imagens OK está ATIVADO?
        ├── NÃO → não arquiva
        └── SIM
              ↓
event_id + categoria + análise ativa válidos?
              ├── NÃO → não arquiva
              └── SIM
                    ↓
mesma evidência de Copiar imagem
                    ↓
public/ok_archive/
```

Regressões obrigatórias do arquivo OK:

- toggle inicia ativado;
- operador pode desativar durante a sessão;
- NG nunca é salvo pelo arquivo OK;
- `source="auto"` nunca gera arquivo OK;
- `source="button"` e `source="xp_keyboard"` podem gerar arquivo OK;
- XP salva exatamente o mesmo frame de `Copiar imagem`;
- MSS salva exatamente o mesmo frame de `Copiar imagem`;
- MSS nunca reutiliza frame XP anterior;
- mesmo `event_id` é salvo no máximo uma vez;
- mesma imagem OK reaparecendo em outro `event_id` não cria outro PNG;
- mesma imagem OK já existente antes de reiniciar o ODIN também não é duplicada;
- um PNG antigo com nome legado bloqueia nova cópia quando o conteúdo visual é idêntico;
- imagens visualmente diferentes continuam sendo preservadas separadamente;
- novo `event_id` com imagem diferente pode ser salvo normalmente;
- categoria vazia não cria `SEM_CATEGORIA`;
- fila grava PNG com o mesmo formato de nome do NG;
- o bloco visual OK permanece imediatamente abaixo do bloco NG e usa o mesmo padrão responsivo.


### Regra de não duplicar a mesma imagem OK

A necessidade operacional é manter apenas uma evidência quando a **mesma imagem**
for julgada como OK repetidas vezes.

Exemplo:

```text
imagem A → operador julga OK → salva 1 PNG
10 outras imagens passam
imagem A reaparece → operador julga OK → NÃO salva outro PNG
imagem A reaparece novamente → operador julga OK → NÃO salva outro PNG
```

A identidade usada nessa regra é o conteúdo exato dos pixels. O nome do arquivo,
horário e `event_id` podem mudar; se os pixels forem idênticos, a evidência já
existe e o novo salvamento é ignorado.

A deduplicação deve sobreviver a reinicializações do ODIN porque a fila
`OKImageArchiveQueue` indexa os PNGs já existentes em `public/ok_archive/`.

Essa regra é exclusiva do arquivo visual OK. O arquivo visual NG não deve adotar
automaticamente essa deduplicação por conteúdo.


#### Validação operacional da deduplicação OK em 02/10/2026

O operador validou em uso real o comportamento de não duplicar a mesma imagem OK.

Foi confirmado que:

- uma imagem julgada OK é salva na primeira ocorrência;
- se a mesma imagem reaparecer vários eventos depois e for julgada OK novamente,
  nenhum novo PNG é criado;
- outras imagens podem passar entre as ocorrências sem quebrar a deduplicação;
- imagens visualmente diferentes continuam sendo salvas normalmente;
- a regra permanece exclusiva do arquivo OK;
- o arquivo NG mantém o comportamento anterior;
- a deduplicação por conteúdo não altera julgamento, dataset, KNN ou ciclo
  produtivo.

Essa validação passa a ser a referência operacional para o arquivo visual OK:
**uma mesma evidência visual exata deve existir apenas uma vez em
`public/ok_archive/`**.

### Padrão de nome dos arquivos visuais OK/NG

OK e NG usam obrigatoriamente o mesmo gerador compartilhado em
`src/services/image_archive_naming.py`.

Formato atual:

```text
YYYY-MM-DD_HHmm_CATEGORIA.png
```

Exemplo:

```text
2026-10-02_0811_FALTANDO.png
```

Esse padrão substitui o formato anterior com dia/mês textual, segundos e
milissegundos. Refatorações futuras não devem criar formatos diferentes entre
`public/ng_archive/` e `public/ok_archive/`.


#### Validação operacional do padrão de nomes em 02/10/2026

Após reiniciar o ODIN com a versão atualizada, o operador confirmou em uso real
que novos arquivos OK e NG passaram a ser gravados no padrão correto:

```text
YYYY-MM-DD_HHmm_CATEGORIA.png
```

Foi confirmado que:

- o formato antigo deixou de ser usado para novos arquivos;
- OK e NG usam o mesmo padrão;
- a categoria continua sendo normalizada no nome;
- arquivos antigos já existentes não são renomeados retroativamente;
- a aplicação precisa carregar a versão atual do módulo de nomeação para usar o
  novo padrão;
- a gravação efetiva de OK e NG chama diretamente o gerador compartilhado
  `build_archive_filename()`.

Essa configuração passa a ser a referência operacional validada para nomes dos
arquivos visuais de auditoria.

### Validação operacional do arquivo visual OK em 02/10/2026

O recurso foi validado em uso real pelo operador após a implementação.

Foi confirmado que:

- o bloco **`Salvar imagens OK`** aparece corretamente logo abaixo de
  **`Salvar imagens NG`**;
- o toggle inicia **ATIVADO** ao abrir o ODIN;
- o operador pode desativar e reativar o recurso durante a sessão;
- julgamentos humanos `OK` geram corretamente a evidência visual;
- a imagem salva corresponde à mesma evidência completa de **`Copiar imagem`**;
- o comportamento funciona sem alterar o fluxo normal de julgamento;
- o visual do bloco permanece consistente com o arquivo NG;
- a gravação não interfere no ciclo produtivo, KNN, dataset ou decisão.

Essa validação passa a ser a referência operacional do arquivo visual OK.
Refatorações futuras devem preservar o mesmo contrato de evidência, o estado
ativado por padrão e a posição visual imediatamente abaixo do arquivo NG.


**Regra visual do painel KNN:**
- A existência de memória e a força do match são conceitos diferentes na interface.
- `PRIMEIRA OCORRÊNCIA` só pode ser exibido quando não existir nenhum registro da categoria consultada.
- Se existirem JSONs da categoria, mas não houver assinatura visual válida/comparável, mostrar `MEMÓRIA CARREGADA` e a quantidade de registros; não representar como primeira ocorrência.
- Se houver comparação visual, a barra amarela `Melhor match` deve usar a melhor similaridade realmente calculada. Caso `best_similarity` não tenha sido propagado por uma camada de telemetria, a UI pode recuperar o mesmo valor a partir da melhor hipótese OK/NG já calculada, sem alterar a decisão.
- Match abaixo do limiar continua visível como `MEMÓRIA ENCONTRADA • abaixo do limiar`; isso não concede influência ao KNN.
- Essa regra é exclusivamente de apresentação. Limiares, classificação, pesos, conflito OK/NG e regra de melhor correspondência permanecem no núcleo.


**Regra visual da barra "Influência dos Motores" para KNN:**
- No KNN, `raw_score`/voto mede direção da decisão: `0 = OK` e `1 = NG`. Esse valor não representa força da memória.
- A barra grande da linha `Memória local KNN` deve representar `best_similarity` (força do melhor match visual), nunca o percentual de voto NG.
- Portanto um caso conhecido como falha falsa pode ter `voto 0% NG` e, ao mesmo tempo, uma barra de match quase cheia.
- O texto da linha deve separar `match`, `voto OK/NG`, `peso` e `efeito`.
- A barra fina amarela continua representando somente o peso efetivo do KNN na fusão.
- O marcador de referência da barra grande do KNN usa o limiar de match da memória, não o corte de decisão física.
- Essa alteração é apenas de apresentação/telemetria; não muda voto, peso, limiar de memória ou resultado da fusão.


## Regra crítica da categoria FALTANDO — ausência física prevalece sobre memória

### Problema observado em 01/10/2026

Foi identificado um caso real em que a AOI enviou uma inspeção válida da categoria
`FALTANDO`: o gabarito mostrava o componente presente e a imagem NG mostrava a
região fisicamente vazia. O intake estava correto:

- `valid_epicenter = True`;
- 6 anomalias brutas;
- epicentro final presente;
- ROI de foco aproximadamente `63 × 122 px`;
- a falha ocorria depois do intake, durante a fusão dos motores.

O problema arquitetural era que o especialista físico `MissingComponentExpert`
podia detectar uma quebra forte da expectativa visual, mas uma correspondência
KNN antiga rotulada como `OK` ainda podia receber peso dominante/100% e vetar
o defeito físico, produzindo `FALHA FALSA`. Além disso, a camada de contraste
OK × NG podia transformar um conflito de memória em `REVISÃO OBRIGATÓRIA`,
mesmo quando o componente estava fisicamente ausente.

Para `FALTANDO`, memória visual é evidência histórica; ela não pode provar a
presença de um componente que o comparador físico confirmou que desapareceu.

### Solução: hard physical absence

`MissingComponentExpert` acrescenta agora:

- `missing_hard_absence`;
- `missing_hard_absence_reason`;
- `missing_hard_absence_thresholds`.

A ausência física forte só é habilitada quando o caso já é defeito físico e não
foi classificado como `DESLOCAMENTO PROVÁVEL`.

Existem três rotas de confirmação:

1. **substituição pelo fundo**
   - score físico >= 72%;
   - área divergente >= 25%;
   - exposição de fundo >= 28%;
   - melhor correspondência próxima < 60%.

2. **colapso estrutural forte**
   - score físico >= 85%;
   - área divergente >= 30%;
   - residual médio >= 45%;
   - perda estrutural >= 30%;
   - melhor correspondência próxima < 60%.

3. **componente removido com footprint/base escura**
   - score físico >= 90%;
   - área divergente >= 45%;
   - residual médio >= 60%;
   - perda de aparência >= 50%;
   - similaridade direta <= 50%;
   - melhor correspondência próxima < 35%;
   - não pode estar classificado como `DESLOCAMENTO PROVÁVEL`.

A terceira rota cobre componentes que, ao desaparecerem, deixam uma área escura,
footprint, cola ou base com aparência parcialmente semelhante ao corpo original.
Nesses casos, `background_exposure` pode permanecer em 0% e
`structure_loss` pode ficar abaixo de 30%, mesmo quando a aparência esperada
foi destruída.

Caso real registrado em 01/10/2026:
- evento `bf6b6a2f1e844cc796ae355ae7ceb7e8`;
- componente R375;
- `missing_score ≈ 96.4%`;
- cobertura ≈ 54.7%;
- residual médio ≈ 67.2%;
- `missing_structure_loss ≈ 24.0%`;
- `background_exposure = 0%`;
- melhor correspondência próxima ≈ 13.2%;
- conflito KNN: NG 90.6% × OK 90.0%;
- consequência antiga: revisão obrigatória.

Esse padrão não deve depender de exposição do fundo vermelho nem exigir perda
estrutural >= 30%. A combinação de score alto, grande cobertura, residual alto,
perda da aparência original e ausência de correspondência próxima é suficiente
para caracterizar ausência física forte.

Esses limites são deliberadamente mais fortes que o limiar comum do motor. O
objetivo é reservar o override apenas para desaparecimento inequívoco, não para
variação de iluminação, ruído, deslocamento ou divergência parcial.

### Hierarquia de decisão quando hard missing = true

Quando `missing_hard_absence == True`:

```text
MissingComponentExpert
        ↓
ausência física forte confirmada
        ↓
fusion_rule = missing_hard_absence
        ↓
motor dominante = missing
        ↓
peso físico = 100%
peso KNN = 0%
        ↓
final_score = 1.0
confidence = 0.99
        ↓
DEFEITO REAL / NG
```

A memória KNN continua sendo calculada e exibida para auditoria, inclusive seu
melhor rótulo e similaridade, mas recebe:

```text
role = AUDITORIA — SEM VETO SOBRE AUSÊNCIA FÍSICA
suppressed_by_hard_missing = true
```

Ela não pode alterar o veredito nesse caso.

### Conflito de memória

Se as melhores memórias OK e NG estiverem quase empatadas, a regra normal ainda
é revisão humana. Porém, quando a ausência física forte já foi confirmada, esse
conflito não pode rebaixar a confiança nem criar revisão obrigatória. O conflito
permanece apenas como telemetria/auditoria.

### Regra de segurança

Não transformar toda categoria `FALTANDO` em NG automático.

Casos abaixo dos critérios de hard absence continuam usando exatamente o fluxo
normal:

- motor físico;
- KNN;
- contraste OK × NG;
- limiares existentes;
- revisão humana quando aplicável.

Um componente deslocado deve continuar sendo distinguido de um componente
ausente. A existência de uma correspondência próxima forte é uma das travas que
impedem o hard missing.

### Debug obrigatório para FALTANDO

O `Copiar debug XP` deve registrar, após a análise:

- categoria normalizada;
- veredito;
- confiança;
- score final e score físico;
- regra de fusão;
- motor dominante;
- `missing_score`;
- `missing_changed_coverage`;
- `missing_structure_loss`;
- `missing_background_exposure`;
- `missing_direct_similarity`;
- `missing_appearance_loss`;
- `missing_edge_mismatch`;
- `missing_residual_p90`;
- `missing_best_similarity`;
- `missing_hard_absence`;
- motivo do hard absence;
- melhor rótulo/similaridade KNN;
- conflito de memória;
- se o KNN foi suprimido pelo hard missing.

Assim um futuro caso de componente realmente ausente não deve ser diagnosticado
somente pelo intake. O debug precisa mostrar também a decisão final e qual camada
teve autoridade sobre ela.

### Regressões obrigatórias

Manter testes que garantam:

- componente completamente removido → `missing_hard_absence=True`;
- componente removido deixando footprint/base escura → `missing_hard_absence=True`;
- diferença parcial com correspondência próxima plausível → não vira hard missing;
- componente deslocado → não vira hard missing;
- memória OK forte não veta hard missing;
- conflito OK × NG não força revisão sobre hard missing;
- hard missing alcança 99% de confiança e pode passar pela trava existente de
  Produção sem reduzir globalmente o limiar de confiança;
- casos ambíguos continuam seguindo a política normal.


### Caso observado em 02/10/2026 — componente presente confundido com ausência

Evento operacional: `a997255818ea489b8afddf3d90b470fd`.

A AOI classificou a ocorrência como `FALTANDO`, mas visualmente o componente
continuava presente. O corpo físico permanecia no mesmo local e com dimensões
compatíveis; a principal diferença estava em aparência, brilho, contraste e
serigrafia interna.

O debug mostrou:

- `missing_score ≈ 100%`;
- cobertura divergente ≈ `56,3%`;
- residual médio ≈ `61,4%`;
- perda estrutural ≈ `86,6%`;
- exposição de fundo ≈ `55,3%`;
- incompatibilidade de bordas ≈ `78,8%`;
- melhor correspondência próxima do motor físico ≈ `34,3%`;
- KNN melhor rótulo = `OK`;
- similaridade OK ≈ `99,43%`;
- melhor NG ≈ `89,68%`;
- sem conflito de memória;
- resultado incorreto antes da correção: `missing_hard_absence=True` e
  `DEFEITO REAL`.

Este caso reforça que **não é suficiente apenas baixar ou subir limiares**.
Uma mudança grande de aparência pode produzir métricas típicas de ausência mesmo
quando o corpo físico continua presente.

#### Nova testemunha de presença em duas escalas

O `MissingComponentExpert` passa a procurar presença física em duas regiões:

1. a ROI interna usada pelo motor `FALTANDO`;
2. o envelope do componente fornecido pelo epicentro final da AOI.

A segunda escala é obrigatoriamente geométrica e procura:

- similaridade de baixa frequência do corpo;
- Dice da silhueta;
- razão de área;
- razão de largura e altura da massa principal;
- deslocamento do centróide.

Existem duas rotas de presença:

1. **coarse + geometria**
   - usa similaridade de baixa frequência junto com silhueta, área e centro;
2. **geometry-only**
   - não exige correlação tonal forte;
   - exige silhueta muito compatível;
   - área compatível;
   - caixa principal com largura/altura compatíveis;
   - centro praticamente preservado.

A rota `geometry-only` existe para componentes como o D5 do caso real, em que
o corpo continua presente, mas acabamento, brilho, cor aparente e serigrafia
mudam o suficiente para derrubar a correlação tonal.

A serigrafia/texto interno e variações globais de brilho não devem, sozinhos,
ser usados como prova de falta física.

Se a ROI interna divergir muito, mas o epicentro final confirmar massa,
silhueta, área e centro coerentes, o sistema deve registrar:

```text
missing_component_body_present = True
missing_body_presence_source = aoi_epicenter
missing_body_presence_veto = True
missing_hard_absence = False
```

Nesse estado, o motor `FALTANDO` não pode classificar o componente como
fisicamente ausente. A divergência de aparência continua disponível para os
outros motores e para a memória KNN.

#### Telemetria obrigatória

O debug de captura deve registrar:

- `missing_component_body_present`;
- `missing_body_presence_veto`;
- `missing_body_presence_source`;
- `missing_body_presence_box`;
- `missing_body_coarse_similarity`;
- `missing_body_silhouette_dice`;
- `missing_body_area_ratio`;
- `missing_body_centroid_shift`;
- `missing_body_box_width_ratio`;
- `missing_body_box_height_ratio`;
- `missing_body_presence_policy`;
- `missing_body_presence_reason`.

A presença geométrica só bloqueia o hard missing quando o motor local está
tentando classificar a ROI como conteúdo ausente/quebra da expectativa. Uma ROI
local conforme, por si só, não pode impedir a análise dual-scale do contexto
maior; isso preserva os casos reais em que um patch pequeno parece normal mas o
componente desapareceu fora dele.

#### Validação operacional da testemunha geométrica em 02/10/2026

O caso real do componente D5 foi retestado após a implementação da testemunha
de presença em duas escalas e o operador confirmou que o falso positivo foi
corrigido.

Foi validado que:

- o componente fisicamente presente deixou de ser tratado como ausência forte;
- a análise do corpo completo pelo epicentro final complementa a ROI interna;
- a rota `geometry_only` consegue preservar a evidência de presença mesmo com
  mudança forte de brilho, acabamento e serigrafia;
- a correção não depende de simplesmente baixar o `missing_score` ou afrouxar
  globalmente os limiares de ausência;
- o hard missing continua disponível para componentes realmente ausentes;
- a análise dual-scale continua preservada nos casos em que a ROI local, sozinha,
  não é suficiente para decidir;
- a memória KNN continua atuando separadamente da testemunha física de presença.

Contrato operacional validado:

```text
componente presente no envelope AOI
        ↓
silhueta/área/caixa/centro coerentes
        ↓
missing_component_body_present = True
missing_body_presence_source = aoi_epicenter
missing_body_presence_policy = geometry_only
        ↓
hard missing bloqueado
        ↓
não classificar como componente fisicamente ausente
```

Essa regra passa a ser a referência operacional para casos de `FALTANDO` em
que a aparência muda fortemente, mas o corpo físico do componente continua
presente.


### Validação operacional — testemunha OK quase exata em FALTANDO em 02/10/2026

Foi validado em uso real um caso da categoria `FALTANDO` em que o detector
físico bruto levantava `missing_hard_absence=True`, porém a memória encontrava
uma ocorrência OK praticamente idêntica ao evento atual.

No caso validado:

- melhor similaridade OK ≈ `99,999999%`;
- melhor similaridade NG ≈ `87,25%`;
- sem conflito de memória efetivo;
- o hard missing bruto continuou registrado para auditoria;
- o hard missing efetivo foi descartado;
- o motor dominante passou a ser o KNN;
- o veredito final passou corretamente para **FALHA FALSA**.

Contrato validado:

```text
raw_hard_missing_evidence = True
hard_missing_evidence = False
hard_missing_contradicted_by_exact_ok = True
fusion_rule = hard_missing_exact_ok_witness
dominant_engine = knn
weights = physical 0% / knn 100%
veredito = FALHA FALSA
```

A causa técnica corrigida estava na camada `best_match_memory.py`: ela recebia
novamente o `missing_hard_absence` bruto depois da fusão base e restaurava
indevidamente a prioridade física, anulando a testemunha OK quase exata.

A regra validada permanece restrita:

- aplica-se à categoria `FALTANDO`;
- exige memória OK quase exata;
- exige vantagem clara sobre a melhor hipótese NG;
- não vale para memória apenas "forte";
- não vale quando existe conflito relevante;
- não enfraquece a guarda transversal de ausência física em outras categorias;
- não altera o detector físico bruto, apenas a autoridade final de fusão.

Essa validação passa a ser a referência operacional para falsos positivos de
hard missing quando existir uma recorrência OK praticamente idêntica.

### Regra de consistência visual — hard missing não pode reaparecer como revisão

Foi identificado um segundo ponto após a proteção de hard missing: a camada de
decisão podia corretamente concluir `missing_hard_absence`, zerar o peso do
KNN e remover a revisão, enquanto a camada visual de memória ainda lia o
`memory_conflict` bruto do KNN e sobrescrevia o texto do veredito com
`CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA`.

Essa divergência entre núcleo e interface é proibida.

Quando qualquer um dos sinais abaixo estiver presente:

- `decision_trace.hard_missing_evidence == True`;
- `decision_trace.fusion_rule == "missing_hard_absence"`;
- `decision_trace.memory.suppressed_by_hard_missing == True`;
- `detail.missing_hard_absence == True`;

a UI deve considerar:

```text
conflito bruto KNN = somente auditoria
revisão visual = False
veredito exibido = o veredito final do núcleo
```

O conflito bruto continua preservado na telemetria para investigação, mas deve
ser distinguido do conflito efetivo que possui autoridade para exigir operador.

O painel KNN deve mostrar nesse estado algo equivalente a:

```text
AUSÊNCIA FÍSICA FORTE
KNN SOMENTE AUDITORIA
```

e nunca `REVISÃO OBRIGATÓRIA`.

A camada `memory_status_model.py` é responsável por separar:

- `raw_conflict`: conflito bruto calculado pela memória;
- `conflict`: conflito efetivo para apresentação;
- `raw_review_required`: pedido bruto de revisão da memória;
- `review_required`: revisão efetiva;
- `hard_missing_override`: autoridade física que suprimiu a memória.

A telemetria do hard missing deve preservar também `best_ok_similarity`,
`best_ng_similarity`, `hypothesis_margin` e `memory_conflict`, mesmo quando
o KNN recebe peso zero, para que o debug continue auditável.

Regressão obrigatória: hard missing + conflito KNN bruto deve manter
`DEFEITO REAL`, confiança 99%, zero peso KNN e nenhuma mensagem visual de
revisão obrigatória.


## Guarda transversal de ausência física — categoria AOI não define a realidade visual

### Problema observado em 01/10/2026 — EMBORCADO com componente ausente

Foi registrado o evento `1ef8605d905e4fac998cee708c89cbf6`.
A AOI classificou a ocorrência como `EMBORCADO`, mas visualmente o componente
do gabarito estava presente e a imagem de teste mostrava somente a região/footprint
onde o componente deveria existir.

Antes desta correção:

```text
Categoria AOI = EMBORCADO
        ↓
MissingComponentExpert não executava
        ↓
motores físicos = defeito forte
        ↓
KNN encontra OK parecido com ~90,6%
        ↓
best_match_strong
        ↓
peso KNN = 100%
        ↓
FALHA FALSA
```

O debug desse caso mostrava:

- `physical_score = 1.0`;
- divergência estrutural ≈ 54%;
- evidência semântica ≈ 71%;
- memória OK ≈ 90,6%;
- `fusion_rule = best_match_strong`;
- `dominant_engine = knn`;
- resultado incorreto: `FALHA FALSA`.

A causa arquitetural era assumir que ausência física só poderia existir quando
o texto da AOI fosse `FALTANDO`. A categoria da AOI é um rótulo do equipamento,
não uma prova de que o componente está presente.

### Solução

Foi criado `src/core/experts/physical_absence_guard.py`.

A `PhysicalAbsenceGuard` é uma guarda visual independente da categoria
`FALTANDO`. Ela atua somente nas categorias:

- `EMBORCADO`;
- `DESLOCADO`;
- `INVERTIDO`.

`MUITO ADESIVO` fica explicitamente fora desta guarda porque possui física e
região de interesse próprias. `FALTANDO` continua usando seu especialista
dedicado e não passa pela guarda transversal.

A guarda não renomeia a categoria. Um evento `EMBORCADO` continua sendo
persistido, consultado e auditado como `EMBORCADO`.

### Contrato mais restritivo

Fora de `FALTANDO`, a ausência física só é aceita quando todas as condições
abaixo são satisfeitas:

- score da guarda >= 82%;
- cobertura alterada >= 45%;
- residual médio >= 38%;
- perda de aparência >= 40%;
- similaridade direta <= 60%;
- incompatibilidade de bordas >= 40%;
- melhor correspondência próxima < 25%;
- classificação não pode ser `DESLOCAMENTO PROVÁVEL`;
- comparador estrutural >= 35%;
- motor semântico >= 60%;
- score físico agregado >= 85%.

A concordância entre a guarda visual, o comparador estrutural e o motor
semântico é obrigatória. Assim um exemplo OK da memória não é ignorado apenas
porque uma única métrica visual subiu.

### Reprodução do caso real

Usando o mesmo frame e a mesma ROI do evento EMBORCADO, a guarda transversal
produziu aproximadamente:

- score da guarda: 84,9%;
- cobertura: 52,4%;
- residual médio: 40,8%;
- perda de aparência: 46,2%;
- similaridade direta: 53,8%;
- incompatibilidade de bordas: 51,0%;
- melhor correspondência próxima: 9,3%;
- estrutural da decisão original: 54%;
- semântico da decisão original: 71%;
- score físico agregado: 100%.

Esse vetor satisfaz o contrato transversal e caracteriza desaparecimento físico
mesmo que a AOI tenha usado o rótulo `EMBORCADO`.

### Hierarquia da decisão

Quando a guarda transversal confirma `missing_hard_absence=True`:

```text
categoria original permanece EMBORCADO/DESLOCADO/INVERTIDO
        ↓
ausência física forte confirmada
        ↓
fusion_rule = missing_hard_absence
        ↓
motor dominante = missing
        ↓
peso físico = 100%
peso KNN = 0%
        ↓
KNN continua visível somente para auditoria
        ↓
DEFEITO REAL / NG
confiança = 99%
```

Essa regra agora também existe na fusão base, antes da ponderação da memória.
Portanto a proteção não depende da ordem de instalação de wrappers de KNN.

### Memória continua isolada por categoria

A guarda transversal não pode:

- mudar `EMBORCADO` para `FALTANDO`;
- consultar memória de outra categoria;
- adicionar `missing_mask` à assinatura KNN de EMBORCADO/DESLOCADO/INVERTIDO;
- treinar o KNN como se o evento pertencesse a FALTANDO.

A memória continua usando a categoria AOI original. A guarda é somente uma
trava física de segurança contra veto incorreto de um desaparecimento
inequívoco.

### Debug obrigatório

Quando a guarda transversal for avaliada, `Copiar debug XP` deve registrar:

- `missing_cross_category_guard`;
- `missing_guard_policy`;
- `missing_guard_source_category`;
- `missing_guard_physical_support`;
- todas as métricas de ausência física;
- se o KNN foi suprimido por hard missing.

O debug deve permitir distinguir um `EMBORCADO` aprendido normalmente pela
memória de um evento rotulado como `EMBORCADO` pela AOI, mas com componente
fisicamente ausente.

### Regressões obrigatórias

Manter testes que garantam:

- vetor real do evento EMBORCADO ausente → guarda transversal confirma ausência;
- memória OK ~90,6% não veta ausência transversal;
- fusão base aplica `missing_hard_absence` antes do peso KNN;
- match próximo plausível bloqueia o override;
- suporte semântico/estrutural fraco bloqueia o override;
- `DESLOCAMENTO PROVÁVEL` nunca é convertido pela guarda;
- `MUITO ADESIVO` e `FALTANDO` não usam essa guarda;
- máscara da guarda transversal não altera assinatura KNN da categoria original.


## INVERTIDO — fusão única, conflito de memória e colapso visual extremo

### Caso real de 01/10/2026

Evento: `299285f21c8448b0bff8a9757fa382af`.

A AOI classificou a ocorrência como `INVERTIDO`, mas visualmente o corpo do
componente presente no gabarito desapareceu no teste. O debug mostrou:

- score físico ≈ 85,7%;
- `missing_score ≈ 94,5%`;
- cobertura alterada ≈ 75,6%;
- residual médio ≈ 59,6%;
- perda de aparência ≈ 66,2%;
- similaridade direta ≈ 33,8%;
- melhor correspondência próxima ≈ 14,6%;
- estrutural ≈ 45,7%;
- semântico ≈ 57,1%;
- match KNN OK ≈ 87,7%.

A guarda transversal executou, mas a rota primária de hard missing não passou
porque o semântico ficou abaixo de 60% e a incompatibilidade de bordas ficou
abaixo de 40%. Esse padrão é típico de um epicentro estreito: quase todo o
conteúdo esperado desaparece, porém parte das bordas locais permanece.

### Rota de colapso visual extremo

A `PhysicalAbsenceGuard` possui uma segunda rota para esse caso. Ela exige
simultaneamente:

- score >= 92%;
- cobertura alterada >= 70%;
- residual médio >= 55%;
- perda de aparência >= 60%;
- similaridade direta <= 40%;
- incompatibilidade de bordas >= 30%;
- melhor correspondência próxima < 20%;
- estrutural >= 40%;
- semântico >= 55%;
- score físico agregado >= 85%;
- classificação diferente de `DESLOCAMENTO PROVÁVEL`.

Essa rota não substitui a rota primária. Ela existe somente para colapso visual
extremo, onde a quantidade de evidências independentes compensa um epicentro
local estreito.

### Causa arquitetural descoberta no INVERTIDO

O módulo `inverted_face_integration.py` mantinha uma segunda implementação de
fusão e importava `_dynamic_fusion` diretamente no carregamento do módulo.
Depois, `best_match_memory` e `memory_hypothesis_contrast` substituíam a
função de fusão no módulo central, mas o módulo INVERTIDO continuava apontando
para a referência antiga.

Isso permitia este fluxo incorreto:

```text
fusão central atualizada
        ↓
hard missing / best-match / conflito OK×NG tratados
        ↓
INVERTIDO executa uma segunda fusão privada antiga
        ↓
resultado e telemetria podem divergir
```

Essa duplicação foi removida.

### Regra obrigatória de arquitetura

Nenhum especialista de categoria pode possuir uma segunda implementação de
pesos KNN/físico.

`INVERTIDO`, `EMBORCADO`, `DESLOCADO`, `FALTANDO` e futuras categorias
devem terminar na mesma função de fusão central.

O módulo INVERTIDO agora:

1. calcula a assinatura específica da face;
2. adiciona suas métricas ao `detail`;
3. consulta a memória da categoria INVERTIDO;
4. chama dinamicamente `anomaly_memory_module._dynamic_fusion`;
5. herda automaticamente best-match, contraste OK×NG, hard missing e futuras
   extensões da fusão.

Não importar `_dynamic_fusion` por valor para manter uma referência antiga.

### Conflito bruto x conflito efetivo no debug

Foi identificada outra divergência: a UI podia ler `detail.memory_conflict`
enquanto o debug lia apenas `decision_trace.memory.memory_conflict`. Quando o
trace não continha a chave, o debug mostrava `False` por padrão mesmo que a UI
estivesse mostrando `CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA`.

O debug deve agora registrar separadamente:

- `raw_memory_conflict`: conflito produzido pela comparação de memórias;
- `memory_conflict`: conflito efetivo depois das regras de autoridade física;
- `raw_operator_review_required`: revisão pedida pela memória;
- `operator_review_required`: revisão efetiva;
- `suppressed_by_hard_missing`: indica que a ausência física retirou o poder
  de veto/revisão do KNN.

O debug e a UI devem sempre concordar sobre o estado efetivo.

### Regressões obrigatórias

Manter testes que garantam:

- vetor real INVERTIDO de colapso extremo → hard missing verdadeiro;
- `INVERTIDO` usa a fusão central atual, nunca uma cópia privada antiga;
- hard missing continua ativo depois do especialista INVERTIDO;
- memória OK forte não veta ausência física;
- conflito bruto pode permanecer na auditoria, mas não cria revisão efetiva sob
  hard missing;
- debug não pode reportar `memory_conflict=False` quando a UI estiver usando
  um conflito bruto verdadeiro sem supressão;
- assinatura e memória continuam isoladas pela categoria original.


## Dual-Scale Presence — epicentro local + contexto físico do componente

### Motivação

Foi identificado que a ROI/epicentro fornecida pela AOI pode representar apenas
uma pequena fração do componente. Nesses casos, uma região interna escura pode
continuar parecida mesmo depois que o corpo inteiro do componente desapareceu.

Evento real de referência: `912c92754da3433d8a2a0980052e2b78`, categoria
`FALTANDO`, em 01/10/2026.

O evento apresentava:

- caixa global aproximadamente `547 × 261 px`;
- caixa de foco aproximadamente `140 × 108 px`;
- área local equivalente a aproximadamente 10,6% da área global;
- `missing_score` local ≈ 34,9%;
- cobertura local ≈ 4,1%;
- similaridade direta local ≈ 87,5%;
- classificação local: `ROI CONFORME`;
- score físico agregado ≈ 85%;
- divergência estrutural ≈ 56%;
- evidência semântica ≈ 56%;
- KNN encontrou memória `OK` ≈ 89,4%;
- consequência antiga: `best_match_intermediate` com KNN dominante e
  resultado incorreto `FALHA FALSA`.

O problema não era somente calibração de thresholds. A escala local estava
respondendo à pergunta 'este pequeno patch ainda parece parecido?', enquanto a
decisão de presença precisava responder 'o componente como um todo ainda
existe?'.

### Arquitetura

Foi criado:

`src/core/experts/dual_scale_presence.py`

A `DualScalePresenceAnalyzer` adiciona uma segunda escala sem remover o
epicentro original.

```text
imagem AOI
    ↓
epicentro / ROI local
    ├── escala local: precisão do defeito
    │
    └── quando necessário
         ↓
       ROI contextual expandida
         ↓
       presença física do componente
         ↓
       hard missing?
         ├── SIM → KNN somente auditoria
         └── NÃO → fluxo normal de fusão/memória
```

Essa funcionalidade deve ser chamada de **Dual-Scale Presence** para não ser
confundida com `dual_scale_memory.py`, que trata da representação/memória KNN.

### Quando a segunda escala é executada

A análise contextual não roda indiscriminadamente em toda inspeção.

Ela é ativada quando:

1. a área do epicentro representa <= 25% da área global do componente; ou
2. a escala local diz que não há defeito, mas os motores estrutural/semântico
   apresentam suporte físico independente suficiente.

O estado de contradição é registrado como `missing_scale_disagreement`.

Para suporte físico contextual, a política atual exige:

- estrutural >= 45% E semântico >= 45%; ou
- um dos dois >= 65%.

### Construção da ROI contextual

A ROI contextual é centrada no epicentro e cresce sem sair da imagem:

- expansão mínima em torno do foco: 2,5× por eixo;
- referência mínima adicional: 55% das dimensões da caixa global;
- largura/altura finais limitadas ao recorte real recebido.

No evento de referência:

```text
local:    140 × 108
contexto: 350 × 270
```

Essa expansão mantém o epicentro como âncora, mas passa a incluir o corpo do
componente e sua relação com pads/footprint ao redor.

### Métricas contextuais

A segunda escala reutiliza o mesmo pipeline determinístico de visão, porém em
uma caixa maior. Ela registra:

- `missing_context_score`;
- `missing_context_coverage`;
- `missing_context_residual_mean`;
- `missing_context_residual_p90`;
- `missing_context_structure_loss`;
- `missing_context_edge_mismatch`;
- `missing_context_direct_similarity`;
- `missing_context_appearance_loss`;
- `missing_context_best_similarity`;
- `missing_context_box`;
- `missing_local_global_area_ratio`;
- `missing_context_hard_absence`;
- `missing_context_hard_reason`.

### Contrato para ausência contextual forte

A rota contextual normal exige simultaneamente:

- score contextual >= 72%;
- cobertura contextual >= 30%;
- residual médio contextual >= 50%;
- perda de aparência contextual >= 35%;
- melhor correspondência próxima < 35%;
- perda estrutural >= 20% OU incompatibilidade de bordas >= 30%;
- suporte físico independente estrutural/semântico conforme regra acima.

Existe também uma rota contextual extrema, reservada para desaparecimento
inequívoco mesmo quando o suporte global não estiver disponível:

- score >= 85%;
- cobertura >= 45%;
- residual médio >= 60%;
- perda de aparência >= 50%;
- melhor correspondência próxima < 25%.

A rota extrema não deve ser usada para simples diferença parcial ou
deslocamento.

### Reprodução do evento real

Com o mesmo frame e a mesma geometria do evento
`912c92754da3433d8a2a0980052e2b78`, a análise contextual produz
aproximadamente:

- score contextual ≈ 79,5%;
- cobertura ≈ 38,8%;
- residual médio ≈ 68,8%;
- P90 ≈ 84,0%;
- perda estrutural ≈ 32,7%;
- incompatibilidade de bordas ≈ 39,2%;
- similaridade direta ≈ 56,2%;
- perda de aparência ≈ 43,8%;
- melhor correspondência próxima ≈ 19,8%.

Com estrutural ≈ 56% e semântico ≈ 56%, esse vetor confirma ausência física
contextual, apesar de a pequena ROI local ter sido classificada como conforme.

### Autoridade sobre memória

Quando `missing_context_hard_absence == True`, o resultado é promovido para:

```text
missing_hard_absence = True
fusion_rule = missing_hard_absence
motor dominante = missing
peso físico = 100%
peso KNN = 0%
confidence = 0.99
```

A memória continua sendo consultada e exibida para auditoria, mas não pode
anular uma ausência física contextual confirmada.

Isso vale tanto para o especialista dedicado de `FALTANDO` quanto para a
`PhysicalAbsenceGuard` transversal de `EMBORCADO`, `DESLOCADO` e `INVERTIDO`.

### Isolamento e segurança

A Dual-Scale Presence:

- não altera a categoria recebida da AOI;
- não mistura memórias entre categorias;
- não injeta máscara contextual na assinatura KNN transversal;
- não muda regras de adesivo;
- não substitui o epicentro local;
- não transforma automaticamente todo `FALTANDO` em NG;
- somente ganha autoridade quando o contrato contextual de ausência física é
  satisfeito.

Se o contexto não confirmar ausência, a decisão continua no fluxo normal com
motores físicos, KNN, contraste OK×NG e revisão humana quando aplicável.

### Debug obrigatório

`Copiar debug XP` deve registrar:

- política dual-scale;
- se a análise contextual foi ativada/executada;
- razão área local/global;
- desacordo entre escalas;
- caixa contextual;
- score/cobertura/residual contextual;
- perda estrutural e incompatibilidade de bordas contextual;
- similaridade/perda de aparência contextual;
- melhor match próximo contextual;
- resultado de hard absence contextual;
- suporte físico independente e motivo final.

O debugger visual de presença deve mostrar explicitamente `LOCAL + CONTEXTO` e
as principais métricas contextuais quando a segunda escala tiver sido
executada.

### Regressões obrigatórias

Manter testes que garantam:

- geometria real 140×108 dentro de 547×261 dispara a segunda escala;
- evento real reproduzido possui razão local/global ≈ 10,6%;
- contexto do evento real confirma hard missing;
- contexto sem suporte físico suficiente não promove ausência, exceto pela
  rota extrema;
- match próximo plausível bloqueia hard missing contextual;
- FALTANDO localmente conforme pode ser promovido por contexto confirmado;
- guarda transversal pode ser promovida pelo contexto;
- hard missing contextual continua zerando o peso KNN na fusão;
- debug e UI expõem claramente as duas escalas.



### Regressão de segurança — DESLOCADO presente não pode virar hard missing

Em 01/10/2026 foi registrado o evento `4128dec4a02f423fbdbcd47fca777108` da categoria `DESLOCADO`. Visualmente o componente `104` estava presente tanto no gabarito quanto no teste; a ocorrência foi validada pelo operador como **falha falsa** do ODIN.

O debug anterior mostrava:

- `missing_score ≈ 98,9%`;
- cobertura local ≈ 78,8%;
- residual local ≈ 70,1%;
- dual-scale contextual ≈ 97,8%;
- cobertura contextual ≈ 69,8%;
- estrutural independente ≈ 48,7%;
- semântico independente ≈ 30,1%;
- `missing_context_physical_support.supported = False`;
- melhor memória KNN `OK ≈ 98,18%`;
- melhor memória NG ≈ 89,76%;
- resultado incorreto antigo: `missing_hard_absence=True`, KNN suprimido e `DEFEITO REAL`.

A causa era a rota contextual de **colapso visual extremo** permitir hard missing mesmo quando os motores físicos independentes não confirmavam a ausência. Em uma categoria de deslocamento, comparar referência e teste em posições fixas pode produzir grande residual apenas porque o mesmo componente mudou de posição/registro.

#### Regra corrigida

Para a guarda transversal usada por `EMBORCADO`, `DESLOCADO` e `INVERTIDO`:

- a rota extrema do Dual-Scale Presence **não pode** promover `missing_hard_absence` sem `missing_context_physical_support.supported=True`;
- `FALTANDO` mantém sua política própria e pode continuar usando a rota contextual extrema conforme seu contrato dedicado;
- a categoria original continua preservada;
- memória KNN continua sendo consultada normalmente quando o hard missing transversal é bloqueado.

No evento de referência, como o suporte transversal era falso, a ausência física deve permanecer falsa e o KNN `OK ≈ 98,18%` volta a participar da fusão. O resultado esperado é **FALHA FALSA / OK**, sem suprimir a memória.

Essa proteção existe para impedir que **deslocamento, mudança de registro, pequena variação geométrica ou iluminação** sejam confundidos com desaparecimento físico apenas porque a comparação fixa local/contextual apresenta residual alto.

Regressões obrigatórias:

- evento DESLOCADO presente + suporte físico transversal falso → `missing_hard_absence=False`;
- rota extrema transversal sem suporte independente → não promove ausência;
- KNN OK forte permanece elegível quando o hard missing foi corretamente bloqueado;
- os casos reais anteriores de ausência física em EMBORCADO/INVERTIDO continuam passando quando possuem suporte físico independente suficiente.

#### Validação operacional da correção

Em 01/10/2026, após o endurecimento da guarda transversal, o mesmo fluxo foi
retestado na AOI real e o operador confirmou que o comportamento ficou correto.

Esse resultado valida especificamente a regra introduzida para o evento
`4128dec4a02f423fbdbcd47fca777108`:

- componente presente em categoria `DESLOCADO`;
- suporte físico transversal insuficiente;
- `missing_hard_absence` não deve ser promovido;
- memória `OK` forte continua elegível;
- o resultado final deve permanecer `FALHA FALSA / OK`.

Essa validação deve ser preservada como regressão operacional. Qualquer mudança
futura no Dual-Scale Presence ou na `PhysicalAbsenceGuard` não pode reintroduzir
o comportamento antigo de transformar deslocamento/variação de registro em
ausência física forte sem confirmação independente.

### Validação operacional em AOI real

Em 01/10/2026, após a implementação da **Dual-Scale Presence**, o fluxo foi
retestado na AOI real e o operador confirmou que o comportamento esperado
funcionou corretamente.

Essa validação operacional complementa as regressões automatizadas e deve ser
preservada como referência de engenharia para futuras alterações no motor de
presença.

Regra de manutenção:

- não remover a análise contextual apenas porque a ROI local apresenta alta
  similaridade;
- não retornar ao modelo de decisão baseado exclusivamente no epicentro para
  presença/ausência física;
- qualquer refatoração futura deve preservar a hierarquia:
  `ROI local → contexto quando necessário → hard missing → KNN somente
  auditoria`;
- alterações de thresholds devem manter as travas contra
  `DESLOCAMENTO PROVÁVEL`, match próximo plausível e ausência de suporte físico;
- se um caso futuro voltar a produzir `FALHA FALSA` com componente fisicamente
  ausente, registrar o debug completo e verificar primeiro se o dual-scale foi
  ativado, qual caixa contextual foi usada e qual métrica bloqueou o
  `missing_context_hard_absence`.


## Identidade visual da interface — ODIN

A identidade exibida ao operador foi padronizada para:

```text
ODIN - Observador Digital Inteligente
```

Essa é uma alteração **visual/de apresentação**. Ela não renomeia o repositório,
schemas, módulos Python, arquivos de pesos, caminhos, protocolos de rede nem
identificadores persistidos que já usam o nome técnico `visionx`.

Superfícies visuais obrigatórias:

- título da janela principal:
  `ODIN - Observador Digital Inteligente - Monitoramento IA`;
- título principal do cabeçalho:
  `ODIN - Observador Digital Inteligente`;
- HUD inicial:
  `ODIN - Observador Digital Inteligente: Inicializando...`;
- janela de calibração:
  `ODIN - Observador Digital Inteligente - Calibrar Zona de Interesse Avançado`;
- relatório técnico copiado pela interface:
  `ODIN - Observador Digital Inteligente - DEBUG DE ENTRADA WINDOWS XP`;
- seção de decisão do relatório:
  `DECISÃO ODIN - Observador Digital Inteligente`.

A fonte única da identidade visual fica em:

```text
src/ui/branding.py
```

Não espalhar novamente strings de marca diretamente pelos widgets. Novas telas
devem reutilizar as constantes de `branding.py`.

### Regra de compatibilidade

Manter inalterados, salvo migração específica e planejada:

- `visionx.network_xp_debug.v1`;
- nome do repositório `visionx-neural`;
- nomes de arquivos/pesos como `visionx_neural_weights.pth`;
- nomes de classes, módulos e APIs já existentes;
- protocolos TCP e comandos usados pelo agente Windows XP.

A troca para ODIN não pode quebrar integração, persistência ou histórico.

### Responsividade da marca

Como `ODIN - Observador Digital Inteligente` é maior que o nome anterior, o
título principal deve aceitar quebra de linha e largura mínima zero para
continuar responsivo em notebooks e monitores menores.


## Fundo dinâmico conforme o veredito da análise

O fundo principal do ODIN acompanha visualmente o **veredito final já calculado
pela IA**, sem participar da lógica de decisão.

Estados obrigatórios:

```text
sem análise / aguardando imagem / processando sem resultado
→ fundo neutro original

FALHA FALSA / OK
→ fundo verde escuro

DEFEITO REAL / NG
→ fundo vermelho escuro
```

Cores atuais do canvas:

- neutro: `#050505`;
- OK: `#0b2f18`;
- NG: `#3a0d12`.

Cores das superfícies principais:

- neutro: `#0d0d0d`;
- OK: `#103d22`;
- NG: `#46131a`.

Os estados OK/NG devem continuar escuros, porém visualmente inequívocos. Tons
tão próximos do preto que só sejam perceptíveis por comparação não atendem ao
objetivo operacional dessa sinalização.

A implementação mantém a propriedade Qt `decisionState` para telemetria e
estado visual e o stylesheet global do ODIN permanece **fixo** durante a
execução. A troca de estado não pode chamar `setStyleSheet()` no painel
principal nem reconstruir/reaplicar o tema completo.

O estado é aplicado ao canvas principal:

- `rootWindow`;
- `rootContent`;
- `rootViewport`;

e também aos grandes painéis que cobrem a maior parte da janela e não possuem
stylesheet local próprio:

- `headerFrame`;
- `sectionPanel`;
- `infoSection`;
- `confidenceFrame`;
- `controlsSection`;
- `statusBar`.

`ngArchiveFrame` e `networkDebugFrame` mantêm seus stylesheets locais
originais para preservar completamente o visual, estados `:hover`, `:focus`
e `:checked` dos respectivos botões.

Os cards internos continuam escuros para manter contraste, hierarquia visual e
legibilidade.

Os estados visuais válidos são:

- `neutral`;
- `ok`;
- `ng`.

O estado é derivado exclusivamente de `analysis["is_defect"]`:

- `False` → `ok`;
- `True` → `ng`;
- análise ausente → `neutral`.

### Regra de ciclo

O fundo colorido só pode permanecer enquanto existe uma análise final ativa para
visualização.

O fundo deve retornar obrigatoriamente ao neutro quando:

- o ODIN inicia;
- uma nova captura começa antes de existir um novo resultado;
- a captura é descartada;
- a análise é limpa;
- o julgamento/ciclo é concluído;
- o sistema fica aguardando a próxima imagem do Windows XP ou uma nova captura
  MSS.

Isso evita que o operador interprete a cor da peça anterior como estado da peça
seguinte.

### Regra crítica de arquitetura

O fundo dinâmico é **somente apresentação**. Ele não pode:

- alterar `is_defect`, score, confiança ou veredito;
- alterar KNN, memória ou dataset;
- mudar os comandos `PRESS_0/PRESS_1`;
- interferir no gate de imagens;
- alterar origem XP/MSS;
- manter estado vermelho/verde depois que a análise atual deixou de existir.

A implementação fica isolada em:

```text
src/ui/decision_background.py
```

e reutiliza o resultado já produzido pelo pipeline. O tema apenas declara as
cores em `src/ui/theme.py`.

### Regressões obrigatórias

Manter testes que garantam:

- sem análise → `neutral`;
- `is_defect=False` → `ok`;
- `is_defect=True` → `ng`;
- reset da confiança/análise → `neutral`;
- conclusão do julgamento/ciclo → `neutral`;
- `rootWindow`, `rootContent` e `rootViewport` recebem o mesmo estado;
- os containers principais recebem `decisionState` sem substituir o
  stylesheet global;
- um `sectionPanel` real renderiza `#1a0b0c` em estado NG;
- o stylesheet existente da janela permanece byte-a-byte inalterado após a
  troca de estado;
- estilos de botão e estados `:hover`, `:focus` e `:checked` permanecem
  preservados;
- o recurso continua exclusivamente visual.

### Correções visuais em 02/10/2026

Foi observado em operação que o veredito podia mostrar **DEFEITO REAL** enquanto
o fundo permanecia visualmente no tema escuro original.

Uma primeira tentativa forçou a reaplicação do stylesheet completo da janela.
Essa estratégia foi descartada porque, em uso real, alterou o visual dos botões
e fez estados de interação como `:hover` deixarem de responder corretamente.

A implementação válida passa a obedecer estas regras:

1. o stylesheet global instalado em `ControlPanelUI.setup_ui()` não é
   substituído durante uma decisão;
2. `apply_decision_background()` altera somente a propriedade dinâmica
   `decisionState` dos containers de fundo;
3. cada container alterado é repolido individualmente;
4. os seletores `[decisionState="ok"]` e `[decisionState="ng"]` já fazem
   parte de `APP_STYLESHEET` desde a criação da interface;
5. frames que possuem stylesheet local próprio não são tocados pela rotina de
   fundo;
6. botões, hover, focus, checked e demais estados interativos devem permanecer
   idênticos ao tema original.

Além dos testes de estado, deve existir regressão de renderização Qt offscreen:
um `sectionPanel` real em estado `ng` precisa renderizar o pixel de fundo
`#46131a` sem que o stylesheet da janela seja modificado.

Assim, `DEFEITO REAL / NG` deve produzir fundo vermelho-escuro visível e
`FALHA FALSA / OK` deve produzir fundo verde-escuro visível, retornando ao
tema neutro ao limpar ou concluir o ciclo, sem regressão visual dos controles.

#### Ajuste de contraste validado por captura em 02/10/2026

Uma captura real mostrou que o estado `OK` estava tecnicamente ativo, porém a
superfície `#0c1a11` era escura demais e visualmente parecia o tema neutro.
Por isso, os tons foram reforçados sem alterar a arquitetura:

- canvas OK: `#0b2f18`;
- superfície OK: `#103d22`;
- canvas NG: `#3a0d12`;
- superfície NG: `#46131a`.

Esse ajuste é somente cromático. Não pode reintroduzir `setStyleSheet()` global
durante a análise nem alterar hover, focus, checked ou estilos dos botões.

#### Validação operacional final em 02/10/2026

Após o reforço cromático, o operador validou o comportamento em uso real e
confirmou que a sinalização ficou correta.

Foi confirmado que:

- `FALHA FALSA / OK` deixa o fundo claramente verde-escuro;
- `DEFEITO REAL / NG` deixa o fundo claramente vermelho-escuro;
- sem análise ativa, o ODIN retorna ao fundo neutro original;
- o contraste é perceptível sem descaracterizar o tema escuro industrial;
- os botões mantêm o visual original;
- os estados `:hover`, `:focus` e `:checked` continuam funcionando;
- a mudança permanece exclusivamente visual e não interfere em decisão,
  confiança, KNN, captura, XP/MSS ou ciclo produtivo.

Essa configuração cromática passa a ser a referência operacional validada para
o fundo dinâmico do ODIN.


## Feedback visual temporário de julgamento 0/1

O ODIN possui um overlay exclusivamente visual para confirmar imediatamente ao operador quando uma tecla de julgamento foi recebida.

Comportamento:

```text
0 → OK
1 → NG
```

Fontes cobertas:

- teclado do próprio ODIN: `0`, `Num+0`, `1` e `Num+1`;
- teclado físico do Windows XP recebido pela rede como `CMD_OK` ou `CMD_NG`.

A apresentação é um quadrado temporário de aproximadamente `180 × 180 px`, posicionado no **canto inferior direito** da interface, com margem aproximada de `24 px` das bordas e acima dos demais componentes.

O visual segue a identidade industrial do ODIN:

- fundo escuro `#101010`;
- borda-base discreta `#303030`;
- cabeçalho `DECISÃO RECEBIDA` em amarelo ODIN `#f5c518`;
- `0 / OK`: detalhe, borda e tipografia de estado em verde `#4ade80`;
- `1 / NG`: detalhe, borda e tipografia de estado em vermelho `#ff6262`;
- origem exibida como `TECLADO ODIN` ou `TECLADO WINDOWS XP`;
- duração aproximada total: `800 ms`;
- entrada suave: fade-in + deslocamento vertical de apenas `8 px` em aproximadamente `120 ms`;
- saída suave: fade-out em aproximadamente `160 ms`;
- desaparece automaticamente.

Implementação:

```text
src/ui/decision_key_feedback.py
```

### Regra crítica de arquitetura

Esse recurso é **somente apresentação**. Ele não pode:

- alterar `save_label()`;
- enviar comandos ao XP;
- decidir OK/NG;
- modificar confiança, score ou memória KNN;
- bloquear o gate de imagens;
- gravar dataset ou evidências;
- capturar foco ou cliques do mouse;
- criar espera ativa, `sleep` ou animação pesada no caminho produtivo.

O overlay reutiliza um único widget, um único `QTimer` single-shot e animações curtas de propriedades Qt. A animação só existe enquanto o alerta está visível: não há loop, animação contínua, thread adicional ou `sleep`. O fade usa `QGraphicsOpacityEffect` somente sobre o pequeno widget de `180 × 180 px`, e o slide altera apenas sua posição em `8 px`.

Ele possui `WA_TransparentForMouseEvents` e `NoFocus`, portanto pode aparecer sobre outros componentes sem impedir interação. A escolha de animações curtas e locais é obrigatória para manter o custo de renderização desprezível diante do pipeline de visão computacional.

### Ordem de acionamento

No teclado local, o feedback aparece imediatamente **antes** do caminho normal do botão OK/NG. Assim a confirmação visual não espera o envio TCP `PRESS_0/PRESS_1`.

No teclado XP, o feedback é exibido somente quando existe uma captura ativa e o ODIN recebe `CMD_OK` ou `CMD_NG`.

### Supressão de eco visual

O agente XP pode devolver pelo hook global a mesma tecla que o ODIN acabou de enviar por `PRESS_0/PRESS_1`. Para não mostrar dois alertas para uma única decisão, repetições do mesmo julgamento dentro de aproximadamente `1,5 s` são suprimidas **somente na camada visual**.

Essa deduplicação não altera nem descarta comandos produtivos; ela apenas impede um segundo flash do overlay.

### Regressões obrigatórias

Manter testes que garantam:

- tecla local `0` aciona `0 / OK`;
- tecla local `1` aciona `1 / NG`;
- `CMD_OK` e `CMD_NG` do XP acionam o mesmo overlay com origem XP;
- botões desabilitados não são burlados pelos atalhos;
- comando XP sem captura ativa não produz confirmação visual de julgamento;
- o eco da mesma decisão não gera um segundo alerta imediato;
- o overlay permanece click-through e sem foco;
- o overlay permanece ancorado no canto inferior direito em telas de tamanhos diferentes;
- fade-in/fade-out são curtos e sem repetição;
- o movimento de entrada permanece pequeno e não desloca outros componentes;
- não existe animação em background quando o alerta está oculto;
- o recurso não altera nenhuma regra de negócio do ciclo.

### Validação operacional

Em 01/10/2026, o feedback visual `0 = OK` / `1 = NG` foi validado em operação e o comportamento esperado foi confirmado.

Após essa validação, a posição visual foi refinada do centro da tela para o canto inferior direito para reduzir interferência visual sobre a inspeção principal. Essa posição passa a fazer parte do contrato da interface.


## Diagnóstico e cópia de evidência por origem

Os controles visuais:

- `Copiar debug`;
- `Copiar imagem`;

não são exclusivos do Windows XP.

Eles devem ficar disponíveis para a **última captura analisada**, independentemente
da origem:

```text
Windows XP / rede
        ou
Captura local MSS
```

A fonte genérica da evidência fica em:

```text
src/services/capture_evidence.py
```

Contrato:

- cada captura possui um `event_id` próprio;
- o relatório e a imagem copiada devem pertencer ao mesmo `event_id`;
- uma captura local nunca pode reutilizar silenciosamente o último frame XP;
- uma captura XP nunca pode reutilizar silenciosamente um frame MSS;
- para rede, `Copiar imagem` continua usando exatamente o frame completo recebido do XP;
- para captura local, `ScreenMonitor` preserva exatamente o frame completo MSS que originou os recortes analisados;
- o relatório identifica a origem como `Windows XP` ou `Captura local MSS`;
- a interface usa o título genérico `DIAGNÓSTICO DA CAPTURA`.

Para captura local, o relatório usa o schema de observabilidade
`visionx.capture_debug.v1` e registra, quando disponíveis:

- frame MSS completo;
- recorte gabarito;
- recorte teste;
- informações AOI;
- decisão final;
- categoria;
- confiança;
- memória/KNN e métricas de ausência física já expostas pelo debug.

Essa camada é somente de observabilidade e não altera classificação, memória,
dataset, gate, confiança ou decisão.

### Estado padrão dos arquivos visuais NG e OK

Os controles `Salvar imagens NG` e `Salvar imagens OK` iniciam em:

```text
ATIVADO
```

em toda abertura do ODIN.

O operador continua podendo desativar cada arquivo independentemente durante a
sessão.

O NG mantém seu contrato atual: arquivamento automático restrito às evidências
de captura recebidas do Windows XP.

O OK possui contrato próprio: arquiva somente julgamentos humanos `OK`
(`button` ou `xp_keyboard`) e pode usar XP ou MSS, sempre através da mesma
evidência completa de `Copiar imagem`.

O arquivo OK não transforma decisões automáticas de Produção em evidência de
operador e não participa do dataset/KNN.

### Estado padrão do arquivo visual NG

O controle `Salvar imagens NG` inicia em:

```text
ATIVADO
```

em toda abertura do ODIN.

A mudança é somente do estado inicial do toggle. O operador continua podendo
desativá-lo a qualquer momento durante a sessão. O arquivo permanece assíncrono
e independente do dataset/KNN.

A regra de arquivamento automático continua restrita às evidências de captura
recebidas do Windows XP, conforme o contrato existente. Tornar o arquivamento
NG local/MSS automático exige uma decisão de produto separada e não deve ser
introduzido implicitamente.

### Regressões obrigatórias

Manter testes que garantam:

- captura XP habilita `Copiar debug` e `Copiar imagem`;
- captura local MSS analisada também habilita os dois controles;
- `Copiar imagem` local copia o frame MSS completo preservado, não o último frame XP;
- relatório local identifica `Captura local MSS`;
- `event_id` da imagem e do relatório sempre coincide;
- o toggle `Salvar imagens NG` inicia marcado/ativado;
- o operador ainda pode desativar o arquivamento durante a sessão.

### Validação operacional em 02/10/2026

O comportamento foi validado em uso real pelo operador após a implementação.

Foi confirmado que:

- imagens obtidas por **`Capturar nova peça (descarta a atual)`** também disponibilizam
  corretamente **`Copiar debug`** e **`Copiar imagem`**;
- a evidência copiada pertence à captura local MSS atual e não reutiliza um frame
  anterior recebido do Windows XP;
- o diagnóstico e a imagem permanecem associados ao mesmo `event_id`;
- o controle **`Salvar imagens NG`** inicia **ATIVADO** ao abrir o ODIN;
- o operador continua podendo desativar o arquivamento durante a sessão;
- o arquivamento automático de NG permanece restrito ao fluxo Windows XP, sem
  introduzir salvamento automático de capturas MSS.

Essa validação deve ser preservada como regressão operacional. Refatorações futuras
na camada de evidência não podem voltar a tornar os botões exclusivos do XP nem
permitir mistura silenciosa entre frames XP e MSS.

### Validação operacional da prévia da captura em 02/10/2026

A nova visualização **`CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA`**, posicionada na
seção **`IMAGENS DA INSPEÇÃO`** antes dos cards de gabarito/teste, foi validada
em uso real pelo operador.

Foi confirmado que:

- a prévia exibe corretamente a captura recebida do **Windows XP**;
- a prévia também exibe corretamente a captura local **MSS**;
- a imagem mostrada é exatamente a mesma evidência utilizada por
  **`Copiar imagem`**, sem criar uma segunda fonte visual independente;
- a proporção original da imagem é preservada durante o redimensionamento;
- o componente se adapta de forma responsiva ao espaço disponível na interface;
- o visual permanece consistente com os demais cards do ODIN;
- a prévia não altera classificação, KNN, decisão, ciclo produtivo nem regras de
  captura;
- permanece proibido reutilizar silenciosamente um frame XP antigo quando a
  evidência atual pertence a uma captura local MSS.

Essa prévia deve continuar sendo apenas uma camada de visualização da evidência
já validada pelo `event_id`. Refatorações futuras não devem duplicar a fonte da
imagem nem desacoplar a prévia do mesmo contrato usado por **`Copiar imagem`**.


### Limpeza visual ao entrar em AGUARDANDO PEÇA

Quando o painel de decisão retorna para:

```text
AGUARDANDO PEÇA
```

nenhuma imagem da inspeção anterior pode permanecer visível na seção de
inspeção.

Devem ser limpos visualmente:

- `CAPTURA RECEBIDA • EVIDÊNCIA COMPLETA`;
- gabarito em visão completa;
- foco do gabarito;
- teste em visão completa;
- foco do teste.

A limpeza é **somente visual**. Ela não apaga a evidência técnica da última
captura e não invalida, por si só:

- `Copiar debug`;
- `Copiar imagem`;
- `event_id`;
- frame completo preservado para auditoria.

O controller mantém o estado visual `_inspection_images_visible`:

- `False` em `AGUARDANDO PEÇA`;
- `True` quando uma nova inspeção válida começa a ser processada.

Enquanto esse estado estiver falso, `resizeEvent` não pode reconstruir os
pixmaps a partir de `current_sample` ou `current_ng`, e a sincronização do
diagnóstico não pode repopular a prévia completa com a evidência anterior.

Essa separação é obrigatória: **evidência preservada para auditoria não significa
imagem antiga visível na inspeção atual**.

Regressões obrigatórias:

- `_reset_confidence_panel()` limpa todos os visuais da peça anterior;
- a prévia completa mostra `Aguardando captura`;
- gabarito/teste mostram apenas placeholders de espera;
- resize durante espera não faz a peça anterior reaparecer;
- `Copiar imagem` continua disponível quando existe evidência válida;
- nova inspeção válida reabilita a exibição das imagens.
