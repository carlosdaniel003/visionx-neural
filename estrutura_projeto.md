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


## Melhoria em andamento — inspeção mult-iluminação para adesivo

### Objetivo

A próxima melhoria do ODIN para a categoria de excesso de adesivo passa a usar
as três iluminações disponíveis na AOI sobre a **mesma peça**.

A motivação visual observada é:

```text
SIDE → adesivo pouco evidente; útil como base geométrica/contextual
TOP  → adesivo muito escuro e bem marcado
MID  → adesivo muito claro/branco e bem marcado
```

A hipótese de trabalho é que `TOP` e `MID` fornecem evidências fotométricas
complementares do adesivo, enquanto `SIDE` continua servindo como imagem
inicial e referência de contexto.

### Mapeamento operacional confirmado da AOI

```text
← seta esquerda → TOP
↓ seta para baixo → SIDE
→ seta direita → MID
```

### Estado do agente Windows XP

Em 06/10/2026, a V5.2 foi copiada manualmente para o Windows XP e validada para
os comandos enviados pelo ODIN. Depois disso, a referência da branch `central`
evoluiu para:

```text
agente_industrial_xp.py V5.3
```

A V5.3 mantém:

```text
PRESS_0
PRESS_1
PRESS_LEFT
PRESS_DOWN
PRESS_RIGHT
```

e acrescenta a telemetria XP → ODIN das setas físicas por
`CMD_TOP/CMD_SIDE/CMD_MID`.

Até nova confirmação operacional, o estado documentado é: **GitHub em V5.3 e
Windows XP ainda precisa receber manualmente essa versão e reiniciar o agente**.

### Controle manual do ODIN compatível com a AOI

Em 06/10/2026, o controle manual de iluminação do ODIN foi corrigido para usar
o mesmo mapeamento operacional do Windows XP:

```text
← → TOP  → PRESS_LEFT
↓ → SIDE → PRESS_DOWN
→ → MID  → PRESS_RIGHT
```

O estado inicial exibido pelo ODIN passa a ser `SIDE`, que é a iluminação
padrão da captura recebida da AOI.

Os três botões de iluminação também seguem a ordem visual e os atalhos corretos:

```text
Luz TOP  | ←
Luz SIDE | ↓
Luz MID  | →
```

A troca manual só atualiza o estado visual depois que o envio TCP do comando ao
agente XP retorna com sucesso. Se o comando falhar, o ODIN não deve fingir que a
iluminação mudou.

O feedback temporário de tecla no canto inferior direito foi ampliado:

- `0/1` continuam exibindo a tecla e `OK/NG`;
- `←/↓/→` reutilizam o mesmo card e mostram explicitamente qual seta foi
  pressionada/enviada;
- feedback de seta não pode disparar o fade-out do veredito da IA.

Existe ainda um terceiro card flutuante em
`src/ui/lighting_status_feedback.py`. Ele fica no canto superior direito,
abaixo do card de veredito, e mostra:

```text
ILUMINAÇÃO ATUAL
TOP  ←
SIDE ↓
MID  →
```

O estado de iluminação é atualizado internamente durante os comandos, mas o
card **permanece oculto enquanto a análise ainda não terminou**. Ele só aparece
quando existe um veredito final válido, no mesmo evento visual em que o card
`FALHA FALSA / DEFEITO REAL / REVISÃO OBRIGATÓRIA` é exibido.

Ao ocorrer o julgamento `0/1`, o card de iluminação é preservado durante o
reset produtivo e inicia o mesmo fade-out sincronizado do veredito e do feedback
de tecla. Portanto os três elementos encerram juntos o ciclo visual da peça.

Esse card é somente de apresentação: não envia comandos, não altera análise,
gate, KNN, dataset ou decisão.

Após o primeiro teste operacional em 06/10/2026 foram encontrados dois pontos:

- os botões `Luz TOP/SIDE/MID` funcionaram e a AOI respondeu corretamente;
- as setas do teclado do próprio ODIN não disparavam de forma confiável porque
  dependiam de `keyPressEvent` do painel e o foco podia estar em widgets filhos;
- as setas físicas do Windows XP mudavam a AOI, mas o agente V5.2 não enviava
  essa mudança de volta ao ODIN.

Correção implementada:

- `src/ui/lighting_shortcuts.py` instala `QShortcut` com
  `WindowShortcut`, tornando `←/↓/→` válidas em toda a janela do ODIN e
  preservando as travas dos botões;
- o agente foi evoluído para V5.3 e passa a enviar `CMD_TOP`, `CMD_SIDE` e
  `CMD_MID` quando as setas são detectadas pelo hook global;
- comandos recebidos da rede atualizam o card fixo e mostram o mesmo feedback
  temporário de tecla, com origem `TECLADO WINDOWS XP`;
- o eco de uma seta que foi originalmente enviada pelo ODIN continua coberto
  pela supressão temporal do feedback.

A correção do ODIN está implementada. A telemetria XP → ODIN depende de copiar a
V5.3 do `agente_industrial_xp.py` para o Windows XP e reiniciar o agente.

### Etapas A e B — interface e alimentação visual implementadas

O ODIN agora possui uma área de inspeção específica para adesivo em
`src/ui/adhesive_multilight_inspection.py`.

Ativação:

```text
ADESIVO
ADHESIVE
MUITO ADESIVO
MUCH ADHESIVE
EXCESS ADHESIVE
ADESIVO EM EXCESSO
        ↓
categoria canônica = MUITO ADESIVO
        ↓
layout multilight
```

As entradas `ADESIVO` e `ADHESIVE` também passam a ser aliases oficiais de
`MUITO ADESIVO` no normalizador. Para qualquer outra categoria, a pilha visual
retorna ao painel normal já existente e o comportamento anterior é preservado.

Cada iluminação possui exatamente três visões:

```text
SIDE
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor

TOP
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor

MID
├── imagem TESTE recebida
├── recorte do retângulo maior
└── recorte do retângulo menor
```

O retângulo maior vem de `global_box_info` produzido por
`detect_anomalies()`. O retângulo menor usa o epicentro selecionado por
`EpicenterExtractor.extract_focus()`. Portanto os recortes visuais reutilizam
a mesma geometria já empregada pelo pipeline atual e não introduzem uma segunda
regra de detecção de caixas.

#### Alimentação das três iluminações

A primeira imagem de uma peça de adesivo continua entrando no pipeline normal e
é registrada visualmente como `SIDE`, conforme o contrato operacional da AOI.

Depois que a análise inicial está ativa, o receptor pode aceitar frames
auxiliares da mesma peça mesmo com o gate principal fechado. Essa exceção é
explicitamente visual:

- `NetworkReceiver.set_auxiliary_image_mode(True)` não reabre o gate principal;
- o frame auxiliar não cria novo `event_id`;
- o frame auxiliar não substitui `current_sample`, `current_ng` ou
  `current_analysis`;
- o frame auxiliar executa uma análise MoE isolada da iluminação para alimentar
  os painéis de especialistas, mas essa análise não substitui
  `current_analysis` nem participa do resultado final;
- a iluminação atribuída ao preview vem do estado atual
  `SIDE/TOP/MID` já comandado/confirmado pelo ODIN;
- ao julgar ou descartar a peça, o modo auxiliar é desligado antes da liberação
  do próximo ciclo.

Assim, quando novas imagens chegarem após o operador mudar a iluminação, elas
preenchem o card correspondente sem transformar TOP/MID em novas peças.

A alimentação visual agora está conectada à automação de aquisição. Depois que a
primeira inspeção válida de adesivo em `SIDE` termina, o ODIN inicia uma máquina
de estados não bloqueante em `src/ui/adhesive_multilight_automation.py`. A
fusão das três iluminações na decisão continua separada e ainda não foi
implementada.

#### Análise dos especialistas por iluminação

A categoria de adesivo também possui uma página visual própria na seção
`ANÁLISE DOS ESPECIALISTAS`, implementada em:

```text
src/ui/adhesive_multilight_analysis.py
```

Ela apresenta três grupos independentes:

```text
ANÁLISE SIDE ↓
ANÁLISE TOP  ←
ANÁLISE MID  →
```

Cada grupo já possui a mesma estrutura de especialistas da interface normal:

- SSIM • textura e calor;
- XOR • tinta e epicentro;
- DNA • assinatura semântica;
- SHIFT • deslocamento;
- FUSÃO • score final como fallback quando não há motor ativo.

**Contrato atual:** `SIDE`, `TOP` e `MID` possuem análises visuais
independentes. SIDE continua usando o `current_analysis` principal produzido
pelo fluxo original. Quando os frames auxiliares TOP e MID chegam, cada um
percorre o mesmo pipeline técnico:

```text
detect_anomalies
        ↓
EpicenterExtractor.extract_focus
        ↓
MoEOrchestrator.inspect
        ↓
painéis de especialistas daquela iluminação
```

O contexto geométrico calculado para TOP/MID é compartilhado entre o preview
visual e o MoE para evitar executar novamente `detect_anomalies` e
`EpicenterExtractor` sobre o mesmo frame.

As análises são armazenadas separadamente em
`adhesive_multilight_analyses["SIDE"|"TOP"|"MID"]`. TOP e MID recebem
metadados explícitos:

```text
multilight_visual_analysis = True
eligible_for_final_decision = False
lighting_mode = TOP | MID
```

Contrato atual:

- TOP não sobrescreve a análise SIDE durante a coleta;
- MID não sobrescreve a análise SIDE durante a coleta;
- os `is_defect`, `confidence` e `verdict` locais continuam sendo saídas
  auditáveis de cada iluminação;
- depois que SIDE, TOP e MID terminam, uma quarta análise lógica é criada:
  a fusão final de `src/core/adhesive_multilight_fusion.py`;
- somente essa análise fundida recebe `eligible_for_final_decision = True` e
  substitui `current_analysis` para o julgamento da peça;
- a decisão 0/1 automática de Produção usa exclusivamente o resultado fundido.

A automação só avança da captura TOP para MID, e de MID para conclusão, depois
que a imagem **e a análise visual** da iluminação esperada foram concluídas.

#### Debug técnico e Copiar imagem para adesivo

O diagnóstico copiável também passa a respeitar o contrato multilight somente
quando o evento atual pertence à categoria canônica `MUITO ADESIVO`.

**Copiar debug**

O relatório técnico original do evento SIDE é preservado e recebe uma seção
adicional:

```text
ANÁLISES MULTILIGHT - ADESIVO

ILUMINAÇÃO SIDE
ILUMINAÇÃO TOP
ILUMINAÇÃO MID
```

Para cada iluminação são registrados, entre outros:

- motores ativos;
- veredito local do MoE, identificado explicitamente como não sendo o resultado
  final multilight;
- flag local de defeito;
- confiança local;
- score final local;
- score físico local;
- regra de fusão local;
- motor dominante local;
- motivo local;
- detalhes técnicos compactos em JSON.

Matrizes e imagens internas não são despejadas pixel a pixel no clipboard.
Arrays NumPy são resumidos por `shape`, `dtype`, mínimo, máximo e média, e
listas muito grandes são resumidas. Isso mantém o debug técnico copiável sem
perder a estrutura necessária para diagnóstico.

O relatório agora inclui também uma seção `JULGAMENTO FINAL MULTILIGHT`
com veredito, score final, score físico máximo, regra de fusão, iluminação
dominante, iluminações positivas, auxiliares fortes e o papel da memória KNN.

**Copiar imagem**

Para o mesmo `event_id` de adesivo, o ODIN preserva também os frames completos
de origem de cada iluminação:

```text
SIDE | TOP | MID
```

O botão `Copiar imagem` só considera o conjunto multilight pronto quando os
três frames estão disponíveis. Em vez de copiar apenas SIDE, ele cria uma única
imagem composta:

```text
┌────────────┬────────────┬────────────┐
│    SIDE    │    TOP     │    MID     │
├────────────┼────────────┼────────────┤
│ frame SIDE │ frame TOP  │ frame MID  │
└────────────┴────────────┴────────────┘
```

As imagens ficam lado a lado, com separadores próprios e sem sobreposição.
Os frames não são redimensionados para montar a composição; o canvas apenas
acomoda as alturas e larguras originais e adiciona um cabeçalho externo com o
nome da iluminação.

A evidência multilight persistida para debug é vinculada ao `event_id` da
captura SIDE original. Se o evento atual mudar, um conjunto multilight antigo
não pode ser reutilizado por `Copiar debug` ou `Copiar imagem`.

Essa mudança é exclusiva do computador novo e não altera
`agente_industrial_xp.py`.

A página normal de especialistas permanece como padrão para todas as categorias
que não sejam adesivo. Imagens multilight e especialistas multilight são
trocados juntos através do mesmo modo condicional de adesivo.

#### Responsividade

As áreas de imagens e especialistas usam páginas empilhadas: o painel antigo
permanece como página padrão e as páginas multilight só são selecionadas para
adesivo.

O layout multilight refluí em dois níveis:

**Imagens**

- largura interna abaixo de `1000 px`: uma iluminação por linha;
- largura interna a partir de `1000 px`: SIDE, TOP e MID em três colunas;
- os nove viewports preservam proporção com
  `KeepAspectRatio + SmoothTransformation`.

**Especialistas**

- abaixo de `1500 px`: SIDE, TOP e MID ficam empilhados verticalmente;
- a partir de `1500 px`: SIDE, TOP e MID ficam em três colunas;
- dentro de cada iluminação, os especialistas usam scroll horizontal para não
  serem comprimidos abaixo de uma largura útil.

Durante uma inspeção de adesivo o splitter principal fica vertical, dando a
largura inteira primeiro para as imagens e depois para os especialistas. Em
notebooks o conteúdo cresce verticalmente dentro do `root_scroll`; em
monitores grandes a largura adicional é aproveitada para três colunas.

Ao sair da categoria de adesivo, o splitter e as páginas normais voltam ao
comportamento responsivo anterior. Portanto a mudança não altera visualmente as
outras categorias.

### Automação implementada da captura mult-iluminação

A aquisição automática de imagens para adesivo está implementada em:

```text
src/ui/adhesive_multilight_automation.py
```

Fluxo atual:

```text
mesma peça / mesmo ciclo
        ↓
SIDE recebido e analisado pelo pipeline atual
        ↓
categoria canônica = MUITO ADESIVO?
   ├── NÃO → fluxo normal atual
   └── SIM
        ↓
preservar SIDE
        ↓
habilitar recepção auxiliar sem reabrir o gate principal
        ↓
PRESS_LEFT
        ↓
AOI seleciona TOP
        ↓
aguardar frame TOP
        ↓
armazenar TOP na mesma sessão
        ↓
PRESS_RIGHT
        ↓
AOI seleciona MID
        ↓
aguardar frame MID
        ↓
armazenar MID na mesma sessão
        ↓
fechar recepção auxiliar
        ↓
PRESS_DOWN
        ↓
restaurar SIDE
        ↓
captura automática concluída
```

Não existe retorno intermediário para SIDE entre TOP e MID, porque as setas da
AOI são tratadas como seletores absolutos:

```text
LEFT  = TOP
DOWN  = SIDE
RIGHT = MID
```

### Validação operacional da automação

Em 06/10/2026, o fluxo automático de troca de iluminação e recebimento das
imagens foi testado na AOI real pelo operador e confirmado como funcional.

O comportamento validado foi:

```text
SIDE inicial
→ TOP automático + foto recebida
→ MID automático + foto recebida
→ retorno automático para SIDE
```

Essa validação confirma a infraestrutura de aquisição na AOI real. A fusão
multilight foi implementada depois dessa validação operacional e ainda precisa
ser validada em novos casos reais OK/NG antes de qualquer calibração adicional.

A máquina de estados somente avança após o frame esperado ter sido recortado e
armazenado. Para cada `TOP` ou `MID`:

- timeout: `8000 ms`;
- após o primeiro timeout, o mesmo seletor absoluto é enviado uma vez novamente;
- após uma segunda falha, a automação é interrompida, SIDE é restaurada e o
  status de rede informa a falha;
- em falha automática, a recepção auxiliar permanece disponível para fallback
  manual da mesma peça.

Enquanto a sequência está ativa, os botões/setas de iluminação do próprio ODIN
não podem trocar manualmente o modo no meio de TOP/MID. Isso evita que uma foto
seja armazenada sob a iluminação errada.

Ao julgar ou descartar a peça antes da conclusão, qualquer timer pendente é
cancelado, a recepção auxiliar é fechada e a AOI é devolvida para SIDE antes do
próximo ciclo.

#### Modo Produção durante a automação

O pipeline SIDE continua calculando seu resultado normalmente. Entretanto, em
`Modo Produção`, o `PRESS_0/PRESS_1` automático não pode avançar a AOI antes
das fotos auxiliares terminarem.

Contrato:

```text
resultado local SIDE calculado
        ↓
não promover SIDE a julgamento final
        ↓
capturar + analisar TOP
        ↓
capturar + analisar MID
        ↓
restaurar SIDE
        ↓
fuse_adhesive_multilight(SIDE, TOP, MID)
        ↓
resultado final único
        ↓
Modo Produção sem revisão?
        ├── NÃO → REVISÃO OBRIGATÓRIA, sem PRESS_0/PRESS_1 automático
        └── SIM → save_label(resultado fundido, source="auto")
                    ↓
                 PRESS_0 / PRESS_1
```

Assim a mesma peça permanece na tela durante toda a coleta. Em `Modo Teste` ou
`Modo Sombra`, ações de julgamento iniciadas no próprio ODIN são recusadas
enquanto a automação está ativa, com mensagem para aguardar a sequência.

O teclado físico do Windows XP continua sendo um controle externo à aplicação;
o operador não deve julgar a peça fisicamente com `0/1` enquanto a sequência
automática TOP/MID estiver em andamento, pois a própria AOI pode avançar antes
que o ODIN consiga impedir a ação.

### Fusão final multilight de adesivo

A primeira política de fusão está implementada em:

```text
src/core/adhesive_multilight_fusion.py
```

Ela é exclusiva da categoria canônica `MUITO ADESIVO` e não usa votação
majoritária nem média simples entre SIDE/TOP/MID.

Motivo: uma iluminação pode ocultar o adesivo sem isso significar que o adesivo
não existe. Portanto um resultado local negativo não tem o mesmo significado de
uma evidência física positiva forte.

Política inicial:

```text
TOP ou MID com:
  adhesive_is_defect = True
  adhesive_score >= 0.80
  physical_score >= 0.80
        ↓
DEFEITO REAL

ou

duas iluminações com adhesive_score >= tolerância do motor
        ↓
DEFEITO REAL

ou

somente uma iluminação positiva, sem força auxiliar suficiente
        ↓
REVISÃO OBRIGATÓRIA

ou

nenhuma das três com evidência física positiva
        ↓
FALHA FALSA
```

TOP e MID são as testemunhas fotométricas prioritárias; SIDE continua útil como
testemunha contextual/corroboradora. O KNN de cada iluminação é preservado para
auditoria, mas a memória local recebe papel `audit_only` na fusão e não pode
vetar evidência física multilight forte.

A análise dominante da fusão é a iluminação com maior `adhesive_score`, usando
`physical_score` e prioridade auxiliar como desempate. O resultado final
registra:

- `lighting_mode = MULTILIGHT`;
- `multilight_final = True`;
- `eligible_for_final_decision = True`;
- `adhesive_multilight_positive_modes`;
- `adhesive_multilight_strong_auxiliary_modes`;
- `adhesive_multilight_dominant_mode`;
- `fusion_rule`;
- `memory_role = audit_only`.

O tempo de análise de adesivo também passa a terminar somente após SIDE, TOP e
MID terem sido analisadas, a fusão final ter sido calculada e o resultado ter
sido pintado na interface.

A etapa seguinte continua separada: melhorar a capacidade específica do motor de
adesivo na iluminação MID. A política de fusão não modifica internamente o
detector MID.

#### Caso real usado como referência da primeira política

No teste de 06/10/2026 que motivou esta fusão, a mesma peça apresentou:

```text
SIDE
  veredito local = FALHA FALSA
  adhesive_score ≈ 0.792
  physical_score = 0.85
  motor de adesivo físico = positivo
  KNN = OK ≈ 96.4%

TOP
  veredito local = DEFEITO REAL
  adhesive_score ≈ 0.972
  physical_score ≈ 0.972
  KNN = NG 100%

MID
  veredito local = FALHA FALSA
  adhesive_score = 0.0
  physical_score ≈ 0.190
```

Pela nova fusão, TOP é uma testemunha auxiliar forte e SIDE também fornece
corroboração física. O único resultado final esperado para esse conjunto é
`DEFEITO REAL`, independentemente do veto KNN que anteriormente fazia a
primeira imagem SIDE encerrar a peça como `FALHA FALSA`.

Esse caso foi usado como referência funcional da regra inicial. Após a
implementação da fusão, o operador repetiu o fluxo na AOI real e confirmou que o
novo julgamento multilight funcionou corretamente. A fusão inicial está,
portanto, **validada operacionalmente** para esse caso real.

Isso não encerra a calibração estatística: novos casos reais OK/NG ainda devem
ser usados para avaliar os limiares e reduzir risco de falso positivo/falso
negativo.

### Detector MID claro — `mid_bright_resin_v1`

A análise real mostrou que a iluminação MID deixava o adesivo visualmente muito
evidente, porém o motor físico retornava:

```text
adhesive_score = 0
adhesive_is_defect = False
reference_area_pct = 0
test_area_pct = 0
```

A causa era fotométrica: o detector original de
`src/core/experts/adhesive_shift_expert.py` foi construído para material
quente, saturado e relativamente escuro. Na MID, a mesma película pode aparecer
quase branca/creme/amarelada e, por isso, desaparecer da máscara antiga.

Foi implementado um segundo perfil, ativado **somente quando
`lighting_mode == "MID"`**:

```text
perfil padrão SIDE/TOP = dark_warm_v1
perfil MID             = mid_bright_resin_v1
```

O perfil MID não classifica simplesmente pixels claros como adesivo. Ele compara
gabarito e teste no espaço LAB e exige uma mudança cromática local coerente:

```text
gabarito MID + teste MID
        ↓
ΔE em LAB
        +
ganho amarelo/vermelho (b*/a*)
        +
presença quente no teste
        +
brilho alto
        -
supressão de cobre saturado
        ↓
mid_bright_resin_witness
```

A testemunha é **diferencial**: uma região branca estável presente nas duas
imagens não gera evidência. Uma mudança apenas neutra de brilho também não deve
ser suficiente. O perfil foi criado para destacar a película clara/creme que
aparece no TESTE e não está no GABARITO.

Integração com o motor existente:

- o detector escuro/quente original continua funcionando e não foi removido;
- SIDE e TOP permanecem no perfil legado, sem novos limiares;
- MID combina o material legado com a nova testemunha clara;
- o limiar de material da máscara TESTE em MID passa a `0.18`; o gabarito
  continua usando `0.22`;
- a cobertura coerente da testemunha MID reforça o `adhesive_score`, mas não
  substitui as métricas físicas já existentes de excesso, padding, expansão,
  espalhamento e vazamento;
- a tolerância final do motor permanece `0.32`.

Novas telemetrias:

```text
adhesive_lighting_mode
adhesive_detector_profile
adhesive_material_threshold
mid_bright_witness_coverage
mid_bright_witness_peak
mid_bright_witness_score
mid_bright_witness_mask
```

Esses valores aparecem no **Copiar debug** da iluminação MID e também no painel
`FLUXO DE ADESIVO`, permitindo verificar se o ODIN realmente passou a enxergar
a película clara.

Regressões adicionadas:

- película clara/amarelada diferencial em MID deve ser detectada;
- a mesma aparência clara não ativa o perfil novo em TOP;
- MID idêntica entre gabarito e teste continua estável;
- mudança neutra de luminosidade sem ganho amarelo/vermelho não deve virar
  adesivo;
- uma MID fisicamente forte pode ser a testemunha dominante da fusão final.

Esta implementação foi validada operacionalmente na AOI real com o caso que
antes deixava a iluminação MID matematicamente cega. O operador confirmou que,
após a inclusão do perfil `mid_bright_resin_v1`, o fluxo passou a funcionar
corretamente também na MID.

Estado atual validado:

```text
SIDE → análise física
TOP  → análise física
MID  → análise física com mid_bright_resin_v1
        ↓
fusão SIDE/TOP/MID
        ↓
um único julgamento final
```

A validação confirma o comportamento funcional do detector MID no caso real
testado. Isso ainda não substitui uma calibração estatística ampla: novos casos
OK/NG devem continuar sendo coletados antes de alterar limiares ou pesos.

### Restrições arquiteturais da melhoria

A implementação atual e as próximas etapas devem preservar os seguintes contratos:

- `SIDE`, `TOP` e `MID` pertencem à mesma peça e não podem virar três
  inspeções independentes;
- as capturas auxiliares não podem ganhar `event_id` de peças diferentes;
- o gate de rede precisa distinguir imagens auxiliares da sessão
  mult-iluminação de uma nova peça real;
- não confiar apenas na ordem temporal; uma captura precisa corresponder ao
  estado de iluminação solicitado antes de ser aceita;
- o ODIN deve restaurar `SIDE` ao final da sequência para não deixar a AOI em
  uma iluminação inesperada;
- falha ao obter `TOP` ou `MID` não deve fabricar evidência ausente nem
  reutilizar silenciosamente um frame anterior;
- cada motor local continua monoimagem e auditável; a fusão acontece em uma
  camada posterior, sem alterar os cálculos internos de SIDE/TOP/MID;
- a memória KNN local permanece disponível para diagnóstico, mas não pode vetar
  evidência física multilight forte;
- a política atual não recalibra ainda o detector específico da iluminação MID.

### Ordem de trabalho

A melhoria será executada por etapas, sem avançar automaticamente:

1. **Concluído — agente XP:** comandos de setas disponíveis; a referência
   atual do GitHub é V5.3 e a atualização operacional continua manual no XP.
2. **Correção implementada — controle manual do ODIN:** botões e
   `QShortcut` usam `← TOP / ↓ SIDE / → MID`; V5.3 suporta retorno
   `CMD_TOP/CMD_SIDE/CMD_MID`.
3. **Concluído — Etapa A:** layout multilight condicional e responsivo para
   adesivo, com nove viewports.
4. **Concluído — Etapa B visual:** SIDE usa a primeira inspeção normal e frames
   auxiliares recebidos depois podem preencher TOP/MID sem criar nova peça nem
   substituir a análise ativa.
5. **Concluído — especialistas por iluminação:** somente para adesivo, a seção
   de especialistas possui SIDE/TOP/MID; SIDE mostra a análise real atual e
   TOP/MID permanecem aguardando análise, sem cálculo artificial.
6. **Concluído e validado na AOI — automação de aquisição:** após SIDE, o
   ODIN comanda TOP, aguarda a foto, comanda MID, aguarda a foto e restaura
   SIDE; timeout, repetição única e cancelamento de ciclo são tratados pela
   máquina de estados.
7. **Concluído — análise visual por iluminação:** SIDE, TOP e MID executam
   análises independentes dos especialistas e alimentam seus próprios painéis;
   TOP/MID não substituem `current_analysis` e não entram no veredito final.
8. **Concluído — debug/evidência multilight:** `Copiar debug` reúne as três
   análises e `Copiar imagem` gera uma única composição SIDE/TOP/MID sem
   sobreposição, vinculada ao mesmo `event_id`.
9. **Concluído e validado operacionalmente — fusão multilight inicial:**
   SIDE/TOP/MID formam um único julgamento final físico; TOP/MID fortes têm
   autoridade, duas iluminações positivas corroboram defeito, caso intermediário
   exige revisão e o KNN local permanece apenas como auditoria na fusão.
10. **Concluído e validado operacionalmente — detector MID claro v1:** o
    perfil `mid_bright_resin_v1` adiciona testemunha diferencial LAB para
    película clara/creme/amarelada exclusivamente na iluminação MID,
    preservando SIDE/TOP no detector legado. O caso real que antes gerava
    `adhesive_score = 0` na MID foi repetido e o comportamento foi confirmado
    como correto pelo operador.
11. **Próxima etapa — calibração com mais casos reais:** coletar novos exemplos
    OK/NG de adesivo em SIDE/TOP/MID antes de reajustar qualquer limiar ou peso.

A aquisição, as três análises visuais, a fusão final e o detector específico de
MID estão implementados e validados operacionalmente nos casos reais testados.
O próximo trabalho recomendado é aumentar a base de validação antes de novas
mudanças de regra.


**Arquivo visual NG opcional:**
- Toggle **ativado por padrão** em toda inicialização do ODIN. O operador pode desativá-lo manualmente durante a sessão.
- Para categorias comuns, um julgamento final `NG` de captura XP arquiva o frame completo do evento atual em `public/ng_archive/`.
- Para a categoria canônica `MUITO ADESIVO`, quando SIDE/TOP/MID pertencem ao mesmo `event_id`, o mesmo julgamento pode arquivar **até três imagens completas separadas**:
  - `..._MUITO_ADESIVO_SIDE.png`
  - `..._MUITO_ADESIVO_TOP.png`
  - `..._MUITO_ADESIVO_MID.png`
- A resolução dessas imagens fica em `src/services/image_archive_candidates.py` e usa `adhesive_multilight_last_source_frames` somente quando o `event_id` do conjunto coincide com o evento julgado.
- Não existe fallback para `current_ng` ou outro recorte. Se a evidência completa do evento não estiver disponível, nenhuma imagem substituta é arquivada.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.
- Deduplicação de eco por `event_id` continua obrigatória: o mesmo julgamento não pode ser arquivado duas vezes se o comando voltar pelo hook do XP.
- **Deduplicação persistente por conteúdo visual:** antes de gravar, a fila indexa os PNGs já existentes e calcula SHA-256 do conteúdo visual exato. Se uma imagem pixel a pixel idêntica já existir em `public/ng_archive/`, ela é ignorada mesmo que venha de outro evento ou após reiniciar o ODIN.
- Imagens realmente diferentes continuam sendo preservadas. Se duas imagens diferentes caírem no mesmo minuto/categoria, o ODIN cria um nome alternativo `_2`, `_3`, etc., em vez de sobrescrever o arquivo anterior.
- O arquivamento só é permitido enquanto existe uma captura de rede ativa, com análise ativa e categoria AOI não vazia.
- `SEM_CATEGORIA` não é um nome de arquivo válido para o fluxo automático de evidências NG.


**Arquivo visual OK opcional:**
- Existe um segundo toggle **`Salvar imagens OK`**, exibido imediatamente abaixo de **`Salvar imagens NG`**.
- O toggle inicia **ATIVADO por padrão** em toda abertura do ODIN e o operador pode desativá-lo durante a sessão.
- Visualmente, o bloco OK deve manter o mesmo layout, dimensões, tipografia, hover, focus e estado checked do bloco NG.
- Quando ativado, cada julgamento humano final `OK` arquiva a evidência do evento em `public/ok_archive/`; para adesivo multilight, o mesmo julgamento pode gerar SIDE/TOP/MID como três PNGs separados.
- Julgamentos humanos aceitos: botão/atalho do ODIN (`source="button"`) e teclado físico do XP (`source="xp_keyboard"`).
- Decisão automática de Produção (`source="auto"`) **não** gera arquivo OK.
- Para categoria comum, a imagem salva continua sendo a evidência completa resolvida por `Copiar imagem`. Para adesivo multilight, em vez da composição visual, são preservados os três frames completos individuais SIDE/TOP/MID da mesma peça.
- O contrato compartilhado de evidência fica em `src/services/capture_evidence.py`, por meio de `current_copy_image_snapshot()` e `current_copy_image_event_id()`.
- O arquivo OK aceita tanto captura recebida do **Windows XP** quanto captura local **MSS**, desde que exista análise ativa, `event_id` válido e categoria AOI válida.
- Uma captura local MSS nunca pode usar como fallback um frame XP anterior.
- Não usar `current_ng`, ROI, foco ou outro recorte como imagem substituta.
- O formato do nome é o mesmo do arquivo NG: `YYYY-MM-DD_HHmm_CATEGORIA.png`.
- A implementação de nome compartilhada fica em `src/services/image_archive_naming.py`.
- `SEM_CATEGORIA` não é permitido no arquivamento automático OK.
- A gravação é assíncrona em fila daemon e não pode bloquear julgamento, envio de tecla, limpeza da interface ou recepção da próxima captura.
- Deduplicação obrigatória por `event_id`: um mesmo evento não pode ser salvo duas vezes caso o julgamento retorne pelo hook do XP.
- **Deduplicação persistente por conteúdo visual em OK e NG:** se uma imagem pixel a pixel idêntica já existir no respectivo arquivo visual, um novo julgamento dessa mesma imagem não cria outro PNG, mesmo que apareça muitos eventos depois ou após reiniciar o ODIN.
- A verificação é feita pelo conteúdo da imagem, não pelo nome do arquivo nem pelo `event_id`. Portanto arquivos antigos com o padrão de nome legado também contam como duplicatas se contiverem exatamente os mesmos pixels.
- A fila OK indexa os PNGs já existentes em background para não bloquear o julgamento. Novas imagens realmente diferentes continuam sendo salvas normalmente.
- O mesmo utilitário compartilhado em `src/services/image_archive_dedup.py` é usado pelas filas OK e NG.
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

### Arquivamento multilight de adesivo após julgamento 0/1

Para a categoria de adesivo, o conjunto SIDE/TOP/MID já existe no computador
novo antes do julgamento final. Ao ocorrer `0 = OK` ou `1 = NG`, e com o
respectivo arquivo visual habilitado, o ODIN usa o mesmo `event_id` da peça
para resolver as três evidências completas.

Fluxo:

```text
SIDE + TOP + MID da mesma peça
        ↓
julgamento 0 ou 1
        ↓
resolver imagens do mesmo event_id
        ↓
para cada imagem:
  fingerprint SHA-256 do conteúdo
        ↓
já existe no arquivo visual?
   ├── SIM → não salvar novamente
   └── NÃO → salvar PNG
```

Os arquivos usam a iluminação no nome:

```text
YYYY-MM-DD_HHmm_MUITO_ADESIVO_SIDE.png
YYYY-MM-DD_HHmm_MUITO_ADESIVO_TOP.png
YYYY-MM-DD_HHmm_MUITO_ADESIVO_MID.png
```

Se duas imagens realmente diferentes produzirem o mesmo nome base no mesmo
minuto, a segunda recebe sufixo incremental `_2`, depois `_3`, evitando
sobrescrita silenciosa.

A deduplicação é exata, por conteúdo pixel a pixel, e é persistente entre
reinicializações porque os PNGs já existentes são indexados ao iniciar a fila.
OK e NG mantêm índices separados, pois representam arquivos de auditoria com
rótulos distintos.

Essa mudança ocorre apenas no computador novo e não exige alteração no agente
Windows XP.

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


### Caso observado em 02/10/2026 — ROI interna estreita escondia o envelope do componente

Evento: `4a05d237e8c44c44a1c2dbd996afbb51`.

A ocorrência foi classificada pela AOI como `FALTANDO`, porém visualmente o
componente continuava presente. O ODIN concluiu incorretamente `DEFEITO REAL`
por `missing_hard_absence`.

O debug mostrou:

- `missing_score ≈ 96,50%`;
- cobertura divergente ≈ `57,43%`;
- residual médio ≈ `59,22%`;
- perda estrutural ≈ `52,61%`;
- similaridade direta ≈ `44,85%`;
- exposição de fundo = `0%`;
- melhor similaridade próxima do motor físico ≈ `26,16%`;
- melhor memória KNN = `OK` com ≈ `91,85%`;
- melhor memória NG ≈ `87,66%`;
- sem conflito de memória efetivo.

A testemunha de presença havia sido executada somente na caixa estreita:

```text
[128, 60, 101, 480]
```

Nessa região interna, que concentra serigrafia/conteúdo do componente, as
métricas ficaram incompatíveis:

- `silhouette_dice ≈ 17,3%`;
- `area_ratio ≈ 3,81`;
- `box_height_ratio ≈ 3,31`.

Entretanto, a própria AOI também havia detectado a caixa verde externa do
componente, aproximadamente:

```text
[26, 26, 307, 514]
```

Essa caixa global não chegava ao especialista porque `detect_anomalies()`
preservava somente `w/h` em `global_box_info`, descartando `x/y`.

#### Correção arquitetural

`src/core/inspection.py` passa a preservar:

```text
global_box_info = {
    x,
    y,
    w,
    h,
    detected
}
```

O `MissingComponentExpert` ganhou uma testemunha complementar de **envelope
global**.

Ela não declara OK e não apaga a divergência física. Sua única função é impedir
que uma ROI interna estreita tenha autoridade absoluta para declarar
`missing_hard_absence` quando o envelope completo do componente permanece
estruturalmente coerente.

A testemunha compara, no envelope global:

- perfil horizontal de baixa frequência;
- perfil vertical de baixa frequência;
- similaridade coarse do corpo;
- exposição de fundo.

Contrato:

```text
ROI interna sugere hard missing
        ↓
envelope global AOI detectado
        ↓
perfis horizontal + vertical preservados
+ coarse similarity compatível
+ sem exposição relevante de fundo
        ↓
missing_global_envelope_support = True
missing_global_envelope_veto = True
        ↓
missing_hard_absence = False
        ↓
divergência física continua existente
        ↓
decisão retorna à fusão normal física + KNN
```

Importante: o envelope global **não produz FALHA FALSA diretamente**. Ele apenas
remove a autoridade especial de ausência física forte. A memória e os demais
motores continuam responsáveis pelo veredito final.

Isso é importante neste caso porque o KNN já possui correspondência OK forte
(acima de 90%) e só estava impedido de atuar pela supressão de hard missing.

#### Segurança

A caixa global só é usada quando foi realmente detectada pela AOI
(`global_box_info.detected=True`). O fallback de frame inteiro não pode servir
como testemunha de presença.

Uma remoção real do componente deve continuar produzindo baixo suporte do
envelope e manter `missing_hard_absence=True`.

#### Telemetria obrigatória

O debug passa a registrar:

- `missing_global_envelope_active`;
- `missing_global_envelope_support`;
- `missing_global_envelope_veto`;
- `missing_global_envelope_box`;
- `missing_global_envelope_row_profile`;
- `missing_global_envelope_col_profile`;
- `missing_global_envelope_coarse_similarity`;
- `missing_global_envelope_background_exposure`;
- `missing_global_envelope_reason`.

#### Validação operacional em 02/10/2026

O caso foi retestado na interface real após a correção e o operador confirmou
que o falso positivo foi resolvido.

Foi validado que:

- o componente presente deixou de ser tratado como ausência física forte;
- o envelope global da AOI passou a participar da verificação de presença;
- a ROI interna estreita não possui mais autoridade isolada para impor
  `missing_hard_absence`;
- o hard missing foi rebaixado quando o envelope completo permaneceu coerente;
- a divergência física continuou registrada, sem ser apagada;
- a decisão retornou à fusão normal física + KNN;
- a memória OK forte pôde voltar a participar da decisão;
- o veredito final passou corretamente para **FALHA FALSA**;
- a correção não exigiu redução global de limiares;
- componentes realmente ausentes continuam protegidos pelas regressões que
  exigem baixo suporte do envelope antes de manter `missing_hard_absence=True`.

Contrato operacional validado:

```text
ROI interna estreita indica ausência
        +
envelope global do componente permanece preservado
        ↓
missing_global_envelope_support = True
missing_global_envelope_veto = True
        ↓
hard missing perde autoridade especial
        ↓
fusão normal física + KNN
        ↓
FALHA FALSA neste caso validado
```

Esse comportamento passa a ser a referência operacional para casos de
`FALTANDO` em que a ROI interna concentra serigrafia/conteúdo e não representa
corretamente o corpo completo do componente.

Status em 02/10/2026: **validado operacionalmente na mesma peça**.

### Caso observado em 02/10/2026 — componente presente com posição interna diferente

Evento: `fb7de76ab04843c3b2ab4ad3e16da3f6`.

A AOI classificou a ocorrência como `FALTANDO`, mas o componente físico estava
presente tanto no gabarito quanto no teste. O ODIN concluiu incorretamente
`DEFEITO REAL` por `missing_hard_absence`.

O caso mostrou um limite diferente do falso positivo anterior:

- caixa global detectada: aproximadamente `[26, 26, 308, 514]`;
- ROI/foco local: aproximadamente `[39, 105, 279, 147]`;
- `missing_score ≈ 99,62%`;
- cobertura local ≈ `68,58%`;
- residual médio ≈ `78,36%`;
- exposição de fundo local ≈ `62,69%`;
- melhor KNN = `OK` com ≈ `92,30%`;
- melhor NG ≈ `91,01%`;
- margem OK × NG ≈ `1,30 p.p.`;
- sem conflito de memória.

Visualmente, o componente continuava presente, porém sua massa escura,
serigrafia e posição interna estavam diferentes dentro do envelope global.

#### Erro de escopo espacial identificado

Foi identificado que `missing_global_envelope_background_exposure` estava
reutilizando diretamente `missing_background_exposure`, calculado na ROI local.

Isso é incorreto porque exposição de fundo é uma métrica espacial:

```text
background_exposure da ROI local
!=
background_exposure do envelope global
```

A partir desta correção, a exposição de fundo do envelope é recalculada
diretamente sobre a própria caixa global com `_background_replacement_signal()`.

Nenhuma métrica espacial da ROI local pode ser apresentada como se tivesse sido
medida no envelope global.

#### Presença global invariável a deslocamento interno

Os perfis horizontal/vertical alinhados do envelope continuam úteis, mas podem
cair quando o mesmo componente muda de posição ou orientação interna.

Foi adicionada uma testemunha auxiliar que mede a distribuição de massa escura
no envelope sem exigir que essa massa esteja nas mesmas linhas/colunas.

Ela registra:

- limiar de massa escura;
- fração escura no gabarito;
- fração escura no teste;
- retenção de massa escura;
- perfil horizontal invariável;
- perfil vertical invariável;
- `missing_global_envelope_invariant_support`.

Os perfis invariáveis são comparados após ordenação de suas distribuições. Isso
reduz a dependência de translação interna e da orientação da serigrafia.

#### Regra de segurança: massa escura sozinha não derruba hard missing

Uma regressão de segurança mostrou que um componente realmente ausente pode
deixar footprint/base escuro e, portanto, também preservar parte da massa
escura.

Por isso:

```text
missing_global_envelope_invariant_support = True
```

**não é suficiente sozinho** para vetar `missing_hard_absence`.

A evidência invariável permanece auxiliar no especialista físico.

#### Nova regra combinada

Para `FALTANDO`, um hard missing bruto pode perder autoridade pela nova rota
somente quando todas as condições abaixo forem satisfeitas:

- `missing_global_envelope_invariant_support=True`;
- memória disponível e confiável;
- melhor rótulo = `OK`;
- melhor OK >= `90%`;
- vantagem OK sobre NG >= `1 ponto percentual`;
- sem conflito de memória.

Contrato:

```text
hard missing bruto
        +
massa física global invariável preservada
        +
melhor memória OK >= 90%
        +
OK - NG >= 1 p.p.
        +
sem conflito
        ↓
hard_missing_contradicted_by_invariant_ok = True
fusion_rule = hard_missing_invariant_presence_ok_witness
hard_missing_evidence = False
        ↓
FALHA FALSA
```

Essa combinação é deliberadamente mais restrita que simplesmente permitir que
uma memória OK forte anule ausência física.

As duas proteções são obrigatórias:

- **OK forte sem presença global invariável** não veta hard missing;
- **massa global invariável sem OK forte dominante** não veta hard missing.

Isso preserva os casos reais de componente removido com footprint escuro.

#### Telemetria adicionada

O debug passa a registrar também:

- `missing_global_envelope_dark_threshold`;
- `missing_global_envelope_reference_dark_fraction`;
- `missing_global_envelope_test_dark_fraction`;
- `missing_global_envelope_dark_retention`;
- `missing_global_envelope_invariant_row_profile`;
- `missing_global_envelope_invariant_col_profile`;
- `missing_global_envelope_invariant_support`;
- `hard_missing_contradicted_by_invariant_ok`.

Status em 02/10/2026: **correção implementada e protegida por regressões;
aguardando validação operacional nesta mesma peça antes de considerar a regra
validada**.

### Caso observado em 05/10/2026 — ROI pequena preservava ocupação, mas hard missing encerrava a decisão

Evento: `c5a70a299d2349fabd0cd078998a70d1`.

A AOI classificou a ocorrência como `FALTANDO`, mas visualmente o componente
estava presente. O ODIN concluiu incorretamente `DEFEITO REAL` por
`missing_hard_absence`.

O debug mostrou:

- caixa global detectada: `[25, 25, 525, 285]`;
- ROI/foco local: `[335, 36, 137, 260]`;
- a ROI local representa aproximadamente `23,8%` da área global;
- `missing_score ≈ 89,38%`;
- cobertura local ≈ `46,34%`;
- residual médio ≈ `69,30%`;
- exposição de fundo local ≈ `47,10%`;
- envelope global com `background_exposure ≈ 5,69%`;
- massa escura do gabarito ≈ `43,38%`;
- massa escura do teste ≈ `50,52%`;
- retenção de massa escura ≈ `116,48%`;
- perfil invariável horizontal ≈ `90,56%`;
- perfil invariável vertical ≈ `92,52%`;
- `missing_global_envelope_invariant_support=True`;
- melhor memória KNN = `OK` com ≈ `89,34%`;
- melhor NG ≈ `87,47%`;
- sem conflito de memória.

A testemunha rígida de corpo não passou, porém a geometria local permaneceu
fortemente ocupada:

- `silhouette_dice ≈ 73,97%`;
- `area_ratio ≈ 88,58%`;
- `centroid_shift ≈ 8,59%`;
- `box_width_ratio = 1,0`;
- `box_height_ratio = 1,0`.

Mesmo com esses sinais, a ROI local encerrava o caso como ausência física forte
e o dual-scale não era executado porque a escala local já havia confirmado
`hard missing`.

#### Correção arquitetural

Foi adicionada uma testemunha composta de presença para `FALTANDO`.

Ela não altera os limiares rígidos existentes e não baixa o corte global da
memória. A nova rota só pode atuar quando todas as condições abaixo forem
verdadeiras:

- a ROI local ocupa no máximo `25%` do envelope global;
- `missing_global_envelope_invariant_support=True`;
- `silhouette_dice >= 70%`;
- `area_ratio` entre `70%` e `135%`;
- `centroid_shift <= 10%`;
- razões de largura e altura entre `75%` e `133%`.

Contrato:

```text
ROI local pequena (<= 25% do envelope)
        +
massa física global invariável preservada
        +
ocupação geométrica local ainda coerente
        ↓
missing_invariant_occupancy_support = True
missing_invariant_occupancy_veto = True
        ↓
hard missing local perde autoridade especial
        ↓
divergência física permanece registrada
        ↓
decisão retorna à fusão normal física + memória
```

A nova testemunha não declara `OK` diretamente. Ela apenas impede que uma ROI
local pequena represente sozinha o desaparecimento do componente inteiro.

No vetor deste evento, após o rebaixamento do hard missing, a memória OK de
aproximadamente `89,34%` volta a participar como `best_match_intermediate`.
Como o peso da memória é superior a 80%, o resultado esperado é
`FALHA FALSA` sem alterar o corte de 90% da regra
`hard_missing_invariant_presence_ok_witness`.

#### Segurança

A presença composta exige simultaneamente evidência global e ocupação local.
Uma massa escura global preservada sem geometria local coerente não pode vetar
hard missing. Isso protege componentes realmente removidos que deixam footprint
escuro.

A telemetria adicionada inclui:

- `missing_invariant_occupancy_support`;
- `missing_invariant_occupancy_veto`;
- `missing_invariant_occupancy_reason`;
- `missing_local_global_area_ratio` mesmo quando o dual-scale não executa.

Status em 05/10/2026: **correção implementada e coberta por regressões;
aguardando validação operacional nesta peça antes de considerar o caso validado**.

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


### Caso observado em 06/10/2026 — ROI testemunha mínima não pode significar inversão por si só

Evento: `bc8b227c749241fcb72fd7e2fd7e0a48`.

A AOI classificou a ocorrência como `INVERTIDO`, porém a peça era OK e deveria
ser tratada como falha falsa após confirmação do operador.

O debug mostrou que o problema não vinha de `FALTANDO` nem da guarda
transversal de ausência física:

- `missing_hard_absence=False`;
- dual-scale executado;
- contexto não confirmou ausência física forte;
- similaridade direta contextual ≈ `84,5%`.

A decisão incorreta estava concentrada no motor `INVERTIDO` e na forma como o
estado de conflito era apresentado:

- caixa global do componente: aproximadamente `278 × 527 px`;
- ROI testemunha: aproximadamente `125 × 44 px`;
- a ROI representa apenas ≈ `3,75%` da área global;
- score bruto do especialista INVERTIDO ≈ `67%`;
- retenção da marca ≈ `44%`;
- perda da marca/feature ≈ `58%`;
- topologia divergente ≈ `29%`;
- orientação divergente ≈ `4%`;
- melhor memória OK ≈ `90,01%`;
- melhor memória NG ≈ `89,36%`;
- margem ≈ `0,65 p.p.`;
- conflito de memória efetivo = `True`;
- revisão obrigatória = `True`.

Apesar disso, o resultado exposto era `DEFEITO REAL`.

#### Causa 1 — piso artificial de 90% no motor INVERTIDO

Na fusão central, qualquer disparo do especialista INVERTIDO era convertido em:

```text
effective_score = max(90%, inverted_score)
```

Portanto:

```text
score bruto ≈ 67%
        ↓
motor disparou
        ↓
score físico efetivo = 90%
```

Isso dava autoridade de defeito forte para uma divergência de marca localizada,
mesmo quando não existia uma segunda evidência clara de inversão.

#### Autoridade da ROI testemunha

O `InvertedWitnessExpert` passa a registrar:

- `inverted_local_global_area_ratio`;
- `inverted_small_witness_roi`;
- `inverted_high_authority`;
- `inverted_corroborated`;
- `inverted_corroboration_reason`.

Uma ROI é considerada pequena para fins de autoridade quando ocupa no máximo
`10%` da caixa global detectada.

Uma ROI pequena só mantém a antiga autoridade física elevada quando existe ao
menos um corroborador forte independente:

1. orientação >= `20%` **e** topologia >= `20%`;
2. sinal de deslocamento >= `40%`;
3. ganho de transformação >= `10%` com similaridade transformada >= `56%`;
4. face alternativa >= `58%`;
5. perda da marca >= `62%`;
6. perda estrutural da marca >= `65%`.

Sem esses corroboradores:

```text
ROI testemunha <= 10% do componente
        +
marca local divergiu
        +
sem evidência forte independente de inversão
        ↓
inverted_high_authority = False
        ↓
não aplicar piso físico de 90%
        ↓
usar o score bruto real do especialista
```

Essa regra **não desativa** o motor INVERTIDO. A divergência continua registrada
e pode participar da fusão. A mudança remove apenas a amplificação automática
de uma evidência local ambígua.

ROIs maiores continuam com o comportamento anterior. ROIs pequenas com
corroboração forte também preservam o piso físico existente.

#### Causa 2 — revisão obrigatória não é um veredito binário

A camada de contraste de memória já concluía corretamente:

```text
fusion_rule = memory_conflict_operator_review
operator_review_required = True
confidence = 50%
```

e a justificativa informava que a decisão automática estava bloqueada.

Mesmo assim, as integrações convertiam `is_defect=True` diretamente em
`DEFEITO REAL`, criando uma contradição semântica.

Foi criado o contrato:

```text
operator_review_required = True
        ↓
verdict = REVISÃO OBRIGATÓRIA
```

Somente quando não houver revisão pendente:

```text
is_defect=True  → DEFEITO REAL
is_defect=False → FALHA FALSA
```

A função central `resolved_analysis_verdict()` é a fonte desse mapeamento nas
integrações de memória e INVERTIDO.

O estado de revisão:

- não significa OK;
- não significa NG;
- não pode ser apresentado como `FALHA FALSA` nem `DEFEITO REAL`;
- deve exibir no overlay superior a mensagem `REVISÃO OBRIGATÓRIA` em vermelho;
- deve permanecer amarelo no painel técnico de decisão, distinguindo alerta operacional de veredito NG;
- mantém `0=OK` e `1=NG` disponíveis para o operador;
- continua preservando `is_defect` e scores brutos para auditoria interna.

No vetor deste evento, a memória continua em conflito por margem inferior a
`1 p.p.`. Portanto o resultado automático esperado após esta correção é
**REVISÃO OBRIGATÓRIA**, e não um OK artificial.

Ao o operador confirmar `0=OK`, a ocorrência pode fortalecer a hipótese OK da
categoria INVERTIDO para recorrências futuras.

#### Debug obrigatório para INVERTIDO

O `Copiar debug` passa a expor explicitamente:

- `inverted_score`;
- classificação;
- retenção/perda da testemunha;
- perda de feature;
- topologia;
- orientação;
- face alternativa;
- transformação e similaridade;
- relocação;
- razão ROI/global;
- se a ROI é pequena;
- se possui alta autoridade;
- se existe corroborador;
- motivo da autoridade/corroborador.

Isso permite distinguir uma marca local diferente de uma evidência física
corroborada de face invertida.

#### Regressões obrigatórias

Manter testes que garantam:

- ROI ≈ `3,75%` do componente, orientação ≈ `4%`, perdas moderadas e sem
  corroborador → `inverted_high_authority=False`;
- esse vetor não recebe piso físico automático de `90%`;
- ROI pequena com orientação + topologia fortes mantém alta autoridade;
- evidência INVERTIDO corroborada continua preservando o piso físico existente;
- conflito OK × NG sem hard missing efetivo → `REVISÃO OBRIGATÓRIA`;
- revisão obrigatória nunca é apresentada como `DEFEITO REAL` ou
  `FALHA FALSA`;
- hard missing verdadeiro continua tendo precedência e não vira revisão;
- debug mostra os campos de autoridade do INVERTIDO.

Status em 06/10/2026: **correção implementada; aguardando validação operacional
nesta mesma peça**.

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


## Fundo neutro permanente do ODIN

O fundo geral do ODIN permanece **sempre no tema escuro neutro**, independentemente
do veredito final da IA.

Contrato atual:

```text
sem análise
→ fundo neutro

FALHA FALSA / OK
→ fundo neutro

DEFEITO REAL / NG
→ fundo neutro

REVISÃO OBRIGATÓRIA
→ fundo neutro
```

O fundo não comunica mais estado operacional. A comunicação visual do resultado
fica concentrada no card de veredito do canto superior direito.

Cores neutras de referência:

- canvas: `#050505`;
- superfícies principais: `#0d0d0d`.

### Alteração de contrato em 02/10/2026

O comportamento anterior, já validado em operação, fazia o fundo principal
mudar para verde-escuro em `FALHA FALSA` e vermelho-escuro em
`DEFEITO REAL`.

Esse comportamento foi **substituído deliberadamente**.

A partir desta alteração:

- o fundo não acompanha `analysis["is_defect"]`;
- o `main.py` não instala mais `install_decision_background(ControlPanel)`;
- o módulo `src/ui/decision_background.py` permanece apenas como compatibilidade
  defensiva e normaliza qualquer solicitação para `neutral`;
- verde/vermelho continuam permitidos no texto do card de veredito;
- botões, hover, focus, checked e demais componentes mantêm o tema original.

A decisão de remover o fundo dinâmico evita que grandes áreas coloridas disputem
atenção com a inspeção e deixa o veredito explícito em um único elemento visual.

### Regra crítica de arquitetura

Nenhum veredito pode alterar a cor geral do canvas ou das superfícies principais.

Regressões obrigatórias:

- `None` → `neutral`;
- `is_defect=False` → `neutral`;
- `is_defect=True` → `neutral`;
- uma chamada defensiva `apply_decision_background(..., "ok")` ou
  `apply_decision_background(..., "ng")` também resulta em `neutral`;
- `main.py` não instala o hook de fundo dinâmico;
- o stylesheet global não é substituído;
- um `sectionPanel` permanece em `#0d0d0d` mesmo se alguém tentar aplicar
  estado NG pelo módulo legado.

## Feedback visual temporário de teclas operacionais

O ODIN possui um overlay exclusivamente visual para confirmar imediatamente ao
operador qual tecla operacional foi pressionada ou enviada.

Comportamento:

```text
0 → OK
1 → NG
← → seta esquerda / TOP
↓ → seta para baixo / SIDE
→ → seta direita / MID
```

Fontes cobertas:

- teclado do próprio ODIN: `0`, `Num+0`, `1` e `Num+1`;
- teclado físico do Windows XP recebido pela rede como `CMD_OK` ou `CMD_NG`.

A apresentação é um quadrado temporário de aproximadamente `180 × 180 px`, posicionado no **canto inferior direito** da interface, com margem aproximada de `24 px` das bordas e acima dos demais componentes.

O visual segue a identidade industrial do ODIN:

- fundo escuro `#101010`;
- borda-base discreta `#303030`;
- cabeçalho `TECLA PRESSIONADA`, `TECLA ENVIADA` ou `TECLA RECEBIDA` em amarelo ODIN `#f5c518`;
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


### Refinamento visual do feedback 0/1 em 02/10/2026

O card de confirmação de tecla continua no canto inferior direito e mantém a
mesma função operacional, porém sua linguagem visual passa a ficar mais próxima
do restante do ODIN:

- fundo escuro próximo ao `SURFACE`;
- borda amarela ODIN, em vez de usar verde/vermelho como moldura principal;
- cabeçalho continua amarelo;
- verde/vermelho ficam restritos ao conteúdo do estado `OK` / `NG`;
- animação, posição, click-through, duração e deduplicação de eco permanecem
  inalterados.

O objetivo é evitar um alerta visual que pareça pertencer a outro sistema ou a
um componente genérico de IA.

## Feedback visual do veredito final da IA

Além do feedback de tecla `0/1`, o ODIN possui um segundo overlay exclusivamente
visual para comunicar o veredito final calculado pela IA.

Esse overlay tem função diferente do feedback de tecla:

```text
feedback 0/1
= confirma ação/comando recebido

feedback de veredito
= comunica o resultado final da análise da IA
```

### Mensagens permitidas

O card de veredito deve exibir somente:

```text
FALHA FALSA
```

ou:

```text
DEFEITO REAL
```

Não exibir nesse card:

- porcentagem;
- score;
- confiança;
- motor dominante;
- regra de fusão;
- categoria;
- justificativa;
- origem da tecla.

Essas informações continuam disponíveis nos painéis detalhados e no debug.

### Posição

O overlay do veredito fica no **canto superior direito** da janela principal.

Essa escolha é deliberada:

- separa visualmente o veredito da IA do feedback de tecla `0/1`, que fica no
  canto inferior direito;
- não compete com a área central de decisão;
- segue um padrão natural de notificação sem ocupar o centro da inspeção;
- mantém leitura imediata em telas largas.

Contrato atual:

- largura aproximada: `300 px`;
- altura aproximada: `88 px`;
- margem direita: `24 px`;
- offset superior aproximado: `84 px`.

### Linguagem visual

O card deve seguir a identidade industrial do ODIN:

- base escura `SURFACE`;
- borda fina amarela `ACCENT`;
- nenhum texto auxiliar: o único texto renderizado é o próprio veredito;
- `FALHA FALSA` em verde;
- `DEFEITO REAL` em vermelho;
- tipografia forte e limpa;
- sem ícones decorativos genéricos;
- sem pills;
- sem gradientes;
- sem barras laterais coloridas;
- sem aparência de componente gerado por IA.

O verde/vermelho é usado apenas para o texto do estado. A estrutura do card
continua escura/amarela. Não exibir cabeçalho, subtítulo ou legenda dentro do
card: a moldura amarela já comunica que o elemento pertence ao ODIN.

### Relação com o fundo neutro

O card de veredito é agora o **único elemento global de alto nível** que comunica
o resultado binário da IA.

```text
FALHA FALSA
        ↓
fundo permanece neutro
        +
card superior direito com texto verde

DEFEITO REAL
        ↓
fundo permanece neutro
        +
card superior direito com texto vermelho
```

Não existe mais dependência de ordem entre overlay e fundo dinâmico, porque o
fundo não reage ao resultado.

Quando o ODIN volta para `AGUARDANDO PEÇA`, o card deve desaparecer e o fundo
continua neutro.

### Persistência, entrada e saída sincronizada

Enquanto existe uma análise ativa, o card de veredito permanece fixo no canto
superior direito.

Entrada:

- fade-in curto de aproximadamente `140 ms`;
- pequeno slide horizontal de aproximadamente `10 px`;
- depois da entrada, permanece estático e visível;
- não existe `QTimer` próprio de auto-hide;
- sem thread adicional;
- sem `sleep`;
- sem loop contínuo;
- `WA_TransparentForMouseEvents`;
- `NoFocus`.

#### Saída sincronizada com o julgamento 0/1

Quando o operador confirma a decisão por `0 / OK` ou `1 / NG`, o card de
veredito não pode desaparecer imediatamente por causa do reset interno do ciclo.

O contrato é:

```text
operador pressiona 0 ou 1
        ↓
feedback 0/1 aparece
        +
card FALHA FALSA / DEFEITO REAL é marcado para saída
        ↓
ciclo produtivo pode salvar/resetar normalmente
        ↓
card de veredito permanece visível durante a confirmação 0/1
        ↓
feedback 0/1 inicia fade-out
        ↓ mesmo evento
card de veredito inicia fade-out
        ↓
ambos desaparecem juntos
```

A saída dos dois overlays usa:

- início no mesmo callback lógico;
- duração de fade-out: aproximadamente `160 ms`;
- easing `InOutQuad`;
- nenhuma animação contínua.

O `QTimer` continua pertencendo somente ao feedback temporário de `0/1`. O
card de veredito não cria um segundo timer: ele apenas aguarda o sinal de
fade-out emitido pelo overlay de tecla.

Durante essa janela, chamadas internas de reset/`save_label()` não podem
apagar o veredito instantaneamente. Elas respeitam o estado de saída pendente.

Se não existir feedback `0/1` ativo, um reset normal para
`AGUARDANDO PEÇA` continua limpando o card imediatamente.

Para comandos físicos vindos do Windows XP, o feedback visual é preparado antes
do handler produtivo consumir `CMD_OK/CMD_NG`. Isso garante a mesma
sincronização mesmo quando o processamento do comando encerra o ciclo
imediatamente.

### Estado de revisão obrigatória

Se a análise exigir revisão humana, o overlay não deve inventar uma decisão
binária. Em vez disso, ele deve comunicar explicitamente o estado operacional:

```text
operator_review_required = True
        ↓
REVISÃO OBRIGATÓRIA
        ↓
texto vermelho no card superior direito
```

Portanto:

- `REVISÃO OBRIGATÓRIA` → mostrar exatamente `REVISÃO OBRIGATÓRIA` em vermelho;
- nunca substituir esse estado por `FALHA FALSA` ou `DEFEITO REAL`, mesmo que
  `is_defect` bruto ainda esteja presente para auditoria;
- análises legadas sem texto de veredito só podem usar `is_defect` como
  fallback quando não existir revisão humana pendente.

### Arquitetura

Implementação:

```text
src/ui/decision_verdict_feedback.py
```

O recurso é exclusivamente visual e não pode:

- alterar `analysis`;
- alterar `is_defect`;
- alterar confiança ou score;
- alterar KNN;
- alterar dataset;
- alterar arquivos OK/NG;
- enviar `PRESS_0/PRESS_1`;
- responder pelo operador;
- bloquear o ciclo de imagens.

### Regressões obrigatórias

Manter testes que garantam:

- `FALHA FALSA` → texto verde, sem percentual;
- `DEFEITO REAL` → texto vermelho, sem percentual;
- posição no canto superior direito;
- card click-through e sem foco;
- entrada curta e não bloqueante;
- ausência de `QTimer` próprio no card de veredito;
- card permanece visível após a animação de entrada;
- julgamento `0/1` prepara a saída antes do reset produtivo;
- reset/`save_label()` não apagam o veredito enquanto a saída sincronizada está pendente;
- fade-out do veredito e do feedback `0/1` iniciam no mesmo evento;
- ambos usam aproximadamente `160 ms` e easing `InOutQuad`;
- comando `CMD_OK/CMD_NG` do XP prepara o feedback antes do handler produtivo;
- revisão obrigatória exibe `REVISÃO OBRIGATÓRIA` em vermelho e não inventa um veredito binário;
- reset sem feedback `0/1` ativo continua limpando o overlay imediatamente;
- o fundo permanece neutro durante toda a análise;
- feedback de tecla continua independente no canto inferior direito;
- o visual do feedback `0/1` usa moldura amarela/escura e conserva
  verde/vermelho apenas no conteúdo do estado.

### Validação operacional do fundo neutro e card persistente em 02/10/2026

O comportamento atualizado foi validado em uso real pelo operador.

Foi confirmado que:

- o fundo do ODIN permanece escuro/neutro durante `FALHA FALSA`;
- o fundo permanece escuro/neutro durante `DEFEITO REAL`;
- o fundo também permanece neutro em `AGUARDANDO PEÇA`;
- o card do canto superior direito continua exibindo somente o veredito;
- `FALHA FALSA` permanece visível em verde enquanto a análise atual continua
  ativa;
- `DEFEITO REAL` permanece visível em vermelho enquanto a análise atual
  continua ativa;
- o card não desaparece mais por tempo;
- não existe `QTimer` de auto-hide nem fade-out automático;
- ao encerrar/resetar o ciclo e retornar para `AGUARDANDO PEÇA`, o card é
  removido;
- uma nova análise pode substituir o conteúdo do card pelo novo veredito;
- o feedback temporário de tecla `0/1` continua independente no canto inferior
  direito;
- nenhum desses elementos altera decisão, confiança, KNN, dataset ou comandos XP.

Essa configuração continua sendo a referência para o fundo neutro e a posição do
card. Em 05/10/2026, o contrato de **saída** do card foi refinado: após um
julgamento `0/1`, ele deixa de ser removido imediatamente pelo reset e passa a
desaparecer sincronizado com o feedback temporário de tecla.

A sincronização de saída está **implementada e aguardando validação operacional
na interface real**. A validação anterior do fundo neutro permanece válida.


## Tempo de análise end-to-end

O card `TEMPO DE ANÁLISE` mede o tempo operacional percebido entre a entrada
da imagem no ODIN e o resultado já atualizado visualmente na interface.

### Contrato

Para imagem recebida do Windows XP:

```text
payload completo terminou de chegar ao ODIN
        ↓
descompressão / decode
        ↓
confirmação de estabilidade
        ↓
validação da AOI / epicentro
        ↓
extração / OCR / normalização
        ↓
motores físicos + memória + fusão
        ↓
widgets de resultado atualizados
        ↓
Qt processa a pintura pendente
        ↓
fim do Tempo de análise
```

O marco inicial da rede usa `time.perf_counter()` imediatamente após o payload
completo ser recebido, antes de descompressão e `cv2.imdecode()`.

O `NetworkReceiver` associa esse timestamp ao candidato estável efetivamente
entregue ao painel através de `last_delivered_image_received_at`.

Para captura local MSS:

```text
frame MSS terminou de ser capturado
        ↓
detecção da interface / recortes / OCR
        ↓
análise
        ↓
resultado pintado
        ↓
fim do Tempo de análise
```

O clique em `Capturar local (MSS)` e a espera anterior ao primeiro frame válido
não fazem parte do tempo de análise.

### Relógio

Usar exclusivamente relógio monotônico de alta resolução (`time.perf_counter()`)
para duração. `time.time()` não deve ser usado para calcular esse intervalo,
porque alterações no relógio do sistema podem distorcer a duração.

### Momento final

Depois de atualizar veredito, motivo, imagens, painéis técnicos e overlay, o
ODIN chama `QApplication.processEvents()` com input do operador excluído.
Somente depois desse processamento de pintura é registrado
`analysis_displayed_at`.

Assim, o valor não termina apenas quando a IA retorna: inclui a preparação e a
entrega visual do resultado ao operador.

### Telemetria

`analysis["detail"]` registra:

- `analysis_time_seconds`;
- `analysis_time_start_source`;
- `analysis_time_contract`.

Origens válidas:

- `network_payload_received`;
- `local_mss_frame_received`;
- `process_entry_fallback` apenas como proteção de compatibilidade.

O `Copiar debug` também deve exibir esses campos para auditoria.

### Regra de apresentação

O título permanece `TEMPO DE ANÁLISE` e o valor é mostrado em segundos, por
exemplo:

```text
1.27 s
```

Não rotular esse valor como simples `Latência`, pois ele representa o ciclo
end-to-end descrito acima.

Status em 06/10/2026: **contrato corrigido e implementação concluída;
aguardando validação operacional com medições reais no ODIN**.

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
