# Estrutura do Projeto: VisionX Neural

**Módulos Existentes:**
- `src/config/settings.py`: Centralização de todas as variáveis de ambiente, caminhos e constantes mágicas.
- `src/services/ng_image_archive.py`: Arquivo visual opcional de decisões finais NG em fila de background, independente do dataset e da memória KNN.

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
- Toggle desligado por padrão; desativado mantém o fluxo atual sem criar cópias extras.
- Ativado: cada evento válido do Windows XP pode gerar **no máximo uma** evidência final `NG` em `public/ng_archive/`, usando exatamente o mesmo frame completo disponibilizado pelo botão `Copiar imagem XP`.
- Nome: `DDdMMmAAAA_HHhMMminSSsmmmms_CATEGORIA.png`, por exemplo `01d10m2026_10h22min21s943ms_FALTANDO.png`. O formato mantém dia, mês, ano, hora, minuto, segundo e milissegundo visualmente identificáveis sem deixar o nome excessivamente longo.
- A fonte é única: `src/services/network_xp_frame.py` valida que o `event_id` do frame preservado é o mesmo do diagnóstico atual. O botão `Copiar imagem XP` e o arquivo visual NG usam essa mesma função.
- Não existe fallback para `current_ng` ou outro recorte. Se o frame XP do evento atual não estiver disponível, nenhuma imagem substituta é arquivada.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.
- Deduplicação obrigatória por `event_id`: o mesmo evento XP nunca pode gerar duas imagens de arquivo, mesmo se o `PRESS_1` enviado pelo VisionX reaparecer pelo hook global do XP como `CMD_NG`.
- O arquivamento só é permitido enquanto existe uma captura de rede ativa, com análise ativa e categoria AOI não vazia.
- `SEM_CATEGORIA` não é um nome de arquivo válido para o fluxo automático de evidências NG. Se a categoria já tiver sido limpa, o evento não deve ser salvo novamente.


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

A apresentação é um quadrado temporário de aproximadamente `180 × 180 px`, centralizado sobre a interface e acima dos demais componentes:

- `0 / OK`: destaque verde;
- `1 / NG`: destaque vermelho;
- origem exibida como `TECLADO ODIN` ou `TECLADO WINDOWS XP`;
- duração aproximada: `800 ms`;
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

O overlay reutiliza um único widget e um único `QTimer` single-shot. Ele possui `WA_TransparentForMouseEvents` e `NoFocus`, portanto pode aparecer sobre outros componentes sem impedir interação.

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
- o recurso não altera nenhuma regra de negócio do ciclo.
