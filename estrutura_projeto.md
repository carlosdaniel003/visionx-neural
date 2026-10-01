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
- Ativado: toda decisão final `NG` de uma captura do Windows XP salva em `public/ng_archive/` exatamente o mesmo frame completo disponibilizado pelo botão `Copiar imagem XP`.
- Nome: `AAAA-MM-DD_HH-MM-SS-ms_CATEGORIA.png`.
- A fonte é única: `src/services/network_xp_frame.py` valida que o `event_id` do frame preservado é o mesmo do diagnóstico atual. O botão `Copiar imagem XP` e o arquivo visual NG usam essa mesma função.
- Não existe fallback para `current_ng` ou outro recorte. Se o frame XP do evento atual não estiver disponível, nenhuma imagem substituta é arquivada.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.


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

Existem duas rotas de confirmação:

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
- componente deslocado → não vira hard missing;
- memória OK forte não veta hard missing;
- conflito OK × NG não força revisão sobre hard missing;
- hard missing alcança 99% de confiança e pode passar pela trava existente de
  Produção sem reduzir globalmente o limiar de confiança;
- casos ambíguos continuam seguindo a política normal.
