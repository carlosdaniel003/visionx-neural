## 15/10/2026 — OCR de Value apenas, remoção de Board/Parts e responsividade Qt

**Falha comprovada de inicialização:** `diagnostico_inicio.txt` do operador registrou `UnicodeEncodeError` ainda no import `main.py -> src.ui.control_panel` (linha que imprime `👁️ OCR Ativado com sucesso`). Sob `cp1252`, `sys.stdout` redirecionado com política `strict` não aceita o emoji; **nenhuma janela foi criada**. O bootstrap de `main.py` agora configura `errors='backslashreplace'` em stdout/stderr antes de importar PyQt6 e outros módulos. Isso preserva a codificação do console e transforma glifos incompatíveis em escapes em vez de derrubar o processo; não exige `PYTHONIOENCODING` nem permissões elevadas.

**Escopo de OCR solicitado:** `Placa/Máquina (Board)` e `Componente (Parts)` não são utilizados pelo operador nesta estação. O `ScreenMonitor` deixa de procurar seus padrões no resultado Tesseract e, sobretudo, deixa de chamar `recover_parts_from_text_zone`, que nos casos ambíguos iniciava `parts_anchor_data + parts_full + parts_numeric` (três subprocessos Tesseract adicionais). Os campos legados `board` e `parts` continuam existindo **sempre vazios** apenas para não quebrar contratos de memória/report histórico; os cartões e labels correspondentes foram removidos da interface e do controlador. O campo **`Value` continua integralmente reconhecido**, normalizado e usado para extrair **categoria do defeito**, sem alterar decisões CNN/KNN nem a máquina de estados SIDE/TOP/MID. Para preservar a localização e a precisão de `Value`, a **única leitura geral** do texto acima da barra azul continua usando a mesma região e as mesmas configurações Tesseract; caso ela seja completamente vazia, a releitura geral sobre barra vermelha permanece (máximo duas chamadas). O texto bruto pode conter caracteres das demais linhas AOI, mas eles não são processados como campos Board/Parts.

**Responsividade:** se `_ocr_fast` executar na thread GUI Qt, o Tesseract é enviado a um único worker Python (`max_workers=1`) e a GUI processa os eventos através de `QEventLoop` com pulsos de 25 ms enquanto espera o **mesmo resultado**. O processamento retorna sincronicamente para o contrato de extração atual, portanto SIDE termina completamente antes de avançar para TOP/MID. Em threads de fundo (MSS e observador do acervo), permanece o OCR direto sem criar event loop Qt aninhado. Testes exigem que um `QTimer` continue sendo executado enquanto a chamada OCR simulada dorme em worker. **Limite importante:** outros trechos síncronos da inspeção (geometria/índice KNN, escrita de PNGs, validação) ainda podem bloquear a UI, e uma event loop aninhada exige cuidado com eventos de usuário e timeout; o gate existente permanece responsável por rejeitar uma nova SIDE enquanto a atual está ativa. Esta alteração não torna todo o pipeline assíncrono.

O A/B offline e testes de telemetria são atualizados para considerar `Parts` desativado (`parts_syntax_valid=null`, não erro), manter monitoramento de `Value/categoria` e continuar recusando aprovação automática quando não há verdade humana. **Expectativa de ganho:** inspeções antigas com `Parts=FICE~S` efetuavam quatro chamadas (4,0 s de subprocessos na 7ª inspeção); com a nova implementação esperam-se normalmente **uma chamada** (~1,3 s naquele evento), sendo a economia exata dependente de captura/carga e sujeita à medição em estação. Não garantir tempo fixo de 5 s nem alteração de veredito baseada apenas em ganho temporal.

## 15/10/2026 — 7 ciclos após PR #29: janela entre peças

Medições sequenciais FALTANDO Modo Teste: **4,58 / 7,27 / 16,92 / 8,96 / 11,81 / 6,05 / 14,70 segundos** (média 10,04 s, mediana 8,96 s; não há crescimento monótono). A 7ª inspeção, evento `c026ddf6cb9147a5aab7b78568838a7e`, demorou 14,703 s. OCR da SIDE foi 4,306 s em **4 Tesseracts**, retornando `Parts=FICE~S` sem candidato válido. A fase `parts_anchor_data` levou 1,443 s e o total das chamadas foi 4,004 s. Recortes SIDE levaram 1,697 s, dos quais 821 ms na gravação de PNGs de debug (incluídos na etapa); validação SIDE 1,735 s; router KNN SIDE 1,479 s, reconstrução 1,191 s por `legacy_records_changed` e warmup invalidado por `orchestrator_reload_memory`. A CNN pura SIDE levou 28 ms. Não somar etapas que se sobrepõem.

**PR #29 funcionou parcialmente:** o observador KNN registrou `deferred_inspection` com **0 ms de concorrência em TOP/MID**, mas na SIDE a pausa começou quando `extracting_png` já estava em andamento, ocasionando **6,965 s de sobreposição**. A medição anterior de liberação do gate até a nova SIDE foi **2,121 s**. Como a liberação do gate fazia o observador acordar imediatamente, um OCR de acervo podia começar entre duas peças próximas e continuar mesmo após o sinal de prioridade.

**Mitigação de agendamento:** após liberar o gate de uma peça de rede, o observador permanece em `deferred_between_inspections` por até **3 s**, desde que uma varredura real tenha sido iniciada nos últimos **30 s**. Isso evita iniciar um Tesseract de acervo no intervalo típico até a próxima SIDE. Se a inspeção seguinte entrar nesse intervalo, `pause_for_inspection` mantém a prioridade. Quando a peça seguinte não chega, o backlog recomeça ao fim dos 3 s, mesmo sem aviso de arquivo novo; se a fila estiver em ciclos sucessivos com folga curta, a proteção de 30 s permite que o observador faça progresso na próxima folga, sem cortar execução durante uma peça.

A nova opção é configurável na classe (`post_inspection_quiet_seconds=3.0`, `max_idle_deferral_seconds=30.0`). As métricas `idle_grace_active`, `idle_grace_remaining_ms` e `idle_grace_deferrals` indicam o efeito no diagnóstico. O mecanismo não altera OCR, verificações de PNG, SHA, rótulos, memória append-only, CNN, KNN, cálculo do veredito ou comandos XP. A importação automática pode sofrer atraso de 3 s entre peças; **um OCR do acervo iniciado antes do gate e ainda em execução não é cancelado**. Tempo de ciclo e consumo real de RAM precisam ser validados no PC industrial; meta de 5 s não está garantida.

## 15/10/2026 — Prioridade de OCR durante inspeções consecutivas

No evento `b1b999332d08443391cdbb654bf7a33e`, o ciclo em Modo Teste FALTANDO demorou **13.309 ms**, embora a SIDE tenha resolvido `Parts=C375` com **somente 1 chamada** Tesseract (1.498 ms da chamada; OCR total 1.672 ms). O atraso incluiu recortes SIDE 1.800 ms (738 ms de gravações PNG de debug), validação 2.349 ms, geometria 1.093 ms, inferência de especialistas 1.041 ms e CNN pura SIDE 69 ms. A indexação KNN exigiu 558 ms somente no SIDE. Esses tempos podem se sobrepor e **não devem ser somados**.

A telemetria do observador KNN registrou **7.931 ms de trabalho simultâneo** à SIDE, mais 1.266 ms TOP e 1.051 ms MID. Ele estava na fase `extracting_png` durante parte importante do ciclo apesar de não ter importado PNGs novos na sessão. A prioridade Win32 de background estava ativa, mas subprocessos OCR do worker ainda podem concorrer com o OCR de inspeção, RAM e disco.

**Correção de agendamento cooperativo:** em `begin_cycle` da primeira SIDE por rede, o observador de acervo recebe um sinal instantâneo `pause_for_inspection`. A thread não inicia novos scans nem novos arquivos/OCR dirigidos do acervo enquanto o gate de inspeção estiver ativo. Se um PNG já estava sendo analisado, ele termina normalmente, e entradas já extraídas são persistidas com as mesmas regras: nunca se interrompe Tesseract à força nem se perde backlog. Em decisão OK/NG, skip, watchdog, falha ou liberação normal do gate, `resume_after_inspection` remove o sinal e acorda o worker de imediato, mesmo sem chegar uma nova imagem no diretório. A interface não espera `thread.join`, bloqueios, jobs ou I/O da thread KNN.

A sincronização continua independente do modo (Teste/Produção/Sombra), com inspeções prioritárias na frente dos trabalhos em segundo plano. O observador **não apaga** pendências, imagens, retries, registros ou a verificação append-only. Durante a análise ou enquanto a peça estiver aguardando decisão do operador em Modo Teste, novas imagens arquivadas podem demorar mais para serem incorporadas na memória persistente; a atualização ocorrerá após a liberação do gate. O `activity()` acrescenta `inspection_priority_active`, `deferred_scans` e `deferred_candidates` para identificar claramente essa contenção.

**Limite:** uma operação de OCR de acervo já iniciada antes da chegada SIDE não é interrompida. Portanto, esta mudança reduz principalmente a concorrência acumulativa e a degradação ao longo de ciclos, mas **não garante 5 s**. É preciso medir três a seis inspeções com idêntico acervo/carga e comparar OCR, recortes, validação, KNN e consumo total de RAM. A aceleração da chamada Tesseract operacional ainda exige benchmark independente, sem comprometer `Board/Parts/Value`.

## 15/10/2026 — Campo Parts: referência E1~9 confundida com EL~~~9

Evidência da interface AOI SIDE/TOP/MID fornecida e confirmada visualmente pelo operador: a referência `E1~9` foi exibida pelo ODIN como `EL~~~9`. A rotina `normalize_parts_ocr()` antiga procurava o **primeiro dígito reconhecido**. Com `EL~~~9`, só encontrava o dígito final `9`, preservava `EL~~~` como prefixo e o valor resultante não atendia à gramática; isso provocava releituras dirigidas adicionais.

Correção em `aoi_ocr_fields.py` **estritamente delimitada**: se e somente se o token tiver formato `E[1IL|]~{2,}[0-9]+`, normalizar para `E1~<sufixo>`. O caso confirmado `EL~~~9` passa a `E1~9`, sem leituras adicionais caso seja a saída do OCR geral. Não se faz `L→1` globalmente, não se altera outro designador, nem se reescreve `EL~9` com separador único (continua ambíguo). Campos `Board` e `Value`, motor Tesseract, CNN/KNN e decisões não foram alterados.

**Limite:** esse reparo não resolve `FICE~S`, encontrado em **3/6** arquivos FALTANDO do benchmark posterior, ainda inválido após 4 chamadas Tesseract. A mediana passou de 1123,96 ms para 1062,40 ms com OMP=1, ganho de **5,48%** apenas no experimento offline; não justifica mudar threads de produção. As seis capturas tinham `expected=null`, portanto suas saídas não foram validadas com verdade humana. Além disso, `RI75` passa na gramática mas pode representar leitura errada de `R175` ou outra referência; sintaxe válida não prova a identidade física.

A correção `EL~~~9` é suportada **somente pela confirmação deste caso específico**. Antes de generalizar para outros tipos, projetar e comparar reconhecimento de campo Parts por ROI, com anotação humana dos campos e inclusão de casos fáceis e difíceis. Se a extração fiel não for possível, preservar a ambiguidade em vez de atribuir componente não verificado. A regra de fusão e o uso de memória não devem confiar cegamente em valores de OCR plausíveis.

# OCR da AOI — avaliação de desempenho sem alterar Produção

## Evidência e hipótese

No sexto ciclo pós-PR #25 (evento `147704ee1d28488fbac2f42a1993ab34`), a inspeção FALTANDO durou **18.955 ms** depois de duas leituras em ~5s. Só o OCR da SIDE consumiu **5.859 ms**, incluindo **4 Tesseracts** (geral 1.722 ms, ancoragem Parts 1.904 ms, leitura integral 1.167 ms e numérica 911 ms). `Parts` foi lido como `FICE~S` e a recuperação não encontrou candidato válido (`directed_candidates_invalid`). A CNN SIDE consumiu ~118 ms. O observador KNN estava configurado corretamente em background, porém executava `extracting_png` simultaneamente à inspeção. Não é prova de que OCR é a única causa de lentidão.

A documentação oficial do Tesseract recomenda experimentar a variável `OMP_THREAD_LIMIT=1` para reduzir overhead de OpenMP em tarefas OCR pequenas:
[FAQ oficial](https://tesseract-ocr.github.io/tessdoc/FAQ.html).
No Windows, isso pode melhorar **ou piorar** o desempenho conforme hardware, tamanho de ROI, simultaneidade e versão. A precisão/parse pode variar; não promover sem validar os quatro campos.

## Experimento executável no computador do ODIN

Este PR não altera `screen_monitor.py`, `aoi_ocr_fields.py`, `main.py`, o processo da interface nem os parâmetros da CNN/PyTorch. O script inicia subprocessos **Python separados**, em ordem **controle → OMP=1 → OMP=1 → controle**. O limite OMP é aplicado **apenas aos subprocessos de benchmark**. Em cada subprocesso, a leitura usa o `ScreenMonitor._extract_text_info()` **inalterado**, portanto mede as mesmas fases gerais/dirigidas (Board/Parts/Value) em cada captura AOI integral.

1. Separe de 10 a 30 capturas **integrais** das telas AOI do Windows XP (840×1165 é um exemplo), incluindo leituras simples e leituras em que Parts vira `FICE~S`, `RI~5` ou outros valores ambíguos. Evite usar apenas crops, que não contêm as barras de referência. Os arquivos **nunca são alterados**.
2. Pare de executar inspeções reais enquanto roda o benchmark. Faça o teste fora de Produção, sem bloquear a estação ou interferir com o XP.
3. No PowerShell, na pasta `C:\visionx-neural-main`, execute:

```powershell
python -m src.services.aoi_ocr_thread_trial --images "C:\caminho\captura1.png" "C:\caminho\captura2.png" --repeats 2 --output "C:\caminho\ocr_omp_ab_20261015.json"
```

Opcional: para comparação com verdade humana, salve manifesto JSON **fora do repositório**:

```json
[
  {
    "path": "captura1.png",
    "expected": {
      "board": "P22-22200 (PRINCIPAL) L13",
      "parts": "D10~5",
      "value": "0 <= 5 <= 10 FALTANDO",
      "category": "FALTANDO"
    }
  }
]
```

Depois:

```powershell
python -m src.services.aoi_ocr_thread_trial --manifest "C:\caminho\manifesto.json" --repeats 2 --output "C:\caminho\ocr_omp_com_rotulos.json"
```

Os valores humanos acima são **somente exemplo**, não correspondem às peças reais de 15/10. A referência precisa ser verificada olhando a tela original; a saída da IA **não** deve ser copiada como rótulo humano. O manifesto segue o mesmo esquema e as mesmas proteções contra caminhos externos do benchmark OCR existente.

O JSON reporta medianas OCR, contagem de chamadas, `Parts` válido pela gramática, variação do resultado entre rodadas, divergência de Board/Parts/Value/Category e discordância com rótulos humanos. A mediana **não** inclui custo de criar os subprocessos Python de benchmark, mas inclui cada subprocesso Tesseract e o processamento do OCR, como no código original. O teste roda os motores **sequencialmente**, para não agravar os 8 GB de RAM.

**Não conclua que a variante é segura se `different_from_baseline_cases=0`**: o baseline pode estar errado. A regra para uma implementação futura é não regredir o desempenho dos quatro campos em capturas reais rotuladas, incluindo falsos negativos de Parts, e obter ganho consistente em múltiplas capturas/rodadas. Nenhuma variante tem permissão de ativação automática.

## Tecnologias candidatas (ordem de avaliação)

| Opção | Possível vantagem | Risco/condição |
|---|---|---|
| Tesseract atual com `OMP_THREAD_LIMIT=1` **por processo OCR** | Reduzir overhead de OpenMP sem novo modelo | Exige benchmark de campos + integração isolada apenas nos subprocessos Tesseract; aplicar variável global em `main.py` poderia afetar PyTorch |
| Tesseract via biblioteca em processo (`tesserocr`) | Evitar iniciar 1–4 executáveis por inspeção | Extensão nativa e DLLs no Windows corporativo; concorrência, inicialização e liberação de memória precisam ser testadas |
| [RapidOCR + ONNX Runtime CPU](https://github.com/RapidAI/RapidOCR) | Reconhecimento e detecção OCR com modelos compactos locais; pode se beneficiar de executor persistente | Novos modelos/dependências e outro comportamento para componentes industriais (I/1, O/0, `~`, intervalos); precisa teste offline completo em 8 GB RAM |
| ROI/gramática específica da interface AOI | Menos pixels e potencialmente menos subprocessos | Qualquer omissão de `Parts` ambíguo deve resultar em revisão ou fallback validado, nunca em referência inventada |

**Decisão sugerida:** medir a variante de thread única primeiro; se não alcançar robustez e latência alvo, preparar um *shadow benchmark* de RapidOCR e Tesseract em processo, sem mexer nos veredictos. A troca de OCR só é justificável quando medidas por referência humana cobrirem tanto a rapidez quanto a confiabilidade industrial.

Nenhuma alteração de OCR neste PR altera CNN, KNN, máscaras, comandos XP, emissão 0/1, rótulos da memória ou fluxo multilight.
