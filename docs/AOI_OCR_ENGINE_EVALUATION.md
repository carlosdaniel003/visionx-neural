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
