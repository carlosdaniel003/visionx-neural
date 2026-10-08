# Preparação do dataset neural — FALTANDO

**Etapa atual:** preparação/qualificação offline, sem treinamento ou substituição de motores.
**Branch:** central. **Máquina de execução:** Windows 10 do ODIN (não o XP).
**Fonte:** `public/ok_archive` e `public/ng_archive`, somente leitura.

## Inventário informado pelo operador (08/10/2026)

- 117 PNGs FALTANDO: 107 OK e 10 NG.
- NG: 10 sem sufixo de luz (SIDE histórico presumido); nenhum NG TOP/MID.
- OK: 32 sem sufixo + 25 SIDE, 25 TOP e 25 MID explícitos.
- SHA-256 distintos: 117; nenhuma duplicata exata, nenhum hash comum OK/NG.
- 25 trincas por **nome** (sufixos _2/_3 incluídos), sem manifesto event_id.
- Isso **não prova** 117 observações independentes, rótulos perfeitos ou cobertura
  suficiente para treinamento seguro de NG multilight.

## Objetivo desta etapa

Converter os screenshots AOI em pares gabarito/teste completos, sem treino e
sem mudar os rótulos. Reutilizar exclusivamente
`ScreenMonitor.process_external_image`, o extrator da produção. Proteger
a imagem inteira, evitar recortes limitados ao epicentro, e gerar relatório
de qualificação humana antes de qualquer divisão treino/validação/teste.

**Não chamar:** MoEOrchestrator, KNN, CNN, pesos, replay do startup, comando XP
ou API de Produção. A extração não executa julgamento e não publica modelo.

### Comando

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.faltando_neural_dataset
```

Saída local sob `reports/faltando_neural/run_<timestamp>/`:

- `manifest.json`: hash SHA-256 da origem, rótulo **provisório** da pasta,
  iluminação, event_id se comprovado por manifesto real, status, pendências de
  qualificação, caminhos de imagens derivadas, tamanhos e OCR observado;
- `summary.txt`: total extraído, falhas, vínculos de evento confirmados e
  candidatos por nome;
- `pairs/<label>_<hash>/reference.png`: gabarito completo;
- `pairs/<label>_<hash>/test.png`: teste completo.

Nenhum arquivo em `public` é editado, renomeado ou substituído. A pasta
`reports/faltando_neural` é excluída do Git. A saída é um staging técnico,
**não** um dataset de treinamento aprovado.

## Qualificação obrigatória antes do treino

- Confirmar que a pasta OK/NG corresponde a **julgamento humano**, não
  etiqueta automática, em cada imagem selecionada.
- Inspecionar se gabarito e teste foram corretamente recortados (sem
  barras/textos da interface misturados às regiões físicas).
- Conferir a categoria via OCR quando disponível; preservar o nome como
  pista, não reclassificar automaticamente.
- Validar que arquivos nomeados SIDE/TOP/MID pertencem ao mesmo evento:
  **sufixo e horário não bastam**. Um manifesto com hashes/event_id permite
  vínculo confiável; os demais são somente sugestões de grupo.
- Identificar duplicatas **perceptuais**, capturas sucessivas da mesma peça
  e casos de PCB/componentes equivalentes; SHA distinto não garante
  independência.
- Identificar diversidade de defeitos NG. Com apenas 10 PNGs NG SIDE, o
  conjunto ainda não sustenta declarar CNN multilight segura para operação.
- Somente depois da qualificação, definir splits por **evento/peça/lote**
  sem vazamento entre treino e avaliação.

A execução informa `training_ready=false` em todos os candidatos. Esta
etapa não altera a decisão operacional nem o startup gate.

## Qualificação visual assistida — próxima etapa implementada

Use, depois da preparação e atualização do repositório:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.ui.faltando_neural_review
```

Para escolher uma preparação específica:
`python -m src.ui.faltando_neural_review --manifest "C:\visionx-neural-main\reports\faltando_neural\run_XXXXX\manifest.json"`.

A interface offline lista casos individuais e trincas candidatas. Gabarito e
teste são exibidos inteiros, sem crop oculto. O usuário confirma cada par
como `CONFIRMED_OK` (componente presente), `CONFIRMED_NG` (componente
ausente) ou `REJECTED` (recorte impróprio); requer checkbox humano, e
registra eventual divergência contra o rótulo original, sem alterar a fonte.

Uma trinca SIDE/TOP/MID só é validada manualmente depois de todos os três
pares terem rótulo confirmado equivalente ao rótulo arquivado; gera
`human_group_id`, nunca falsifica `event_id` original da AOI.
As sugestões de semelhança dHash consideram ambos gabarito/teste na mesma
iluminação e **não** removem imagens ou alteram rótulos.

Cada decisão é salva imediatamente em
`reports/faltando_neural/run_*/qualification.json`, com confirmação
SHA-256 da origem, em escrita atômica. O estado é restaurado ao reabrir.
Os originais em `public`, o dataset e o `manifest.json` não mudam.

Esta revisão é preparatória: `training_ready=False` continua intacto,
não treina CNN, não cria splits e não interfere no ODIN em Produção.

## Treinamento experimental aprovado pelo operador (08/10/2026)

O operador confirmou que os rótulos do acervo são válidos e **autorizou
treinamento imediato sem revisão manual obrigatória**. O painel visual
anterior continua disponível para auditoria opcional. O treinamento acessa
os recortes **já extraídos**; não renomeia ou move imagens originais.

O modelo `src/core/neural/faltando_cnn.py` é uma CNN comparativa
(gabarito versus teste) com extrator compartilhado e seleção por
`SIDE/TOP/MID`. Eventos monoimagem legados usam máscara SIDE; cada
trinca de três luzes OK com OCR `board/parts/value` coerente
é agrupada provisoriamente como um evento. As imagens da trinca
não são três eventos independentes. A saída é um único logit da
presença/ausência; o maior score NG das iluminações disponíveis
determina o score do evento. KNN e motores físicos não são consultados.

O treino é executado na máquina nova, no **mesmo ambiente Python**:

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -c "import torch; print(torch.__version__)"
python -m src.scripts.train_faltando_cnn --epochs 25 --batch-size 4 --size 160 --device cpu
```

Caso `import torch` falhe, é necessário disponibilizar o pacote PyTorch
compatível nesse ambiente (sem instalar software com privilégio de
administrador). Evite alterar o ambiente Python do Windows XP.
O treinamento pode levar vários minutos na CPU corporativa.

**Saídas do treino** (nenhum arquivo será enviado ao GitHub):

```text
reports/faltando_neural/models/experiment_<timestamp>/
    faltando_cnn_candidate.pt
    training_report.json
    training_summary.txt
```

A avaliação utiliza holdout com separação por board/parts e similaridade
de imagem, evitando que uma trinca apareça parcialmente no treino e no
holdout. O relatório inclui `FN_NG_as_OK`: NG verdadeiro liberado
erroneamente como OK. Se não for possível formar holdout com OK/NG
independentes, o treino falha sem salvar modelo.

**Limites e estado:**
- Apenas 10 NG SIDE históricos e nenhum NG TOP/MID: métrica pequena, sem
  cobertura de falhas reais multilight.
- Rede pequena treinada do zero: demonstrador/linha de base; não alegar
  transfer learning nem generalização comprovada.
- Checkpoint `experimental=True`, `production_approved=False`.
  Nenhuma alteração em `main.py`, no roteador KNN, no modo Produção ou
  na regressão de startup. Substituição de motores e automação de decisão
  serão etapas posteriores após avaliar os resultados reais.


## Resultado real da v1 — 08/10/2026

O operador treinou a v1 com `python -m src.scripts.train_faltando_cnn
--epochs 25 --batch-size 4 --size 160 --device cpu` e enviou
`training_report.json` e `training_summary.txt`. A execução local
resultou em:

| Medida | Resultado |
|---|---:|
| Capturas preparadas | 117 |
| Eventos (25 trincas OK + 42 monoimagem) | 67 (57 OK / 10 NG) |
| Treinamento | 54 (46 OK / 8 NG) |
| Desenvolvimento (holdout) | 13 (11 OK / 2 NG) |
| NG detectados (TP) | 0 |
| NG erroneamente considerados OK (FN) | **2** |
| OK corretamente considerados OK (TN) | 11 |
| OK erroneamente considerados NG (FP) | 0 |
| Acurácia | 84,62% |
| Recall NG | **0%** |
| Perda treino época 1 → 25 | 1,337919 → 0,000206 |
| Perda no holdout época 25 | 0,637349 |

A rede respondeu **OK para todos os 13 eventos** e errou os dois NG
do holdout: `2026-10-01_07-53-17-716_FALTANDO.png` e
`2026-10-01_09-53-18-089_FALTANDO.png`. Ambos possuem componente
OCR aproximado R475; esse detalhe sugere dificuldade de generalizar
com poucos NG, sem provar um único defeito de mecanismo universal.
Todas as 25 épocas obtiveram **TP_NG=0** na validação.
Os 84,62% expressam prevalência OK, não detecção confiável.
O declínio quase completo de perda apenas no treinamento
é compatível com sobreajuste.

**v1: REPROVADA para julgamento automático de FALTANDO.**
Nenhum motor de Produção foi substituído.

## CNN FALTANDO v2 — mudanças e comando

A v2 implementa uma CNN comparativa de duas escalas:
imagem integral e recorte central de 70% redimensionado,
preservando mais detalhes da região de inspeção. Cada uma recebe
RGB de gabarito, RGB de teste e diferença absoluta dos dois.
Mantém SIDE/TOP/MID agrupados, respeitando máscara de luz
para imagens legadas SIDE. O modelo utiliza mapas espaciais 2×2
antes da cabeça de classificação, sampler OK/NG balanceado,
perda auxiliar por iluminação, dropout e early stopping.
Não introduz regras físicas de decisão nem usa a memória KNN.

A v2 registra **probabilidades por evento e por iluminação**,
erros FP/FN e melhor época. O mesmo split (seed 42) permite
comparação de desenvolvimento com a v1; **não é teste cego**,
pois já vimos as falhas da v1 nesse conjunto. Para validar
generalização serão necessários eventos inéditos, especialmente
NG reais em TOP/MID, antes de qualquer substituição operacional.

```powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.scripts.train_faltando_cnn_v2 --epochs 25 --batch-size 4 --size 160 --device cpu
```

Salva localmente em `reports/faltando_neural/models/experiment_v2_*/`:

- `faltando_cnn_v2_candidate.pt` — pesos **candidatos**;
- `training_report_v2.json` — treino, split e resultados;
- `holdout_predictions_v2.json` — cada caso, verdade,
  probabilidade NG, iluminação e erro;
- `training_summary_v2.txt` — resumo para leitura humana.

**Ainda não executado nos dados reais nesta etapa.**
A v2 permanece desligada no ODIN, com
`production_approved=False`. Enviar os três relatórios
para avaliação antes de discutir roteador KNN/CNN.
