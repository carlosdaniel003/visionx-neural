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
