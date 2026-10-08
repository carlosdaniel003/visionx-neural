# Preparação do dataset neural — FALTANDO

**Etapa atual:** preparação/qualificação offline, sem treinamento ou substituição de motores.
**Branch:** central. **Máquina de execução:** Windows 10 do ODIN (não o XP).
**Fonte:** \`public/ok_archive\` e \`public/ng_archive\`, somente leitura.

## Inventário informado pelo operador (08/10/2026)

- 117 PNGs FALTANDO: 107 OK e 10 NG.
- NG: 10 sem sufixo de luz (SIDE histórico presumido); nenhum NG TOP/MID.
- OK: 56 sem sufixo + 17 SIDE, 17 TOP e 17 MID explícitos.
- SHA-256 distintos: 117; nenhuma duplicata exata, nenhum hash comum OK/NG.
- 17 trincas por **nome**, ainda sem comprovação por manifesto event_id.
- Isso **não prova** 117 observações independentes, rótulos perfeitos ou cobertura
  suficiente para treinamento seguro de NG multilight.

## Objetivo desta etapa

Converter os screenshots AOI em pares gabarito/teste completos, sem treino e
sem mudar os rótulos. Reutilizar exclusivamente
\`ScreenMonitor.process_external_image\`, o extrator da produção. Proteger
a imagem inteira, evitar recortes limitados ao epicentro, e gerar relatório
de qualificação humana antes de qualquer divisão treino/validação/teste.

**Não chamar:** MoEOrchestrator, KNN, CNN, pesos, replay do startup, comando XP
ou API de Produção. A extração não executa julgamento e não publica modelo.

### Comando

\`\`\`powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.faltando_neural_dataset
\`\`\`

Saída local sob \`reports/faltando_neural/run_<timestamp>/\`:

- \`manifest.json\`: hash SHA-256 da origem, rótulo **provisório** da pasta,
  iluminação, event_id se comprovado por manifesto real, status, pendências de
  qualificação, caminhos de imagens derivadas, tamanhos e OCR observado;
- \`summary.txt\`: total extraído, falhas, vínculos de evento confirmados e
  candidatos por nome;
- \`pairs/<label>_<hash>/reference.png\`: gabarito completo;
- \`pairs/<label>_<hash>/test.png\`: teste completo.

Nenhum arquivo em \`public\` é editado, renomeado ou substituído. A pasta
\`reports/faltando_neural\` é excluída do Git. A saída é um staging técnico,
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

A execução informa \`training_ready=false\` em todos os candidatos. Esta
etapa não altera a decisão operacional nem o startup gate.
