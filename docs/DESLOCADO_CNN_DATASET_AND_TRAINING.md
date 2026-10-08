# CNN especializada DESLOCADO — dataset, bootstrap e aprendizado incremental

## 08/10/2026 — Solicitação

A categoria \`DESLOCADO\` deve seguir o mesmo macrofluxo da CNN FALTANDO:
**AOI → memória KNN verificada → CNN especializada se novo →
rótulo humano → treino incremental → teste de regressão → modelo ativo**.

**Estado inicial declarado pelo operador:**
- NG reais em \`public/ng_archive\`: **nenhum DESLOCADO**.
- OK SIDE histórico em \`public/ok_archive\`, por exemplo:
  \`2026-10-02_1349_DESLOCADO.png\`,
  \`2026-10-02_1350_DESLOCADO.png\`,
  \`2026-10-02_1351_DESLOCADO.png\`.
- OK SIDE/TOP/MID atuais, por exemplo:
  \`2026-10-08_0754_DESLOCADO_SIDE.png\`,
  \`2026-10-08_0754_DESLOCADO_TOP.png\`,
  \`2026-10-08_0754_DESLOCADO_MID.png\`.
- **As contagens reais ainda dependem de rodar o inventário no Windows 10**;
  estes nomes representam exemplos, não uma contagem integral verificada.

## Estrutura implementada

### Preparação do dataset

\`src/services/deslocado_neural_dataset.py\` reutiliza a extração **real**
da AOI de \`AOIPairExtractor\`: imagem integral → gabarito/teste.
Filtra somente DESLOCADO do inventário de \`ng_archive\` e \`ok_archive\`,
confere validade PNG, SHA-256 e separa SIDE legado de SIDE/TOP/MID.
Mantém candidatos de trinca somente por nome; não inventa
\`event_id\`. Nenhuma imagem do acervo original é alterada.

\`\`\`powershell
cd "C:\visionx-neural-main"
git pull origin central
python -m src.services.deslocado_neural_dataset
\`\`\`

Gera \`reports/deslocado_neural/run_<timestamp>/manifest.json\`,
\`summary.txt\` e \`pairs/<id>/reference.png, test.png\`.
Se houver erro de extração, não prosseguir para treinamento.

### CNN DESLOCADO v1 (pesos separados)

\`src/core/neural/deslocado_cnn.py\` utiliza a **mesma arquitetura base
comparativa de duas escalas da FALTANDO v2**, mas a instância tem
checkpoint, treinamento e schema distintos:
\`visionx.deslocado_comparative_cnn.v1\`.
Entradas: gabarito/teste/diferença, imagem completa + região central,
SIDE/TOP/MID com máscara; três luzes coerentes são um evento.

**Zero NG DESLOCADO reais significa que ainda não é possível validar
supervisionadamente um classificador OK×NG real.** Para iniciar o
aprendizado da arquitetura sem inventar NG reais,
\`src/scripts/train_deslocado_cnn.py\` treina em **OK reais vs
deslocamentos locais sintéticos** da região central do teste.
Cada proxy fica identificado como \`SYNTHETIC_PROXY_SHIFT\`, separado
dos futuros NG reais. Esses proxies podem conter artefatos que não
existem em defeitos industriais; seus resultados **não medem recall NG**
nem autorizam decisões na produção. Amostras multilight são agrupadas
quando nome e OCR são coerentes; grupos board/parts são separados
no desenvolvimento quando possível.

\`\`\`powershell
python -m src.scripts.train_deslocado_cnn --epochs 15 --batch-size 4 --size 160
\`\`\`

Saída em \`reports/deslocado_neural/models/experiment_*/\`:

- \`deslocado_cnn_candidate.pt\`: checkpoint isolado,
  \`production_approved=False\`,
  \`allow_automatic_classification=False\`;
- \`training_report_deslocado.json\`: indicadores separados
  para OK real e proxy sintético, grupos e limitações;
- \`training_summary_deslocado.txt\`: resumo legível.

**Este checkpoint não é carregado em \`main.py\`.**
O ODIN continua decidindo DESLOCADO por seus motores físicos
existentes; nenhuma alteração de automação de OK/NG nesta categoria.

### Memória KNN e incremento em Teste / Produção / Sombra

O roteador KNN primeiro permanece igual: par visual
humano conhecido → memória KNN, sem especialistas.
Se o par DESLOCADO for **novo**, a rota continua
\`NEW_EXPERTS\` (especialistas físicos), com
\`detail.specialist_candidate =
DESLOCADO_CNN_V1_BOOTSTRAP_NOT_ACTIVE\`.
O tooltip explica que a CNN está em treinamento
e ainda não substitui os motores.

Quando o operador confirma **OK ou NG** no caso novo
em qualquer modo:
1. O \`DatasetManager\` salva normalmente, incluindo gabarito
   e teste mesmo quando a IA havia concordado;
2. \`DecisionPersistenceQueue\` só solicita o treino se a gravação
   humana tiver sucesso;
3. \`OnlineLearningQueue\` grava evento durável em
   \`reports/neural_online/events/\`, com hashes e até três luzes;
4. \`SPECIALIST_TRAINERS["DESLOCADO"]\` chama
   \`src.scripts.train_deslocado_cnn_online\` num processo CPU
   separado do ciclo AOI;
5. O treinador junta os OK históricos e todos os casos
   DESLOCADO humanos novos, agora incluindo **NG reais** se
   aparecerem. Futuros NG são marcados \`REAL_NG\`,
   nunca falsificados como sintéticos. O próximo candidato
   é salvo em \`reports/deslocado_neural/models/\`.

**Bloqueio intencional:** ao contrário da CNN FALTANDO v2,
\`DESLOCADO\` ainda não possui qualificação com NG reais.
Seu treinador online **não promove** checkpoint nem cria
\`reports/neural_online/live_active.json\` para esta categoria.
A ativação exigirá evidências NG reais independentes,
holdout rigoroso, replay de todos os OK/NG e validação
de NG SIDE/TOP/MID antes de liberar previsões em operação.
Mesmo quando surgirem NG online, o candidato permanece
experimental até essa etapa explícita.

O registro da fila é extensível para futuras especialidades;
os eventos de categorias distintas permanecem isolados.

## Verificações / pendências

- Testes automatizados utilizam imagens sintéticas: não avaliam o
  dataset real do computador corporativo.
- Antes de prosseguir, executar preparação local e enviar
  **\`manifest.json\` e \`summary.txt\`**.
- Executar treino local apenas se a preparação tiver extraído
  todos os pares esperados. Enviar então
  \`training_report_deslocado.json\` e
  \`training_summary_deslocado.txt\`.
- Quando houver NG real de DESLOCADO, registrá-lo com
  confirmação humana e luminosidade identificada. O dataset
  não deve incluir proxies como defeitos verificados.
