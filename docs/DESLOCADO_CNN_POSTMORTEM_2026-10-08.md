# CNN DESLOCADO — encerramento e retrospectiva técnica

**Data:** 08/10/2026  
**Decisão:** **DESENVOLVIMENTO ABORTADO / EXPERIMENTOS ARQUIVADOS** por solicitação do operador.  
**Motivo determinante:** não existe no acervo avaliado nenhum **NG DESLOCADO real, confirmado e independente**. Portanto não é possível aferir se a CNN identifica deslocamento físico verdadeiro ou se libera defeitos como OK.  
**Escopo do encerramento:** desenvolvimento/treino/revisão de máscaras/ajuste de limiares e proposta de ativação da CNN especializada DESLOCADO. **Não significa desativar a inspeção DESLOCADO atualmente existente no ODIN.**  
**Ação nesta atualização:** apenas documentação; não alterar motores, checkpoints, imagens, arquivos de treino, integração, KNN nem a CNN FALTANDO.

> **IMPORTANTE — documento histórico:** os scripts e os comandos das etapas abaixo continuam no repositório para rastreabilidade, **não são etapas pendentes para o operador executar**. As antigas recomendações de “próximo experimento” em outros documentos foram superadas por esta decisão. Não iniciar novos experimentos DESLOCADO sem nova decisão explícita e dados de validação adequados.

## 1. Material e pergunta da pesquisa

- Inventário utilizado em 08/10/2026: **34 imagens/observações DESLOCADO rotuladas OK**, equivalentes a **24 eventos candidatos**: **19 SIDE legados** mais **5 trincas SIDE/TOP/MID** (15 observações). Os vínculos das cinco trincas vêm de nome/OCR e não equivalem a `event_id` verificado por origem.
- **0 NG reais DESLOCADO** no inventário. Consequentemente, `real_NG_recall=null`; não há sensibilidade, precisão operacional ou taxa de falsos OK de NG real medível.
- Comparação principal: **gabarito vs. teste**, nos quais a marcação do componente (letras, números, tinta branca) pode variar sem que o corpo se desloque.
- Os **três OK SIDE históricos** que a CNN v2 chamou de NG: `2026-10-02_1349_DESLOCADO.png`, `2026-10-02_1400_DESLOCADO.png`, `2026-10-02_1415_DESLOCADO.png`.
- Referência auditável do acervo: `run_20261008T173116_094174Z/manifest.json`, SHA-256 `6af07429aed97292eb47c14c12669d49936ef470e48c2f643b292c9a0421bd94`.

**Pergunta ainda sem resposta experimental:** “A CNN separa um componente **realmente deslocado** de um OK com inscrições, iluminação e reflexos diferentes?” **Sem NG reais, nenhuma tentativa respondeu a essa pergunta.**

## 2. Placar das tentativas — critérios de avaliação não equivalentes

| Método / tentativa | Resultado efetivamente observado | Proximidade ao objetivo | Limitação que impediu avanço |
| --- | --- | --- | --- |
| **CNN DESLOCADO v1**, comparativa e proxies sintéticos deslocando região central | Treino: **18/18 OK** e **18/18 proxies**. Desenvolvimento agrupado: **0/6 OK reconhecidos** e **6/6 proxies artificiais detectados**, 50% no conjunto combinado. | **Longe da generalização OK.** | Aprendeu pistas artificiais/locais e tratou todos os seis OK reservados como NG; os seis proxies não equivalem a defeitos reais. Forte sobreajuste. |
| **CNN DESLOCADO v2**, duas escalas + máscara heurística/“shift” com inpainting | Desenvolvimento: **6/6 OK** e **2/2 OK recompostos** corretos, mas **0/2 proxies** detectados. Treino: **15/18 OK** e **5/12 proxies**. **Replay integral de OK conhecidos: 31/34**, isto é, **91,18%**, com **3 falsos NG**. | **Tentativa que mais se aproximou de preservar OK conhecidos**, sem demonstrar detecção NG. | Replay usa acervo já envolvido no desenvolvimento, **não é teste cego**. Prévia visual mostrou deslocamento da inscrição “104”, não do corpo; só **14/24 eventos** permitiram alguma simulação; não se sabe recall de NG real. |
| **Investigação visual dos três falsos NG v2** | Os casos 13:49, 14:00 e 14:15 eram OK; marcação interna diferente entre gabarito/teste e geometria externa aparentemente semelhante. Scores NG-proxy `0.50294089`, `0.50141501`, `0.50511873`. | **Aprendizado qualitativo útil** sobre erro de representação, não melhoria quantitativa. | Não há prova de qual região guiou a CNN sem ablação/atribuição; scores próximos a 0,5 não são probabilidades calibradas. |
| **Máscaras v3: propostas e aprovação de caixas** | **34 caixas retangulares** estruturalmente aprovadas para gabarito/teste; auditoria de seis prévias identificou possível omissão de terminais, fundo/pads dentro do retângulo e crops cortados. | **Melhor rastreabilidade/anotação**, mas **não chegou à segmentação física apta**. | Caixa não é máscara de pixels; aprovação estrutural não demonstra contorno correto ou deslocamento real. |
| **Máscaras binárias v3.1: GrabCut + editor humano** | Serviço e editor implementados, com propostas não aprovadas e validação. **Não há relatório de 34 máscaras binárias aprovadas nem treinamento CNN v3.** Procedimento manual foi rejeitado por custo de operação. | **Caminho descartado por inviabilidade de fluxo**, não um classificador avaliado. | Perda de tempo desenhando componentes; segmentação automática de inscrições, terminais e pads não comprovada. |
| **Diagnóstico geométrico OK-only v1: correlação de fase + bordas por zona** | **7/34** pares com **métricas descritivas disponíveis**; **27/34** com `EVIDENCIA_INSUFICIENTE`. SIDE **6/24**, TOP **0/5**, MID **1/5**. Três falsos NG v2 continuaram sem registro confiável. | **Cobertura exploratória parcial (20,6%)**, não classificador CNN. | Regiões externas não foram comprovadas como pads fixos; correlação de fase não mede deslocamento da peça. **7/34 não é taxa de OK/NG correto.** |
| **Diagnóstico geométrico v1.1: ORB + AKAZE + RANSAC** | **0/34** registros geométricos aceitos; `NO_GEOMETRICALLY_VALID_FEATURE_REGISTRATION` nos **34/34**, em SIDE/TOP/MID. Em comparação com v1, **7 casos antes com métricas** deixaram de passar pelo novo critério. | **Mais distante na cobertura de registro (0%)**, com bloqueio explícito em vez de uma conclusão falsa. | Correspondências e inliers inconsistentes em todos os pares nos limiares adotados. Testes automatizados com dados sintéticos passaram, mas **não provam adequação ao acervo real**. |
| **One-Class / CNN de normalidade somente OK** | **Planejada, não treinada e não validada**. | **Sem resultado medido; não pontuar como sucesso nem fracasso.** | Somente OK pode modelar normalidade, mas não certifica sensibilidade a NG nem segurança de liberação. |

**Como interpretar “mais perto” e “mais longe”:** na **única tentativa de classificação CNN com replay integral**, a v2 foi a que mais reconheceu OK conhecidos (31/34). Na **tarefa diferente de registro geométrico**, a v1 obteve métricas em 7/34 e a v1.1 em 0/34. Esses números **não devem ser combinados nem ranqueados como uma única acurácia**, e nenhum autoriza afirmar que NG real seria detectado.

## 3. Causas e lições técnicas

1. **Ausência de classe negativa verdadeira:** o impedimento principal não é a arquitetura nem um limiar; é a falta de exemplos representativos de **deslocamento físico real**, idealmente sob cada iluminação relevante. Com apenas OK, “responder OK para todos” poderia parecer ótimo e seria inseguro.
2. **Proxy errado:** deslocar texto/contraste produz um atalho visual; a CNN aprende artefatos que não correspondem à mecânica do componente. Os proxies da v2 foram inspecionados e mostraram movimento da inscrição, não da peça.
3. **Distância entre gabarito e teste não é necessariamente defeito:** tinta, numeração, brilho, sombras, reflexos e recorte variam mesmo quando a peça está posicionada corretamente.
4. **Segmentação física não foi resolvida:** caixas retangulares incluem fundo; os terminais móveis e os pads soldados/fixos não foram isolados com confiabilidade demonstrada. A exigência de desenhar contornos é incompatível com o fluxo operacional desejado.
5. **Registro global não demonstra deslocamento relativo:** ORB/AKAZE/RANSAC e correlação de fase podem tentar alinhar imagens, mas não comprovam posição do corpo em relação aos pads fixos. No acervo real, os gates foram insuficientes.
6. **Evitar vazamento e conclusões infladas:** SIDE/TOP/MID de uma mesma peça e referências repetidas não são evidências independentes. Separar por evento/placa/componente e nunca usar replay do treino como teste final de generalização.
7. **Limiar não resolve falta de NG:** os três falsos NG v2 tiveram scores próximos de 0,5, porém elevar o limiar para eliminá-los pode permitir NG reais; isso seria impossível de medir com o acervo atual.
8. **Sucesso em teste de código != sucesso industrial:** GitHub Actions validou funcionalidades, invariantes e exemplos artificiais, não detecção segura de deslocamentos em placas reais.

## 4. Artefatos que devem ser preservados (não executados)

- Fonte e extração: `src/services/deslocado_neural_dataset.py`, `reports/deslocado_neural/run_*/manifest.json`.
- CNN v1: `src/scripts/train_deslocado_cnn.py`, `training_report_deslocado.json`, `training_summary_deslocado.txt`.
- CNN v2: `src/scripts/train_deslocado_cnn_v2.py`, `src/services/deslocado_proxy_v2.py`, `holdout_predictions_deslocado_v2.json`, `deslocado_ok_replay_v2.json`, `deslocado_ok_replay_v2.txt`; checkpoint experimental `experiment_v2_20261008T174851_435041Z/deslocado_cnn_v2_candidate.pt` (não aprovado para Produção).
- V3 retangular: `src/services/deslocado_body_masks_v3.py`, `src/scripts/prepare_deslocado_body_masks_v3.py`, `validated_body_masks.json` (34 caixas).
- V3.1 máscaras: `src/services/deslocado_pixel_masks_v3.py`, `src/scripts/refine_deslocado_body_masks_v3.py` — ferramentas sem validação final do acervo e descontinuadas.
- Diagnóstico OK v1: `src/services/deslocado_ok_geometry.py`, `src/scripts/diagnose_deslocado_ok_geometry.py`, `deslocado_ok_geometry.json` e `.txt`.
- Diagnóstico OK v1.1: `src/services/deslocado_ok_geometry_v11.py`, `src/scripts/diagnose_deslocado_ok_geometry_v11.py`, `deslocado_ok_geometry_v11.json` e `.txt`.
- Revisões e GitHub Actions originais permanecem vinculados à documentação cronológica em `docs/DESLOCADO_CNN_DATASET_AND_TRAINING.md` e `estrutura_projeto.md`.

Os caminhos de `reports/` correspondem a **artefatos locais da estação da fábrica**: citar o nome no repositório não significa que todos estejam versionados no GitHub. Não excluir logs, capturas, pesos ou relatórios históricos por causa deste encerramento.

## 5. Estado final e hipótese de reabertura

- **CNN DESLOCADO: desenvolvimento encerrado / sem novas etapas aprovadas.**
- **CNN DESLOCADO v1/v2: pesos existentes apenas como candidatos históricos**, sem autorização de classificação/auto-OK em Produção; CNN v3/one-class não treinadas.
- **Especialistas atuais de DESLOCADO, CNN FALTANDO, KNN operacional, roteador, startup gate, arquivos OK/NG: sem alterações por esta decisão.**
- **Nenhum treinamento, ajuste de limiar, anotação adicional ou diagnóstico DESLOCADO deve ser solicitado automaticamente.**
- Uma futura reabertura é **uma nova decisão**, não uma etapa pendente. Exigiria NG reais confirmados com variedade de deslocamentos, iluminação e componente, separação independente treino/teste, métricas de **falsos OK em NG** e **falsos NG em OK**, e aprovação explícita antes de qualquer ativação.

**Síntese:** a **CNN v2 foi a melhor aproximação para reconhecer os OK já conhecidos (31/34)**; a **v1 falhou em todos os seis OK de desenvolvimento**; a **geometria v1 teve cobertura descritiva parcial (7/34)** e **a v1.1 nenhuma (0/34)**; as tentativas de máscaras não produziram uma segmentação física final validada. **Nenhuma abordagem provou detecção de NG DESLOCADO real. A linha de pesquisa foi interrompida por insuficiência de dados, não por uma falha universal da técnica CNN.**
