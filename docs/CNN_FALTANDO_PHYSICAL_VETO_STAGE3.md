# Etapa 3 — Contraprova física independente para veto do falso OK CNN FALTANDO

**15/10/2026.** Esta etapa protege exclusivamente a hipótese `FALTANDO` no fluxo `NEW_CNN` que produz três votos **OK** em `SIDE/TOP/MID`. Não altera as etapas concluídas de KNN e treinamento da CNN, nem o escopo de EMBORCADO, INVERTIDO, DESLOCADO ou ADESIVO.

## Problema e natureza do novo sinal

No evento `f39c34cd51bb48abb7538afd6eec550e`, a CNN FALTANDO V2 votou OK nas três luzes, mas o operador confirmou fisicamente NG. O consenso, por si só, reproduziu um erro sistemático da mesma CNN; as saídas `physical_score=0` não representam uma segunda rede física ativa.

Já havia um `ROIPatchExpectationExpert` independente da CNN e do KNN. A Etapa 3 o utiliza somente como **contraprova física** aplicada ao par completo gabarito/teste de cada iluminação após a fusão multilight. A nova função `physical_absence_probe` analisa duas ROIs centrais (58% e 76% da imagem), com limite de 256 pixels no maior lado para limitar custo. Examina presença e estrutura esperada (cores/bordas, cobertura, exposição de fundo, distinção visual e compatibilidade em regiões próximas).

Os limiares `MIN_SCORE=0.72`, `MIN_COVERAGE=0.32`, `MIN_BACKGROUND=0.30`, `MIN_REFERENCE_DISTINCTNESS=0.25`, `MAX_DIRECT_SIMILARITY=0.65`, `MAX_NEARBY_SIMILARITY=0.35` são **hipóteses conservadoras para um veto** e não estão calibrados em holdout industrial. Uma contraprova abaixo desses valores significa **ausência de evidência física forte**, e NÃO prova de que o componente está presente.

## Integração e garantias de segurança

1. `src/ui/adhesive_multilight_inspection.py::_finalize_multilight_decision` utiliza as três amostras já capturadas em `adhesive_multilight_learning_samples` (mesmo ciclo), antes de publicar o resultado no painel ou chamar o controlador de Produção.
2. `src/core/physical_missing_veto.py` só executa a sonda quando a fusão produziu `NEW_CNN + FALTANDO + consenso OK completo de 3 luzes`. Não roda para CNN com NG, KNN conhecido, adesivo ou outras categorias.
3. `SUSPECT_ABSENCE` em **qualquer** luz ou `UNAVAILABLE` por erro, ROI ausente, luz faltante ou medida inválida convertem o resultado final em `REVISÃO OBRIGATÓRIA`, sem criar um rótulo NG. O operador é quem decide 0/1. `NO_STRONG_EVIDENCE` não habilita nenhuma aprovação: apenas preserva a política anterior, que já exigia três scores CNN válidos e SHA consistente.
4. O `production_decision_policy` exige **explicitamente** a presença do registro `physical_missing_veto.status=NO_STRONG_EVIDENCE` antes de sequer considerar AUTO-OK CNN FALTANDO. Registro ausente, inválido, `SUSPECT_ABSENCE` ou `UNAVAILABLE` bloqueia `auto_allowed`, mesmo se campos superficiais do veredito forem reescritos antes do disparo. Este status não certifica a peça: prova somente que a sonda executou sem disparar seus critérios de suspeita.
5. A auditoria inclui `detail.physical_missing_veto`: por luz, status, duas ROIs e métricas numéricas. Nunca grava máscaras grandes, altera dados humanos ou contamina o treino. Imagens brutas do Windows XP não são reutilizadas como pares de inspeção.

## Auditoria do NG confirmado na estação (sem XP e sem treino)

Depois do merge, com o ODIN **fechado** e na mesma estação que possui `reports/neural_online/events/`, execute:

```powershell
cd C:\visionx-neural-main
python -X utf8 -m src.services.startup_regression.cnn_physical_veto_replay --aoi-event-id f39c34cd51bb48abb7538afd6eec550e
```

O comando verifica `source_aoi_event_id`, rótulo humano e SHA-256 de cada PNG. Cada iluminação usa o **seu próprio** par recortado. Saída em `reports/startup_regression/cnn_physical_veto_*.json` com `suspect_lights`, `unavailable_lights`, `human_label` e `would_veto_cnn_all_ok`. O nome indica teste contrafactual: **não é uma nova inferência da CNN**.

Se sair `NO_JOURNAL`, a confirmação pode ter sido salva na memória KNN/dataset, mas não existe prova do journal supervisionado no disco examinado; recuperar as evidências auditáveis antes de ajustar limiares. Se o NG não for suspeito pela sonda, não declarar que a proteção detecta esse caso; examinar as métricas, geometria e capturas reais. É proibido reduzir limiares só para ajustar essa amostra.

## Aceitação técnica exigida

- Testes unitários e de integração: suspeita em apenas uma luz impede 0/OK, erro/incompletude falha fechado, política de produção revalida o veto antes do envio, sem transformar OK em NG; pares visuais idênticos não geram suspeita; adesivo/outros defeitos não mudam; a finalização da interface realmente executa a trava.
- Testar na estação o NG confirmado e uma amostra independente de OK/NG para **cada** luz (e famílias distintas de componente). Medir falsos alarmes de revisão em OK, taxa de suspeitas no NG e latência extra; ainda não há métricas reais.
- `NO_STRONG_EVIDENCE` não atesta ausência de falso OK; **não há garantia industrial de detecção de todas as ausências**. A política de AUTO-OK já existente continua habilitada quando CNN e contraprova não apontam revisão. Uma implantação que exija zero auto-OK até certificação precisa de autorização específica para desativá-lo completamente.
- Etapa 3 NÃO altera pesos CNN, KNN, thresholds CNN, ROI original da AOI, UX de decisão humana, comando do agente XP ou hardware. Nenhum teste sintético certifica desempenho com placas reais.

Arquivos principais: `src/core/physical_missing_veto.py`, `src/ui/adhesive_multilight_inspection.py`, `src/ui/production_confidence_gate.py`, `src/services/startup_regression/cnn_physical_veto_replay.py`, testes e GitHub Actions.
