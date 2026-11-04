# ODIN AOI — Relato do operador, problemas recorrentes e critérios de sucesso

**Registro:** 04/11/2026  
**Origem:** observações do operador responsável pelo desenvolvimento do ODIN VisionX, ao longo de meses de testes, alterações de regras, aprendizado de memória e treinamento CNN.  
**Natureza:** requisitos e problemas observados. Este documento **não** afirma que os problemas estão solucionados, nem determina previamente que uma tecnologia específica seja a solução.

## 1. Relato do operador — manter visível nas próximas alterações

### 1.1. Alternância entre acerto de OK e acerto de NG

> Há momentos, após muitas alterações de estratégia, em que o ODIN reconhece corretamente os defeitos NG, mas passa a classificar peças OK como NG. Em outros momentos, reconhece os OK, mas libera defeitos reais como OK.

- A melhoria de uma classe não pode ser considerada sucesso se degradar significativamente a outra.
- Não selecionar modelo ou mudança somente pela taxa global de acerto.
- Verificar separadamente **NG → OK (defeito não detectado)** e **OK → NG (falso alarme)**; contabilizar também ambos por categoria AOI e iluminação.
- O comportamento precisa se manter em inspeções novas, não apenas nos exemplos usados para ajustar o código.

### 1.2. Correções humanas esquecidas após centenas de eventos

> Um evento é corrigido manualmente como OK ou NG; quando reaparece após cerca de 500 inspeções, o ODIN às vezes não o reconhece, tratando-o como algo novo.

- Investigar persistência, recuperação, reindexação, prioridade temporal da decisão humana, similaridade robusta entre capturas e possível dependência do cache da sessão.
- Testar reocorrências após 500, 1.000 e mais eventos, depois de reiniciar o ODIN, em outra data e com diferenças de iluminação/captura.
- **Não confundir** identidade RGB exata com recorrência visual do mesmo padrão ou defeito físico.
- O julgamento humano mais recente deve ser rastreável e considerado pelas políticas de aprendizagem aplicáveis; não presumir que uma correção foi preservada apenas porque o registro foi salvo em disco.
- Evitar esquecimento, colisão de rótulos e sobreposição silenciosa por novas decisões automáticas.

### 1.3. A memória foi a abordagem de melhor resultado prático até agora

> O investimento em memória e treinamento com decisões humanas trouxe os melhores resultados observados pelo operador, mais consistentes do que apenas regras determinísticas ou CNN.

- Tratar essa informação como **observação empírica do operador**, ainda sem benchmark cego conclusivo.
- Manter a memória disponível como mecanismo distinto e auditável da CNN, sem permitir que seu desempenho aparente esconda falhas de classificação física.
- Investigar por que eventos semelhantes retornam como `vi semelhante` ou `não encontrei par exato` apesar de existirem imagens equivalentes no acervo.
- Diferenciar recuperação por similaridade, identificação de um exemplo anteriormente julgado e inferência generalizável de qualidade.

### 1.4. Crescimento de regras e exceções gerando regressões

> Diversas regras determinísticas, ajustes de limiar e exceções funcionaram para alguns exemplos, mas quebraram outros. O motor ficou complexo sem resolver de forma estável a classificação.

- Não adicionar um `if` específico para cada novo evento sem justificar generalização e comprovar ausência de regressões.
- Priorizar módulos com responsabilidades claras: extração dos pares, CNN, memória, evidência física, fusão, apresentação e ação XP.
- Remover duplicações, caminhos residuais e regras contraditórias mediante testes de regressão, em mudanças separadas e revisáveis.
- Uma classificação visual exibida como OK ou NG não equivale automaticamente à autorização para pressionar `0` ou `1` na máquina.

### 1.5. O acervo histórico é o ativo mais confiável disponível

> O operador possui as imagens OK e NG em `ok_archive` e `ng_archive`, incluindo registros mais antigos, salvos e julgados **antes** de existir a funcionalidade atual de captura/arquivamento.

- Preservar e aproveitar as capturas antigas; não descartar imagens apenas por precederem a estrutura atual de memória.
- Preservar SHA-256, caminho original, data, categoria, iluminação, proveniência e correções humanas.
- Diferenciar **imagem fisicamente inspecionada com rótulo confirmado** de **imagem salva em pasta OK/NG com rótulo provisório**. A presença em `ng_archive` pode refletir um NG falso da AOI.
- Deduplicar sem perder proveniência; evitar vazamento de cópias ou quase duplicatas entre treino, validação e teste.
- Confirmar os vínculos SIDE/TOP/MID por evidência do evento real, não apenas pelo nome ou minuto da captura.
- O objetivo é aprender a reconhecer o padrão físico de qualidade em **novas capturas**, não apenas comparar igualdade de pixels com o acervo.

### 1.6. Objetivo final expresso pelo operador

> O que eu preciso é que o ODIN julgue corretamente o que é OK e o que é NG.

Requisito operacional: veredito **binário**, estável, auditável e generalizável; taxa de falso NG e taxa de NG liberado precisam ser medidas juntas. O objetivo de produção é autonomia 0/1, mas um candidato CNN não passa a ter autorização automática sem critérios de qualidade e segurança demonstrados.

## 2. Critérios mínimos de aceite para uma melhoria futura

1. **Matriz de confusão com ambos os erros:** registrar quantos OK foram rejeitados e quantos NG reais foram liberados; não mascarar casos difíceis por acurácia agregada ou mudança de limiar.
2. **Validação em casos independentes:** reservar inspeções novas, com rótulo humano conferido, sem seleção por desempenho observado no mesmo teste; comparar com o modelo anterior nas mesmas amostras.
3. **Estratificação obrigatória:** apresentar resultados para FALTANDO, EMBORCADO, INVERTIDO, DESLOCADO e MUITO ADESIVO, separados por SIDE, TOP e MID onde houver rótulos suficientes; sinalizar células sem exemplos.
4. **Persistência de aprendizagem:** demonstrar que um julgamento humano continua recuperável depois de centenas de inspeções, reinício, mudança de sessão e variação visual, sem transformar semelhança em identidade exata.
5. **Regressão de ponta a ponta:** toda correção relevante deve manter uma suíte congelada de eventos OK e NG representativos, inclusive os que anteriormente voltaram a falhar.
6. **Separação de responsabilidades:** a CNN aprende padrões nos pares gabarito/teste; a memória recupera experiências; regras físicas e fusão justificam evidências; a política XP governa a emissão final. Alterar uma não deve silenciosamente redefinir outra.
7. **Acervo íntegro e representativo:** preservar a história, verificar a proveniência dos rótulos e reconhecer ausência de exemplos de uma classe/categoria/iluminação antes de declarar melhoria.
8. **Sem promessa de 100% baseada só no acervo:** a meta de maximizar ambos os acertos não equivale a garantia estatística de ausência de erros em toda produção.

## 3. Evidência associada — primeiro treinamento offline CNN V3 de 04/11/2026

O operador anexou o relatório local `training_report(5).json`, com:
- `schema=visionx.aoi_quality_cnn_v3_training_report.v1`;
- execução UTC `2026-11-04T17:35:25.801668+00:00`;
- candidato `reports/aoi_quality_cnn_v3/candidates/20261104T173525_801668Z_90bed0/aoi_quality_cnn_v3_candidate.pt`;
- `candidate_sha256=781d5d4d6cb80461a990e9a325560b75688f24f4647846ef12023e3898e15e1c`.

**Composição do acervo após preparação:**
- 1.112 pares de origem; 1.096 pares após deduplicação exata/remoção de um caso ignorado;
- 780 rótulos provisórios/corrigidos NG e 316 OK;
- 1.060 exemplos com rótulo de **pasta provisório**, 13 rótulos NG provenientes de contato humano revisado e 23 rótulos com confirmação humana SHA;
- validação humana completa: **não**; grupos físicos multilight confirmados: **não**.

**Teste separado registrado pelo script (215 exemplos, rótulos majoritariamente provisórios):**

| Modelo | Acertos / total | NG → OK (defeito liberado) | OK → NG (falso alarme) |
|---|---:|---:|---:|
| CNN V2 pinada — comparação no mesmo teste | 120 / 215 (55,8%) | 95 / 152 | 0 / 63 |
| CNN V3 candidata — época selecionada 4 | 200 / 215 (93,0%) | 10 / 152 | 5 / 63 |

**Interpretação restrita:** o relatório mostra redução importante de NG → OK no conjunto rotulado, porém surgimento de 5 OK → NG que a comparação V2 não apresentava. Essa troca reflete precisamente a preocupação do operador em não melhorar uma classe sacrificando a outra. A comparação é **diagnóstica**, não aprovação para produção: `blind_holdout=false`, `all_quality_labels_operator_verified=false`, `physical_event_independence_proven=false`, `production_eligible=false`. As métricas dependem em grande parte de **rótulos de pasta não auditados fisicamente**.

Também há cobertura insuficiente em categorias/luzes (por exemplo, MUITO ADESIVO não tem amostras OK em TOP ou MID), o que impede conclusão abrangente sobre essas condições.

Este relatório **não** é autorização para promover V3, modificar o checkpoint V2, alterar regras de emissão XP ou declarar resolução definitiva dos problemas acima.

## 4. Como usar este documento

- Antes de mudar a CNN, memória, limiares ou fusão, confrontar a proposta com os problemas de §1 e os critérios de §2.
- Ao finalizar uma experiência, registrar em relatório qual problema ela abordou, o que foi medido, qual erro melhorou, qual piorou e o que permanece desconhecido.
- Novas correções humanas não devem ser usadas como único teste de sucesso da alteração que as incorporou.
- **Estado em 04/11/2026:** o histórico e os requisitos foram formalizados; a V3 foi treinada offline segundo relatório anexado, **sem comprovação de generalização industrial**.

Documentos técnicos associados:
- [Treinamento da CNN V3](AOI_QUALITY_CNN_V3_20261104.md).
- [Auditoria histórica de regressões](AOI_UNIVERSAL_HISTORICAL_REGRESSIONS_20261026.md).
- [Preparação do dataset](FALTANDO_CNN_DATASET_PREPARATION.md).
