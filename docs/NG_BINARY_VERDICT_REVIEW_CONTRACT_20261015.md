# 15/10/2026 — Separar veredito NG de confirmação humana

## Evidência do evento
Evento XP `0e87f2ab7d0947248c2fa674df723fc5`, modo Teste, 5,178 s.

O log anexo mostra **`decision.verdict=DEFEITO REAL`**, **`decision.is_defect=true`** e `fusion_rule=knn_visual_cnn_physical_corroboration_v1`, porém também `operator_review_required=true`. A CNN V2 votou OK nas três luzes; a fusão encontrou NG semelhantes em SIDE/TOP/MID e contraprova física de ausência. Esses dois fatos (classificação e pendência de comando) não são mutuamente exclusivos.

A UI prévia priorizava `production_review_required` e mostrava **REVISÃO OBRIGATÓRIA** no cartão mesmo com veredito final NG. A política `production_decision_policy` também rebaixava o texto para revisão diante da trava física. Isso é uma **inconsistência de representação**, não um erro comprovado da fusão neste evento.

## Contrato alterado

1. `corroborated_ng_verdict` exige simultaneamente veredito DEFEITO REAL, `is_defect=true`, rota NEW_CNN, categoria FALTANDO, regra KNN+física, ao menos 2 luzes corroboradas por vizinhos NG e status físico `SUSPECT_ABSENCE` com luz suspeita documentada. Não cria classificação nova.
2. Quando o contrato é satisfeito, a interface mostra **DEFEITO REAL (NG)** em vermelho mesmo com confirmação humana pendente; o estado de intervenção permanece no painel operacional.
3. O gate XP preserva `verdict=DEFEITO REAL`, `proposed_decision=NG` e `operator_review_required=true`, porém `auto_allowed=false`. O operador continua decidindo a emissão de `1` (ou `0` se discordar). `_record_policy` passa a determinar o flag de pendência por `operator_review_required`, e não pelo texto do veredito.
4. Fora desse contrato estrito, revisão continua revisão. Faltou informação física, conflito OK/NG, rótulo exato não conhecido ou resultado ambíguo: o módulo não promove automaticamente NG nem libera 0/OK.
5. A captura de debug serializa **comparações efetivas** da KNN aproximada: contagem e melhor similaridade OK/NG por SIDE/TOP/MID, status, suporte NG e comparação cruzada de categorias. Não copia PNGs nem infere desempenho a partir da quantidade bruta dos diretórios.

## O papel do `ok_archive`

O operador relatou milhares de fotos iguais/parecidas no `public/ok_archive` apesar de identificar este evento como NG. **Não está demonstrado pelo log** se os arquivos são peças boas equivalentes, registros de outros componentes, casos rotulados incorretamente ou duplicatas. `ok_archive` contém exemplos rotulados OK pelo fluxo de arquivamento; o nome do diretório não confirma por si só que o conteúdo é uma peça boa verificada por humano. Vizinhos OK próximos devem ser contraprova ao NG e não podem ser descartados arbitrariamente.

O novo relatório deixa visível a comparação real: `decision.knn_visual.lights.SIDE|TOP|MID.ok_count/ng_count/best_ok_similarity/best_ng_similarity`. Sem essa distribuição, não alterar limiares ou promover pesos CNN. Os testes deste PR são de consistência de código/telemetria e **não certificam a acurácia em produção**.

## Testar

Depois de integrar a branch, repetir o mesmo grupo de 6 inspeções reais, incluindo este NG e os 5 anteriormente corretos. No cartão visual, observar se o evento NG corroborado exibe DEFEITO REAL e mantém a pendência humana na produção. Copiar o debug e conferir distribuições OK/NG nas três luzes. A medição SIDE→resultado deve ser comparada ao baseline 5,178 s. As mudanças não adicionam modelos, acesso de disco na inspeção nem inferência nova.
