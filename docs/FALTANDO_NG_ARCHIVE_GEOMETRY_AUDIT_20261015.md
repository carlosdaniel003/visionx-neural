# 15/10/2026 — FALTANDO: evidência física indisponível e vizinhos NG do acervo

## Evento real

A inspeção `239dc35fdce44b8190b6b69200ceee90`, em Modo Teste, traz os seguintes fatos no debug:

- A CNN FALTANDO V2 original, checkpoint SHA `6e4a31e8...`, votou OK em SIDE/TOP/MID, com scores NG não calibrados de 0,001158 / 0,001083 / 0,0000154. Isso é **falso OK local** frente à avaliação humana visual do usuário. Não interpretar scores como certeza.
- A fusão final corretamente produziu `REVISÃO OBRIGATÓRIA` pela contraprova física indisponível: nenhum comando XP foi autorizado.
- O log de extração SIDE traz `sample_crop = 477×570` e `test_crop = 477×567`; `physical_absence_probe` exigia imagens exatamente da mesma forma. Isso é **uma causa compatível provável**, não prova da causa específica em TOP/MID — o debug não inclui `physical_missing_veto.lights`/geometria por luz.
- A memória KNN recuperou **zero pares pixel-idênticos** de 1.604 registros vistos no índice, apesar de 436 registros duráveis. `NEW_CNN` significa `NOT_FOUND` na busca de identidade exata; **não afirma** que não existem imagens semelhantes no arquivo.
- Reconstrução do índice legado foi 5.970,75 ms SIDE, 2.516,58 ms TOP, 2.578,53 ms MID. O motivo após SIDE é a invalidação `orchestrator_reload_memory`. Não desligar a invalidação sem demonstrar que metadados legados e rótulos continuam íntegros.
- Arquivos de `public/ng_archive` são automaticamente importados para memória KNN (se imagem/categoria/rótulo estiverem íntegros). O botão `TREINAMENTO • VALIDAÇÃO DO ACERVO` **somente reproduz** a CNN sobre os PNGs; não altera o checkpoint. Treino CNN incremental exige caso humano `NEW_CNN` persistido em `reports/neural_online/events` e promoção após gates. O log atual não prova a existência do journal nem promoção desse evento.

## Correção da contraprova, sem alterar o veredito por similaridade

`physical_absence_probe` agora mantém a mesma escala e faz **somente aparagem central** quando ambos os recortes BGR uint8 têm pelo menos 32 px em cada dimensão e divergências não ultrapassam **6 px e 2%** em largura/altura. Não interpola pixels ou modifica arquivos históricos. Ambos os recortes da comparação física são aparados para a interseção geométrica central; `geometry.method=center_trim_no_rescale` registra dimensões antes e depois.

O motor `ROIPatchExpectationExpert` executa duas análises ROI no par aparado para registrar métricas físicas. Se qualquer escala suspeitar de ausência, status `SUSPECT_ABSENCE`; se não suspeitar **mas houve aparagem**, status `GEOMETRY_REVIEW` (não `NO_STRONG_EVIDENCE`). Geometria fora dos limites, PNG inválido ou erro continuam `UNAVAILABLE`. Os status `GEOMETRY_REVIEW` e `UNAVAILABLE` exigem confirmação humana, sem liberar AUTO-OK ou inferir NG. O gate de produção reaplica a trava caso veredito seja sobrescrito.

Os limites de ROI físicos da etapa 3 **não foram alterados**, e novas métricas não certificam presença: aparagem central não substitui validação de alinhamento físico nem holdout industrial.

## Nova auditoria local de vizinhos de arquivo KNN + aprendizado CNN

Com o ODIN fechado, na mesma instalação que possui as memórias e **após confirmar NG pelo botão 1 no ODIN** (para criar evento humano), execute:

```powershell
cd C:\visionx-neural-main
python -X utf8 -m src.services.startup_regression.faltando_archive_neighbors_audit --aoi-event-id 239dc35fdce44b8190b6b69200ceee90 --top-k 5
```

O comando confere o evento humano por `source_aoi_event_id`, valida SHA-256 dos PNGs gabarito/teste da iluminação correta, carrega a memória KNN local e compara assinaturas de divergência, separando os cinco vizinhos NG e OK mais próximos por luz. Também informa `learning_job_status`, `checkpoint_promoted_by_job` e `active_checkpoint_points_to_this_job`.

Salva somente o JSON em `reports/startup_regression/cnn_knn_ng_neighbors_*.json`. `JOURNAL_FOUND` significa que o rótulo humano está no journal, **não** que a CNN o aprendeu. `NO_JOURNAL` significa que o evento não foi localizado nos dados do PC examinado (não preencher com outros eventos pelo nome da peça). A KNN apenas fornece **diagnóstico não decisório**, sem converter `NEW_CNN` em `KNOWN_KNN` e sem alterar pesos, rótulos ou decisões XP.

Para testar a contraprova nos pares persistidos, utilize também:

```powershell
python -X utf8 -m src.services.startup_regression.cnn_physical_veto_replay --aoi-event-id 239dc35fdce44b8190b6b69200ceee90
```

O replay agora reporta também `geometry_review_lights`.

**Próximos passos condicionados a resultados locais:** avaliar distribuição de vizinhos NG versus OK por iluminação, confirmar que os exemplos no `ng_archive` foram importados e têm assinatura válida, verificar o status e gates de treinamento do evento, e **só então** considerar candidato de novos pesos com replay de todos os NG/OK e holdout separado por evento/peça. Sem esse relatório, não é seguro promover CNN apenas porque o arquivo contém exemplos parecidos. Não declarar melhora da acurácia neural a partir dos testes sintéticos deste PR.

## Segurança

Esta mudança não altera pesos CNN, limiares CNN, KNN exata, promoção de checkpoints, desenho de telas, modelos físicos, contrato do Windows XP nem auto-NG. O único novo diagnóstico decisório é a obrigatoriedade de revisão quando a aparagem central foi necessária. A arquitetura permanece modular e usa os pares do mesmo ciclo/luz.
