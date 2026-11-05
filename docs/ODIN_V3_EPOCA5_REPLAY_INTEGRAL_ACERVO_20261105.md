# ODIN VisionX AOI — Repetir auditoria integral com CNN V3 época 5, sem MSS

**Data:** 05/11/2026.

## Objetivo da repetição

A avaliação anterior com V3 época 3 analisou 1.111 arquivos PNG originais
em `ok_archive` e `ng_archive`, obtendo 1.083 acordos contra as pastas,
2 OK→NG e 26 NG→OK. Depois, o operador confirmou 25 desses NG como
defeitos físicos, excluiu uma captura e escolheu a **época 5** da V3.

A época 5 acertou 24/24 pares NG humanos **no treino**, mas gerou
2 OK→NG na validação. Precisamos repetir os 1.110+ (contagem dinâmica
do inventário) PNGs originais para saber quantos erros persistem.
**Não atribuir resultados da época 3 à época 5.**

### As três verificações são diferentes

1. **CNN época 5:** gabarito/teste extraídos de cada PNG original
   pelo mesmo extrator do ODIN; classifica independentemente como OK/NG
   por SIDE/TOP/MID; mede OK→NG e NG→OK sem ler rótulo antes da decisão.
2. **Índice operacional XP:** envia o próprio PNG original à rotina
   `ArchiveVisualPresenceIndex.inspect(..., exact_only=True)`.
   Testa a mesma estrutura SHA de tela inteira do XP, separa
   `EXACT_FRAME` de `EXACT_FRAME_CONFLICT` e
   `NO_EXACT_FRAME`. Este modo eficiente pula APENAS a busca
   por semelhantes; não modifica o caminho normal do XP.
   Um acerto por autoidentidade da própria captura **não é
   generalização nem validação de rótulo físico**.
3. **Semelhança leave-one-out:** procura outros PNGs (exclui a
   própria imagem), restritos à categoria e luz AOI do arquivo
   e usando `archive_visual_descriptor` e
   `near_duplicate_visual`. É uma sonda aproximada com 12
   candidatos e não prova ausência de vizinho quando falha.
   Rótulos conflitantes impedem concluir uma classe.

Não misturar os três resultados. O arquivo de `MUITO ADESIVO` é
pontuado pela V3 **comparativamente**, mas a Produção conserva
especialista específico. O escopo é apenas AOI/VisionX, não F2/F3.

## Execução na estação local

Fechar ODIN, deixar XP desconectado do envio automático.

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

# Se a época 5 ainda não foi ativada localmente:
python -X utf8 -m src.scripts.activate_aoi_quality_v3_epoch5 --root . --activate

# Repetir a auditoria usando obrigatoriamente o modelo da época 5:
python -X utf8 -m src.scripts.audit_v3_full_archive --root . --require-epoch5 --verify-runtime-archive
```

O `--require-epoch5` bloqueia a execução se ponteiro, SHA do
relatório, SHA dos pesos ou metadados da época 5 estiverem ausentes.
Também confere época e SHA de cada inferência, impedindo aceitar
silenciosamente a época 3. Um acervo incompleto/PNG inválido é
registrado; a CLI retorna 2 se o inventário não puder ser avaliado
integralmente. As fontes do acervo e pesos nunca são escritos.

**Saída:** `reports/aoi_v3_full_archive_audit/run_<timestamp>/`

- `full_archive_v3.txt`: matriz de confusão, SHA/época reais,
  resumo do replay exato e lista de todas as divergências.
- `full_archive_v3.csv`: um registro por PNG, incluindo época,
  score não calibrado, recortes, recall exato e candidatos próximos.
- `full_archive_v3.json`: dados estruturados, provas e agrupamentos.

Para repetir o resultado de época 3, **não** remover a flag
`--require-epoch5` no teste atual: a comparação precisa da mesma
identidade de modelo e dados. Se a época 5 foi desativada, reative
somente com o comando indicado, após verificar arquivos locais.

## Critérios de interpretação

- `classifier.v3_checkpoint_epoch == [5]` e
  `classifier.v3_checkpoint_sha256 == [eff1d3bf...]`.
- `summary.OK_false_NG` e `summary.NG_false_OK`:
  ambos precisam ser observados; 100% contra pastas não é exigência
  que possa ser implementada como hardcode.
- `runtime_archive_exact_replay.scanned`: todos os PNGs originais
  consultados pelo índice SHA XP; `exact_unambiguous` deve excluir
  conflitos e recuperação não exata.
- `leave_one_out_archive_probe.same_label_near` informa apenas
  outros PNGs visualmente próximos encontrados pela sonda, não
  reconhecimento do mesmo evento físico reencaminhado.
- `coverage.source_changed_during_run` precisa permanecer vazio.
- `all_files_evaluated == true` significa apenas cobertura técnica
  sem falhas, não que todos os acertos CNN sejam verdade física.

## Próxima etapa (depende dos resultados)

Quando o relatório de época 5 estiver disponível, confrontar
todas as divergências e os conflitos contra os rótulos confirmados
pelo operador. Depois decidir separadamente como incorporar
**recuperação EXATA verificada** e **semelhança visual comprovada**
à política do XP, sem substituir a CNN por um rótulo de pasta
potencialmente incorreto, sem confundir um evento novo com o próprio
arquivo pesquisado e sem liberar 0/OK por correspondência ambígua.

Este teste **não** retreina, não altera KNN/memória/arquivos,
não aciona 0/1 e não muda a Produção.


## 05/11/2026 — Índice XP: nomes históricos HH-MM-SS-ms

O replay da época 5 encontrou **15 arquivos NG com CNN correta, mas
`NO_EXACT_FRAME` no índice operacional**. Todos tinham nomes no padrão
`YYYY-MM-DD_HH-MM-SS-ms_CATEGORIA.png`; o parser
`archive_category_from_filename` só reconhecia `HHMM`, por isso não
criava o escopo `CATEGORIA_SIDE` para esses arquivos legados.

A leitura de nomes agora admite ambos os formatos, sem renomear imagens
ou modificar pixels, hashes, categorias, rótulos, CNN, KNN, memória
humana, política XP ou envio 0/1. A iluminação ausente continua SIDE,
e os sufixos `_2`/`_3` e as luzes explícitas continuam suportados.

`tests/test_aoi_v3_full_archive_batch.py` contém regressão com os
15 nomes do relatório, teste `exact_only`, separação de categorias,
recusa de pixels alterados e replay integral sintético.

Após `git pull origin central`, repetir o comando de auditoria acima
na estação com ODIN fechado e XP desconectado. **Esperado, se os
1.110 arquivos originais estiverem intactos e sem novos conflitos:**
`runtime_archive_exact_replay.exact_unambiguous = 1110` e
`not_exact_or_ambiguous = 0`. Isto é uma expectativa a conferir,
não um resultado medido na estação. O reconhecimento da própria
fonte no índice continua sendo diagnóstico, sem provar acerto
em novas capturas nem autorizar decisão automática.
