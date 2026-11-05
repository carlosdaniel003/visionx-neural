# ODIN AOI — Fine-tuning offline de 25 NG físicos confirmados (05/11/2026)

## Evidência e problema

A auditoria direta de 1.111 PNGs pela época 3 encontrou
1.083 concordâncias com pasta, 2 OK→NG e 26 NG→OK.
O operador examinou os 26 NG→OK: confirmou 25 NG reais e informou
que 1 arquivo havia sido excluído. O caso excluído **não** pode
ser usado no treino. A confirmação humana substitui o rótulo
provisório **somente das 25 fontes conferidas**.

Quatro PNGs originais foram anexados pelo operador e tiveram SHA
verificado contra o relatório de auditoria:

- `2026-10-30_1359_FALTANDO.png`: gota de adesivo escuro sobre
  o pad direito, score NG V3 original de 0,000006.
- `2026-10-14_0859_INVERTIDO_SIDE.png`: marca de polaridade do
  diodo na posição oposta, score NG de 0,000033.
- `2026-10-14_0859_INVERTIDO_TOP.png`: polaridade invertida também
  na TOP de menor contraste, score NG de 0,000972.
- `2026-10-21_1446_UNKNOWN.png`: adesivo escuro no pad direito,
  score NG de 0,420825. A AOI o chamou de TERMINAL ALTO, mas
  **categoria não é atributo-alvo da CNN**.

Os outros 21 foram confirmados NG pelo operador por nome/arquivo,
sem subtipo físico especificado. Não inventar diagnóstico por nome.

O rótulo NG é da qualidade física; não provém de classificador KNN
nem de decisão automática. SIDE/TOP do mesmo minuto podem ser
um evento físico comum, mas esse vínculo ainda não está provado.

## Código adicionado

- `docs/aoi_v3_operator_ng_escape_review_20261105.json`:
  25 decisões NG com nomes e SHA exatos, sem o caso apagado.
- `src/services/aoi_v3_ng_escape_reviews.py`:
  importação rigorosa com hash do PNG, conferência contra o
  staging AOI e dois recortes, sem regras por categoria/nome.
- `src/services/aoi_universal_quality_archive.py`: opt-in
  de qualidade humana, com conflitos detectados. Duas fontes
  SIDE/TOP têm o mesmo par RGB exato: retém as **25 confirmações**
  sobre **24 pares únicos** sem duplicar o gradiente.
- `src/scripts/train_aoi_quality_cnn_v3_offline.py`:
  dois parâmetros novos: `--initialize-from-verified-epoch3`
  (lê apenas o checkpoint experimental SHA-verificado) e
  `--use-operator-ng-escape-reviews` (adiciona as 25 revisões).
  Exige AMBOS para estudar os NG recentes.
- O treinamento continua usando exclusivamente o par visual
  gabarito/teste em duas escalas e iluminação; rótulo AOI não é
  entrada do classificador nem justificativa para um veredito.
- Os 14 julgamentos antigos permanecem, sem reclassificação.
  Ocorre uma sobreposição entre os 14 antigos e os 25 novos,
  conservando uma única amostra por fonte e evitando vazamento.

### Gate de qualidade

1. As 25 fontes verificadas ficam obrigatoriamente no treino e fora
   de validação/teste, junto dos 14 exemplos humanos anteriores.
2. O modelo usa `AOIQualityCNNV3` compatível com a época 3,
   inicializada por `validate_trial_snapshot` (SHA e metadados),
   sem alterar peso/pointer ativo.
3. A validação separa erros OK→NG e NG→OK.
4. Se uma época piora **qualquer uma das duas taxas absolutas de erros**
   contra a inicial, ela não pode substituir o modelo selecionado.
5. Os resultados dos 25 NG no próprio treino são apresentados
   por época, como `in-sample`, jamais como teste independente.
6. O teste é lido após a seleção de época e os checkpoints de
   cada época continuam experimentais, incompatíveis com
   ativação automática na Produção.
7. Sem ganho não regressivo, `selected_epoch=0` mantém os pesos
   de referência; isso é diagnóstico de que os novos exemplos
   não foram aprendidos com segurança nesta configuração.

Esta mudança **não altera arquitetura da V3 ativa**. Registra os
erros físicos e habilita fine-tuning seguro para avaliar se os
pesos existentes podem aprender os pequenos defeitos antes de
introduzir outra arquitetura. Não altera filtro óptico, iluminação,
regras XP, KNN, acervo ou checkpoint ativo.

## Executar SOMENTE offline (XP desconectado e ODIN fechado)

A captura excluída faz o staging anterior ficar desatualizado.
Gerar novo staging com todas as categorias, sem reutilizar manifesto
que contenha o PNG excluído:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

python -X utf8 -m src.services.faltando_neural_dataset --root . --all-aoi-categories
```

Copiar o caminho do `manifest.json` que o comando imprimir, e
executar (substitua apenas `<RUN_GERADO>` pelo diretório real):

```powershell
python -X utf8 -m src.scripts.train_aoi_quality_cnn_v3_offline --root . --manifest "reports\faltando_neural\<RUN_GERADO>\manifest.json" --epochs 6 --allow-provisional-archive-labels --allow-verified-trial-pointer-readonly --initialize-from-verified-epoch3 --use-operator-ng-escape-reviews
```

O console apresenta `NG_revistos=X/24 pares únicos (25 fontes)`,
`val_OK_falsos_NG`, `val_NG_falsos_OK`, `gate` e `melhor_epoca`.

Relatório:
`reports/aoi_quality_cnn_v3/candidates/<nova_execucao>/training_report.json`.

Avaliar:
- `operator_ng_escape_review_verified == 25`;
- `operator_verified_ng_escape_training_replay.NG_total == 24`;
- `training.selected_epoch_using_validation_only`;
- `validation.NG_false_OK`, `validation.OK_false_NG`;
- `previous_v3_ng_escape_training_replay` versus
  `operator_verified_ng_escape_training_replay` (ambos **treino**);
- `verified_previous_v3_baseline` no mesmo fold histórico;
- `safety.production_eligible == false`.

Se houver divergência de SHA, ausência de imagem ou manifesto,
**não** remover as proteções nem substituir as confirmações por
rótulos de pasta: investigar a fonte exata.

## Próxima decisão

Apenas após observar o candidato offline poderemos decidir se
é necessário um novo comparador espacial com maior resolução
ou supervisão explícita de região/polaridade. Não inserir
heurísticas por nome de imagem/categoria, limiares de scores
de teste ou liberação automática 0/XP.
