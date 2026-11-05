# ODIN AOI — Auditoria integral do acervo OK/NG pela CNN V3 (05/11/2026)

## Problema investigado

Dois PNGs existentes no `ok_archive` foram exibidos em um visualizador e
fotografados pelo MSS, produzindo NG e `SEM PAR EXATO` no ODIN.
O primeiro, `2026-10-02_1029_FALTANDO.png`, quando lido diretamente
pelo extrator da estação, retornou **FALHA FALSA / OK**
(`NG_score=0.00222958`) no checkpoint V3 época 3; não foi erro
da rede naquele PNG original. O segundo,
`2026-10-02_1350_DESLOCADO.png`, teve MSS com recortes
600×345 (ambos `BOTH_UNBOUNDED`) e CNN V3 NG=0.743022.
Esse último ainda **não teve inferência direta do PNG original**.

Repetir screenshots isolados não mede a qualidade total e confunde
variação do visualizador/MSS com inferência sobre o arquivo original.

## Implementação

- `src/services/aoi_v3_archive_batch.py`: avalia TODOS os registros do
  `inventory_archives`, com caminho, SHA original, categoria, luz,
  recortes, score V3 e status individual. Não ignora falhas.
- `src/scripts/audit_v3_full_archive.py`: CLI e três relatórios
  íntegros por execução: JSON, CSV e TXT.
- O extrator `AOIPairExtractor` usa `ScreenMonitor.process_external_image`
  (mesmo algoritmo para PNG/XP), sem fotografar a tela e sem abrir UI.
- `AOIQualityV3Primary` é instanciada UMA vez e lê o checkpoint local
  V3 ativado e verificado. Não executa V2.
- A CNN vê exclusivamente os pares de pixels e luz, nunca o rótulo da pasta.
- São contabilizados `OK_false_NG` e `NG_false_OK`, com tabelas
  estratificadas por categoria, iluminação e combinação das duas.
- Uma sonda visual leave-one-out exclui o próprio arquivo da busca,
  utiliza o mesmo descritor e `near_duplicate_visual` do acervo e
  reporta candidatos e possíveis contradições. **Não é KNN humana e
  não substitui o veredito físico nem a CNN.**
- Rótulo de diretório é **provisório** até conferência física.
  Correspondência de pixels da própria imagem com o próprio índice é
  trivial; não equivale a generalização nem a teste cego.
- Em `MUITO ADESIVO` e demais categorias fora do roteamento
  principal da V3, o resultado é um **comparativo experimental da V3**,
  não o veredito do especialista efetivamente usado em produção.

## Como executar na estação (XP desconectado)

Com o ODIN fechado:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
python -X utf8 -m src.scripts.audit_v3_full_archive --root .
```

A CLI imprime progresso `CNN V3 ACERVO 1/N`, `2/N` ...
sem alterar arquivos operacionais. Os resultados ficam em:

```text
reports/aoi_v3_full_archive_audit/run_<data>/
  full_archive_v3.json
  full_archive_v3.csv
  full_archive_v3.txt
```

O TXT destaca a quantidade total, as divergências por classe, as
categorias/luzes e os caminhos dos PNGs discordantes. JSON e CSV
trazem **uma linha/objeto por arquivo** para análise posterior.

A execução termina com código de saída 2 quando houver PNG, extração
ou inferência não avaliada, preservando os relatórios gerados.
Isso não deve ser interpretado como 100% de concordância.

### O que avaliar no resultado

1. `coverage.png_in_inventory == summary.evaluated`, sem fontes alteradas
   ou erros. Caso contrário, a cobertura está incompleta.
2. `OK_false_NG`: imagens de `ok_archive` que a V3 classificou NG.
3. `NG_false_OK`: imagens de `ng_archive` que a V3 classificou OK.
4. `by_category_light`: identificar regressões concentradas.
5. `leave_one_out_archive_probe`: qualidade e conflitos da recuperação
   *sem deixar o arquivo se consultar a si mesmo*.
6. `full_frame_exact`: identifica cópias exatas e inconsistência OK/NG,
   **não é o par AOI humano verificado**.

## Limites e próximo estágio

Esta etapa apenas **mede e diagnostica**. Não treina, não troca pesos,
não registra novas decisões humanas, não modifica memória KNN e não
envia `0/1` ao XP. Não tenta forçar 100% de acerto via exceções.

A meta de produção inclui reconhecer uma nova captura exatamente igual
a um PNG do acervo e encontrar eventos muito semelhantes, mas o
julgamento da similaridade exige medir conflitos OK/NG e confiabilidade
dos rótulos. Se houver falsos NG e NG liberados na própria V3, esses
números não desaparecem porque o arquivo existe no acervo.

Depois de revisar o relatório real é possível planejar, **em etapa
separada**, um índice de memória no caminho XP com evidência de arquivo
exato/semelhante, precedência humana auditada, proteção contra rótulos
contraditórios e testes independentes antes de alterar o veredito
operacional. Não autorizar auto-OK da linha somente por pasta/score.
