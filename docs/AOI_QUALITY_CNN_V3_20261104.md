## 04/11/2026 — Correção de manifesto renovado (1.112 capturas)

O inventário real da estação foi atualizado:
- 1.112 capturas, 784 com rótulo de pasta NG e 328 com rótulo OK;
- 679 SIDE, 216 TOP, 217 MID;
- 0 falhas de extração;
- 212 conjuntos de trincas com vínculo apenas pelo nome, NÃO validadas como evento físico;
- nenhuma captura automaticamente aprovada para treino físico.

O treinamento inicial parou em
`verified_validation_labels -> Revisão humana não corresponde ao manifesto original`.
Motivo: o documento histórico de 26/10 contém um SHA-256 do **manifesto
inteiro antigo**, não da versão de 04/11. As nove fontes do documento
permanecem presentes no inventário enviado, todas com SHA-256 de PNG
original idêntico. A presença/identidade dos recortes derivados
ainda depende da execução real na estação.

**Solução limitada à CNN V3 offline:** `verified_validation_labels`
tem parâmetro explícito `allow_refreshed_manifest=False`.
O modo histórico permanece estrito; a V3 permite revalidar cada um
dos 9 julgamentos pelo SHA-256 do PNG original, do gabarito extraído
e do teste extraído, além de conferir esquema, categoria, veredito e
presença das nove fontes. Não usa igualdade de nome ou um rótulo
provisório para presumir que uma imagem permaneceu igual. Em caso
de qualquer divergência, aborta. O importador relata o reuso
com `operator_validation_evidence_reused_on_refreshed_manifest`.
Não modifica documentos, recortes, acervo, pesos ou XP.

**Retomar:** após `git pull origin central`, NÃO precisa gerar um
manifesto novo. O caminho abaixo aponta para o manifesto de 04/11
que já foi extraído com sucesso:

```powershell
python -X utf8 -m src.scripts.train_aoi_quality_cnn_v3_offline --root . --manifest "C:\visionx-neural-main\reports\faltando_neural\run_20261104T163621_087850Z\manifest.json" --epochs 6 --allow-provisional-archive-labels --allow-verified-trial-pointer-readonly
```

Se aparecer uma divergência de SHA de gabarito/teste, não
substituir o documento histórico nem desativar a validação:
revisar a nova extração versus o par originalmente confirmado.

O arquivo `latest_event.json` enviado pelo operador indica
`state=QUEUED` e `promoted=null`, o que NÃO comprova
que ocorreu treinamento ou ativação de uma CNN. Esse status
pertence à fila online, separada do treinamento V3 offline.

---

# 04/11/2026 — CNN AOI V3 por pares: recuperação de generalização, sem memória

## Origem: evento EMBORCADO 12:11, registrado em Teste

Evento `c93e28847255450bb552b0cc53373440`. O operador mostrou
um componente fisicamente presente nos recortes AOI de SIDE/TOP/MID
e reportou qualidade **OK / FALHA FALSA**. A CNN V2 original de ausência
(classificador `faltando_cnn_v2`) emitiu scores NG **não calibrados**:

| Luz | Score CNN NG | Score físico | CNN conclusiva |
|---|---:|---:|---|
| SIDE | 0,9991123 | 0 | DEFEITO REAL |
| TOP | 0,9664062 | 0 | DEFEITO REAL |
| MID | 0,9974234 | 0 | DEFEITO REAL |

A fusão `multilight_strong_single` herdou os scores máximos e classificou
DEFEITO REAL, apesar de não existir NG físico corroborado. O checkpoint
carregado foi `c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c`,
melhor época 18. Não há dados para afirmar que os scores sejam
probabilidades reais de defeito. Não confundir `par RGB exato`/KNN
com capacidade de generalização de CNN: a exigência de identidade
de pixels pertence a outra fronteira e não foi alterada.

**Causa estrutural provável (não prova exclusiva):** o experimento
CNN FALTANDO V2 original teve poucos exemplos físicos NG (e nenhum
NG TOP/MID naquele conjunto), mas os mesmos pesos são roteados para
EMBORCADO, INVERTIDO, DESLOCADO e FALTANDO. Os relatórios de 2026
registram 13 OK→NG em 43 OK do benchmark histórico, 19 falsos NG
em 117 vistas OK de outra regressão e NG→OK de experimentos universais.
Não há evidência suficiente para promover um fine-tuning por acerto
aparente no acervo já visto.

## Mudança: motor experimental AOIQualityCNNV3

Arquivos:
- `src/core/neural/aoi_quality_cnn_v3.py`: nova arquitetura de comparação
  de pares, separada da V2; encoder **compartilhado** para referência e
  teste, combinação simétrica de distância absoluta e correlação de
  features espaciais, em duas escalas (quadro e zoom central).
  Mantém máscara explícita SIDE/TOP/MID e um logit por luz.
  Nenhuma consulta KNN/memória na inferência.
- `src/scripts/train_aoi_quality_cnn_v3_offline.py`: treino efetivo
  AdamW CPU com backpropagation e augmentation **somente no treino**,
  com ganho/brilho diferentes para gabarito e teste e a mesma rotação
  para os dois lados do par. Adiciona regularização fraca de par
  `referência == referência` para penalizar falso defeito em
  imagens idênticas. Não deriva rótulos de similaridade.
- `tests/test_aoi_quality_cnn_v3_offline.py`: prova de simetria da
  arquitetura, máscaras, gradientes reais, geração de checkpoint
  isolado, partição por grupos sem associação física presumida,
  imutabilidade do acervo e do ponteiro ativo.
- `.github/workflows/aoi-quality-cnn-v3-offline-tests.yml`: pipeline
  de teste da arquitetura e do treino em CPU.

A V3 usa o **par extraído** `gabarito/teste` com o mesmo extrator AOI
do runtime. Os PNGs da pasta `public/ok_archive`/`ng_archive`
contêm capturas AOI; não são introduzidos como exemplos brutos
de uma classe sem extrair o par. A categoria AOI é metadado
de avaliação e balanceamento de amostras, NÃO entrada da CNN
nem sinônimo do defeito físico.

O split em treino/validação/teste usa cercas temporais e de quase
duplicados, evitando avaliar no teste a mesma captura treinada.
A seleção de época usa apenas validação, **o teste é lido após
escolher os pesos**. Relatório apresenta erros OK→NG/NG→OK,
por categoria e iluminação, proveniência de rótulos (humanos
versus pasta provisória), cobertura, e comparação contra o
checkpoint V2 original verificado por SHA e avaliado com
seus próprios `image_size`/`focus_fraction`.

## Execução SOMENTE offline na estação com o ODIN/XP encerrados

No PowerShell, com ambiente Python de treino ativado:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central

# Recriar staging para incluir TODO o acervo atual e suas três luzes.
python -X utf8 -m src.services.faltando_neural_dataset --root . --all-aoi-categories
```

O comando acima imprime o caminho absoluto do manifesto novo,
por exemplo `reports/faltando_neural/run_<execucao>/manifest.json`.
Use **o caminho real** impresso, sem reaproveitar automaticamente
o manifesto antigo de outubro. A preparação deve terminar sem
recortes inutilizáveis.

```powershell
# Substituir somente o caminho do manifesto pelo que apareceu no console.
python -X utf8 -m src.scripts.train_aoi_quality_cnn_v3_offline --root . --manifest "reports/faltando_neural/run_<execucao>/manifest.json" --epochs 6 --allow-provisional-archive-labels --allow-verified-trial-pointer-readonly
```

A opção `--allow-verified-trial-pointer-readonly` **somente** permite
ler um ponteiro de ensaio já validado: não o modifica e recusa
ponteiro arbitrário. `--allow-provisional-archive-labels` é
obrigatória porque a maioria dos rótulos nas pastas não recebeu
conferência física individual. Não implica autorização para instalar
o modelo. A comparação com V2 original é feita por padrão; se
o checkpoint V2 pinado estiver ausente ou adulterado, a execução
falha, em vez de inventar desempenho do modelo anterior.

Saída isolada:
`reports/aoi_quality_cnn_v3/candidates/<execucao>/training_report.json`
e `aoi_quality_cnn_v3_candidate.pt`, com SHA-256.

## Gates de segurança, limites e próxima decisão

**Não instalamos nada em Produção nesta etapa.** A CNN V2 ativa
continua intacta; não há escrita no `live_active.json`, ajustes
na fusão/UI, arquivos de memória KNN/JSON, arquivos PNG ou comandos
XP. O schema do novo candidato é incompatível com o carregador V2.

Um candidato só pode ser discutido para Produção após:
1. Demonstrar que reduz OK→NG **sem aumentar NG→OK** em capturas
   de qualidade física confirmada e eventos de peças independentes.
2. Cobrir OK e NG por categoria AOI e iluminação, com atenção aos
   defeitos de orientação (EMBORCADO/INVERTIDO) e ausência real.
3. Excluir rótulos incorretos das pastas e vazamento por imagens
   semelhantes da validação/teste; não confiar em métricas do
   histórico já usado para otimização.

Sem isso, a V3 é um ensaio de treinamento, não uma CNN consertada
no chão de fábrica. Enviar o `training_report.json` para a
comparação técnica antes de qualquer decisão de ativação.
