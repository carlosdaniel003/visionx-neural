# Benchmark controlado do Tesseract AOI — PR #19

## Propósito e segurança

O ciclo FALTANDO de 14/10/2026 foi de **5,44 s**; a SIDE gastou **1.269 ms** no OCR, dos quais **1.251 ms** foram UMA única chamada Tesseract `general_blue`. Portanto, remover uma releitura redundante não é opção neste caso.

O benchmark é um comando OFFLINE, separado de `main.py`: não abre receptor Windows XP, não muda pesos, não treina, não atualiza memória, não altera a detecção/categoria, não modifica o fluxo de produção e **nunca ativa variantes**. A cada captura chama primeiro o `ScreenMonitor._extract_text_info` real, com pré-processamento OCR operacional intacto, e depois alternativas de escala e corte de região apenas no subprocesso OCR principal (o reconhecimento dirigido de `Parts` continua idêntico). `normalize_aoi_text` é o mesmo parser de categoria do painel.

Variantes: `baseline_original` (escala dinâmica original, mínimo 3), escalas **2, 3, 4 e 5**, escala 3 com **60 pixels removidos do topo** da região de texto e escala 3 com **20 pixels de cada lado** removidos. Cortes usam pixels da região original, antes do redimensionamento. `--repeats 3` executa 3 passes por imagem, com candidatos em ordem rotativa para reduzir viés temporal; tempo é duração do OCR completo, incluindo parsing de `Parts`. Não são tempos garantidos em Produção.

## Rodar com imagens reais da AOI (original integral)

Usar screenshot **inteiro** do XP, com as barras azul/vermelha, e **não** o recorte `*_source.png` do dataset CNN; a extração de texto depende da posição das barras. Primeiro parar a inspeção no ODIN e escolher período ocioso para não competir pela CPU com Produção.

No PowerShell, dentro do repositório:

```powershell
git checkout central
git pull origin central
python -X utf8 -m src.services.aoi_ocr_benchmark --images "public/debug_ocr/frame_completo.png" --repeats 3 --output "reports/startup_regression/ocr_benchmark/ensaio_01.json"
```

Esse exemplo só vale se o arquivo local de imagem integral existir. É possível fornecer também um diretório de capturas AOI PNG/JPG/BMP (`--images "C:\\AOI\\capturas"`). As imagens jamais são modificadas. O programa falha se não encontrar barras, faltar Tesseract, imagem inválida, controle com campo vazio/desconhecido ou se o relatório já existir.

**A equivalência com o controle NÃO comprova correção:** o controle pode ler `RI75` em vez de `R175`, por exemplo. É preciso comparação com referência humana conferida, não apenas OCR anterior.

## Manifesto com identificação humana (recomendado)

Criar `C:\\AOI\\auditoria\\manifesto.json` contendo caminhos relativos **à pasta desse manifesto**, por exemplo:

```json
[
  {
    "path": "capture_001.png",
    "expected": {
      "board": "P22-22200 (PRINCIPAL) L13",
      "parts": "R175",
      "value": "12 <= 0 <= 60 FALTANDO",
      "category": "FALTANDO"
    }
  }
]
```

**Os valores acima são apenas um EXEMPLO, não rótulo confirmado do evento RI75.** O operador deve inspecionar os pixels da AOI e conferir manualmente todos os quatro campos, incluindo dígitos/intervalos e unidade física. A pasta `OK/NG` e o resultado original do OCR **não** são tratados como verdade humana.

Executar:

```powershell
python -X utf8 -m src.services.aoi_ocr_benchmark --manifest "C:\\AOI\\auditoria\\manifesto.json" --repeats 3 --output "reports/startup_regression/ocr_benchmark/ensaio_rotulado_01.json"
```

No JSON, `summary` informa mediana do OCR por variante, diferença percentual relativa ao controle, número de capturas em que algum campo divergiu do controle e divergências da referência fornecida; cada entrada contém SHA-256 da imagem e medições individuais. Não armazena nem multiplica a imagem ou o `raw_text` completo. Os relatórios ficam em `reports/startup_regression/` e são ignorados pelo Git.

Mesmo com **100% de igualdade**, `eligible_for_automatic_activation=false`: uso industrial exige auditoria de amostras variadas e confirmação humana explícita em PR separado. O benchmark não pode ser usado para elevar o status de qualquer checkpoint CNN, nem para aprovar 0/1 automáticos.

## Interpretação

- **Escala 2 mais rápida com campo divergente:** reprovar, sem considerar o ganho de tempo.
- **Igualdade com controle mas ausência de referência humana:** serve apenas para triagem, não para liberar alteração de produção.
- **Controle divergente de referência humana:** existe problema prévio de OCR. A variante não deve ser ativada pelo simples fato de concordar com ele.
- **Divergência em `Parts`, `Board`, `Value` ou categoria:** tratar separadamente e conferir screenshot original com operador.
- **Imagens de telas diferentes:** reter várias placas, posições, categorias e caracteres ambíguos no manifesto; uma peça repetida não comprova generalização.
