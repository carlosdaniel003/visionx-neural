# Correção 28/10/2026 — remover falso NG do acervo

**Operador confirmou:** `2026-10-21_1414_FALTANDO.png`, armazenado
indevidamente em `public/ng_archive`, é falha falsa e será removido
localmente. A fonte de verdade para esse evento é
`public/ok_archive/2026-10-21_1416_FALTANDO.png`, já confirmada OK.

## Comportamento corrigido

- A KNN C **não indexa** a imagem NG retirada, ainda que apareça em
  inventário/cache anterior. O warmup reconstrói assinaturas a partir
  dos PNGs atuais; apagar a imagem não remove seu histórico de auditoria
  em `docs/odin_operator_confirmed_quality_20261026.json`.
- A regra antiga de aceitação por referência NG corrigida é substituída
  por referência **OK real**. A origem OK é checada por SHA-256 de bytes,
  pela decisão humana preexistente e pela ausência da fonte NG errada.
- SIDE: os dois descritores KNN C A/B devem recuperar como vizinho
  principal a imagem OK 14:16 (pixel SHA verificado), com similaridade
  mínima 0.85/0.82 e sem concorrente NG forte. TOP/MID precisam ter
  maioria 4/5 OK nos dois descritores e contraprova favorável a OK
  nas assinaturas independentes.
- A CNN deve estar sem consenso válido e as três imagens completas,
  sem prova física de ausência. A regra não afirma que a captura atual
  tenha um par de pixels exatamente idêntico.
- Se toda essa evidência for verificada no **Modo Produção contínuo**,
  a política existente pode transmitir 0/OK automaticamente após
  janela de 5 s para pausa; caso contrário, há rejeição preventiva
  ou parada por erro de aquisição. NÃO afrouxar para outras peças.
- Os pipelines offline não exigem mais que o PNG descartado continue
  presente quando o documento histórico e a fonte OK confirmada
  são verificáveis. Outras fontes ausentes ou adulteradas continuam
  erro; o manifesto original conserva o registro de auditoria.

## Regressão

`tests/test_operator_corrected_archive_live_ok.py` cobre exclusão do
PNG indevido, não reaproveitamento do cache KNN, referência OK validada
por SHA, reprodução multilight, autorização apenas em Produção,
falha por adulteração / evidência NG física, e rejeição na ausência
de origem OK confiável.

**Operação:** fechar ODIN, excluir apenas
`public/ng_archive/2026-10-21_1414_FALTANDO.png`, executar
`git pull origin central`, reiniciar e confirmar KNN C READY.
Não apagar a fonte 14:16 do OK.
