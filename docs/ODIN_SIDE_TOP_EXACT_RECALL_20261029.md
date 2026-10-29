# ODIN — SIDE/TOP e par RGB exato (29/10/2026)

## Ocorrência real

Evento `a062c9850a144976a62591e5bfa726a4`, 29/10 às 16:00.
Operador informou **falha falsa** de categoria AOI FALTANDO e identificou
`public/ok_archive/2026-10-08_0725_FALTANDO_TOP.png` como referência
histórica OK. A estação informou `lighting_mode=SIDE`, embora a imagem
apresentada na AOI parecesse ser da iluminação TOP.

A memória estrita devolveu `NEW_CNN/NOT_FOUND` (0 pares). A CNN experimental
`c13ca329...` emitiu score NG **não calibrado** de `0.9999750853` em
SIDE. A política de Produção antecipou o veredito SIDE, dispensou TOP/MID
e confirmou **comando XP 1/NG**. Isso não prova ausência física.

O banco KNN C estava READY. Mesmo com voto agregado A/B NG, o vizinho
mais próximo em ambos os descritores era OK:
- A: OK 0.91192131, NG seguinte 0.86743278.
- B: OK 0.86549428, NG seguinte 0.80695629.

O arquivo TOP informado pelo operador **não apareceu** nos cinco vizinhos
SIDE do log. Os PNGs locais originais (captura atual e histórico TOP)
não estão versionados no GitHub; portanto **a igualdade de seus pixels
não foi comprovada** neste desenvolvimento. Similaridade e par exato são
conceitos diferentes.

## Contratos alterados

1. `PersistentArchiveKNN.lookup_exact_any_light` consulta as três
   iluminações **somente para categoria idêntica e hashes SHA-256
   dos dois recortes RGB idênticos**. Contradição OK/NG entre quaisquer
   luzes retorna CONFLICT, nunca autoriza 0/OK. O lookup legado por luz
   permanece inalterado.
2. `VerifiedKNNMemory` usa esse fallback se a consulta da luz atual
   não recupera o par, exige índice KNN legado integral e auditado,
   confere contradições humanas nas três luzes e anota luz da fonte,
   caminho e o fato de o par ser exato entre iluminações. Cache
   indisponível ou inconsistente bloqueia o atalho.
3. Quando a SIDE CNN FALTANDO julga NG **sem evidência física de
   ausência** e os vizinhos mais próximos A e B são OK com vantagem
   de pelo menos 0.04 sobre NG (A>=0.90, B>=0.85), o encerramento
   antecipado SIDE é adiado para adquirir TOP/MID. **Não corrige
   a classificação para OK e não autoriza 0/XP por semelhança.**
4. Com prova RGB exata TOP confirmada no índice, a memória reconhece
   o caso como KNOWN_KNN/EXACT_PAIR, com sinalização da origem TOP.
   Quando os pixels variam, a rota continua não exata e o julgamento
   deverá usar as demais verificações sem liberação artificial.

Não modifica pesos da CNN, nem conteúdo dos arquivos OK/NG, nem
protocolo XP 0/1. Os testes usam pares RGB sintéticos e os números do
log como regressão; não equivalem a validação na estação industrial.

## Validação operacional necessária

- Com ODIN fechado, atualizar a branch aprovada, preservar backup
  da memória `data/knn_archive_memory/memory.json` e executar os
  testes de memória/SIDE antes de conectar XP.
- Conferir na estação o arquivo TOP original, o par gabarito/teste
  extraído do evento SIDE e os respectivos hashes RGB de ambos.
- Observar se o diagnóstico reporta `EXACT_PAIR` com
  `recognition_memory_cross_light_exact=True`; se não, a diferença
  é visual/aproximada e **não pode ser rotulada como exata**.
- Em replay sem XP, testar peças OK e NG de geometria parecida, o
  tempo de TOP/MID, a política final e se a janela de pausa protege
  a apresentação. Não alegar que o evento específico foi corrigido
  na estação antes de reproduzi-lo com os PNGs reais.
