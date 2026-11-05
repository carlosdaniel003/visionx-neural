# ODIN AOI — CNN V3 principal e reconhecimento MSS do acervo (05/11/2026)

## Decisão e limites

Por autorização explícita do operador, a CNN AOI V3 (época 3) passa a ser
**o classificador visual principal nas categorias que anteriormente usavam
a CNN FALTANDO V2**: FALTANDO, EMBORCADO, INVERTIDO e DESLOCADO.
Aplica-se aos modos Teste (inclusive MSS), Sombra e Produção.

A substituição é feita na construção do `MoEOrchestrator` em
`main.py`: o slot antes ocupado pelo `FaltandoCNNLive` recebe
`AOIQualityV3Primary`. Assim, os casos NOVOS usam V3 sem executar a
inferência V2. O roteador KNN de par humano exato continua antes do
classificador e preserva a última decisão física confirmada.
A CNN V3 experimental também continua atuando como contraprova NG nas
rotas de especialistas em categorias fora do escopo da antiga V2.
MUITO ADESIVO preserva o especialista físico próprio, que nunca foi
substituído pela V2.

Os scripts, checkpoints e testes V2 não são apagados: são históricos/
diagnósticos e não são usados como inferência normal do painel. A
auditoria histórica `Validar acervo OK/NG` ainda verifica a CNN V2
antiga e a KNN; NÃO mede o checkpoint V3 principal.

### Ativação e falha

O checkpoint permanece o snapshot época 3 SHA-verificado pelo
`AOIQualityV3ProductionTrial`, com o ponteiro local
`reports/aoi_quality_cnn_v3/production_v3_trial.json`.
A ativação já foi confirmada na estação pelo operador em 05/11.
O mesmo ponteiro, criado originalmente como `ng_veto_only`, continua
servindo de **autorização de carregamento do checkpoint experimental**
para a nova rota visual principal. A primazia está definida no código
do painel, não no nome histórico do ponteiro.

Se o ponteiro, modelo, SHA ou voto estiverem ausentes/adulterados,
a nova rota retorna `REVISÃO OBRIGATÓRIA` e status `ERROR`;
**não** faz fallback silencioso para a V2. A política binária
interrompe o envio se o contrato V3 ficar inválido.

### Segurança de decisões XP

**A V3 ainda não está homologada para liberar OK automaticamente.**
O checkpoint época 3 perdeu 1 NG real entre 14 exemplos humanos
revisados e 3 NG entre 219 exemplos na validação, sem holdout
independente por peça. Portanto:

- Um veredito visual V3 `FALHA FALSA` em Teste ou MSS é apresentado
  como inferência, não como confirmação física do rótulo.
- Em Produção, um **OK isolado da V3 não autoriza 0/OK**. A política
  exige contraprova humana verificada conforme os contratos existentes.
- Um NG V3 pode gerar `1/NG` no ciclo de produção completo, por
  política binária; isso é rejeição operacional de segurança,
  **não diagnóstico físico garantido**.
- A rota antiga de Produção não pode contornar esse bloqueio.
- A lógica de pausa, identidade do evento, envio único e ACK XP não foi
  alterada. Não conectar o XP antes de verificar numa estação isolada.

O objetivo continua ser 0/1 binário, mas o estágio atual **não
certifica produção autônoma 0/1 sem risco de liberar NG físico**.

## MSS e acervo OK/NG

O scanner `ArchiveVisualPresenceIndex` já usado pelo XP
é agora chamado após cada captura MSS válida, em worker próprio.
O frame MSS original completo é consultado nas luzes SIDE/TOP/MID,
pois uma imagem mostrada na tela pode ter metadados de luz
diferentes dos do arquivo. A busca mostra:

- `EXACT_FRAME`: SHA-256 dos **pixels da tela completa** idêntico
  ao PNG do acervo, mesma categoria/iluminação do índice;
- `SIMILAR`: semelhança de descritores do frame, **não igualdade**;
- `AMBIGUOUS` ou conflito de OK/NG: não inferir verdade física;
- `NO_SIMILAR`: nenhum candidato na busca atual.

**FRAME EXATO ≠ PAR AOI EXATO.** O `KNOWN_KNN/EXACT_PAIR` exige
igualdade RGB dos dois recortes gabarito/teste e evidência humana.
O `ok_archive` e o `ng_archive` podem conter rótulos de pasta
provisórios. Reconhecer uma imagem OK do acervo **não deve reescrever
o veredito V3 nem liberar 0/OK** sem conferência física.

A abertura de um PNG dentro de um visualizador pode acrescentar bordas,
mudar escala/resolução ou modificar pixels do frame que o MSS capturou:
não prometer `FRAME EXATO` nesse caso. Uma repetição real da mesma
tela, pixel-idêntica, é reconhecível. O debug MSS passa a registrar
`cnn_v3`, `archive_visual_presence`, rótulo, luzes e hash da tela,
sem ocultar a distinção entre quadro completo e par AOI.

## Aprendizado e regressões

A memória humana KNN mantém suas correções persistentes.
O treino incremental automático **V2** não recebe novos casos
quando V3 está ativa; também é bloqueado pelo marcador
`cnn_v3_primary`. Não há treinamento online V3 homologado
nesta alteração. O retreinamento de V3 continua offline, com
validação explícita antes da troca de pesos.

Os testes do CI exercitam o roteamento visual V3, erros de checkpoint,
ausência de fallback V2, painel V3, integridade do XP, consulta do mesmo
scanner em MSS, conflito e exatidão do frame. Testes de contrato não
substituem o replay das imagens reais e uma avaliação física cega.

## Atualização e teste supervisionado local

Fechar ODIN e manter o XP desconectado do envio automático:

```powershell
cd C:\visionx-neural-main
git switch central
git pull origin central
python -X utf8 main.py
```

Verificar no Modo Teste/MSS uma imagem OK e uma NG conhecidas, e
comparar o frame MSS salvo com o PNG original (mesmos pixels/escala).
Verificar o status V3, a rota KNN, `EXACT_FRAME` versus
`EXACT_PAIR` e a ausência de comandos XP durante o teste.
Depois testar SIDE/TOP/MID com peças **fisicamente confirmadas** e
as duas classes de erro. **Não habilitar operação autônoma de
liberação 0/OK com base apenas no percentual V3**.

Para suspender a V3, parar o ODIN e usar o comando de desativação
documentado em `AOI_V3_PRODUCTION_NG_VETO_TRIAL_20261104.md`.
Após desativar, esta versão nova **não retorna automaticamente à V2**;
os casos novos ficam bloqueados em revisão por falta de checkpoint.
