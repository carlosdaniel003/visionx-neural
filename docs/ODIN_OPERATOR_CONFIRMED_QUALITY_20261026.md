# 26/10/2026 — Correção de rótulos humanos para auditoria CNN (sem ativação)

Confirmados visualmente pelo operador:
- Casos 1–6: NG físico, componente INVERTIDO (6 capturas, não presumir 6 peças).
- Caso 7: OK físico apesar de estar em \`ng_archive\`; a CNN isolada acertou.
- Caso 8: NG físico; serigrafia escura, tipo específico incerto.
- Sete controles adicionais confirmados: 2 OK e 5 NG, inclusive faltando e adesivo.
- Os tipos físicos são independentes do roteamento por categoria AOI.

Registro com SHA-256 em \`docs/odin_operator_confirmed_quality_20261026.json\`.
\`src.services.aoi_universal_quality_archive.universal_archive_events\` agora utiliza
a qualidade humana para essas 15 fontes **somente após verificar SHA-256 dos PNGs**.
Especialmente o caso 7 não pode mais entrar no candidato universal como NG.
Divergência entre revisões ou arquivos interrompe a carga. Todos os outros
rótulos continuam provisórios; treino universal segue restrito a candidato
experimental, sem autorização de produção.

Os seis casos de inversão possuem pares gabarito/teste observados pelo
operador e podem ser persistidos pelo \`QualificationStore\`, sem afirmar
que as três luzes são da mesma peça física. Os outros nove não recebem
qualificação de par automaticamente.

Com o ODIN **fechado**, no PowerShell do repositório atualizado:

\`\`\`powershell
python -m src.services.odin_operator_confirmed_labels --apply
\`\`\`

O comando verifica as 15 capturas e os seis pares antes de escrever
\`reports/faltando_neural/run_20261016T154441_358349Z/qualification.json\`.
Não move PNGs, não treina CNN, não altera a CNN produtiva ou o KNN,
não muda comando \`0/1\` do XP e não confirma vínculos multilight.
Sem \`--apply\` roda só a auditoria.

A CNN original não foi corrigida em produção: a etapa seguinte exige gerar e
avaliar um candidato offline usando separação por grupos, revisões independentes
dos 78 possíveis OK→NG e proteção dos NG, antes de qualquer nova aprovação.
