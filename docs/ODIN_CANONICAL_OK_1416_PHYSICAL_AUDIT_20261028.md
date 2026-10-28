# ODIN — 28/10/2026 — Recuperação da referência OK 14:16 e sonda física

## Evidência real (event_id 8674e908a7b24ea082963bb3e2515069)

Operador confirmou FALHA FALSA. Em Produção o ODIN apresentou NG PREVENTIVO.
O log mostra a referência humana `public/ok_archive/2026-10-21_1416_FALTANDO.png`
na primeira posição SIDE KNN C A=0.86618201 / B=0.83100362.
A KNN exata retornou `NOT_FOUND` porque não encontrou pares
gabarito/teste com pixels idênticos. Os rótulos de origem não são
transferidos apenas pela proximidade; referência visual não é par exato.

CNN FALTANDO V2 (score NG bruto, não calibrado):
SIDE=0.03436757 (falha falsa), TOP=0.25600529 (revisão),
MID=0.85979676 (revisão); consenso `REVIEW/INVALID_LIGHT_EVIDENCE`.
A fusão permaneceu em revisão, físico=0 e nenhuma luz NG corroborada.
A KNN visual independente teve SIDE OK=0.977865 vs NG=0.988808,
TOP OK=0.972946 vs NG=0.935611 e MID OK=0.960225 vs NG=0.932796.

## Causa-raiz

`apply_physical_missing_veto` só inspecionava quando havia
AUTO-OK CNN ou o padrão de CNN *dois OK e uma revisão NG*.
Este evento tem apenas SIDE OK e TOP/MID em revisão.
Por isso a sonda nem rodou; `_operator_corrected_side_ok_recall`
retornou `PHYSICAL_VETO_NOT_CLEAR` antes das verificações restantes.

## Correção

- Uma pré-seleção leve e específica da referência canônica OK pelo
  caminho e SHA de pixels SIDE A/B aciona a sonda de ausência física
  em SIDE, TOP e MID. A pré-seleção **não classifica** a captura.
- A recuperação OK só aceita auditoria física identificada e íntegra,
  sem ausência suspeita nem iluminação indisponível, e mantém as
  provas KNN, CNN, SHA de fonte, três luzes e contraprova visual
  existentes. A saída `FALHA FALSA` representa reconhecimento por
  **referência visual aproximada**, não pixels idênticos.
- `NO_STRONG_EVIDENCE` nas três luzes permite que a política
  supervisada existente recompute o contrato e, em Produção contínua,
  autorize 0/OK após sua janela de pausa. Não é certificação da peça.
- `GEOMETRY_REVIEW` permite mostrar `FALHA FALSA` por memória
  histórica quando o resto do contrato é válido, mas exige o operador:
  não transmite 0/OK e não transforma a peça em NG preventivo
  automático. `UNAVAILABLE` e `SUSPECT_ABSENCE` não recuperam OK.
- Os recortes reportados têm tamanhos 318x570 (gabarito) e 317x567
  (teste). A sonda atual considera aparagem geométrica motivo de revisão;
  só a execução real na estação determina o estado físico final.

## Garantias

Nenhuma alteração de checkpoint CNN, índice KNN, rótulo histórico, arquivos
de acervo, roteador de par estritamente exato ou interpretação de 0/1 do XP.
O teste em `tests/test_operator_corrected_archive_live_ok.py` cobre
pré-seleção, execução física de três luzes, geometria em revisão,
ausência/sensor indisponível e referência não qualificada.
O replay sintético não valida generalização; testar na estação
supervisionada com XP isolado antes de conectar comandos físicos.
