# ODIN | 28/10/2026 | Evento ec843a9ae7f04b26ab0681eebec8a1f8

Operador: FALHA FALSA. Painel: NG PREVENTIVO.

## Causa-raiz comprovada por proveniência

A referência que a KNN C classificou como NG com
0.97620107 (descritor A) e 0.96378327 (B) foi
`public/ng_archive/2026-10-21_1414_FALTANDO.png`,
pixel SHA `4481318c87aa3568959554fa95b438ef1b1c74a9d2b4876c2bd14c7ea751f9a3`.
Essa MESMA fonte aparece no arquivo de decisões humanas
`docs/odin_operator_confirmed_quality_20261026.json` como
`quality_label=OK`, `physical_type=COMPONENT_PRESENT`,
arquivo SHA `b33a8a9b9a9f7ef2684564f851ee989c5a76783ab158095cd5b910c52311dc65`.
A KNN C, entretanto, indexava o rótulo da pasta AOI (`NG`)
em vez da qualidade física confirmada.

## Correção técnica

1. A memória KNN C aplica uma sobreposição somente quando
   documento + PNG original + SHA de bytes + SHA de pixels coincidem.
   A origem permanece no `ng_archive` (sem alterar acervo), mas
   o rótulo efetivo no índice vira OK. O cache anterior NG é
   invalidado para esse caminho pela comparação de rótulo.
   Sem a fonte/hashes intactos, nenhum override acontece.
2. Reconhecimento multilight revisa o evento antigo e o novo índice:
   SIDE recupera a fonte confirmada entre primeiros A/B com
   similaridades >=0.97/0.96, todos outros NG SIDE <0.91;
   TOP e MID têm >=4 de 5 vizinhos OK em A e B e apontam
   OK na comparação visual; SIDE pode ter pequena margem NG,
   mas não há NG forte corroborado em nenhuma luz.
   CNN SIDE <=0.12, TOP <=0.30, MID <=0.90 (scores não
   calibrados), consenso CNN REVIEW/INVALID_LIGHT_EVIDENCE;
   físico ausente (score=0), 3 luzes válidas e mesmo checkpoint.
   O resultado é `FALHA FALSA`.
3. O modo Produção contínuo já solicitado pelo operador autoriza
   `0/OK` somente com o contrato de três luzes revalidado
   imediatamente antes do comando. A janela de pausa de 5 segundos
   e controle de geração/peça existentes permanecem.
4. Esse OK automático não se converte em verdade humana para
   treinamento: a fila de quarentena guarda a evidência e não
   importa para KNN/CNN os rótulos automáticos.
5. Não foi declarado par exato: os índices de memória exata
   permaneceram `NOT_FOUND`. A qualidade da **fonte** foi
   comprovada, a correspondência com a captura nova é aproximada.

### Segurança

O override não autoriza todos os eventos da categoria. Se outro
NG real aparecer com evidência física, mudança de PNG,
maior margem NG ou voto TOP/MID desfavorável, a rota OK não
passa. O fallback de produção é NG preventivo apenas em
capturas completas; falhas técnicas param o ciclo.

### Provas de regressão

`tests/test_operator_corrected_archive_live_ok.py` testa
validação de arquivo/hash, indexação efetiva KNN C,
reprodução dos scores das três iluminações, autorização
0/OK no modo Produção e bloqueios de adulteração/NG.
