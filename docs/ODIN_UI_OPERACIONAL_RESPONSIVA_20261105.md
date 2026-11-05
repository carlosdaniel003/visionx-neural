# ODIN — organização de interface, status e responsividade (05/11/2026)

## Escopo e integridade

A atualização é **somente de interface/observabilidade**. Não altera a captura AOI,
a fusão SIDE/TOP/MID, a CNN V2/V3, a memória KNN, o contrato de pausa por Espaço,
o julgamento 0/1 nem o agente Windows XP.

A marca visual continua `ODIN - Observador Digital Inteligente`, com fundo
neutro escuro e resultado de qualidade visível nos respectivos cartões.

## Hierarquia de ações

Em `CONTROLES DE INSPEÇÃO`, as ações foram organizadas por função:

1. **Iluminação da câmera:** TOP / SIDE / MID, mantendo os atalhos existentes;
   o botão da luz ativa conserva uma aparência distinta.
2. **Julgamento da peça:** `Aprovar como OK` e `Confirmar defeito NG`,
   no mesmo grupo, sem misturar com tarefas auxiliares.
3. **Captura e ações auxiliares:** captura local MSS, reanálise do evento XP
   quando elegível e descarte, preservando o controle real de habilitação.
4. **Manutenção do dataset:** permanece isolada pelos contratos existentes.

O estado hover, pressionado, foco de teclado e desabilitado usa estilos
distintos. O retorno de clique diz **solicitado** quando não existe prova de
que o comando foi efetivamente executado e persistido. Permissões e eventos
continuam sob a lógica operacional pré-existente.

### Responsividade

Os perfis gerais permanecem em `src/ui/responsive_layout.py`:
`compact` abaixo de 1500 px, `standard` de 1500 a 1799 px e
`wide` a partir de 1800 px, com base na **largura da janela/viewport**
(e não necessariamente na resolução física do monitor).

- O julgamento OK/NG usa duas colunas dentro de seu grupo.
- As ações auxiliares usam uma, duas ou três colunas de acordo com a largura
  disponível, inclusive quando o redimensionamento permanece no mesmo perfil.
- Os cards de inspeção, veredito, memória e indicadores de rodapé preservam seus
  layouts adaptativos existentes; imagens mantêm `KeepAspectRatio`.
- A auditoria do acervo também rearranja suas ações independentemente do tamanho
  geral da janela: uma coluna antes de 590 px, duas antes de 1020 px e três
  acima disso.

## Painel de auditoria

O painel agora se chama **AUDITORIA • MEMÓRIA E MODELOS** e oferece somente:

- **Validar acervo OK/NG:** replay histórico somente leitura da KNN e da
  CNN FALTANDO V2, sem treinar uma nova rede.
- **Comparar CNN × KNN • diagnóstico:** estudo experimental sem promoção
  de modelo ou efeito na Produção.
- **Cancelar verificação:** fica visível apenas durante uma execução.

Foram removidos da interface os controles:

- `Comparar geometria de bordas (experimental, somente acervo)` (checkbox);
- `Testar KNN por similaridade`;
- `Replay operacional por evento`.

Os motores/CLIs de diagnóstico permanecem em
`src/services/startup_regression/` para investigação técnica específica,
mas não ocupam a navegação normal do operador. Não apagar esses módulos nem
confundir a retirada dos botões com remoção dos cálculos internos de geometria
utilizados pelos especialistas.

Uma auditoria **parcial ou cancelada** não apresenta resultado como conclusão.
Sucesso histórico no acervo tampouco representa precisão comprovada em peças
novas, nem ativação da CNN V3.

## Status operacional

`src/ui/operational_status_model.py` centraliza a interpretação de
mensagens para a camada visual. Critérios:

- `Ouvindo AOI`, `porta 5001` e `aguardando` são **espera**,
  não conexão confirmada.
- Recepção/conexão explícita é diferenciada de falha/desconexão.
- IA aguardando peça não é marcada como processamento efetivo.
- `Última decisão` diferencia **automática sem confirmação física**,
  operador ODIN e operador XP; descarte é mostrado como **última ação**
  e não como peça OK/NG.

A exibição dessas mensagens não é um gate de segurança nem prova de
que a inspeção industrial está corretamente conectada. Os logs e
a política XP continuam sendo as referências operacionais.

## Validação e execução

Após atualização da branch `central`, reiniciar a interface. Fazer
observação com o XP isolado, comparando janela estreita de notebook e
janela ampla de monitor; verificar luzes, botões habilitados, reanálise,
pausa, estados em espera, uso dos atalhos e conclusão dos relatórios.

A ativação da CNN V3 por ponteiro local permanece independente dessa
alteração. O GitHub valida contratos e testes sintéticos; não reproduz
a interação real com a estação Windows XP.
