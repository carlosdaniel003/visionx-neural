# Estrutura do Projeto: VisionX Neural

**Módulos Existentes:**
- `src/config/settings.py`: Centralização de todas as variáveis de ambiente, caminhos e constantes mágicas.
- `src/services/ng_image_archive.py`: Arquivo visual opcional de decisões finais NG em fila de background, independente do dataset e da memória KNN.

**Fluxos Principais (Planejados):**
1. **Pilar 1 (Extrator Visual):** Monitoramento contínuo da tela usando `mss` para detectar a janela da AOI.
2. **Pilar 2 (Cérebro Comparativo):** Rede siamesa avaliando propostas de defeitos.
3. **Pilar 3 (Display HUD):** Janela transparente sobreposta sinalizando as anomalias detectadas.
4. **Pilar 4 (Active Learning):** Salvamento local de recortes aprovados/rejeitados em `public/dataset/`.

**Dependências Base:**
- PyTorch (Redes Neurais)
- OpenCV (Visão Clássica / Tratamento de Imagem)
- mss (Captura de tela ultrarrápida)
- PyQt6 (Criação do HUD transparente)

**Arquivo visual NG opcional:**
- Toggle desligado por padrão; desativado mantém o fluxo atual sem criar cópias extras.
- Ativado: toda decisão final `NG` salva a imagem bruta de teste em `public/ng_archive/`.
- Nome: `AAAA-MM-DD_HH-MM-SS-ms_CATEGORIA.png`.
- O arquivo é evidência/auditoria e não participa de treinamento, protótipos ou votação KNN.
- A gravação é assíncrona para não bloquear o julgamento, o gate de rede nem a próxima imagem da AOI.


**Regra visual do painel KNN:**
- A existência de memória e a força do match são conceitos diferentes na interface.
- `PRIMEIRA OCORRÊNCIA` só pode ser exibido quando não existir nenhum registro da categoria consultada.
- Se existirem JSONs da categoria, mas não houver assinatura visual válida/comparável, mostrar `MEMÓRIA CARREGADA` e a quantidade de registros; não representar como primeira ocorrência.
- Se houver comparação visual, a barra amarela `Melhor match` deve usar a melhor similaridade realmente calculada. Caso `best_similarity` não tenha sido propagado por uma camada de telemetria, a UI pode recuperar o mesmo valor a partir da melhor hipótese OK/NG já calculada, sem alterar a decisão.
- Match abaixo do limiar continua visível como `MEMÓRIA ENCONTRADA • abaixo do limiar`; isso não concede influência ao KNN.
- Essa regra é exclusivamente de apresentação. Limiares, classificação, pesos, conflito OK/NG e regra de melhor correspondência permanecem no núcleo.
