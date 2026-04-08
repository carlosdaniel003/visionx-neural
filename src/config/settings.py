# C:\Users\cdaniel\visionx-neural\src\config\settings.py
"""
Módulo de configurações centralizadas do VisionX Neural.
Define caminhos, constantes da interface e hiperparâmetros da IA.
"""
from pathlib import Path

# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
    # --- Paths Locais ---
    PUBLIC_DIR = BASE_DIR / "public"
    DATASET_DIR = PUBLIC_DIR / "dataset"
    ANOMALY_DIR = DATASET_DIR / "anomalia"
    NORMAL_DIR = DATASET_DIR / "nao_anomalia"
    MODEL_PATH = BASE_DIR / "models" / "visionx_siamese.pth"

    # --- Configurações do Template Matching (Gatilho) ---
    TEMPLATE_IMAGE_PATH = BASE_DIR / "public" / "template_padrao.png"
    MATCHING_THRESHOLD = 0.60 # Limiar para encontrar a interface

    # --- Configurações do Extrator Visual (Pilar 1) ---
    SCREEN_CAPTURE_FPS = 30
    
    # --- Configurações da IA (Pilar 2) ---
    AI_CONFIDENCE_THRESHOLD = 0.70 # Confiança mínima para marcar como defeito real
    IMAGE_SIZE = (224, 224) # Tamanho padrão para entrada na rede neural
    
    # --- Configurações do HUD (Pilar 3) ---
    HUD_BORDER_COLOR_OK = (0, 255, 0) # Verde
    HUD_BORDER_COLOR_NG = (0, 0, 255) # Vermelho
    HUD_BORDER_THICKNESS = 4

# Instância global para uso em todo o projeto
settings = Config()

# Garante que as pastas de dados existam ao iniciar
settings.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
settings.NORMAL_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)