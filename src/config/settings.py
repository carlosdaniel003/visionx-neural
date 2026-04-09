"""
Módulo de configurações centralizadas do VisionX Neural.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
    PUBLIC_DIR = BASE_DIR / "public"
    DATASET_DIR = PUBLIC_DIR / "dataset"
    ANOMALY_DIR = DATASET_DIR / "anomalia"
    NORMAL_DIR = DATASET_DIR / "nao_anomalia"
    TEMPLATE_IMAGE_PATH = BASE_DIR / "public" / "template_padrao.png"

    SCREEN_CAPTURE_FPS = 15 # Reduzimos para 15 FPS pois a busca por cor é imediata
    
    HUD_BORDER_COLOR_OK = (0, 255, 0)
    HUD_BORDER_COLOR_NG = (0, 0, 255)
    HUD_BORDER_THICKNESS = 4

    # --- NOVO: Assinaturas de Cor do Layout da AOI (Formato HSV do OpenCV) ---
    # Azul (Barra Sample)
    COLOR_BLUE_LOWER = (100, 150, 50)
    COLOR_BLUE_UPPER = (130, 255, 255)
    
    # Vermelho (Barra NG) - Vermelho no HSV tem duas faixas (início e fim do espectro)
    COLOR_RED1_LOWER = (0, 150, 50)
    COLOR_RED1_UPPER = (10, 255, 255)
    COLOR_RED2_LOWER = (170, 150, 50)
    COLOR_RED2_UPPER = (180, 255, 255)
    
    # Verde (Quadrados de ROI)
    COLOR_GREEN_LOWER = (40, 100, 50)
    COLOR_GREEN_UPPER = (80, 255, 255)

settings = Config()
settings.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
settings.NORMAL_DIR.mkdir(parents=True, exist_ok=True)