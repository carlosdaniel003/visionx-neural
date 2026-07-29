# src\config\settings.py
"""Configurações centralizadas do VisionX Neural."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Config:
    PUBLIC_DIR = BASE_DIR / "public"
    DATASET_DIR = PUBLIC_DIR / "dataset"
    ANOMALY_DIR = DATASET_DIR / "anomalia"
    NORMAL_DIR = DATASET_DIR / "nao_anomalia"
    TEMPLATE_IMAGE_PATH = BASE_DIR / "public" / "template_padrao.png"

    SCREEN_CAPTURE_FPS = 15

    HUD_BORDER_COLOR_OK = (0, 255, 0)
    HUD_BORDER_COLOR_NG = (0, 0, 255)
    HUD_BORDER_THICKNESS = 4

    # Assinaturas de cor do layout da AOI, no formato HSV do OpenCV.
    COLOR_BLUE_LOWER = (100, 150, 50)
    COLOR_BLUE_UPPER = (130, 255, 255)

    COLOR_RED1_LOWER = (0, 150, 50)
    COLOR_RED1_UPPER = (10, 255, 255)
    COLOR_RED2_LOWER = (170, 150, 50)
    COLOR_RED2_UPPER = (180, 255, 255)

    # Tons reais medidos na moldura verde menor da AOI. A faixa anterior
    # H=40..80 aceitava grande parte da própria placa verde.
    AOI_GREEN_RGB_SAMPLES = (
        (22, 149, 21),
        (15, 124, 13),
        (86, 188, 105),
        (60, 202, 74),
        (12, 203, 12),
        (153, 246, 133),
        (88, 203, 66),
    )
    COLOR_GREEN_LOWER = (52, 108, 92)
    COLOR_GREEN_UPPER = (70, 255, 255)
    AOI_GREEN_MAX_RGB_DISTANCE = 68.0
    AOI_GREEN_MIN_EXCESS = 10

    # Fundo cinza da interface da AOI.
    AOI_GRAY_THRESHOLD = 45
    AOI_GRAY_MIN = 100
    AOI_GRAY_MAX = 245

    PORTA_RECEPTORA = 5001

    TESSERACT_CMD = (
        r"C:\Users\cdaniel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    )


settings = Config()
settings.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
settings.NORMAL_DIR.mkdir(parents=True, exist_ok=True)
