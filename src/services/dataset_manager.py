# C:\Users\cdaniel\visionx-neural\src\services\dataset_manager.py
"""
Módulo responsável por gerenciar a persistência de imagens para o Dataset.
Salva recortes categorizados para futuro treinamento da Rede Neural.
"""
import cv2
import numpy as np
from datetime import datetime
from src.config.settings import settings

class DatasetManager:
    @staticmethod
    def save_sample(image_bgr: np.ndarray, label: str) -> str:
        """
        Salva um recorte de imagem na pasta correta com timestamp único.
        :param image_bgr: Recorte da tela em formato OpenCV (BGR).
        :param label: "OK" para não anomalias, "NG" para anomalias.
        :return: Caminho do arquivo salvo ou string vazia em caso de erro.
        """
        if image_bgr is None or image_bgr.size == 0:
            return ""

        # Decide a pasta destino com base na label
        folder = settings.ANOMALY_DIR if label == "NG" else settings.NORMAL_DIR
        
        # Gera um nome de arquivo único baseado no relógio (milissegundos)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"sample_{label}_{timestamp}.png"
        filepath = folder / filename

        # Salva o arquivo no disco
        cv2.imwrite(str(filepath), image_bgr)
        return str(filepath)