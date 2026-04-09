"""
Módulo responsável por gerenciar a persistência de imagens para o Dataset.
Salva recortes categorizados + metadados JSON para futuro treinamento.
"""
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from src.config.settings import settings


class DatasetManager:
    @staticmethod
    def save_sample(image_bgr: np.ndarray, label: str,
                    aoi_info: dict = None, analysis: dict = None) -> str:
        """
        Salva um recorte de imagem na pasta correta com timestamp único.
        Também salva um arquivo .json com os metadados (Board, Parts, Value,
        métricas de análise, veredito, etc).

        :param image_bgr: Recorte da tela em formato OpenCV (BGR).
        :param label: "OK" para não anomalias, "NG" para anomalias.
        :param aoi_info: Dict com Board, Parts, Value extraídos via OCR.
        :param analysis: Dict com métricas da análise do Juiz Neural.
        :return: Caminho do arquivo salvo ou string vazia em caso de erro.
        """
        if image_bgr is None or image_bgr.size == 0:
            return ""

        # Pasta destino
        folder = settings.ANOMALY_DIR if label == "NG" else settings.NORMAL_DIR

        # Nome único com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"sample_{label}_{timestamp}"
        filepath_img = folder / f"{filename}.png"
        filepath_json = folder / f"{filename}.json"

        # Salva a imagem
        cv2.imwrite(str(filepath_img), image_bgr)

        # Monta os metadados
        metadata = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "image_file": f"{filename}.png",
            "aoi_info": {
                "board": "",
                "parts": "",
                "value": "",
            },
            "analysis": {}
        }

        # Preenche info da AOI se disponível
        if aoi_info:
            metadata["aoi_info"]["board"] = aoi_info.get("board", "")
            metadata["aoi_info"]["parts"] = aoi_info.get("parts", "")
            metadata["aoi_info"]["value"] = aoi_info.get("value", "")

        # Preenche métricas da análise se disponível
        if analysis:
            detail = analysis.get("detail", {})
            metadata["analysis"] = {
                "verdict": analysis.get("verdict", ""),
                "is_defect": analysis.get("is_defect", False),
                "confidence": analysis.get("score_text", ""),
                "reason": analysis.get("reason", ""),
                "ssim": detail.get("ssim", 0),
                "pct_changed": detail.get("pct_changed", 0),
                "edge_change": detail.get("edge_change", 0),
                "hist_corr": detail.get("hist_corr", 0),
                "local_score": detail.get("local_score", 0),
                "ctx_score": detail.get("ctx_score", 0),
                "db_score": detail.get("db_score", 0),
                "final_score": detail.get("final_score", 0),
            }

        # Salva o JSON
        try:
            with open(filepath_json, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar metadados JSON: {e}")

        return str(filepath_img)