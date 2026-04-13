# src/services/dataset_manager.py
"""
Módulo responsável por gerenciar a persistência de imagens para o Dataset.
v2: Salva NG inteira separada (não pareada) + embedding cache em JSON.
"""
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from src.config.settings import settings


class DatasetManager:
    @staticmethod
    def save_sample(ng_image: np.ndarray, label: str,
                    sample_image: np.ndarray = None,
                    aoi_info: dict = None, analysis: dict = None) -> str:
        """
        Salva a imagem NG (ou OK) inteira na pasta correta.
        NÃO faz mais hstack — salva a imagem individual.

        :param ng_image: Imagem NG (ou OK) completa em BGR.
        :param label: "OK" ou "NG".
        :param sample_image: Imagem Sample (padrão) — salva separada para referência.
        :param aoi_info: Dict com Board, Parts, Value.
        :param analysis: Dict com métricas da análise.
        :return: Caminho do arquivo salvo.
        """
        if ng_image is None or ng_image.size == 0:
            return ""

        folder = settings.ANOMALY_DIR if label == "NG" else settings.NORMAL_DIR

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"sample_{label}_{timestamp}"
        filepath_img = folder / f"{filename}.png"
        filepath_json = folder / f"{filename}.json"

        # Salva a imagem NG/OK inteira (NÃO pareada)
        cv2.imwrite(str(filepath_img), ng_image)

        # Se tiver sample, salva também como referência (para visualização)
        if sample_image is not None and sample_image.size > 0:
            filepath_sample = folder / f"{filename}_sample.png"
            cv2.imwrite(str(filepath_sample), sample_image)

        # Metadados
        metadata = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "image_file": f"{filename}.png",
            "image_type": "single_ng",  # marca que é imagem individual
            "aoi_info": {
                "board": "",
                "parts": "",
                "value": "",
            },
            "analysis": {}
        }

        if aoi_info:
            metadata["aoi_info"]["board"] = aoi_info.get("board", "")
            metadata["aoi_info"]["parts"] = aoi_info.get("parts", "")
            metadata["aoi_info"]["value"] = aoi_info.get("value", "")

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

        try:
            with open(filepath_json, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar metadados JSON: {e}")

        return str(filepath_img)