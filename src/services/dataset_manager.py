# src/services/dataset_manager.py
"""
Módulo responsável por gerenciar a persistência de imagens para o Dataset.
v4: Salvamento estruturado em subpastas por Categoria do OCR (Mixture of Experts Prep).
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
                    aoi_info: dict = None, analysis: dict = None,
                    save_images: bool = True) -> str:
        """
        Salva a imagem NG (ou OK) inteira na pasta correta, estruturada por Categoria.
        """
        if ng_image is None or ng_image.size == 0:
            return ""

        # 1. Define a Raiz (Anomalia ou Falha Falsa)
        base_folder = settings.ANOMALY_DIR if label == "NG" else settings.NORMAL_DIR

        # 2. Define a Subpasta baseada na Categoria do OCR
        category = "Unknown"
        if aoi_info and "category" in aoi_info:
            category = aoi_info["category"]
            # Limpa caracteres bizarros que o SO não gosta em nomes de pasta, só por precaução
            category = "".join(c for c in category if c.isalnum() or c in (' ', '_', '-')).strip()
            if not category:
                category = "Unknown"
                
        # Junta a raiz com a subcategoria
        target_folder = base_folder / category
        
        # Garante que a subpasta exista no HD (cria se não existir)
        target_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"sample_{label}_{timestamp}"
        filepath_img = target_folder / f"{filename}.png"
        filepath_json = target_folder / f"{filename}.json"

        # Apenas salva os pixels no HD se a curadoria (Active Learning) autorizar
        if save_images:
            # Salva a imagem NG/OK inteira (NÃO pareada)
            cv2.imwrite(str(filepath_img), ng_image)

            # Se tiver sample, salva também como referência (para visualização)
            if sample_image is not None and sample_image.size > 0:
                filepath_sample = target_folder / f"{filename}_sample.png"
                cv2.imwrite(str(filepath_sample), sample_image)

        # Metadados
        metadata = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            # Se a imagem não foi salva, avisamos o sistema deixando em branco
            "image_file": f"{filename}.png" if save_images else "",
            "image_type": "single_ng",  # marca que é imagem individual
            "aoi_info": {
                "board": "",
                "parts": "",
                "category": category, # NOVO: Salva a categoria para uso futuro da IA
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
                
                # O "código de barras matemático" da placa (Vetores Semânticos)
                "embedding": detail.get("embedding", [])
            }

        try:
            with open(filepath_json, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar metadados JSON: {e}")

        # Retorna o caminho do PNG (se salvou a foto) ou do JSON (se salvou só o texto)
        return str(filepath_img) if save_images else str(filepath_json)