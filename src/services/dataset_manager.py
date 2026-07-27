"""Persistência das amostras e dos diagnósticos de aprendizado ativo."""

from __future__ import annotations

import cv2
import json
import numpy as np
from datetime import datetime

from src.config.settings import settings


class DatasetManager:
    @staticmethod
    def _json_safe(value):
        """Converte estruturas NumPy residuais em valores serializáveis."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): DatasetManager._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DatasetManager._json_safe(item) for item in value]
        return value

    @staticmethod
    def save_sample(
        ng_image: np.ndarray,
        label: str,
        sample_image: np.ndarray = None,
        aoi_info: dict = None,
        analysis: dict = None,
        save_images: bool = True,
    ) -> str:
        """
        Salva a imagem e um JSON técnico com as assinaturas usadas pela IA.

        O embedding KNN continua sendo a memória utilizada pelo classificador.
        O bloco ``semantic_debug`` registra o embedding 128D, seus deltas e a
        reconstrução espacial 4x4 para auditoria posterior.
        """
        if ng_image is None or ng_image.size == 0:
            return ""

        base_folder = settings.ANOMALY_DIR if label == "NG" else settings.NORMAL_DIR

        category = "Unknown"
        if aoi_info and "category" in aoi_info:
            category = aoi_info["category"]
            category = "".join(
                character
                for character in category
                if character.isalnum() or character in (" ", "_", "-")
            ).strip()
            if not category:
                category = "Unknown"

        target_folder = base_folder / category
        target_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"sample_{label}_{timestamp}"
        filepath_img = target_folder / f"{filename}.png"
        filepath_json = target_folder / f"{filename}.json"

        if save_images:
            cv2.imwrite(str(filepath_img), ng_image)
            if sample_image is not None and sample_image.size > 0:
                filepath_sample = target_folder / f"{filename}_sample.png"
                cv2.imwrite(str(filepath_sample), sample_image)

        metadata = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "image_file": f"{filename}.png" if save_images else "",
            "image_type": "single_ng",
            "status_treinamento": "pendente",
            "aoi_info": {
                "board": "",
                "parts": "",
                "category": category,
                "value": "",
            },
            "analysis": {},
        }

        if aoi_info:
            metadata["aoi_info"]["board"] = aoi_info.get("board", "")
            metadata["aoi_info"]["parts"] = aoi_info.get("parts", "")
            metadata["aoi_info"]["value"] = aoi_info.get("value", "")

        if analysis:
            detail = analysis.get("detail", {})
            knn_embedding = detail.get("query_embedding", [])
            semantic_reference = detail.get("ref_emb", [])
            semantic_query = detail.get("query_emb", [])
            semantic_debug = detail.get("semantic_debug") or {}

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
                "embedding": knn_embedding,
                "semantic": {
                    "schema": semantic_debug.get(
                        "schema",
                        "visionx.semantic.legacy",
                    ),
                    "distance_cosine": detail.get(
                        "semantic_distance_cosine",
                        0,
                    ),
                    "semantic_loss": detail.get("semantic_loss", 0),
                    "reference_embedding": semantic_reference,
                    "query_embedding": semantic_query,
                    "debug": semantic_debug,
                },
            }

        safe_metadata = DatasetManager._json_safe(metadata)
        try:
            with open(filepath_json, "w", encoding="utf-8") as file:
                json.dump(
                    safe_metadata,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            print(f"⚠️ Erro ao salvar metadados JSON: {exc}")

        return str(filepath_img) if save_images else str(filepath_json)
