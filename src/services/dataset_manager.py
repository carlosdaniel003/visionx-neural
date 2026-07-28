"""Persistência compacta da memória de anomalias e auditoria opcional."""

from __future__ import annotations

import json
from datetime import datetime

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import (
    build_anomaly_signature,
    valid_anomaly_signature,
)


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
    def _safe_category(aoi_info: dict | None) -> str:
        raw = str((aoi_info or {}).get("category", "Unknown"))
        category = "".join(
            character
            for character in raw
            if character.isalnum() or character in (" ", "_", "-")
        ).strip()
        return category or "Unknown"

    @staticmethod
    def save_sample(
        ng_image: np.ndarray,
        label: str,
        sample_image: np.ndarray = None,
        aoi_info: dict = None,
        analysis: dict = None,
        save_images: bool = False,
        source: str = "",
        ai_decision: str = "",
    ) -> str:
        """Salva o JSON da anomalia; imagens são apenas auditoria opcional."""
        normalized_label = str(label or "").strip().upper()
        if normalized_label not in {"OK", "NG"}:
            return ""

        detail = (analysis or {}).get("detail", {})
        anomaly_memory = (
            detail.get("anomaly_signature")
            or detail.get("query_anomaly_signature")
            or {}
        )
        if not valid_anomaly_signature(anomaly_memory):
            focus_box = (
                detail.get("semantic_focus_box")
                or detail.get("adhesive_roi_box")
                or detail.get("missing_roi_box")
                or detail.get("roi_box")
                or (analysis or {}).get("bounding_box")
            )
            anomaly_memory = build_anomaly_signature(
                sample_image,
                ng_image,
                detail,
                aoi_info,
                focus_box,
            )

        if not valid_anomaly_signature(anomaly_memory):
            return ""

        category = DatasetManager._safe_category(aoi_info)
        base_folder = (
            settings.ANOMALY_DIR
            if normalized_label == "NG"
            else settings.NORMAL_DIR
        )
        target_folder = base_folder / category
        target_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"memory_{normalized_label}_{timestamp}"
        filepath_test = target_folder / f"{filename}_test.png"
        filepath_reference = target_folder / f"{filename}_reference.png"
        filepath_json = target_folder / f"{filename}.json"

        test_image_file = ""
        reference_image_file = ""
        if save_images and isinstance(ng_image, np.ndarray) and ng_image.size > 0:
            if cv2.imwrite(str(filepath_test), ng_image):
                test_image_file = filepath_test.name
            if (
                isinstance(sample_image, np.ndarray)
                and sample_image.size > 0
                and cv2.imwrite(str(filepath_reference), sample_image)
            ):
                reference_image_file = filepath_reference.name

        info = aoi_info if isinstance(aoi_info, dict) else {}
        semantic_debug = detail.get("semantic_debug") or {}
        semantic_reference = detail.get("ref_emb", [])
        semantic_query = detail.get("query_emb", [])
        legacy_embedding = detail.get("query_embedding", [])

        metadata = {
            "schema": "visionx.memory.v2",
            "label": normalized_label,
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "mode": "json_plus_audit_images" if save_images else "json_only",
                "test_image_file": test_image_file,
                "reference_image_file": reference_image_file,
                "images_required_for_knn": False,
            },
            "image_file": test_image_file,
            "image_type": "anomaly_signature",
            "status_treinamento": "memoria_ativa",
            "decision": {
                "operator_label": normalized_label,
                "source": str(source or ""),
                "ai_label": str(ai_decision or ""),
                "disagreement": bool(
                    ai_decision
                    and str(ai_decision).upper() != normalized_label
                ),
            },
            "aoi_info": {
                "board": info.get("board", ""),
                "parts": info.get("parts", ""),
                "category": category,
                "value": info.get("value", ""),
            },
            "analysis": {
                "operator_label": normalized_label,
                "verdict": (analysis or {}).get("verdict", ""),
                "is_defect": (analysis or {}).get("is_defect", False),
                "confidence": (analysis or {}).get("confidence", 0.0),
                "reason": (analysis or {}).get("reason", ""),
                "final_score": detail.get("final_score", 0.0),
                "physical_score": detail.get("physical_score", 0.0),
                "fusion_rule": detail.get("fusion_rule", ""),
                "anomaly_memory": anomaly_memory,
                "embedding": legacy_embedding,
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
                    "semantic_global_loss": detail.get(
                        "semantic_global_loss",
                        detail.get("semantic_loss", 0),
                    ),
                    "semantic_local_evidence": detail.get(
                        "semantic_local_evidence",
                        0,
                    ),
                    "reference_embedding": semantic_reference,
                    "query_embedding": semantic_query,
                    "debug": semantic_debug,
                },
                "engines": {
                    "adhesive": {
                        "score": detail.get("adhesive_score", 0),
                        "excess_coverage": detail.get("excess_coverage", 0),
                        "padding_overlap": detail.get("padding_overlap", 0),
                        "area_growth_ratio": detail.get("area_growth_ratio", 0),
                        "spread_growth_ratio": detail.get(
                            "spread_growth_ratio",
                            0,
                        ),
                        "lower_leakage_ratio": detail.get(
                            "lower_leakage_ratio",
                            0,
                        ),
                    },
                    "missing": {
                        "score": detail.get("missing_score", 0),
                        "expectation_mode": detail.get(
                            "missing_expectation_mode",
                            "unknown",
                        ),
                        "classification": detail.get(
                            "missing_classification",
                            "",
                        ),
                        "structure_loss": detail.get(
                            "missing_structure_loss",
                            0,
                        ),
                        "extra_structure": detail.get(
                            "missing_extra_structure",
                            0,
                        ),
                        "coverage": detail.get(
                            "missing_changed_coverage",
                            detail.get("missing_coverage", 0),
                        ),
                        "appearance_loss": detail.get(
                            "missing_appearance_loss",
                            0,
                        ),
                        "background_exposure": detail.get(
                            "missing_background_exposure",
                            0,
                        ),
                        "presence_retention": detail.get(
                            "missing_presence_retention",
                            1,
                        ),
                        "direct_similarity": detail.get(
                            "missing_direct_similarity",
                            1,
                        ),
                        "best_nearby_similarity": detail.get(
                            "missing_best_similarity",
                            1,
                        ),
                        "displacement": {
                            "dx": detail.get("missing_displacement_dx", 0),
                            "dy": detail.get("missing_displacement_dy", 0),
                            "pixels": detail.get(
                                "missing_displacement_pixels",
                                0,
                            ),
                            "normalized": detail.get(
                                "missing_displacement_pct",
                                0,
                            ),
                        },
                        "reference_distinctness": detail.get(
                            "missing_reference_distinctness",
                            0,
                        ),
                    },
                    "structural": {
                        "error": detail.get("silk_error_pct", 0),
                        "extra": detail.get("extra_pct", 0),
                        "missing": detail.get("missing_pct", 0),
                        "matched": detail.get("matched_pct", 0),
                    },
                    "texture": {
                        "ssim": detail.get("ssim", 0),
                        "pct_changed": detail.get("pct_changed", 0),
                        "edge_change": detail.get("edge_change", 0),
                        "hist_corr": detail.get("hist_corr", 0),
                        "local_score": detail.get("local_score", 0),
                        "ctx_score": detail.get("ctx_score", 0),
                    },
                    "semantic": {
                        "score": detail.get("semantic_loss", 0),
                        "global": detail.get("semantic_global_loss", 0),
                        "local": detail.get("semantic_local_evidence", 0),
                    },
                },
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
            print(f"Erro ao salvar memória JSON: {exc}")
            return ""

        return str(filepath_json)
