"""Persistência compacta da memória de anomalias e auditoria opcional."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import (
    build_anomaly_signature,
    valid_anomaly_signature,
)
from src.services.image_archive_dedup import image_fingerprint


class DatasetManager:
    _fingerprint_lock = RLock()
    _folder_fingerprint_cache: dict[str, dict[str, str]] = {}

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
    def _lighting_key(value: str) -> str:
        normalized = str(value or "").strip().upper()
        return normalized if normalized in {"SIDE", "TOP", "MID"} else "SIDE"

    @staticmethod
    def _dedup_scope(aoi_info: dict | None) -> str:
        info = aoi_info if isinstance(aoi_info, dict) else {}
        board = "".join(
            char for char in str(info.get("board", "") or "").upper()
            if char.isalnum()
        )
        parts = "".join(
            char for char in str(info.get("parts", "") or "").upper()
            if char.isalnum()
        )
        return f"{board}|{parts}"

    @classmethod
    def _fingerprint_key(
        cls,
        fingerprint: str,
        lighting_mode: str,
        aoi_info: dict | None,
    ) -> str:
        lighting = cls._lighting_key(lighting_mode)
        scope = cls._dedup_scope(aoi_info)
        return f"{lighting}:{scope}:{fingerprint}"

    @classmethod
    def _fingerprint_index(cls, target_folder: Path) -> dict[str, str]:
        """Indexa memórias por iluminação + conteúdo visual exato."""
        key = str(Path(target_folder).resolve())
        with cls._fingerprint_lock:
            cached = cls._folder_fingerprint_cache.get(key)
            if cached is not None:
                return cached

            index: dict[str, str] = {}
            folder = Path(target_folder)
            folder.mkdir(parents=True, exist_ok=True)

            for json_path in folder.glob("*.json"):
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    storage = data.get("storage", {})
                    fingerprint = str(
                        storage.get("test_image_fingerprint", "") or ""
                    ).strip()
                    info = data.get("aoi_info", {}) or {}
                    lighting = cls._lighting_key(
                        info.get("lighting_mode", "")
                    )
                    if fingerprint:
                        index.setdefault(
                            cls._fingerprint_key(
                                fingerprint,
                                lighting,
                                info,
                            ),
                            str(json_path),
                        )
                except Exception:
                    continue

            # Compatibilidade com dataset anterior: calcula o fingerprint dos
            # *_test.png que ainda não possuíam hash no JSON.
            for image_path in folder.glob("*_test.png"):
                try:
                    image = cv2.imread(
                        str(image_path),
                        cv2.IMREAD_UNCHANGED,
                    )
                    fingerprint = image_fingerprint(image)
                    if not fingerprint or fingerprint in index:
                        continue
                    base_name = image_path.name[:-9]
                    json_path = image_path.with_name(f"{base_name}.json")
                    lighting = "SIDE"
                    legacy_info = {}
                    if json_path.exists():
                        try:
                            with open(
                                json_path,
                                "r",
                                encoding="utf-8",
                            ) as file:
                                legacy_data = json.load(file)
                            legacy_info = legacy_data.get("aoi_info", {}) or {}
                            lighting = cls._lighting_key(
                                legacy_info.get("lighting_mode", "")
                            )
                        except Exception:
                            lighting = "SIDE"
                            legacy_info = {}
                    # Um PNG sem JSON não é memória KNN utilizável.
                    # Nesse caso não bloqueamos a criação de um registro novo.
                    if json_path.exists():
                        index.setdefault(
                            cls._fingerprint_key(
                                fingerprint,
                                lighting,
                                legacy_info,
                            ),
                            str(json_path),
                        )
                except Exception:
                    continue

            cls._folder_fingerprint_cache[key] = index
            return index

    @classmethod
    def _register_fingerprint(
        cls,
        target_folder: Path,
        fingerprint: str,
        json_path: Path,
        lighting_mode: str = "",
        aoi_info: dict | None = None,
    ) -> None:
        if not fingerprint:
            return
        index = cls._fingerprint_index(target_folder)
        key = cls._fingerprint_key(
            fingerprint,
            lighting_mode,
            aoi_info,
        )
        with cls._fingerprint_lock:
            index.setdefault(key, str(json_path))

    @staticmethod
    def _upgrade_duplicate_record(
        json_path: str,
        *,
        fingerprint: str,
        lighting_mode: str,
        event_id: str,
        anomaly_memory: dict,
        source: str,
    ) -> str:
        """Enriquece um JSON antigo sem duplicar a imagem já conhecida."""
        if not json_path:
            return ""
        path = Path(json_path)
        if not path.exists():
            return ""

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            storage = data.setdefault("storage", {})
            storage["test_image_fingerprint"] = fingerprint

            info = data.setdefault("aoi_info", {})
            if lighting_mode:
                info["lighting_mode"] = lighting_mode

            if event_id or lighting_mode:
                multilight = data.setdefault("multilight", {})
                if event_id:
                    multilight["event_id"] = event_id
                if lighting_mode:
                    multilight["lighting_mode"] = lighting_mode
                multilight["same_piece_single_judgement"] = True

            analysis = data.setdefault("analysis", {})
            stored_memory = analysis.get("anomaly_memory")
            if (
                valid_anomaly_signature(anomaly_memory)
                and (
                    not valid_anomaly_signature(stored_memory)
                    or (
                        "full_frame_signature" not in stored_memory
                        and "full_frame_signature" in anomaly_memory
                    )
                )
            ):
                analysis["anomaly_memory"] = DatasetManager._json_safe(
                    anomaly_memory
                )

            duplicate = data.setdefault("deduplication", {})
            duplicate["policy"] = "exact_visual_content"
            duplicate["duplicate_observations"] = int(
                duplicate.get("duplicate_observations", 0) or 0
            ) + 1
            duplicate["last_duplicate_at"] = datetime.now().isoformat()
            duplicate["last_duplicate_source"] = str(source or "")

            with open(path, "w", encoding="utf-8") as file:
                json.dump(
                    DatasetManager._json_safe(data),
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            return str(path)
        except Exception as exc:
            print(f"Erro ao enriquecer memória duplicada: {exc}")
            return ""

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
        lighting_mode: str = "",
        event_id: str = "",
        source_frame: np.ndarray = None,
        final_analysis: dict = None,
    ) -> str:
        """Salva o JSON da anomalia; imagens são apenas auditoria opcional."""
        normalized_label = str(label or "").strip().upper()
        if normalized_label not in {"OK", "NG"}:
            return ""

        normalized_lighting = str(
            lighting_mode
            or (aoi_info or {}).get("lighting_mode", "")
            or ""
        ).strip().upper()
        normalized_event_id = str(event_id or "").strip()

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
                or detail.get("inverted_roi_box")
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

        fingerprint = image_fingerprint(ng_image)
        duplicate_visual_of = ""
        if fingerprint:
            fingerprint_key = DatasetManager._fingerprint_key(
                fingerprint,
                normalized_lighting,
                aoi_info,
            )
            existing = DatasetManager._fingerprint_index(
                target_folder
            ).get(fingerprint_key)
            if existing is not None:
                if normalized_label == "OK":
                    return DatasetManager._upgrade_duplicate_record(
                        existing,
                        fingerprint=fingerprint,
                        lighting_mode=normalized_lighting,
                        event_id=normalized_event_id,
                        anomaly_memory=anomaly_memory,
                        source=source,
                    )
                # NG continua sendo uma observação protegida individual. Apenas
                # a imagem pesada repetida é deduplicada.
                duplicate_visual_of = str(existing or "")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        lighting_suffix = (
            f"_{normalized_lighting}"
            if normalized_lighting
            else ""
        )
        filename = (
            f"memory_{normalized_label}{lighting_suffix}_{timestamp}"
        )
        filepath_test = target_folder / f"{filename}_test.png"
        filepath_reference = target_folder / f"{filename}_reference.png"
        filepath_source = target_folder / f"{filename}_source.png"
        filepath_json = target_folder / f"{filename}.json"

        test_image_file = ""
        reference_image_file = ""
        source_image_file = ""
        if (
            save_images
            and not duplicate_visual_of
            and isinstance(ng_image, np.ndarray)
            and ng_image.size > 0
        ):
            if cv2.imwrite(str(filepath_test), ng_image):
                test_image_file = filepath_test.name
            if (
                isinstance(sample_image, np.ndarray)
                and sample_image.size > 0
                and cv2.imwrite(str(filepath_reference), sample_image)
            ):
                reference_image_file = filepath_reference.name
            if (
                isinstance(source_frame, np.ndarray)
                and source_frame.size > 0
                and cv2.imwrite(str(filepath_source), source_frame)
            ):
                source_image_file = filepath_source.name

        info = aoi_info if isinstance(aoi_info, dict) else {}
        semantic_debug = detail.get("semantic_debug") or {}
        semantic_reference = detail.get("ref_emb", [])
        semantic_query = detail.get("query_emb", [])
        legacy_embedding = detail.get("query_embedding", [])

        metadata = {
            "schema": "visionx.memory.v3",
            "label": normalized_label,
            "timestamp": datetime.now().isoformat(),
            "storage": {
                "mode": "json_plus_audit_images" if save_images else "json_only",
                "test_image_file": test_image_file,
                "reference_image_file": reference_image_file,
                "source_image_file": source_image_file,
                "test_image_fingerprint": fingerprint,
                "images_required_for_knn": False,
                "full_test_area_preserved": bool(test_image_file),
                "raw_aoi_frame_preserved": bool(source_image_file),
                "visual_deduplicated": bool(duplicate_visual_of),
                "duplicate_visual_of_json": duplicate_visual_of,
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
                "lighting_mode": normalized_lighting,
            },
            "multilight": {
                "event_id": normalized_event_id,
                "lighting_mode": normalized_lighting,
                "same_piece_single_judgement": bool(
                    normalized_event_id and normalized_lighting
                ),
                "shared_ocr": {
                    "board": info.get("board", ""),
                    "parts": info.get("parts", ""),
                    "category": category,
                    "value": info.get("value", ""),
                },
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
                "lighting_mode": normalized_lighting,
                "final_multilight": {
                    "verdict": (final_analysis or {}).get("verdict", ""),
                    "is_defect": (final_analysis or {}).get(
                        "is_defect",
                        False,
                    ),
                    "confidence": (final_analysis or {}).get(
                        "confidence",
                        0.0,
                    ),
                    "fusion_rule": (
                        ((final_analysis or {}).get("detail", {}) or {}).get(
                            "fusion_rule",
                            "",
                        )
                    ),
                },
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
                    "inverted": {
                        "score": detail.get("inverted_score", 0),
                        "classification": detail.get(
                            "inverted_classification",
                            "",
                        ),
                        "signature_strength": detail.get(
                            "inverted_signature_strength",
                            0,
                        ),
                        "direct_similarity": detail.get(
                            "inverted_direct_similarity",
                            1,
                        ),
                        "feature_loss": detail.get(
                            "inverted_feature_loss",
                            0,
                        ),
                        "extra_structure": detail.get(
                            "inverted_extra_structure",
                            0,
                        ),
                        "topology_mismatch": detail.get(
                            "inverted_topology_mismatch",
                            0,
                        ),
                        "orientation_mismatch": detail.get(
                            "inverted_orientation_mismatch",
                            0,
                        ),
                        "alternate_face_signal": detail.get(
                            "inverted_alternate_face_signal",
                            0,
                        ),
                        "changed_coverage": detail.get(
                            "inverted_changed_coverage",
                            0,
                        ),
                        "orientation": {
                            "expected_angle": detail.get(
                                "inverted_expected_angle",
                                0,
                            ),
                            "observed_angle": detail.get(
                                "inverted_observed_angle",
                                0,
                            ),
                            "reference_histogram": detail.get(
                                "inverted_orientation_hist_reference",
                                [],
                            ),
                            "test_histogram": detail.get(
                                "inverted_orientation_hist_test",
                                [],
                            ),
                        },
                        "best_transform": {
                            "name": detail.get(
                                "inverted_best_transform",
                                "none",
                            ),
                            "similarity": detail.get(
                                "inverted_best_transform_similarity",
                                0,
                            ),
                            "gain": detail.get(
                                "inverted_transform_gain",
                                0,
                            ),
                        },
                        "edge_grid_reference": detail.get(
                            "inverted_edge_grid_reference",
                            [],
                        ),
                        "edge_grid_test": detail.get(
                            "inverted_edge_grid_test",
                            [],
                        ),
                        "polarity_grid_reference": detail.get(
                            "inverted_polarity_grid_reference",
                            [],
                        ),
                        "polarity_grid_test": detail.get(
                            "inverted_polarity_grid_test",
                            [],
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

        DatasetManager._register_fingerprint(
            target_folder,
            fingerprint,
            filepath_json,
            normalized_lighting,
            aoi_info,
        )
        return str(filepath_json)
