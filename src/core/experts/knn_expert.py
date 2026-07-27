# src/core/experts/knn_expert.py
"""Especialista KNN baseado em embeddings MobileNetV2."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from numpy.linalg import norm

from src.config.settings import settings


class KNNExpert:
    def __init__(self):
        print("Inicializando K-NN Expert (cv2.dnn)...")
        self.net = self._load_mobilenet_model()
        self.signatures_ok: list[dict] = []
        self.signatures_ng: list[dict] = []
        self._load_all()

    def _load_mobilenet_model(self):
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mobilenetv2-7.onnx"

        if not model_path.exists():
            print("Modelo não encontrado. Baixando MobileNetV2 (13 MB)...")
            url = (
                "https://github.com/onnx/models/raw/main/validated/vision/"
                "classification/mobilenet/model/mobilenetv2-7.onnx"
            )
            urllib.request.urlretrieve(url, str(model_path))
        return cv2.dnn.readNetFromONNX(str(model_path))

    @staticmethod
    def _clean_string(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"[^A-Z0-9]", "", str(text).upper())

    @staticmethod
    def _resolve_record_label(data: dict, folder_label: str) -> str:
        """Prioriza o rótulo persistido no JSON e usa a pasta como fallback."""
        explicit_label = str(data.get("label", "")).strip().upper()
        if explicit_label in {"OK", "NG"}:
            return explicit_label

        analysis = data.get("analysis", {})
        if isinstance(analysis, dict) and "operator_label" in analysis:
            analysis_label = str(analysis.get("operator_label", "")).strip().upper()
            if analysis_label in {"OK", "NG"}:
                return analysis_label

        normalized_folder = str(folder_label or "").strip().upper()
        return normalized_folder if normalized_folder in {"OK", "NG"} else "OK"

    @staticmethod
    def _weighted_vote(neighbors: list[tuple[float, str, str]]) -> float:
        """Calcula o voto de classe; similaridade e voto permanecem conceitos distintos."""
        if not neighbors:
            return 0.5

        votes_ng = 0.0
        votes_total = 0.0
        for distance, label, _path in neighbors:
            weight = 1.0 / max(float(distance), 0.0001)
            votes_total += weight
            if str(label).upper() == "NG":
                votes_ng += weight

        if votes_total <= 0.0:
            return 0.5
        return float(np.clip(votes_ng / votes_total, 0.0, 1.0))

    def _load_all(self):
        print("Varredura Semântica KNN...")
        loaded_paths: set[str] = set()

        sources = (
            (settings.NORMAL_DIR, "OK"),
            (settings.ANOMALY_DIR, "NG"),
        )
        for folder, folder_label in sources:
            if not folder.exists():
                continue

            for json_path in folder.rglob("*.json"):
                if not json_path.is_file():
                    continue
                canonical_path = str(json_path.resolve())
                if canonical_path in loaded_paths:
                    continue

                try:
                    with open(json_path, "r", encoding="utf-8") as json_file:
                        data = json.load(json_file)

                    embedding_list = data.get("analysis", {}).get("embedding", [])
                    if not embedding_list:
                        continue

                    signature = np.asarray(embedding_list, dtype=np.float32).reshape(-1)
                    if signature.size == 0 or not np.all(np.isfinite(signature)):
                        continue

                    aoi_info = data.get("aoi_info", {})
                    part_name = self._clean_string(aoi_info.get("parts", ""))
                    category_name = self._clean_string(
                        aoi_info.get("category", "Unknown")
                    )
                    image_file = str(data.get("image_file", ""))
                    match_path = (
                        str(json_path.parent / image_file)
                        if image_file
                        else str(json_path)
                    )
                    resolved_label = self._resolve_record_label(data, folder_label)
                    record = {
                        "part": part_name,
                        "category": category_name,
                        "sig": signature,
                        "path": match_path,
                        "json_path": str(json_path),
                        "label": resolved_label,
                        "folder_label": folder_label,
                    }

                    if resolved_label == "NG":
                        self.signatures_ng.append(record)
                    else:
                        self.signatures_ok.append(record)
                    loaded_paths.add(canonical_path)
                except Exception as exc:
                    print(f"KNN ignorou JSON inválido {json_path}: {exc}")

    def reload_memory(self):
        self.signatures_ok = []
        self.signatures_ng = []
        self._load_all()

    def _compute_embedding(self, img: np.ndarray) -> np.ndarray | None:
        if (
            img is None
            or img.size == 0
            or img.shape[0] < 5
            or img.shape[1] < 5
        ):
            return None
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        return self.net.forward().flatten().astype(np.float32)

    @staticmethod
    def _cosine_similarity(query_sig: np.ndarray, stored_sig: np.ndarray) -> float | None:
        if query_sig.size != stored_sig.size:
            return None
        denominator = float(norm(query_sig) * norm(stored_sig))
        if denominator <= 1e-12:
            return None
        similarity = float(np.dot(query_sig, stored_sig) / denominator)
        if not np.isfinite(similarity):
            return None
        return float(np.clip(similarity, -1.0, 1.0))

    def analyze(
        self,
        full_gab: np.ndarray,
        full_test: np.ndarray,
        crop_gab: np.ndarray = None,
        crop_test: np.ndarray = None,
        aoi_info: dict = None,
        top_k: int = 5,
    ) -> dict:
        try:
            query_img = (
                full_test
                if full_test is not None and full_test.size > 0
                else crop_test
            )
            part_name = aoi_info.get("parts", "") if aoi_info else ""
            raw_category = aoi_info.get("category", "") if aoi_info else ""
            target_part = self._clean_string(part_name)
            target_category = self._clean_string(raw_category)

            strict_categories = {"SHIFTED", "UPSIDEDOWN", "REVERSE"}
            valid_ok = [
                item
                for item in self.signatures_ok
                if (not target_part or target_part in item["part"])
                and (not target_category or target_category == item.get("category", ""))
            ]
            valid_ng = [
                item
                for item in self.signatures_ng
                if (not target_part or target_part in item["part"])
                and (not target_category or target_category == item.get("category", ""))
            ]

            total_valid = len(valid_ok) + len(valid_ng)
            if total_valid == 0:
                if target_category in strict_categories:
                    print(
                        "KNN: fallback bloqueado. "
                        f"Categoria {raw_category} exige memória estrita."
                    )
                else:
                    valid_ok = [
                        item
                        for item in self.signatures_ok
                        if not target_part or target_part in item["part"]
                    ]
                    valid_ng = [
                        item
                        for item in self.signatures_ng
                        if not target_part or target_part in item["part"]
                    ]
                    total_valid = len(valid_ok) + len(valid_ng)

                    if total_valid == 0:
                        valid_ok = list(self.signatures_ok)
                        valid_ng = list(self.signatures_ng)
                        total_valid = len(valid_ok) + len(valid_ng)

            query_sig = self._compute_embedding(query_img)
            if total_valid == 0 or query_sig is None:
                return {
                    "has_memory": False,
                    "vote_defect": 0.5,
                    "best_similarity": 0.0,
                    "n_neighbors": 0,
                    "best_match_label": "",
                    "neighbor_details": [],
                    "query_embedding": query_sig.tolist() if query_sig is not None else [],
                }

            distances: list[tuple[float, str, str]] = []
            for label, records in (("OK", valid_ok), ("NG", valid_ng)):
                for item in records:
                    similarity = self._cosine_similarity(query_sig, item["sig"])
                    if similarity is None:
                        continue
                    distances.append((1.0 - similarity, label, item["path"]))

            if not distances:
                return {
                    "has_memory": False,
                    "vote_defect": 0.5,
                    "best_similarity": 0.0,
                    "n_neighbors": 0,
                    "best_match_label": "",
                    "neighbor_details": [],
                    "query_embedding": query_sig.tolist(),
                }

            distances.sort(key=lambda item: item[0])
            neighbors = distances[: max(1, int(top_k))]
            vote_defect = self._weighted_vote(neighbors)

            best_dist, best_label, best_path = neighbors[0]
            best_similarity = float(np.clip(1.0 - best_dist, 0.0, 1.0))

            # Um único vizinho representa integralmente a classe daquele registro.
            # A similaridade controla o peso da memória na fusão, não troca seu rótulo.
            if len(neighbors) == 1:
                vote_defect = 1.0 if best_label == "NG" else 0.0

            neighbor_details = []
            for distance, label, path in neighbors:
                similarity = float(np.clip(1.0 - distance, 0.0, 1.0))
                neighbor_details.append(
                    {
                        "label": label,
                        "similarity": similarity,
                        "distance": float(distance),
                        "path": str(path),
                    }
                )

            return {
                "has_memory": True,
                "vote_defect": float(vote_defect),
                "best_similarity": best_similarity,
                "n_neighbors": int(len(neighbors)),
                "best_match_path": str(best_path),
                "best_match_label": str(best_label),
                "neighbor_details": neighbor_details,
                "memory_label_counts": {
                    "OK": int(sum(item[1] == "OK" for item in neighbors)),
                    "NG": int(sum(item[1] == "NG" for item in neighbors)),
                },
                "query_embedding": query_sig.tolist(),
            }
        except Exception as exc:
            print(f"Erro no KNNExpert: {exc}")
            return {
                "has_memory": False,
                "vote_defect": 0.5,
                "best_similarity": 0.0,
                "n_neighbors": 0,
                "best_match_label": "",
                "neighbor_details": [],
                "query_embedding": [],
            }
