# src/core/experts/knn_expert.py
"""KNN de memória visual orientado à anomalia, com fallback legado."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from numpy.linalg import norm

from src.config.settings import settings
from src.core.anomaly_signature import (
    compare_anomaly_signatures,
    valid_anomaly_signature,
)


class KNNExpert:
    """Compara primeiro a divergência teste-gabarito, não a peça completa."""

    def __init__(self):
        print("Inicializando K-NN de anomalias...")
        self.net = None
        self.signatures_ok: list[dict] = []
        self.signatures_ng: list[dict] = []
        self._load_all()

    def _ensure_legacy_model(self):
        if self.net is not None:
            return self.net
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mobilenetv2-7.onnx"

        if not model_path.exists():
            print("Modelo legado não encontrado. Baixando MobileNetV2 (13 MB)...")
            url = (
                "https://github.com/onnx/models/raw/main/validated/vision/"
                "classification/mobilenet/model/mobilenetv2-7.onnx"
            )
            urllib.request.urlretrieve(url, str(model_path))
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        return self.net

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
    def _weighted_vote(neighbors: list[tuple]) -> float:
        """Vota pela classe usando a distância, sem confundir classe e similaridade."""
        if not neighbors:
            return 0.5

        votes_ng = 0.0
        votes_total = 0.0
        for neighbor in neighbors:
            distance = float(neighbor[0])
            label = str(neighbor[1]).upper()
            weight = 1.0 / max(distance, 0.0001)
            votes_total += weight
            if label == "NG":
                votes_ng += weight

        if votes_total <= 0.0:
            return 0.5
        return float(np.clip(votes_ng / votes_total, 0.0, 1.0))

    @staticmethod
    def _extract_anomaly_memory(data: dict) -> dict | None:
        analysis = data.get("analysis", {})
        candidates = (
            analysis.get("anomaly_memory") if isinstance(analysis, dict) else None,
            analysis.get("anomaly_signature") if isinstance(analysis, dict) else None,
            data.get("anomaly_memory"),
        )
        for candidate in candidates:
            if valid_anomaly_signature(candidate):
                return candidate
        return None

    def _load_all(self):
        print("Varredura da memória de anomalias...")
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

                    aoi_info = data.get("aoi_info", {})
                    part_name = self._clean_string(aoi_info.get("parts", ""))
                    category_name = self._clean_string(
                        aoi_info.get("category", "Unknown")
                    )
                    resolved_label = self._resolve_record_label(data, folder_label)
                    anomaly_memory = self._extract_anomaly_memory(data)

                    analysis = data.get("analysis", {})
                    legacy_embedding = (
                        analysis.get("embedding", [])
                        if isinstance(analysis, dict)
                        else []
                    )
                    legacy_signature = None
                    if legacy_embedding:
                        candidate = np.asarray(
                            legacy_embedding,
                            dtype=np.float32,
                        ).reshape(-1)
                        if candidate.size and np.all(np.isfinite(candidate)):
                            legacy_signature = candidate

                    if anomaly_memory is None and legacy_signature is None:
                        continue

                    image_file = str(data.get("image_file", ""))
                    match_path = (
                        str(json_path.parent / image_file)
                        if image_file
                        else str(json_path)
                    )
                    record = {
                        "part": part_name,
                        "category": category_name,
                        "path": match_path,
                        "json_path": str(json_path),
                        "label": resolved_label,
                        "folder_label": folder_label,
                        "mode": "anomaly" if anomaly_memory is not None else "legacy_image",
                        "anomaly_signature": anomaly_memory,
                        "sig": legacy_signature,
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

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray | None:
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
            or image.shape[0] < 5
            or image.shape[1] < 5
        ):
            return None
        net = self._ensure_legacy_model()
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        return net.forward().flatten().astype(np.float32)

    @staticmethod
    def _cosine_similarity(
        query_sig: np.ndarray,
        stored_sig: np.ndarray,
    ) -> float | None:
        if query_sig.size != stored_sig.size:
            return None
        denominator = float(norm(query_sig) * norm(stored_sig))
        if denominator <= 1e-12:
            return None
        similarity = float(np.dot(query_sig, stored_sig) / denominator)
        if not np.isfinite(similarity):
            return None
        return float(np.clip(similarity, -1.0, 1.0))

    @staticmethod
    def _filter_by_mode(records: list[dict], mode: str) -> list[dict]:
        return [record for record in records if record.get("mode") == mode]

    def _context_candidates(
        self,
        mode: str,
        target_part: str,
        target_category: str,
    ) -> tuple[list[dict], list[dict], str]:
        ok_records = self._filter_by_mode(self.signatures_ok, mode)
        ng_records = self._filter_by_mode(self.signatures_ng, mode)

        def select(predicate):
            return (
                [item for item in ok_records if predicate(item)],
                [item for item in ng_records if predicate(item)],
            )

        # A categoria da anomalia é o filtro principal. O componente não limita
        # a busca quando já existem memórias da mesma classe de defeito.
        if target_category:
            category_ok, category_ng = select(
                lambda item: item.get("category") == target_category
            )
            if category_ok or category_ng:
                return category_ok, category_ng, "categoria"

        if target_part:
            part_ok, part_ng = select(
                lambda item: target_part in item.get("part", "")
            )
            if part_ok or part_ng:
                return part_ok, part_ng, "componente"

        return list(ok_records), list(ng_records), "global"

    @staticmethod
    def _empty_result(
        query_anomaly_signature: dict | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> dict:
        return {
            "has_memory": False,
            "vote_defect": 0.5,
            "best_similarity": 0.0,
            "n_neighbors": 0,
            "best_match_label": "",
            "neighbor_details": [],
            "memory_mode": "none",
            "memory_scope": "none",
            "query_anomaly_signature": query_anomaly_signature or {},
            "query_embedding": (
                query_embedding.tolist()
                if isinstance(query_embedding, np.ndarray)
                else []
            ),
        }

    def _analyze_anomaly_memory(
        self,
        query_signature: dict,
        valid_ok: list[dict],
        valid_ng: list[dict],
        top_k: int,
        scope: str,
    ) -> dict:
        distances: list[tuple[float, str, str, dict]] = []
        for label, records in (("OK", valid_ok), ("NG", valid_ng)):
            for item in records:
                stored = item.get("anomaly_signature")
                if not valid_anomaly_signature(stored):
                    continue
                similarity, breakdown = compare_anomaly_signatures(
                    query_signature,
                    stored,
                )
                distances.append(
                    (
                        1.0 - similarity,
                        label,
                        item["path"],
                        breakdown,
                    )
                )

        if not distances:
            return self._empty_result(query_anomaly_signature=query_signature)

        distances.sort(key=lambda item: item[0])
        neighbors = distances[: max(1, int(top_k))]
        vote_defect = self._weighted_vote(neighbors)

        best_dist, best_label, best_path, best_breakdown = neighbors[0]
        best_similarity = float(np.clip(1.0 - best_dist, 0.0, 1.0))
        if len(neighbors) == 1:
            vote_defect = 1.0 if best_label == "NG" else 0.0

        neighbor_details = []
        for distance, label, path, breakdown in neighbors:
            neighbor_details.append(
                {
                    "label": label,
                    "similarity": float(np.clip(1.0 - distance, 0.0, 1.0)),
                    "distance": float(distance),
                    "path": str(path),
                    "similarity_breakdown": breakdown,
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
            "memory_mode": "anomaly",
            "memory_scope": scope,
            "memory_schema": str(query_signature.get("schema", "")),
            "similarity_breakdown": best_breakdown,
            "query_anomaly_signature": query_signature,
            "query_embedding": [],
        }

    def _analyze_legacy_memory(
        self,
        query_image: np.ndarray,
        valid_ok: list[dict],
        valid_ng: list[dict],
        top_k: int,
        scope: str,
        query_anomaly_signature: dict | None,
    ) -> dict:
        query_sig = self._compute_embedding(query_image)
        if query_sig is None:
            return self._empty_result(
                query_anomaly_signature=query_anomaly_signature,
            )

        distances: list[tuple[float, str, str]] = []
        for label, records in (("OK", valid_ok), ("NG", valid_ng)):
            for item in records:
                stored = item.get("sig")
                if not isinstance(stored, np.ndarray):
                    continue
                similarity = self._cosine_similarity(query_sig, stored)
                if similarity is None:
                    continue
                distances.append((1.0 - similarity, label, item["path"]))

        if not distances:
            return self._empty_result(
                query_anomaly_signature=query_anomaly_signature,
                query_embedding=query_sig,
            )

        distances.sort(key=lambda item: item[0])
        neighbors = distances[: max(1, int(top_k))]
        vote_defect = self._weighted_vote(neighbors)
        best_dist, best_label, best_path = neighbors[0]
        best_similarity = float(np.clip(1.0 - best_dist, 0.0, 1.0))
        if len(neighbors) == 1:
            vote_defect = 1.0 if best_label == "NG" else 0.0

        return {
            "has_memory": True,
            "vote_defect": float(vote_defect),
            "best_similarity": best_similarity,
            "n_neighbors": int(len(neighbors)),
            "best_match_path": str(best_path),
            "best_match_label": str(best_label),
            "neighbor_details": [
                {
                    "label": label,
                    "similarity": float(np.clip(1.0 - distance, 0.0, 1.0)),
                    "distance": float(distance),
                    "path": str(path),
                }
                for distance, label, path in neighbors
            ],
            "memory_label_counts": {
                "OK": int(sum(item[1] == "OK" for item in neighbors)),
                "NG": int(sum(item[1] == "NG" for item in neighbors)),
            },
            "memory_mode": "legacy_image",
            "memory_scope": scope,
            "query_anomaly_signature": query_anomaly_signature or {},
            "query_embedding": query_sig.tolist(),
        }

    def analyze(
        self,
        full_gab: np.ndarray,
        full_test: np.ndarray,
        crop_gab: np.ndarray = None,
        crop_test: np.ndarray = None,
        aoi_info: dict = None,
        top_k: int = 5,
        anomaly_signature: dict | None = None,
    ) -> dict:
        try:
            info = aoi_info if isinstance(aoi_info, dict) else {}
            target_part = self._clean_string(info.get("parts", ""))
            target_category = self._clean_string(info.get("category", ""))

            if valid_anomaly_signature(anomaly_signature):
                valid_ok, valid_ng, scope = self._context_candidates(
                    "anomaly",
                    target_part,
                    target_category,
                )
                if valid_ok or valid_ng:
                    return self._analyze_anomaly_memory(
                        anomaly_signature,
                        valid_ok,
                        valid_ng,
                        top_k,
                        scope,
                    )

            valid_ok, valid_ng, scope = self._context_candidates(
                "legacy_image",
                target_part,
                target_category,
            )
            if not valid_ok and not valid_ng:
                return self._empty_result(
                    query_anomaly_signature=anomaly_signature,
                )

            query_image = (
                full_test
                if isinstance(full_test, np.ndarray) and full_test.size > 0
                else crop_test
            )
            return self._analyze_legacy_memory(
                query_image,
                valid_ok,
                valid_ng,
                top_k,
                scope,
                anomaly_signature,
            )
        except Exception as exc:
            print(f"Erro no KNNExpert: {exc}")
            return self._empty_result(
                query_anomaly_signature=anomaly_signature,
            )
