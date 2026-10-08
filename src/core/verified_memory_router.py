"""Roteamento: memória KNN *verificada* primeiro, especialidade para casos novos.

Um match aproximado de embedding/anomaly_signature NÃO significa caso visto.
O atalho só reconhece RGB idêntico do PAR gabarito/teste arquivado,
com board, parts, categoria, iluminação e rótulo do operador coerentes.
Não modifica a memória nem retreina a CNN. Instalar após faltando_live.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.image_archive_dedup import image_fingerprint

RECOGNITION_SCHEMA = "visionx.memory_first_routing.v1"


def _clean(value) -> str:
    return "".join(c for c in str(value or "").upper() if c.isalnum())


def _image_ok(image) -> bool:
    return (
        isinstance(image, np.ndarray) and image.ndim == 3
        and image.shape[-1] == 3 and min(image.shape[:2]) >= 12
        and image.dtype == np.uint8 and image.size > 0
    )


def _file_within(parent: Path, name: str) -> Path | None:
    """Somente o PNG em seu próprio diretório, sem traversal ou links."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    target = parent / name
    if target.suffix.lower() != ".png" or target.is_symlink() or not target.is_file():
        return None
    return target


def _fingerprint_png(path: Path) -> str:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if not _image_ok(image):
        return ""
    return image_fingerprint(image)


class VerifiedKNNMemory:
    """Índice da memória KNN, apenas entradas humanas com dois PNGs auditáveis."""

    def __init__(self):
        self._lock = RLock()
        self._index: dict[tuple, list[dict]] | None = None
        self._record_identity: tuple | None = None
        self.error = ""

    def invalidate(self) -> None:
        with self._lock:
            self._index = None
            self._record_identity = None
            self.error = ""

    @staticmethod
    def _key(info: dict, reference_hash: str, test_hash: str) -> tuple | None:
        category = canonical_memory_category(info.get("category", ""))
        board = _clean(info.get("board"))
        parts = _clean(info.get("parts"))
        light = canonical_memory_lighting(info.get("lighting_mode", ""))
        if not all((category, board, parts, reference_hash, test_hash)):
            return None
        return category, light, board, parts, reference_hash, test_hash

    @staticmethod
    def _entry(record: dict) -> tuple[tuple, dict] | None:
        if record.get("mode") != "anomaly":
            return None
        path = Path(str(record.get("json_path", "")))
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "visionx.memory.v3":
            return None
        label = str(data.get("label", "")).strip().upper()
        operator = str((data.get("decision") or {}).get("operator_label", "")).strip().upper()
        if (label not in ("OK", "NG")
                or operator != label
                or str(record.get("label", "")).strip().upper() != label
                or str(record.get("folder_label", "")).strip().upper() != label):
            return None
        info = data.get("aoi_info", {})
        storage = data.get("storage", {})
        if not isinstance(info, dict) or not isinstance(storage, dict):
            return None
        test_path = _file_within(path.parent, storage.get("test_image_file", ""))
        ref_path = _file_within(path.parent, storage.get("reference_image_file", ""))
        if test_path is None or ref_path is None:
            # JSON-only / dedup visual legado: não assumir mesmo gabarito.
            return None
        reference_hash = _fingerprint_png(ref_path)
        test_hash = _fingerprint_png(test_path)
        if not reference_hash or not test_hash:
            return None
        declared = str(storage.get("test_image_fingerprint", "")).strip()
        if declared and declared != test_hash:
            return None
        key = VerifiedKNNMemory._key(info, reference_hash, test_hash)
        if key is None:
            return None
        if (
            canonical_memory_category(record.get("category")) != key[0]
            or canonical_memory_lighting(record.get("lighting_mode")) != key[1]
            or _clean(record.get("part")) != key[3]
        ):
            return None
        return key, {"label": label, "source_json": str(path)}

    def lookup(self, knn, reference, test, info: dict) -> dict:
        """Retorna KNOWN/NEW/CONFLICT/UNAVAILABLE, sem similaridade aproximada."""
        if not _image_ok(reference) or not _image_ok(test):
            return {"status": "UNAVAILABLE", "reason": "Par AOI inválido"}
        key = self._key(
            info, image_fingerprint(reference), image_fingerprint(test)
        )
        if key is None:
            return {"status": "NEW", "reason": "OCR/categoria ou par incompleto"}
        if knn is None:
            return {"status": "NEW", "reason": "Nenhuma memória KNN carregada"}

        lock = getattr(knn, "_memory_lock", None)
        if lock is not None:
            with lock:
                records = list(getattr(knn, "signatures_ok", ()) or ()) + list(
                    getattr(knn, "signatures_ng", ()) or ()
                )
        else:
            records = list(getattr(knn, "signatures_ok", ()) or ()) + list(
                getattr(knn, "signatures_ng", ()) or ()
            )
        identity = (id(knn), id(getattr(knn, "signatures_ok", None)),
                    id(getattr(knn, "signatures_ng", None)),
                    len(records))
        with self._lock:
            if self._index is None or identity != self._record_identity:
                built = defaultdict(list)
                for record in records:
                    try:
                        entry = self._entry(record)
                        if entry is not None:
                            built[entry[0]].append(entry[1])
                    except (OSError, ValueError, TypeError, cv2.error):
                        # Um registro ruim não autoriza atalho de memória.
                        continue
                self._index = dict(built)
                self._record_identity = identity

            matched = list(self._index.get(key, ()))
        if not matched:
            return {
                "status": "NEW",
                "reason": "Par exato não encontrado nos PNGs humanos da memória KNN",
            }
        labels = {item["label"] for item in matched}
        if len(labels) != 1:
            return {
                "status": "CONFLICT", "reason": "Par idêntico rotulado como OK e NG",
                "matches": len(matched),
            }
        return {
            "status": "KNOWN", "label": labels.pop(),
            "source_json": matched[0]["source_json"], "matches": len(matched),
            "reason": "Par exato confirmado no índice de memória KNN",
        }


def _recognized_result(match: dict, mode: str) -> dict:
    label = match["label"]
    ng = label == "NG"
    score = 1.0 if ng else 0.0
    trace = {
        "schema": RECOGNITION_SCHEMA,
        "dominant_engine": "knn_verified_exact",
        "fusion_rule": "knn_verified_pair_exact",
        "final_score": score,
        "physical_score": 0.0,
        "cutoff": .5,
        "operator_review_required": False,
        "weights": {"physical": 0.0, "knn": 1.0, "cnn": 0.0},
        "recognition_route": "KNOWN_KNN",
        "memory": {
            "has_memory": True, "best_similarity": 1.0,
            "best_match_label": label, "best_ok_similarity": 0.0 if ng else 1.0,
            "best_ng_similarity": 1.0 if ng else 0.0,
            "role": "Correspondência exata verificada, sem votação aproximada",
        },
    }
    return {
        "is_defect": ng, "confidence": .99,
        "verdict": "DEFEITO REAL" if ng else "FALHA FALSA",
        "reason": (
            f"MEMÓRIA KNN: caso já analisado e confirmado como {label}. "
            "Gabarito/teste, placa, componente, categoria e iluminação idênticos."
        ),
        "production_review_required": False,
        "active_engines": ["knn_expert.py"],
        "bounding_box": None, "all_boxes": {},
        "lighting_mode": mode,
        "detail": {
            "recognition_route": "KNOWN_KNN",
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_match": "EXACT_PAIR",
            "recognition_known_label": label,
            "recognition_memory_path": match["source_json"],
            "recognition_memory_matches": match["matches"],
            "recognition_memory_verified": True,
            "recognition_specialists_skipped": True,
            "has_memory": True, "memory_available": True,
            "best_similarity": 1.0, "best_match_label": label,
            "best_ok_similarity": 0.0 if ng else 1.0,
            "best_ng_similarity": 1.0 if ng else 0.0,
            "memory_scope": "categoria_e_luz_exata",
            "memory_mode": "verified_pair",
            "vote_defect": score,
            "final_score": score, "physical_score": 0.0,
            "fusion_rule": "knn_verified_pair_exact",
            "dominant_engine": "knn_verified_exact",
            "operator_review_required": False,
            "decision_trace": trace,
        },
    }


def _review_result(reason: str, mode: str) -> dict:
    return {
        "is_defect": False, "confidence": .5,
        "verdict": "REVISÃO OBRIGATÓRIA",
        "reason": "MEMÓRIA KNN: " + reason,
        "production_review_required": True,
        "active_engines": ["knn_expert.py"],
        "bounding_box": None, "all_boxes": {},
        "lighting_mode": mode,
        "detail": {
            "recognition_route": "MEMORY_CONFLICT",
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_specialists_skipped": True,
            "final_score": .5, "physical_score": 0.0,
            "dominant_engine": "knn_verified_exact",
            "fusion_rule": "knn_label_conflict_review",
            "operator_review_required": True,
            "decision_trace": {
                "operator_review_required": True,
                "fusion_rule": "knn_label_conflict_review",
                "final_score": .5,
                "physical_score": 0.0,
            },
        },
    }


def install_memory_first_router(orchestrator_cls, *, memory=None) -> None:
    """Último wrapper: KNOWN usa rótulo humano KNN; NEW especialista sem KNN."""
    if getattr(orchestrator_cls, "_memory_first_router_installed", False):
        return
    original = orchestrator_cls.inspect
    original_reload = orchestrator_cls.reload_memory

    def inspect(
        self, full_gab, full_test, raw_anomalies,
        aoi_info, global_box_info, aoi_epicenters,
    ):
        info = aoi_info if isinstance(aoi_info, dict) else {}
        if (
            info.get("_replay_without_memory", False)
            or type(self).__name__ == "PhysicalOnlyOrchestrator"
        ):
            return original(
                self, full_gab, full_test, raw_anomalies,
                aoi_info, global_box_info, aoi_epicenters
            )

        index = getattr(self, "_memory_first_index", None)
        if index is None:
            index = memory if memory is not None else VerifiedKNNMemory()
            self._memory_first_index = index

        try:
            match = index.lookup(
                getattr(self, "experts", {}).get("knn"),
                full_gab, full_test, info,
            )
        except Exception as exc:
            # Índice indisponível: não reconhecer; especialista permanece
            # necessário. Não usar KNN por semelhança como atalho.
            match = {"status": "UNAVAILABLE", "reason": str(exc)}
        mode = canonical_memory_lighting(info.get("lighting_mode", "SIDE"))
        if match["status"] == "UNAVAILABLE":
            return _review_result(
                "Verificação exata indisponível: " + str(match.get("reason", "")),
                mode,
            )
        if match["status"] == "KNOWN":
            return _recognized_result(match, mode)
        if match["status"] == "CONFLICT":
            return _review_result(match["reason"], mode)

        category = canonical_memory_category(info.get("category", ""))
        if category == "FALTANDO":
            route = "NEW_CNN"
            # O FaltandoCNNLive é o wrapper anterior, não consulta KNN.
            requested_info = dict(info)
        else:
            route = "NEW_EXPERTS"
            requested_info = dict(info)
            # A extensão física aceita este marcador: dispensa KNN.
            # Em replay offline original o flag é preservado.
            requested_info["_replay_without_memory"] = True
        result = original(
            self, full_gab, full_test, raw_anomalies,
            requested_info, global_box_info, aoi_epicenters,
        )
        if not isinstance(result, dict):
            return _review_result("O especialista não retornou um resultado", mode)
        detail = result.setdefault("detail", {})
        if not isinstance(detail, dict):
            detail = {}
            result["detail"] = detail
        detail.update({
            "recognition_route": route,
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "recognition_specialists_skipped": False,
            "recognition_reason": str(match.get("reason", "")),
        })
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["recognition_route"] = route
        if route == "NEW_CNN":
            result["reason"] = "CASO NOVO • CNN FALTANDO v2. " + str(result.get("reason", ""))
        else:
            result["reason"] = "CASO NOVO • motores da categoria. " + str(result.get("reason", ""))
        return result

    def reload_memory(self, *args, **kwargs):
        outcome = original_reload(self, *args, **kwargs)
        cached = getattr(self, "_memory_first_index", None)
        if cached is not None:
            cached.invalidate()
        return outcome

    orchestrator_cls.inspect = inspect
    orchestrator_cls.reload_memory = reload_memory
    orchestrator_cls._memory_first_router_installed = True


__all__ = ["VerifiedKNNMemory", "install_memory_first_router", "RECOGNITION_SCHEMA"]
