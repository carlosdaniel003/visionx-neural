"""Recuperação visual de julgamentos humanos duráveis, sem reclassificar por si só.

Fonte: registros OPERATOR_CONFIRMED do snapshot persistente. A KNN C
indexa somente PNGs do acervo, então esses julgamentos podem faltar em
seus vizinhos. Esta consulta mantém uma visão RAM por geração do JSON
e compara assinaturas das diferenças gabarito/teste mesmo com variação
mínima de pixels ou metadado de iluminação.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import RLock

import numpy as np

from src.core.anomaly_signature import (
    GROUP_WEIGHTS, RANGES, VECTOR_SIZE,
    build_anomaly_signature, valid_anomaly_signature,
)
from src.core.strict_category_memory import canonical_memory_category

SCHEMA = "visionx.human_visual_recall.v1"
MIN_VISUAL_SCORE = 0.975
MIN_OPPOSITE_MARGIN = 0.04


class ConfirmedVisualRecall:
    """Busca vetorizada, somente leitura, de assinaturas humanas em RAM."""

    def __init__(self, store):
        self.store = store
        self._lock = RLock()
        self._stamp = None
        self._groups: dict[str, tuple[np.ndarray, list[dict]]] = {}

    def _ensure_index(self) -> None:
        path = getattr(self.store, "path", None)
        if not isinstance(path, Path):
            raise ValueError("Snapshot persistente sem caminho verificável")
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except FileNotFoundError:
            stamp = None
        with self._lock:
            if self._stamp == stamp and self._groups:
                return
            # records() verifica checksum, IDs, identidades e proveniência.
            records = self.store.records()
            grouped = defaultdict(list)
            for record in records:
                if not isinstance(record, dict):
                    continue
                label = record.get("label")
                sources = record.get("sources")
                key = record.get("identity")
                signature = record.get("anomaly_signature")
                if (label not in {"OK", "NG"} or not isinstance(key, list)
                        or len(key) != 4 or not isinstance(sources, list)
                        or not valid_anomaly_signature(signature)
                        or not any(
                            isinstance(source, dict)
                            and source.get("source_kind") == "OPERATOR_CONFIRMED"
                            and source.get("source_label") == label
                            for source in sources
                        )):
                    continue
                category = canonical_memory_category(key[0])
                if not category or category in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
                    continue
                grouped[category].append((
                    np.asarray(signature["vector"], dtype=np.float32),
                    {"label": label, "lighting_mode": key[1],
                     "record_id": record["id"],
                     "source": "OPERATOR_CONFIRMED"},
                ))
            built = {}
            for category, rows in grouped.items():
                built[category] = (
                    np.stack([row[0] for row in rows]),
                    [row[1] for row in rows],
                )
            # Uma escrita concorrente nunca publica uma geração parcial.
            try:
                updated = path.stat()
                new_stamp = (updated.st_mtime_ns, updated.st_ctime_ns, updated.st_size)
            except FileNotFoundError:
                new_stamp = None
            if new_stamp != stamp:
                raise ValueError("Snapshot modificado durante indexação")
            self._groups = built
            self._stamp = stamp

    @staticmethod
    def _similarities(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Equivalente vetorizado a compare_anomaly_signatures."""
        count = candidates.shape[0]
        weighted = np.zeros(count, dtype=np.float64)
        total = np.zeros(count, dtype=np.float64)
        for name, (start, stop) in RANGES.items():
            q = query[start:stop].astype(np.float64)
            values = candidates[:, start:stop].astype(np.float64)
            qnorm = float(np.linalg.norm(q))
            norms = np.linalg.norm(values, axis=1)
            active = (norms > 1e-8) | (qnorm > 1e-8)
            cosine = np.zeros(count, dtype=np.float64)
            good = (norms > 1e-9) & (qnorm > 1e-9)
            if good.any():
                cosine[good] = np.clip(
                    (values[good] @ q / (norms[good] * qnorm) + 1) / 2,
                    0.0, 1.0,
                )
            magnitude = np.clip(
                1.0 - np.mean(np.abs(values - q), axis=1), 0.0, 1.0,
            )
            score = cosine * 0.60 + magnitude * 0.40
            weight = GROUP_WEIGHTS[name]
            weighted += score * weight * active
            total += weight * active
        return np.divide(
            weighted, total, out=np.ones(count, dtype=np.float64),
            where=total > 1e-9,
        )

    def lookup(self, reference, test, info: dict) -> dict:
        result = {
            "schema": SCHEMA, "status": "NO_MATCH",
            "diagnostic_only": True, "auto_ok_authorized": False,
            "source": "PERSISTENT_HUMAN_LABELS",
        }
        category = canonical_memory_category(info.get("category", ""))
        if category in {"", "UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
            return {**result, "status": "UNAVAILABLE", "reason": "UNKNOWN_CATEGORY"}
        try:
            query = build_anomaly_signature(
                reference, test, {}, {"category": category},
            )
            if not valid_anomaly_signature(query):
                raise ValueError("INVALID_VISUAL_SIGNATURE")
            # Um mapa de diferença praticamente vazio não distingue OK/NG.
            if (query["summary"]["map_peak"] < 0.1
                    or query["summary"]["map_coverage"] < .015):
                return {**result, "reason": "NON_DISCRIMINATIVE_VISUAL_MAP"}
            self._ensure_index()
            with self._lock:
                pool = self._groups.get(category)
                if pool is None:
                    return {**result, "reason": "NO_HUMAN_RECORDS_IN_CATEGORY"}
                vectors, metadata = pool
                scores = self._similarities(
                    np.asarray(query["vector"], dtype=np.float32), vectors,
                )
                best = {}
                for i, entry in enumerate(metadata):
                    label = entry["label"]
                    if label not in best or scores[i] > best[label][0]:
                        best[label] = (float(scores[i]), dict(entry))
            score_ok, ok = best.get("OK", (0.0, None))
            score_ng, ng = best.get("NG", (0.0, None))
            if (score_ok >= MIN_VISUAL_SCORE
                    and score_ok - score_ng >= MIN_OPPOSITE_MARGIN):
                status = "HUMAN_OK_VISUAL_CANDIDATE"
            elif (score_ng >= MIN_VISUAL_SCORE
                    and score_ng - score_ok >= MIN_OPPOSITE_MARGIN):
                status = "HUMAN_NG_VISUAL_CANDIDATE"
            elif max(score_ok, score_ng) >= MIN_VISUAL_SCORE:
                status = "AMBIGUOUS"
            else:
                status = "NO_MATCH"
            return {
                **result, "status": status, "category": category,
                "best_ok_similarity": round(score_ok, 8),
                "best_ng_similarity": round(score_ng, 8),
                "best_ok_source": ok, "best_ng_source": ng,
                "records_considered": len(metadata),
                "cross_light_candidate": bool(
                    ok is not None and ok["lighting_mode"] !=
                    str(info.get("lighting_mode", "")).upper()
                ) if status == "HUMAN_OK_VISUAL_CANDIDATE" else False,
                "reason": "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY",
            }
        except (ValueError, OSError, TypeError, KeyError) as exc:
            return {**result, "status": "UNAVAILABLE",
                    "reason": type(exc).__name__}


__all__ = ["ConfirmedVisualRecall", "SCHEMA"]
