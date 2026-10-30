"""Concilia KNN C (PNGs) e memória humana com UMA mesma assinatura de anomalia.

A/B originais usam foco e geometria; por isso seus percentuais não são
comparáveis ao JSON humano. Este índice paralelo utiliza apenas assinaturas
persistidas de arquivos verificados e julgamentos OPERATOR_CONFIRMED.
Não reescala scores A/B nem cria identidade de pixels por aproximação.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from threading import RLock

import numpy as np

from src.core.anomaly_signature import build_anomaly_signature, valid_anomaly_signature
from src.core.confirmed_visual_recall import ConfirmedVisualRecall
from src.core.strict_category_memory import canonical_memory_category
from src.utils.text_normalizer import CATEGORIES

SCHEMA = "visionx.knn_c_archive_alignment.v1"
MIN_SHARED_SIMILARITY = 0.97
MIN_OPPOSITE_MARGIN = 0.04
CATEGORIES_CANONICAL = frozenset(canonical_memory_category(c) for c in CATEGORIES)


class KNNArchiveAlignment:
    """Índice vetorizado persistente por categoria, atualizado por geração."""

    def __init__(self, store):
        self.store = store
        self._lock = RLock()
        self._stamp = None
        self._pool = {}
        self._human = {}

    def _ensure(self):
        path = getattr(self.store, "path", None)
        if not isinstance(path, Path):
            raise ValueError("MISSING_MEMORY_PATH")
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except FileNotFoundError:
            raise ValueError("MISSING_MEMORY_FILE")
        with self._lock:
            if self._stamp == stamp:
                return
            archives, humans = {}, {}
            # Leitura já verifica checksum, fonte humana/PNG, ID e rótulos.
            for row in self.store.records():
                key, label = row["identity"], row["label"]
                category = canonical_memory_category(key[0])
                signature = row.get("anomaly_signature")
                if category not in CATEGORIES_CANONICAL or not valid_anomaly_signature(signature):
                    continue
                sources = row.get("sources") or []
                v = np.asarray(signature["vector"], dtype=np.float32)
                for source in sources:
                    kind = source.get("source_kind", "ARCHIVE_PNG")
                    if kind == "OPERATOR_CONFIRMED" and source.get("source_label") == label:
                        humans[row["id"]] = (v, category, label)
                    elif kind == "ARCHIVE_PNG":
                        p = source.get("source_path", "")
                        prefix = "public/" + ("ok_archive" if label == "OK" else "ng_archive") + "/"
                        if not isinstance(p, str) or not p.startswith(prefix):
                            continue
                        meta = {
                            "record_id": row["id"], "path": p, "label": label,
                            "lighting_mode": key[1], "source_sha256": source["source_sha256"],
                            "identity": key[2:],
                        }
                        archives.setdefault(category, []).append((v, meta))
            built = {
                c: (np.stack([v for v, _ in pairs]),
                    [info for _, info in pairs])
                for c, pairs in archives.items() if pairs
            }
            after = path.stat()
            if (after.st_mtime_ns, after.st_ctime_ns, after.st_size) != stamp:
                raise ValueError("MEMORY_CHANGED_DURING_INDEX")
            self._pool, self._human = built, humans
            self._stamp = stamp

    def diagnose(self, reference, test, info: dict, human: dict) -> dict:
        out = {
            "schema": SCHEMA, "status": "NO_MATCH",
            "diagnostic_only": True, "auto_decision_authorized": False,
            "source": "VERIFIED_ARCHIVE_JSON",
            "threshold": MIN_SHARED_SIMILARITY,
        }
        category = canonical_memory_category(info.get("category", ""))
        if category not in CATEGORIES_CANONICAL:
            return {**out, "status": "UNAVAILABLE", "reason": "UNKNOWN_CATEGORY"}
        if not isinstance(human, dict) or human.get("status") not in {
            "HUMAN_OK_VISUAL_CANDIDATE", "HUMAN_NG_VISUAL_CANDIDATE",
        } or human.get("schema") != "visionx.human_visual_recall.v1":
            return {**out, "reason": "NO_CONFIRMED_HUMAN_CANDIDATE"}
        if human.get("category") != category:
            return {**out, "status": "UNAVAILABLE", "reason": "CATEGORY_MISMATCH"}
        label = "OK" if human["status"] == "HUMAN_OK_VISUAL_CANDIDATE" else "NG"
        source = human.get("best_" + label.lower() + "_source")
        human_id = source.get("record_id") if isinstance(source, dict) else None
        if (not isinstance(human_id, str) or
                source.get("source") != "OPERATOR_CONFIRMED" or
                source.get("label") != label):
            return {**out, "status": "UNAVAILABLE", "reason": "UNVERIFIED_HUMAN_SOURCE"}
        try:
            query = build_anomaly_signature(reference, test, {}, {"category": category})
            if not valid_anomaly_signature(query) or (
                query["summary"]["map_peak"] < .1
                or query["summary"]["map_coverage"] < .015
            ):
                return {**out, "reason": "NON_DISCRIMINATIVE_MAP"}
            self._ensure()
            with self._lock:
                pair = self._pool.get(category)
                operator = self._human.get(human_id)
                if pair is None or operator is None:
                    return {**out, "reason": "MISSING_ARCHIVE_OR_OPERATOR_RECORD"}
                hv, human_category, human_label = operator
                if human_category != category or human_label != label:
                    return {**out, "status": "UNAVAILABLE", "reason": "OPERATOR_RECORD_CONFLICT"}
                vectors, metadata = pair
                scores = ConfirmedVisualRecall._similarities(
                    np.asarray(query["vector"], dtype=np.float32), vectors,
                )
                linked = ConfirmedVisualRecall._similarities(hv, vectors)
                matches = {
                    kind: sorted(
                        ((float(scores[i]), float(linked[i]), dict(meta))
                         for i, meta in enumerate(metadata) if meta["label"] == kind),
                        key=lambda row: row[0], reverse=True,
                    ) for kind in ("OK", "NG")
                }
            opposite = "NG" if label == "OK" else "OK"
            best = matches[label][0] if matches[label] else None
            other = matches[opposite][0] if matches[opposite] else None
            score = best[0] if best else 0.
            link = best[1] if best else 0.
            conflict_score = other[0] if other else 0.
            accepted = bool(
                best and score >= MIN_SHARED_SIMILARITY
                and link >= MIN_SHARED_SIMILARITY
                and score - conflict_score >= MIN_OPPOSITE_MARGIN
                and all(isfinite(x) for x in (score, link, conflict_score))
            )
            return {
                **out, "status": (
                    "RECONCILED_" + label if accepted else
                    "AMBIGUOUS" if best and score >= MIN_SHARED_SIMILARITY
                    and score - conflict_score < MIN_OPPOSITE_MARGIN else "NO_MATCH"
                ), "category": category, "label": label,
                "human_record_id": human_id,
                "archive_source": best[2] if best else None,
                "archive_similarity": round(score, 8),
                "human_archive_similarity": round(link, 8),
                "best_opposite_archive_similarity": round(conflict_score, 8),
                "archive_records_considered": sum(map(len, matches.values())),
                "source_relation": (
                    "SHARED_ANOMALY_SIGNATURE_NOT_RGB_EXACT"
                ),
                "reason": (
                    "HUMAN_AND_ARCHIVE_ALIGNED" if accepted
                    else "ARCHIVE_ALIGNMENT_INSUFFICIENT_OR_CONFLICTING"
                ),
            }
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return {**out, "status": "UNAVAILABLE", "reason": type(exc).__name__}


__all__ = ["KNNArchiveAlignment", "SCHEMA", "MIN_SHARED_SIMILARITY"]
