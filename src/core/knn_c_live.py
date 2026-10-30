"""Estratégia C ao vivo: A visual + B geométrica, com revisão humana.

A memória exata continua tendo prioridade. Em casos novos, C fornece
corroboração ou alerta de conflito; nunca habilita XP 0/1, nem sobrescreve
automaticamente CNN/física com um OK. O banco visual é preparado em thread
de fundo a partir dos PNGs OK/NG já existentes, sem treinar nem criar dados.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter
import logging
import math

import numpy as np
import cv2

from src.config.settings import BASE_DIR
from src.services.knn_c_signature_cache import read_cache, write_cache
from src.core.operator_confirmed_archive_labels import RETIRED_SOURCE
from src.services.knn_c_fast_inventory import inventory_for_knn_c
from src.core.strict_category_memory import canonical_memory_lighting
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.knn_archive_augmented_replay import (
    _judge, _neighbors,
)
from src.services.startup_regression.knn_archive_concordance_experiment import (
    select_concordance_candidate,
)
from src.services.startup_regression.knn_archive_focus_signature import (
    build_focus_signature, valid_focus_signature,
)
from src.services.startup_regression.knn_archive_geometry_experiment import (
    build_geometry_signature,
)

LOG = logging.getLogger(__name__)
SCHEMA = "visionx.knn_c_live_review_gated.v1"
LIGHTS = frozenset({"SIDE", "TOP", "MID"})
MAX_ARCHIVE_IMAGES = 2000


class LiveKNNStrategyC:
    def __init__(self, *, root=None, inventory=None, extractor=None,
                 frame_loader=None):
        self.root = Path(root or BASE_DIR).resolve()
        self._inventory = inventory
        self._extractor = extractor
        self._frame_loader = frame_loader or _archive_image
        self._lock = Lock()
        self._state = "NOT_READY"
        self._reason = "MEMORIA_VISUAL_EM_PREPARACAO"
        self._pools = {}
        self._thread = None
        self._progress = {
            "phase": "NOT_READY", "total": 0, "processed": 0,
            "usable": 0, "excluded": 0, "cached": 0,
            "skipped_duplicates": 0, "elapsed_ms": 0.0,
            "failure_reasons": {}, "sample_failures": [],
        }

    @property
    def status(self) -> str:
        with self._lock:
            return self._state

    def progress(self) -> dict:
        """Snapshot rápido; pode ser exibido mesmo durante LOADING."""
        with self._lock:
            return dict(self._progress, failure_reasons=dict(
                self._progress.get("failure_reasons", {})),
                sample_failures=list(self._progress.get("sample_failures", [])),
            )

    def _publish_progress(self, **kwargs):
        with self._lock:
            self._progress.update(kwargs)

    def warmup(self) -> dict:
        """Índice tolerante a PNG isolado inválido, com fallback e cache seguro.

        Uma amostra com leitura/descritor inválido não deve apagar todo o
        acervo. Conflito OK/NG com pixels idênticos nunca pode ser escolhido
        como referência; exclui AMBOS os rótulos.
        """
        begin = perf_counter()
        with self._lock:
            self._state = "LOADING"
            self._reason = "MEMORIA_VISUAL_EM_PREPARACAO"
            self._progress = {
                "phase": "INVENTORY", "total": 0, "processed": 0,
                "usable": 0, "excluded": 0, "cached": 0,
                "skipped_duplicates": 0, "elapsed_ms": 0.0,
                "failure_reasons": {}, "sample_failures": [],
            }
        failures = defaultdict(int)
        samples = []
        try:
            cached = read_cache(self.root) if self._inventory is None else {}
            # Com cache válido, verificar SHA bruto é suficiente para
            # reutilizar identidade visual já decodificada. PNG novo /
            # modificado continua exigindo CRC, decode e pixel SHA completos.
            # Progresso INVENTORY distingue inventário de assinatura.
            def inventory_progress(done, total, path):
                self._publish_progress(
                    phase="INVENTORY", inventory_processed=done,
                    inventory_total=total, inventory_current_path=path,
                    inventory_elapsed_ms=round(
                        (perf_counter() - begin) * 1000, 2),
                )
            inv = (
                self._inventory if self._inventory is not None else
                inventory_for_knn_c(
                    self.root, cached, progress=inventory_progress,
                )
            )
            inventory_ms = round((perf_counter() - begin) * 1000, 2)
            items = inv.get("images")
            if (inv.get("schema") != "visionx.archive_inventory.v1"
                    or not isinstance(items, list) or not items
                    or len(items) > MAX_ARCHIVE_IMAGES
                    or inv.get("summary", {}).get("png_count") != len(items)):
                raise ValueError("Inventário incompleto ou excede limite")

            bad_digests = {
                group.get("pixel_sha256")
                for group in inv.get("cross_label_conflicts", ())
                if isinstance(group, dict) and group.get("pixel_sha256")
            }
            bad_paths = {
                path for group in inv.get("cross_label_conflicts", ())
                if isinstance(group, dict)
                for path in group.get("paths", ())
            }
            # Verificação independente: nem um inventário injetado pode
            # autorizar a mesma imagem como OK e NG.
            labels_by_digest = defaultdict(set)
            for item in items:
                if item.get("path") == RETIRED_SOURCE:
                    continue
                digest = item.get("pixel_sha256")
                if digest:
                    labels_by_digest[digest].add(item.get("expected_label"))
            bad_digests.update(
                d for d, labels in labels_by_digest.items()
                if "OK" in labels and "NG" in labels
            )
            self._publish_progress(
                inventory_elapsed_ms=inventory_ms,
                inventory_reused_pixel_hashes=inv.get(
                    "reused_pixel_hashes_from_sha_verified_cache", 0),
                processed=0,
            )
            extractor = self._extractor
            if extractor is None:
                from src.services.faltando_neural_dataset import AOIPairExtractor
                extractor = AOIPairExtractor()
            pools = defaultdict(list)
            cache_rows = []
            seen = set()
            counts = {"OK": 0, "NG": 0}
            used_cache, skipped = 0, 0
            self._publish_progress(phase="BUILD_SIGNATURES", total=len(items))
            for index, item in enumerate(items, 1):
                path = str(item.get("path") or "")
                digest = item.get("pixel_sha256") or item.get("file_sha256")
                label = item.get("expected_label")
                light = canonical_memory_lighting(item.get("lighting_mode"))
                reason = None
                if path == RETIRED_SOURCE:
                    reason = "RETIRED_WRONG_AOI_NG"
                elif path in bad_paths or item.get("pixel_sha256") in bad_digests:
                    reason = "CROSS_LABEL_CONFLICT"
                elif item.get("status") != "VALID_PNG" or not digest:
                    reason = "INVALID_PNG_OR_SHA"
                elif label not in {"OK", "NG"} or light not in LIGHTS:
                    reason = "INVALID_LABEL_OR_LIGHT"
                elif digest in seen:
                    skipped += 1
                else:
                    try:
                        stored = cached.get(path)
                        if (stored is not None
                                and stored.get("source_pixel_sha256") == digest
                                and stored.get("source_file_sha256") ==
                                    (item.get("file_sha256") or digest)
                                and stored.get("expected_label") == label
                                and stored.get("lighting_mode") == light):
                            entry = stored
                            used_cache += 1
                        else:
                            reference, capture, _info = extractor(
                                self._frame_loader(self.root, item)
                            )
                            focus = build_focus_signature(reference, capture)
                            if not valid_focus_signature(focus):
                                raise ValueError("Assinatura visual inválida")
                            geo = build_geometry_signature(
                                reference, capture, focus["focus"]
                            )
                            entry = {
                                "source_path": path,
                                "source_pixel_sha256": digest,
                                "source_file_sha256": item.get("file_sha256") or digest,
                                "expected_label": label,
                                "lighting_mode": light,
                                "focus_signature": focus,
                                "geometry_signature": geo,
                            }
                        pools[light].append(entry)
                        cache_rows.append(entry)
                        seen.add(digest)
                        counts[label] += 1
                    except (OSError, ValueError, TypeError, KeyError,
                            cv2.error) as exc:
                        reason = "SIGNATURE_" + type(exc).__name__
                if reason:
                    failures[reason] += 1
                    if len(samples) < 12:
                        samples.append({"path": path, "reason": reason})
                if index == 1 or index % 10 == 0 or index == len(items):
                    self._publish_progress(
                        processed=index, usable=len(cache_rows),
                        excluded=sum(failures.values()), cached=used_cache,
                        skipped_duplicates=skipped,
                        elapsed_ms=round((perf_counter() - begin) * 1000, 2),
                        build_signatures_elapsed_ms=round(
                            (perf_counter() - begin) * 1000 - inventory_ms, 2
                        ),
                        failure_reasons=dict(failures),
                        sample_failures=list(samples),
                    )

            if not cache_rows:
                raise ValueError("Nenhuma assinatura visual íntegra no acervo")
            frozen = {light: tuple(rows) for light, rows in pools.items()}
            # A persistência de cache é opcional: jamais suspende julgamento.
            cache_saved = (
                write_cache(self.root, cache_rows)
                if self._inventory is None else False
            )
            elapsed = round((perf_counter() - begin) * 1000, 2)
            with self._lock:
                self._pools = frozen
                self._state = "READY"
                self._reason = "OK"
                self._progress.update(
                    phase="READY", processed=len(items),
                    elapsed_ms=elapsed,
                    inventory_elapsed_ms=inventory_ms,
                    build_signatures_elapsed_ms=round(elapsed-inventory_ms, 2),
                    cache_saved=cache_saved,
                    indexed_by_label=dict(counts),
                    retired_wrong_ng_entries=sum(
                        row.get("path") == RETIRED_SOURCE for row in items
                    ),
                )
            LOG.info(
                "KNN C READY: %s referências, %s cache, %s excluídas, "
                "%s duplicadas, %.1f ms",
                len(cache_rows), used_cache, sum(failures.values()),
                skipped, elapsed,
            )
            return {
                "status": "READY", "images": len(cache_rows),
                "cached": used_cache, "excluded": sum(failures.values()),
                "skipped_duplicates": skipped,
                "indexed_by_label": dict(counts),
                "retired_wrong_ng_entries": sum(
                    row.get("path") == RETIRED_SOURCE for row in items
                ),
                "cache_saved": cache_saved,
                "duration_ms": elapsed,
            }
        except Exception as exc:
            LOG.warning("KNN C indisponível (sem influência no veredito): %s", exc)
            with self._lock:
                self._pools = {}
                self._state = "UNAVAILABLE"
                self._reason = f"{type(exc).__name__}: {exc}"
                self._progress.update(
                    phase="UNAVAILABLE",
                    failure_reasons=dict(failures),
                    sample_failures=list(samples),
                    elapsed_ms=round((perf_counter() - begin) * 1000, 2),
                )
            return {"status": "UNAVAILABLE", "reason": self._reason}

    def start(self) -> None:
        with self._lock:
            if self._state != "NOT_READY":
                return
            self._state = "LOADING"
            self._thread = Thread(
                target=self._background_warmup, name="ODINKNNLiveCArchive", daemon=True,
            )
            thread = self._thread
        thread.start()

    def _background_warmup(self):
        try:
            from src.services.knn_archive_auto_memory import (
                _set_background_thread_mode,
            )
            _set_background_thread_mode(True)
        except (ImportError, AttributeError):
            pass
        self.warmup()

    def diagnose(self, reference, capture, lighting: str) -> dict:
        self.start()
        with self._lock:
            state = self._state
            reason = self._reason
            pool = self._pools.get(canonical_memory_lighting(lighting), ())
        base = {
            "schema": SCHEMA, "status": "REVISAO", "reason": reason,
            "bank_status": state, "reference_count": len(pool),
            "diagnostic_only": True, "auto_decision_authorized": False,
            "production_approved": False, "xp_command": None,
            "uses_aoi_category": False,
            "bank_progress": self.progress(),
        }
        if state != "READY":
            return base
        if not (isinstance(reference, np.ndarray) and
                isinstance(capture, np.ndarray)):
            base["reason"] = "PAR_AOI_AUSENTE"
            return base
        if not pool:
            base["reason"] = "SEM_MEMORIA_NESTA_ILUMINACAO"
            return base
        try:
            focus = build_focus_signature(reference, capture)
            if not valid_focus_signature(focus):
                raise ValueError("Assinatura visual inválida")
            geo = build_geometry_signature(reference, capture, focus["focus"])
            support = {
                lbl: sum(r["expected_label"] == lbl for r in pool)
                for lbl in ("OK", "NG")
            }
            a = _judge(_neighbors(focus, geo, pool, 5, geometry=False),
                       canonical_memory_lighting(lighting), support)
            b = _judge(_neighbors(focus, geo, pool, 5, geometry=True),
                       canonical_memory_lighting(lighting), support)
            c = select_concordance_candidate(
                a["shadow_candidate"], b["shadow_candidate"],
                visual_prediction=a["prediction"],
                geometry_prediction=b["prediction"],
            )
            base.update({
                "status": c["status"], "reason": c["reason"],
                "source": c["source"], "vote_A": a["prediction"],
                "vote_B": b["prediction"],
                "suggestion_A": a["shadow_candidate"]["status"],
                "suggestion_B": b["shadow_candidate"]["status"],
                "similarity_A": a["shadow_candidate"].get("best_similarity"),
                "similarity_B": b["shadow_candidate"].get("best_similarity"),
                # Explicar por que os votos A/B não produziram sugestão
                # seletiva. Os vizinhos são os cinco mais próximos reais.
                "reason_A": a["shadow_candidate"].get("reason"),
                "reason_B": b["shadow_candidate"].get("reason"),
                "weighted_ng_A": a.get("score_ng"),
                "weighted_ng_B": b.get("score_ng"),
                "reference_support": support,
                "neighbors_A": [
                    {
                        "path": n.get("source_path", ""),
                        "label": n.get("label"),
                        "similarity": n.get("similarity"),
                        "source_pixel_sha256": n.get("source_pixel_sha256"),
                    }
                    for n in a.get("neighbors", ())[:5]
                ],
                "neighbors_B": [
                    {
                        "path": n.get("source_path", ""),
                        "label": n.get("label"),
                        "similarity": n.get("similarity"),
                        "source_pixel_sha256": n.get("source_pixel_sha256"),
                    }
                    for n in b.get("neighbors", ())[:5]
                ],
                "bank_status": "READY",
            })
        except Exception as exc:
            LOG.warning("KNN C falhou nesta captura, só revisão: %s", exc)
            base["reason"] = "FALHA_DA_ASSINATURA_" + type(exc).__name__
        return base


def install_live_knn_strategy_c(orchestrator_cls, *, strategy=None):
    """Instalar DEPOIS da memória exata para só avaliar eventos NOVOS."""
    if getattr(orchestrator_cls, "_live_knn_strategy_c_installed", False):
        return
    original = orchestrator_cls.inspect
    bank = strategy if strategy is not None else LiveKNNStrategyC()
    # A memória começa a ser preparada na inicialização, não espera a
    # primeira peça passar na AOI. Nunca bloqueia a abertura do ODIN.
    if strategy is None:
        bank.start()

    def inspect(self, full_gab, full_test, raw_anomalies,
                aoi_info, global_box_info, aoi_epicenters):
        result = original(
            self, full_gab, full_test, raw_anomalies,
            aoi_info, global_box_info, aoi_epicenters,
        )
        info = aoi_info if isinstance(aoi_info, dict) else {}
        if info.get("_replay_without_memory"):
            return result
        if not isinstance(result, dict):
            return result
        detail = result.get("detail")
        if not isinstance(detail, dict):
            return result
        route = detail.get("recognition_route")
        if route not in ("NEW_CNN", "NEW_EXPERTS"):
            return result  # KNOWN_KNN, MEMORY_CONFLICT, outros: intocados
        diagnosis = bank.diagnose(
            full_gab, full_test, info.get("lighting_mode", "SIDE")
        )
        detail["knn_c"] = diagnosis
        # O roteador da memória é interno à KNN C: o alinhamento com
        # assinaturas humanas só pode ser calculado após diagnose().
        if diagnosis.get("bank_status") == "READY":
            index_getter = getattr(self, "_get_verified_memory_index", None)
            human = detail.get("confirmed_visual_recall")
            if callable(index_getter) and isinstance(human, dict):
                from src.core.verified_memory_router import VerifiedKNNMemory
                try:
                    memory = index_getter()
                    if isinstance(memory, VerifiedKNNMemory):
                        diagnosis["aligned_archive"] = memory.recall_knn_c_archive_alignment(
                            full_gab, full_test, info, human,
                        )
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    diagnosis["aligned_archive"] = {
                        "status": "UNAVAILABLE",
                        "reason": type(exc).__name__,
                    }
        selected = diagnosis.get("status")
        if selected not in ("SUGERE_OK", "SUGERE_NG"):
            return result
        label = "OK" if selected == "SUGERE_OK" else "NG"
        # C entra no fluxo REAL como recomendação para revisão. A validação
        # offline não confere autoridade para transmitir comandos 0/1.
        # Guardar o julgamento e a revisão pré-C para impedir que um gate
        # posterior confunda revisão por C com revisão física/CNN prévia.
        original_detail_review = bool(detail.get("operator_review_required"))
        original_trace = detail.get("decision_trace")
        original_trace_review = bool(
            isinstance(original_trace, dict)
            and original_trace.get("operator_review_required")
        )
        original_review = bool(
            result.get("production_review_required")
            or original_detail_review or original_trace_review
        )
        diagnosis["pre_c_review_required"] = original_review
        diagnosis["pre_c_verdict"] = str(result.get("verdict", "")).upper()
        diagnosis["pre_c_is_defect"] = result.get("is_defect")
        diagnosis["requires_human_confirmation"] = True
        diagnosis["proposed_label"] = label
        current = str(result.get("verdict", "")).upper()
        conflict = (label == "NG" and current == "FALHA FALSA") or (
            label == "OK" and current == "DEFEITO REAL"
        )
        diagnosis["conflicts_with_existing_verdict"] = conflict
        detail["operator_review_required"] = True
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["operator_review_required"] = True
            trace["knn_c_review_only"] = True
        result["production_review_required"] = True
        # Nunca converter um NG de CNN/física em OK nem liberar XP por C.
        if conflict and label == "NG":
            result["verdict"] = "REVISÃO OBRIGATÓRIA"
            result["confidence"] = 0.5
        result["reason"] = (
            f"KNN C sugere {label} ({diagnosis.get('source')}); "
            "confirmação humana obrigatória. "
            + str(result.get("reason", ""))
        )
        return result

    orchestrator_cls.inspect = inspect
    orchestrator_cls._live_knn_strategy_c_installed = True
    orchestrator_cls._live_knn_strategy_c_bank = bank


__all__ = ["LiveKNNStrategyC", "install_live_knn_strategy_c", "SCHEMA"]
