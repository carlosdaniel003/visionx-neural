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
import logging
import math

import numpy as np

from src.config.settings import BASE_DIR
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

    @property
    def status(self) -> str:
        with self._lock:
            return self._state

    def warmup(self) -> dict:
        """Carrega uma geração completa; falha fechada se o acervo for inválido."""
        with self._lock:
            self._state = "LOADING"
        try:
            inv = self._inventory if self._inventory is not None else (
                inventory_archives(self.root)
            )
            items = inv.get("images")
            if (inv.get("schema") != "visionx.archive_inventory.v1"
                    or not isinstance(items, list) or not items
                    or len(items) > MAX_ARCHIVE_IMAGES
                    or inv.get("summary", {}).get("png_count") != len(items)
                    or inv.get("cross_label_conflicts")):
                raise ValueError("Inventário incompleto, conflitante ou grande demais")
            extractor = self._extractor
            if extractor is None:
                from src.services.faltando_neural_dataset import AOIPairExtractor
                extractor = AOIPairExtractor()
            pools = defaultdict(list)
            seen = {}
            for item in items:
                label = item.get("expected_label")
                lighting = canonical_memory_lighting(item.get("lighting_mode"))
                if (item.get("status") != "VALID_PNG" or label not in {"OK", "NG"}
                        or lighting not in LIGHTS):
                    raise ValueError("Entrada PNG inválida ou luz desconhecida")
                digest = item.get("pixel_sha256") or item.get("file_sha256")
                if not digest:
                    raise ValueError("Imagem sem identidade SHA")
                if digest in seen:
                    if seen[digest] != label:
                        raise ValueError("Pixels OK e NG idênticos")
                    continue
                seen[digest] = label
                reference, capture, _info = extractor(
                    self._frame_loader(self.root, item)
                )
                focus = build_focus_signature(reference, capture)
                if not valid_focus_signature(focus):
                    raise ValueError("Assinatura visual inválida no acervo")
                geo = build_geometry_signature(reference, capture, focus["focus"])
                pools[lighting].append({
                    "source_path": item["path"],
                    "source_pixel_sha256": digest,
                    "source_file_sha256": item.get("file_sha256") or digest,
                    "expected_label": label,
                    "focus_signature": focus,
                    "geometry_signature": geo,
                })
            if not pools:
                raise ValueError("Acervo sem referências utilizáveis")
            frozen = {light: tuple(rows) for light, rows in pools.items()}
            with self._lock:
                self._pools = frozen
                self._state = "READY"
                self._reason = "OK"
            return {"status": "READY", "images": sum(map(len, frozen.values()))}
        except Exception as exc:
            LOG.warning("KNN C indisponível (sem influência no veredito): %s", exc)
            with self._lock:
                self._pools = {}
                self._state = "UNAVAILABLE"
                self._reason = f"{type(exc).__name__}: {exc}"
            return {"status": "UNAVAILABLE", "reason": self._reason}

    def start(self) -> None:
        with self._lock:
            if self._state != "NOT_READY":
                return
            self._state = "LOADING"
            self._thread = Thread(
                target=self.warmup, name="ODINKNNLiveCArchive", daemon=True,
            )
            thread = self._thread
        thread.start()

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
        selected = diagnosis.get("status")
        if selected not in ("SUGERE_OK", "SUGERE_NG"):
            return result
        label = "OK" if selected == "SUGERE_OK" else "NG"
        # C entra no fluxo REAL como recomendação para revisão. A validação
        # offline não confere autoridade para transmitir comandos 0/1.
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
