# src/core/moe_orchestrator.py
"""Orquestrador da Mistura de Especialistas e rastreamento da decisão."""

from __future__ import annotations

import numpy as np

from src.core.experts.shift_expert import ShiftExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.experts.knn_expert import KNNExpert
from src.core.experts.semantic_expert import SemanticExpert


class MoEOrchestrator:
    DECISION_CUTOFF = 0.45
    DECISION_SCHEMA = "visionx.decision.v1"

    def __init__(self):
        print("🧠 Inicializando Orquestrador MoE (Fusão Completa com Semântica liberada)...")
        self.experts = {
            "shift": ShiftExpert(),
            "silk": SilkExpert(),
            "ssim": SSIMExpert(),
            "semantic": SemanticExpert(),
            "knn": KNNExpert(),
        }

        all_engines = ["shift", "silk", "ssim", "semantic", "knn"]
        self.routing_table = {
            "Shifted": all_engines,
            "Up Side Down": all_engines,
            "Reverse": all_engines,
            "Missing": all_engines,
            "Bridge": all_engines,
            "Little Solder": all_engines,
            "No solder": all_engines,
            "Dust": all_engines,
            "Much Adhesive": all_engines,
        }

    def reload_memory(self):
        if "knn" in self.experts:
            self.experts["knn"].reload_memory()

    @staticmethod
    def _extract_canonical_focus(
        full_gab: np.ndarray,
        full_test: np.ndarray,
        aoi_epicenters: list,
    ) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int, int, int] | None]:
        """Recorta exatamente a primeira ROI escolhida pelo EpicenterExtractor."""
        if not aoi_epicenters or full_gab is None or full_test is None:
            return None, None, None

        ex, ey, ew, eh = aoi_epicenters[0]
        height, width = full_gab.shape[:2]
        x1, x2 = max(0, int(ex)), min(width, int(ex + ew))
        y1, y2 = max(0, int(ey)), min(height, int(ey + eh))
        if x2 <= x1 or y2 <= y1:
            return None, None, None

        focus_gab = full_gab[y1:y2, x1:x2].copy()
        focus_test = full_test[y1:y2, x1:x2].copy()
        if focus_gab.size == 0 or focus_test.size == 0:
            return None, None, None

        if focus_gab.shape != focus_test.shape:
            import cv2

            focus_test = cv2.resize(
                focus_test,
                (focus_gab.shape[1], focus_gab.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return focus_gab, focus_test, (x1, y1, x2 - x1, y2 - y1)

    def inspect(
        self,
        full_gab: np.ndarray,
        full_test: np.ndarray,
        raw_anomalies: list,
        aoi_info: dict,
        global_box_info: dict,
        aoi_epicenters: list,
    ) -> dict:
        category = aoi_info.get("category", "Unknown")
        active_routes = self.routing_table.get(
            category,
            ["shift", "silk", "ssim", "semantic", "knn"],
        )

        results = {
            "is_defect": False,
            "confidence": 1.0,
            "verdict": "FALHA FALSA",
            "reason": "Sem anomalias significativas",
            "active_engines": [],
            "bounding_box": None,
            "all_boxes": {},
            "detail": {},
        }

        shift_res = None
        silk_res = None
        semantic_res = None
        best_ssim_res = None
        best_box = None
        best_local_score = 0.0

        if "shift" in active_routes:
            results["active_engines"].append("shift_expert.py")
            shift_res = self.experts["shift"].analyze(
                full_gab,
                full_test,
                global_box_info,
                aoi_info,
                aoi_epicenters,
            )
            if shift_res and shift_res.get("bounding_box") and shift_res.get("is_defect"):
                results["all_boxes"]["shift"] = shift_res["bounding_box"]

        if "silk" in active_routes:
            results["active_engines"].append("silk_expert.py")
            silk_res = self.experts["silk"].analyze(
                full_gab,
                full_test,
                global_box_info,
                aoi_info,
                aoi_epicenters,
            )
            if silk_res and silk_res.get("bounding_box") and silk_res.get("is_defect"):
                results["all_boxes"]["silk"] = silk_res["bounding_box"]

        if "semantic" in active_routes:
            results["active_engines"].append("semantic_expert.py")
            semantic_res = self.experts["semantic"].analyze(
                full_gab,
                full_test,
                global_box_info,
                aoi_info,
                aoi_epicenters,
            )
            if semantic_res and semantic_res.get("bounding_box") and semantic_res.get("is_defect"):
                results["all_boxes"]["semantic"] = semantic_res["bounding_box"]
                best_box = semantic_res["bounding_box"]

        if "ssim" in active_routes and raw_anomalies:
            results["active_engines"].append("ssim_expert.py")
            focus_gab, focus_test, canonical_box = self._extract_canonical_focus(
                full_gab,
                full_test,
                aoi_epicenters,
            )
            if canonical_box is not None:
                x, y, width, height = canonical_box
                best_ssim_res = self.experts["ssim"].analyze(
                    focus_gab,
                    focus_test,
                    full_gab,
                    full_test,
                    x,
                    y,
                    width,
                    height,
                    aoi_epicenters=None,
                    canonical_focus=True,
                    focus_box=canonical_box,
                )
                best_local_score = float(best_ssim_res.get("local_score", 0.0))
                if not best_box:
                    best_box = canonical_box
            else:
                for x, y, width, height in raw_anomalies:
                    suspect_gab = full_gab[y : y + height, x : x + width]
                    suspect_test = full_test[y : y + height, x : x + width]
                    ssim_res = self.experts["ssim"].analyze(
                        suspect_gab,
                        suspect_test,
                        full_gab,
                        full_test,
                        x,
                        y,
                        width,
                        height,
                        aoi_epicenters=None,
                        canonical_focus=False,
                        focus_box=(x, y, width, height),
                    )
                    if float(ssim_res.get("local_score", 0.0)) > best_local_score:
                        best_local_score = float(ssim_res["local_score"])
                        best_ssim_res = ssim_res
                        if not best_box:
                            best_box = (x, y, width, height)

            if best_box and "semantic" not in results["all_boxes"]:
                results["all_boxes"]["ssim_local"] = best_box

            if best_ssim_res and best_ssim_res.get("global_boxes"):
                largest_box = max(
                    best_ssim_res["global_boxes"],
                    key=lambda box: box[2] * box[3],
                )
                results["all_boxes"]["ssim_global"] = largest_box

        knn_res = None
        if "knn" in active_routes:
            results["active_engines"].append("knn_expert.py")
            focus = best_box
            if not focus and silk_res and silk_res.get("bounding_box"):
                focus = silk_res["bounding_box"]
            elif not focus and shift_res and shift_res.get("bounding_box"):
                focus = shift_res["bounding_box"]

            if focus:
                x, y, width, height = focus
                crop_test = full_test[y : y + height, x : x + width]
                knn_res = self.experts["knn"].analyze(
                    None,
                    crop_test,
                    None,
                    None,
                    aoi_info,
                )
            else:
                knn_res = self.experts["knn"].analyze(
                    None,
                    full_test,
                    None,
                    None,
                    aoi_info,
                )

        (
            final_score,
            is_defect,
            confidence,
            master_reason,
            decision_trace,
        ) = self._master_fusion_score(
            shift_res,
            silk_res,
            semantic_res,
            best_ssim_res,
            knn_res,
        )

        results["is_defect"] = is_defect
        results["confidence"] = confidence
        results["verdict"] = "DEFEITO REAL" if is_defect else "FALHA FALSA"
        results["reason"] = master_reason

        if semantic_res and semantic_res.get("is_defect"):
            results["bounding_box"] = semantic_res.get("bounding_box")
        elif shift_res and shift_res.get("is_defect"):
            results["bounding_box"] = shift_res.get("bounding_box")
        elif silk_res and silk_res.get("is_defect"):
            results["bounding_box"] = silk_res.get("bounding_box")
        elif best_box:
            results["bounding_box"] = best_box

        compiled_details = {}
        if shift_res:
            compiled_details.update(shift_res)
        if silk_res:
            compiled_details.update(silk_res)
        if semantic_res:
            compiled_details.update(semantic_res)
        if best_ssim_res:
            compiled_details.update(best_ssim_res)
        if knn_res:
            compiled_details.update(knn_res)

        compiled_details.update(
            {
                "final_score": final_score,
                "physical_score": decision_trace["physical_score"],
                "decision_cutoff": decision_trace["cutoff"],
                "dominant_engine": decision_trace["dominant_engine"],
                "fusion_rule": decision_trace["fusion_rule"],
                "decision_trace": decision_trace,
            }
        )
        results["detail"] = compiled_details
        return results

    @staticmethod
    def _engine_entry(
        engine_id: str,
        label: str,
        active: bool,
        triggered: bool,
        raw_score: float,
        effective_score: float,
        threshold: float,
        summary: str,
    ) -> dict:
        return {
            "id": engine_id,
            "label": label,
            "active": bool(active),
            "triggered": bool(triggered),
            "raw_score": float(np.clip(raw_score, 0.0, 1.0)),
            "effective_score": float(np.clip(effective_score, 0.0, 1.0)),
            "threshold": float(np.clip(threshold, 0.0, 1.0)),
            "selected": False,
            "final_influence": 0.0,
            "summary": str(summary or ""),
        }

    def _master_fusion_score(
        self,
        shift: dict | None,
        silk: dict | None,
        semantic: dict | None,
        ssim_result: dict | None,
        knn: dict | None,
    ) -> tuple[float, bool, float, str, dict]:
        """Funde os especialistas e registra exatamente como a decisão foi formada."""
        engines: list[dict] = []
        physical_reasons: list[str] = []

        adhesive_active = bool(shift and shift.get("shift_active", False))
        adhesive_raw = float(shift.get("adhesive_score", 0.0)) if shift else 0.0
        adhesive_threshold = float(
            shift.get("adhesive_tolerance", shift.get("tolerance", 0.32))
        ) if shift else 0.32
        adhesive_triggered = bool(
            adhesive_active
            and shift.get("adhesive_is_defect", shift.get("is_defect", False))
        )
        adhesive_effective = (
            max(0.80, min(1.0, adhesive_raw)) if adhesive_triggered else 0.0
        )
        adhesive_summary = shift.get("adhesive_reason", shift.get("reason", "")) if shift else ""
        engines.append(
            self._engine_entry(
                "adhesive",
                "Fluxo de adesivo",
                adhesive_active,
                adhesive_triggered,
                adhesive_raw,
                adhesive_effective,
                adhesive_threshold,
                adhesive_summary,
            )
        )
        if adhesive_triggered and adhesive_summary:
            physical_reasons.append(adhesive_summary)

        structural_active = bool(silk)
        structural_raw = float(silk.get("silk_error_pct", 0.0)) if silk else 0.0
        structural_threshold = float(silk.get("tolerance", 0.08)) if silk else 0.08
        structural_triggered = bool(silk and silk.get("is_defect", False))
        structural_effective = 0.85 if structural_triggered else 0.0
        structural_summary = silk.get("reason", "") if silk else ""
        engines.append(
            self._engine_entry(
                "structural",
                "Comparador estrutural",
                structural_active,
                structural_triggered,
                structural_raw,
                structural_effective,
                structural_threshold,
                structural_summary,
            )
        )
        if structural_triggered and structural_summary:
            physical_reasons.append(structural_summary)

        semantic_active = bool(semantic)
        semantic_raw = float(semantic.get("semantic_loss", 0.0)) if semantic else 0.0
        semantic_threshold = 0.45
        semantic_triggered = bool(semantic and semantic.get("is_defect", False))
        semantic_effective = (
            min(1.0, max(0.85, semantic_raw * 1.5))
            if semantic_triggered
            else 0.0
        )
        semantic_summary = semantic.get("reason", "") if semantic else ""
        engines.append(
            self._engine_entry(
                "semantic",
                "Debug semântico",
                semantic_active,
                semantic_triggered,
                semantic_raw,
                semantic_effective,
                semantic_threshold,
                semantic_summary,
            )
        )
        if semantic_triggered and semantic_summary:
            physical_reasons.append(semantic_summary)

        texture_active = bool(ssim_result)
        texture_raw = 0.0
        if ssim_result:
            texture_raw = float(
                ssim_result.get("local_score", 0.0) * 0.65
                + ssim_result.get("ctx_score", 0.0) * 0.35
            )
        texture_threshold = float(
            ssim_result.get("decision_threshold", 0.45)
        ) if ssim_result else 0.45
        texture_triggered = bool(texture_active and texture_raw > texture_threshold)
        texture_effective = texture_raw if texture_active else 0.0
        texture_summary = ""
        if ssim_result:
            texture_summary = (
                f"SSIM {ssim_result.get('ssim', 1.0):.2f}; "
                f"pixels alterados {ssim_result.get('pct_changed', 0.0):.0%}"
            )
        engines.append(
            self._engine_entry(
                "texture",
                "Laboratório de textura",
                texture_active,
                texture_triggered,
                texture_raw,
                texture_effective,
                texture_threshold,
                texture_summary,
            )
        )
        if texture_active and texture_summary:
            physical_reasons.append(texture_summary)

        physical_candidates = [engine for engine in engines if engine["active"]]
        physical_dominant = (
            max(physical_candidates, key=lambda item: item["effective_score"])
            if physical_candidates
            else None
        )
        physical_score = (
            float(physical_dominant["effective_score"])
            if physical_dominant
            else 0.0
        )
        if physical_dominant:
            physical_dominant["selected"] = True

        physical_defect = any(
            engine["triggered"] for engine in engines
        )

        has_memory = bool(knn and knn.get("has_memory", False))
        memory_vote = float(knn.get("vote_defect", 0.5)) if knn else 0.5
        memory_similarity = float(knn.get("best_similarity", 0.0)) if knn else 0.0
        neighbors = int(knn.get("n_neighbors", 0)) if knn else 0

        physical_weight = 1.0
        memory_weight = 0.0
        fusion_rule = "physical_only"
        memory_role = "SEM MEMÓRIA"
        final_score = physical_score

        if has_memory:
            if physical_defect:
                if memory_similarity >= 0.85:
                    physical_weight, memory_weight = 0.0, 1.0
                    fusion_rule = "memory_veto"
                    memory_role = "VETO DA MEMÓRIA"
                elif memory_vote < 0.30 and memory_similarity >= 0.75:
                    physical_weight, memory_weight = 0.20, 0.80
                    fusion_rule = "memory_priority"
                    memory_role = "MEMÓRIA PRIORITÁRIA"
                else:
                    physical_weight, memory_weight = 0.70, 0.30
                    fusion_rule = "weighted_physical"
                    memory_role = "FUSÃO 70/30"
            else:
                if memory_similarity >= 0.85:
                    physical_weight, memory_weight = 0.0, 1.0
                    fusion_rule = "memory_override"
                    memory_role = "MEMÓRIA DECISIVA"
                elif memory_similarity >= 0.75:
                    physical_weight, memory_weight = 0.50, 0.50
                    fusion_rule = "balanced_fusion"
                    memory_role = "FUSÃO EQUILIBRADA"
                else:
                    physical_weight, memory_weight = 0.70, 0.30
                    fusion_rule = "weighted_low_similarity"
                    memory_role = "MEMÓRIA AUXILIAR"

            final_score = physical_score * physical_weight + memory_vote * memory_weight

        final_score = float(np.clip(final_score, 0.0, 1.0))
        is_defect = bool(final_score > self.DECISION_CUTOFF)
        distance_max = 1.0 - self.DECISION_CUTOFF if is_defect else self.DECISION_CUTOFF
        current_distance = (
            final_score - self.DECISION_CUTOFF
            if is_defect
            else self.DECISION_CUTOFF - final_score
        )
        confidence = float(
            max(
                0.50,
                min(
                    0.99,
                    0.50 + 0.49 * (current_distance / max(distance_max, 1e-6)),
                ),
            )
        )

        for engine in engines:
            if engine["selected"]:
                engine["final_influence"] = float(
                    engine["effective_score"] * physical_weight
                )

        memory_entry = self._engine_entry(
            "knn",
            "Memória local KNN",
            has_memory,
            has_memory and memory_vote > self.DECISION_CUTOFF,
            memory_vote,
            memory_vote,
            self.DECISION_CUTOFF,
            (
                f"{memory_role}; similaridade {memory_similarity:.0%}; "
                f"{neighbors} vizinho(s)"
                if has_memory
                else "Dataset sem amostras compatíveis"
            ),
        )
        memory_entry["selected"] = bool(memory_weight > 0.0)
        memory_entry["final_influence"] = float(memory_vote * memory_weight)
        engines.append(memory_entry)

        if memory_weight >= physical_weight and memory_weight > 0:
            dominant_engine = "knn"
        elif physical_dominant:
            dominant_engine = physical_dominant["id"]
        else:
            dominant_engine = "none"

        if not physical_reasons:
            physical_reasons.append("Sem anomalias físicas significativas")

        reason = " | ".join(item for item in physical_reasons if item)
        if has_memory:
            reason += (
                f" || KNN {memory_vote:.0%} NG; similaridade "
                f"{memory_similarity:.0%}; {memory_role}"
            )

        decision_trace = {
            "schema": self.DECISION_SCHEMA,
            "cutoff": self.DECISION_CUTOFF,
            "final_score": final_score,
            "confidence": confidence,
            "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
            "physical_score": physical_score,
            "physical_defect": bool(physical_defect),
            "dominant_engine": dominant_engine,
            "fusion_rule": fusion_rule,
            "weights": {
                "physical": float(physical_weight),
                "knn": float(memory_weight),
            },
            "memory": {
                "has_memory": has_memory,
                "vote_defect": float(memory_vote),
                "best_similarity": float(memory_similarity),
                "n_neighbors": neighbors,
                "role": memory_role,
            },
            "engines": engines,
        }
        return final_score, is_defect, confidence, reason, decision_trace
