# src/core/moe_orchestrator.py
"""Orquestrador da Mistura de Especialistas (Mixture of Experts - MoE)."""

import numpy as np

from src.core.experts.shift_expert import ShiftExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.experts.knn_expert import KNNExpert
from src.core.experts.semantic_expert import SemanticExpert


class MoEOrchestrator:
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
        x1, x2 = max(0, ex), min(width, ex + ew)
        y1, y2 = max(0, ey), min(height, ey + eh)
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
        best_local_score = 0

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

            # Fonte prioritária: exatamente o mesmo epicentro usado nos cards
            # GABARITO • EPICENTRO e TESTE • EPICENTRO da interface.
            focus_gab, focus_test, canonical_box = self._extract_canonical_focus(
                full_gab,
                full_test,
                aoi_epicenters,
            )
            if canonical_box is not None:
                x, y, w, h = canonical_box
                best_ssim_res = self.experts["ssim"].analyze(
                    focus_gab,
                    focus_test,
                    full_gab,
                    full_test,
                    x,
                    y,
                    w,
                    h,
                    aoi_epicenters=None,
                    canonical_focus=True,
                    focus_box=canonical_box,
                )
                best_local_score = best_ssim_res.get("local_score", 0.0)
                if not best_box:
                    best_box = canonical_box
            elif raw_anomalies:
                # Fallback legado: só é usado quando não há epicentro válido.
                for x, y, w, h in raw_anomalies:
                    suspect_gab = full_gab[y : y + h, x : x + w]
                    suspect_test = full_test[y : y + h, x : x + w]
                    ssim_res = self.experts["ssim"].analyze(
                        suspect_gab,
                        suspect_test,
                        full_gab,
                        full_test,
                        x,
                        y,
                        w,
                        h,
                        aoi_epicenters=None,
                        canonical_focus=False,
                        focus_box=(x, y, w, h),
                    )
                    if ssim_res.get("local_score", 0.0) > best_local_score:
                        best_local_score = ssim_res["local_score"]
                        best_ssim_res = ssim_res
                        if not best_box:
                            best_box = (x, y, w, h)

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
                x, y, w, h = focus
                crop_test = full_test[y : y + h, x : x + w]
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

        final_score, is_defect, confidence, master_reason = self._master_fusion_score(
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

        compiled_details = {"final_score": final_score}
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

        results["detail"] = compiled_details
        return results

    def _master_fusion_score(
        self,
        shift: dict,
        silk: dict,
        semantic: dict,
        ssim_result: dict,
        knn: dict,
    ) -> tuple:
        is_physical_defect = False
        physical_reason = []
        physical_score = 0.0

        if shift and shift.get("is_defect"):
            is_physical_defect = True
            physical_score = max(0.80, min(1.0, shift.get("shift_pct", 0) * 10))
            physical_reason.append(shift.get("reason", ""))

        if silk and silk.get("is_defect"):
            is_physical_defect = True
            physical_score = max(physical_score, 0.85)
            physical_reason.append(silk.get("reason", ""))

        if semantic and semantic.get("is_defect"):
            is_physical_defect = True
            semantic_score = min(
                1.0,
                max(0.85, semantic.get("semantic_loss", 0.0) * 1.5),
            )
            physical_score = max(physical_score, semantic_score)
            physical_reason.append(semantic.get("reason", ""))

        if ssim_result:
            ssim_score = (
                ssim_result["local_score"] * 0.65
                + ssim_result["ctx_score"] * 0.35
            )
            physical_score = max(physical_score, ssim_score)
            physical_reason.append(
                f"SSIM={ssim_result.get('ssim', 1):.2f} | "
                f"Δpix={ssim_result.get('pct_changed', 0):.0%}"
            )

        if not physical_reason:
            physical_reason.append("Sem anomalias significativas")

        final_reason = " | ".join(reason for reason in physical_reason if reason)
        final_score = physical_score

        if knn and knn.get("has_memory"):
            db_score = knn.get("vote_defect", 0.5)
            best_similarity = knn.get("best_similarity", 0.0)
            final_reason += (
                f" || Dataset: {db_score:.0%} NG "
                f"(Sim: {best_similarity:.0%})"
            )

            if is_physical_defect:
                if best_similarity >= 0.85:
                    final_score = db_score
                    final_reason += " => [VETO: Aprendizado Ativo Aplicado]"
                elif db_score < 0.30 and best_similarity >= 0.75:
                    final_score = physical_score * 0.20 + db_score * 0.80
                else:
                    final_score = physical_score * 0.70 + db_score * 0.30
            else:
                if best_similarity >= 0.85:
                    final_score = db_score
                elif best_similarity >= 0.75:
                    final_score = physical_score * 0.50 + db_score * 0.50
                else:
                    final_score = physical_score * 0.70 + db_score * 0.30

        cutoff = 0.45
        is_defect = bool(final_score > cutoff)
        distance_max = (1.0 - cutoff) if is_defect else cutoff
        current_distance = (
            final_score - cutoff if is_defect else cutoff - final_score
        )
        confidence = float(
            max(
                0.50,
                min(0.99, 0.50 + 0.49 * (current_distance / distance_max)),
            )
        )
        return final_score, is_defect, confidence, final_reason
