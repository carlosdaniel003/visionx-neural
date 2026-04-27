# src/core/moe_orchestrator.py
"""
Módulo Orquestrador da Mistura de Especialistas (Mixture of Experts - MoE).
Ajuste: Motores liberados para TODAS as categorias. Fim das restrições!
NOVO AJUSTE: Active Learning com Veto de Operador. Se o K-NN encontrar uma 
similaridade extrema (>85%), ele anula a física e aplica a decisão salva no Dataset.
"""
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
            "knn": KNNExpert()
        }
        
        # TODOS OS MOTORES LIBERADOS PARA TODAS AS CATEGORIAS
        ALL_ENGINES = ["shift", "silk", "ssim", "semantic", "knn"]
        self.routing_table = {
            "Shifted": ALL_ENGINES, 
            "Up Side Down": ALL_ENGINES, 
            "Reverse": ALL_ENGINES,      
            "Missing": ALL_ENGINES,      
            "Bridge": ALL_ENGINES,
            "Little Solder": ALL_ENGINES,
            "No solder": ALL_ENGINES,
            "Dust": ALL_ENGINES, 
            "Much Adhesive": ALL_ENGINES
        }

    def reload_memory(self):
        if "knn" in self.experts:
            self.experts["knn"].reload_memory()

    def inspect(self, full_gab: np.ndarray, full_test: np.ndarray, raw_anomalies: list, aoi_info: dict, global_box_info: dict, aoi_epicenters: list) -> dict:
        category = aoi_info.get("category", "Unknown")
        active_routes = self.routing_table.get(category, ["shift", "silk", "ssim", "semantic", "knn"])
        
        results = {
            "is_defect": False,
            "confidence": 1.0, 
            "verdict": "FALHA FALSA",
            "reason": "Sem anomalias significativas",
            "active_engines": [],
            "bounding_box": None,
            "all_boxes": {}, 
            "detail": {}
        }

        shift_res = None
        silk_res = None
        semantic_res = None
        best_ssim_res = None
        best_box = None
        best_local_score = 0

        if "shift" in active_routes:
            results["active_engines"].append("shift_expert.py")
            shift_res = self.experts["shift"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            if shift_res and shift_res.get("bounding_box") and shift_res.get("is_defect"):
                results["all_boxes"]["shift"] = shift_res["bounding_box"]

        if "silk" in active_routes:
            results["active_engines"].append("silk_expert.py")
            silk_res = self.experts["silk"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            if silk_res and silk_res.get("bounding_box") and silk_res.get("is_defect"):
                results["all_boxes"]["silk"] = silk_res["bounding_box"]

        if "semantic" in active_routes:
            results["active_engines"].append("semantic_expert.py")
            semantic_res = self.experts["semantic"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            if semantic_res and semantic_res.get("bounding_box") and semantic_res.get("is_defect"):
                results["all_boxes"]["semantic"] = semantic_res["bounding_box"]
                best_box = semantic_res["bounding_box"]

        if "ssim" in active_routes and raw_anomalies:
            results["active_engines"].append("ssim_expert.py")
            for (x, y, w, h) in raw_anomalies:
                suspect_gab = full_gab[y:y+h, x:x+w]
                suspect_test = full_test[y:y+h, x:x+w]
                
                ssim_res = self.experts["ssim"].analyze(suspect_gab, suspect_test, full_gab, full_test, x, y, w, h, aoi_epicenters)
                
                if ssim_res["local_score"] > best_local_score:
                    best_local_score = ssim_res["local_score"]
                    best_ssim_res = ssim_res
                    if not best_box: 
                        best_box = (x, y, w, h)
            
            if best_box and "semantic" not in results["all_boxes"]:
                results["all_boxes"]["ssim_local"] = best_box

            if best_ssim_res and best_ssim_res.get("global_boxes"):
                maior_buraco = max(best_ssim_res["global_boxes"], key=lambda b: b[2] * b[3])
                results["all_boxes"]["ssim_global"] = maior_buraco 

        knn_res = None
        if "knn" in active_routes:
            results["active_engines"].append("knn_expert.py")
            foco = best_box 
            if not foco and silk_res and silk_res.get("bounding_box"):
                foco = silk_res["bounding_box"]
            elif not foco and shift_res and shift_res.get("bounding_box"):
                foco = shift_res["bounding_box"]
            
            if foco:
                x, y, w, h = foco
                crop_test = full_test[y:y+h, x:x+w]
                knn_res = self.experts["knn"].analyze(None, crop_test, None, None, aoi_info)
            else:
                knn_res = self.experts["knn"].analyze(None, full_test, None, None, aoi_info)

        final_score, is_defect, confidence, master_reason = self._master_fusion_score(shift_res, silk_res, semantic_res, best_ssim_res, knn_res)

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
        if shift_res: compiled_details.update(shift_res)
        if silk_res: compiled_details.update(silk_res)
        if semantic_res: compiled_details.update(semantic_res) 
        if best_ssim_res: compiled_details.update(best_ssim_res)
        if knn_res: compiled_details.update(knn_res) 
        
        results["detail"] = compiled_details

        return results

    def _master_fusion_score(self, shift: dict, silk: dict, semantic: dict, ssim: dict, knn: dict) -> tuple:
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
            sem_score = min(1.0, max(0.85, semantic.get("semantic_loss", 0.0) * 1.5))
            physical_score = max(physical_score, sem_score)
            physical_reason.append(semantic.get("reason", ""))

        if ssim:
            ssim_score = ssim["local_score"] * 0.65 + ssim["ctx_score"] * 0.35
            physical_score = max(physical_score, ssim_score)
            physical_reason.append(f"SSIM={ssim.get('ssim',1):.2f} | Δpix={ssim.get('pct_changed',0):.0%}")

        if not physical_reason:
            physical_reason.append("Sem anomalias significativas")

        final_reason = " | ".join([r for r in physical_reason if r])
        final_score = physical_score

        if knn and knn.get("has_memory"):
            db_score = knn.get("vote_defect", 0.5)
            best_sim = knn.get("best_similarity", 0.0)
            
            final_reason += f" || Dataset: {db_score:.0%} NG (Sim: {best_sim:.0%})"
            
            if is_physical_defect:
                # =============================================================
                # A MÁGICA AQUI: VETO DO OPERADOR (OVERRIDE DO DATASET)
                # =============================================================
                if best_sim >= 0.85:
                    final_score = db_score # A memória assume o controle absoluto! A nota passa a ser exatamente o que você ensinou.
                    final_reason += " => [VETO: Aprendizado Ativo Aplicado]"
                elif db_score < 0.30 and best_sim >= 0.75:
                    final_score = (physical_score * 0.20) + (db_score * 0.80)
                else:
                    final_score = (physical_score * 0.70) + (db_score * 0.30)
            else:
                if best_sim >= 0.85:
                    final_score = db_score # Se for muito parecido, usa a nota do banco
                elif best_sim >= 0.75:
                    final_score = physical_score * 0.50 + db_score * 0.50
                else:
                    final_score = physical_score * 0.70 + db_score * 0.30
        
        cutoff = 0.45
        is_defect = bool(final_score > cutoff)
        
        dist_max = (1.0 - cutoff) if is_defect else (cutoff - 0.0)
        dist_atual = (final_score - cutoff) if is_defect else (cutoff - final_score)
        
        confidence = float(max(0.50, min(0.99, 0.50 + (0.49 * (dist_atual / dist_max)))))
        
        return final_score, is_defect, confidence, final_reason