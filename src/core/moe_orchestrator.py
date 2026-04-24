# src/core/moe_orchestrator.py
"""
Módulo Orquestrador da Mistura de Especialistas (Mixture of Experts - MoE).
Ajuste: Fim do Gatekeeper Rígido. Agora o Orquestrador funde as notas físicas (Shift/Silk) 
com a Memória Semântica (K-NN) antes de bater o martelo.
Ajuste 2: Categoria "Dust" agora aciona a rota de Silkscreen para checar alterações estruturais do chassi/tinta.
Ajuste 3: Categoria "Missing" (e inversões) agora aciona a Trindade Completa (SSIM + Silk + KNN).
Ajuste 4: Multilayer Overlay. O Orquestrador agora empacota as bounding_boxes de todos os Experts ativos 
para o painel frontal desenhar o Mapa Tático de Defeitos (Várias cores).
Ajuste 5: Injeção do SemanticExpert para resolver o problema de reflexos/sombras (Parede Branca).
"""
import numpy as np

# Importando os Experts Isolados
from src.core.experts.shift_expert import ShiftExpert
from src.core.experts.silk_expert import SilkExpert
from src.core.experts.ssim_expert import SSIMExpert
from src.core.experts.knn_expert import KNNExpert
from src.core.experts.semantic_expert import SemanticExpert # <- IMPORTANTE

class MoEOrchestrator:
    def __init__(self):
        print("🧠 Inicializando Orquestrador MoE (Fusão Completa com Semântica)...")
        self.experts = {
            "shift": ShiftExpert(),
            "silk": SilkExpert(),
            "ssim": SSIMExpert(),
            "semantic": SemanticExpert(), # <- NOVO MOTOR DE ENGRENAGENS
            "knn": KNNExpert()
        }
        
        # O Livro de Receitas (Quais algoritmos usar para cada defeito)
        self.routing_table = {
            "Shifted": ["shift", "knn"], 
            # Inversões agora usam a Semântica! O texto (silk) não é mais a única fonte de verdade.
            "Up Side Down": ["silk", "semantic", "knn"], 
            "Reverse": ["silk", "semantic", "knn"],      
            # Missing agora usa a Semântica primeiro para ignorar a parede branca!
            "Missing": ["semantic", "ssim", "silk", "knn"],      
            "Bridge": ["ssim", "knn"],
            "Little Solder": ["ssim", "knn"],
            "No solder": ["ssim", "knn"],
            "Dust": ["silk", "knn"], # NOVO: Dust recruta apenas Silk e KNN. SSIM desligado para evitar falsos alarmes em poeira.
            "Much Adhesive": ["ssim", "knn"]
        }

    def reload_memory(self):
        self.experts["knn"].reload_memory()

    def inspect(self, full_gab: np.ndarray, full_test: np.ndarray, raw_anomalies: list, aoi_info: dict, global_box_info: dict, aoi_epicenters: list) -> dict:
        category = aoi_info.get("category", "Unknown")
        
        # Decide a rota
        active_routes = self.routing_table.get(category, ["shift", "silk", "ssim", "semantic", "knn"])
        
        results = {
            "is_defect": False,
            "confidence": 1.0, 
            "verdict": "FALHA FALSA",
            "reason": "Sem anomalias significativas",
            "active_engines": [],
            "bounding_box": None,
            "all_boxes": {}, # NOVO: Guarda as caixas de todos os experts para Multilayer Overlay
            "detail": {}
        }

        # ---------------------------------------------------------
        # COBRANÇA DAS NOTAS DE TODOS OS EXPERTS
        # ---------------------------------------------------------
        shift_res = None
        silk_res = None
        semantic_res = None
        best_ssim_res = None
        best_box = None
        best_local_score = 0

        # 1. Rota de Deslocamento
        if "shift" in active_routes:
            results["active_engines"].append("shift_expert.py")
            shift_res = self.experts["shift"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            if shift_res and shift_res.get("bounding_box") and shift_res.get("is_defect"):
                results["all_boxes"]["shift"] = shift_res["bounding_box"]

        # 2. Rota de Silkscreen
        if "silk" in active_routes:
            results["active_engines"].append("silk_expert.py")
            silk_res = self.experts["silk"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            if silk_res and silk_res.get("bounding_box") and silk_res.get("is_defect"):
                results["all_boxes"]["silk"] = silk_res["bounding_box"]

        # ==========================================================
        # NOVO: Rota Semântica (Características Estruturais - ORB)
        # ==========================================================
        if "semantic" in active_routes:
            results["active_engines"].append("semantic_expert.py")
            semantic_res = self.experts["semantic"].analyze(full_gab, full_test, global_box_info, aoi_info, aoi_epicenters)
            
            # Se a semântica achou o erro, ela é a deusa soberana. O quadrado dela vai pra tela!
            if semantic_res and semantic_res.get("bounding_box") and semantic_res.get("is_defect"):
                results["all_boxes"]["semantic"] = semantic_res["bounding_box"]
                best_box = semantic_res["bounding_box"]

        # 3. Rota de Textura (Manchas SSIM Clássico)
        if "ssim" in active_routes and raw_anomalies:
            results["active_engines"].append("ssim_expert.py")
            for (x, y, w, h) in raw_anomalies:
                suspect_gab = full_gab[y:y+h, x:x+w]
                suspect_test = full_test[y:y+h, x:x+w]
                
                ssim_res = self.experts["ssim"].analyze(suspect_gab, suspect_test, full_gab, full_test, x, y, w, h, aoi_epicenters)
                
                if ssim_res["local_score"] > best_local_score:
                    best_local_score = ssim_res["local_score"]
                    best_ssim_res = ssim_res
                    if not best_box: # Só usa a caixa do SSIM se a semântica falhar
                        best_box = (x, y, w, h)
            
            if best_box and "semantic" not in results["all_boxes"]:
                results["all_boxes"]["ssim_local"] = best_box

            if best_ssim_res and best_ssim_res.get("global_boxes"):
                maior_buraco = max(best_ssim_res["global_boxes"], key=lambda b: b[2] * b[3])
                results["all_boxes"]["ssim_global"] = maior_buraco # O Elefante

        # 4. Rota de KNN (Sempre convocado para ponderar)
        knn_res = None
        if "knn" in active_routes:
            results["active_engines"].append("knn_expert.py")
            
            # Se a mancha for achada pelo SSIM, usamos a box dele. 
            # Mas se a categoria for DUST (sem SSIM), nós usamos a box do Silk Expert!
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


        # ---------------------------------------------------------
        # FUSÃO MASTER (O Veredito Final)
        # ---------------------------------------------------------
        final_score, is_defect, confidence, master_reason = self._master_fusion_score(shift_res, silk_res, semantic_res, best_ssim_res, knn_res)

        results["is_defect"] = is_defect
        results["confidence"] = confidence
        results["verdict"] = "DEFEITO REAL" if is_defect else "FALHA FALSA"
        results["reason"] = master_reason

        # Atualiza a hierarquia da caixa principal
        if semantic_res and semantic_res.get("is_defect"):
            results["bounding_box"] = semantic_res.get("bounding_box")
        elif shift_res and shift_res.get("is_defect"):
            results["bounding_box"] = shift_res.get("bounding_box")
        elif silk_res and silk_res.get("is_defect"):
            results["bounding_box"] = silk_res.get("bounding_box")
        elif best_box:
            results["bounding_box"] = best_box

        # Funde todos os dicionários num só para a Tela de Telemetria ler
        compiled_details = {"final_score": final_score}
        if shift_res: compiled_details.update(shift_res)
        if silk_res: compiled_details.update(silk_res)
        if semantic_res: compiled_details.update(semantic_res) # <- Puxa a nota semântica
        if best_ssim_res: compiled_details.update(best_ssim_res)
        if knn_res: compiled_details.update(knn_res)
        
        results["detail"] = compiled_details

        return results


    def _master_fusion_score(self, shift: dict, silk: dict, semantic: dict, ssim: dict, knn: dict) -> tuple:
        """
        Pondera os avisos dos algoritmos físicos/texturais contra a base de dados histórica (K-NN).
        """
        is_physical_defect = False
        physical_reason = []
        physical_score = 0.0

        # Verifica se os Experts Físicos acharam algo ruim
        if shift and shift.get("is_defect"):
            is_physical_defect = True
            physical_score = max(0.80, min(1.0, shift.get("shift_pct", 0) * 10)) # Aumenta o score base
            physical_reason.append(shift.get("reason", ""))
            
        if silk and silk.get("is_defect"):
            is_physical_defect = True
            physical_score = max(physical_score, 0.85)
            physical_reason.append(silk.get("reason", ""))

        # O NOVO PESO MATADOR: Se a semântica disser que sumiu peça, sobrepõe tudo.
        if semantic and semantic.get("is_defect"):
            is_physical_defect = True
            sem_score = min(1.0, max(0.85, semantic.get("semantic_loss", 0.0) * 1.5))
            physical_score = max(physical_score, sem_score)
            physical_reason.append(semantic.get("reason", ""))

        if ssim:
            # Score normal do SSIM
            ssim_score = ssim["local_score"] * 0.65 + ssim["ctx_score"] * 0.35
            physical_score = max(physical_score, ssim_score)
            physical_reason.append(f"SSIM={ssim.get('ssim',1):.2f} | Δpix={ssim.get('pct_changed',0):.0%}")

        if not physical_reason:
            physical_reason.append("Sem anomalias significativas")

        final_reason = " | ".join([r for r in physical_reason if r])
        final_score = physical_score

        # -------------------------------------------------
        # A PALAVRA DO HISTÓRICO (O Voto do Operador Antigo)
        # -------------------------------------------------
        if knn and knn.get("has_memory"):
            db_score = knn.get("vote_defect", 0.5)
            best_sim = knn.get("best_similarity", 0.0)
            
            final_reason += f" || Dataset: {db_score:.0%} NG (Sim: {best_sim:.0%})"
            
            if is_physical_defect:
                # Se a máquina acha que é erro, mas o banco diz "Sempre Aprovamos Isso (OK)":
                # Proteção adicionada: A Similaridade Semântica precisa ser muito alta para o histórico derrubar o erro físico
                if db_score < 0.30 and best_sim > 0.80:
                    final_score = (physical_score * 0.40) + (db_score * 0.60) # O banco puxa a nota pra baixo (Falso Defeito)
                else:
                    final_score = (physical_score * 0.70) + (db_score * 0.30)
            else:
                # Fusão Padrão do SSIM
                if best_sim >= 0.90:
                    final_score = physical_score * 0.20 + db_score * 0.80
                elif best_sim >= 0.75:
                    final_score = physical_score * 0.50 + db_score * 0.50
                else:
                    final_score = physical_score * 0.70 + db_score * 0.30
        
        # -------------------------------------------------
        # CORTE DE CONFIANÇA
        # -------------------------------------------------
        cutoff = 0.45
        is_defect = bool(final_score > cutoff)
        
        dist_max = (1.0 - cutoff) if is_defect else (cutoff - 0.0)
        dist_atual = (final_score - cutoff) if is_defect else (cutoff - final_score)
        
        # Gera uma porcentagem de 50% a 99% baseada na distância do corte
        confidence = float(max(0.50, min(0.99, 0.50 + (0.49 * (dist_atual / dist_max)))))
        
        return final_score, is_defect, confidence, final_reason