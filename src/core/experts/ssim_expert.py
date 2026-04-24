# src/core/experts/ssim_expert.py
"""
Módulo Especialista em Textura e Manchas.
Visão Biocular: Analisa o micro-foco da AOI (Local) e caça anomalias gigantes na peça (Macro).
Ajuste Foco Extremo: A IA agora recorta a imagem exatamente na caixinha verde menor da AOI
para não ser enganada por reflexos estourados ao redor. Retorna os recortes exatos para o Debugger.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class SSIMExpert:
    def __init__(self):
        pass

    def analyze(self, crop_gab: np.ndarray, crop_test: np.ndarray, full_gab: np.ndarray = None, full_test: np.ndarray = None, box_x: int = 0, box_y: int = 0, box_w: int = 0, box_h: int = 0, aoi_epicenters: list = None) -> dict:
        try:
            if crop_gab is None or crop_test is None or crop_test.size == 0:
                return {"is_defect": False, "local_score": 0, "reason": "Imagem nula", "global_boxes": []}

            # =================================================================
            # FOCO EXTREMO: Restringe a análise EXATAMENTE ao Epicentro Verde
            # =================================================================
            is_epicenter = False
            focus_gab = crop_gab
            focus_test = crop_test
            
            # Se a máquina enviou a caixinha verde menor e nós temos a imagem completa
            if aoi_epicenters and full_gab is not None and full_test is not None:
                for (ex, ey, ew, eh) in aoi_epicenters:
                    # Checa se o retângulo verde toca/intersecciona o componente atual
                    x_right, x_left = min(box_x + box_w, ex + ew), max(box_x, ex)
                    y_bottom, y_top = min(box_y + box_h, ey + eh), max(box_y, ey)
                    
                    if x_right > x_left and y_bottom > y_top:
                        is_epicenter = True
                        
                        # Extração Segura do Epicentro (Garante que não passa da imagem)
                        h_f, w_f = full_gab.shape[:2]
                        y1, y2 = max(0, ey), min(h_f, ey + eh)
                        x1, x2 = max(0, ex), min(w_f, ex + ew)
                        
                        if y2 > y1 and x2 > x1:
                            focus_gab = full_gab[y1:y2, x1:x2].copy()
                            focus_test = full_test[y1:y2, x1:x2].copy()
                        break

            # Se o recorte deu errado ou ficou muito pequeno, volta pro crop normal
            if focus_gab.size < 10 or focus_test.size < 10:
                focus_gab = crop_gab
                focus_test = crop_test

            # =================================================================
            # TRAVA ANTI-ERRO DO OPENCV (-209: Sizes of input arguments do not match)
            # =================================================================
            # Garante que as imagens tenham exatamente o mesmo tamanho na memória antes de continuar
            if focus_gab.shape != focus_test.shape:
                focus_test = cv2.resize(focus_test, (focus_gab.shape[1], focus_gab.shape[0]))

            # =================================================================
            # 1. VISÃO MICRO (Análise Local Focada)
            # =================================================================
            size = (64, 64)
            gab = cv2.GaussianBlur(cv2.resize(focus_gab, size), (3, 3), 0)
            test = cv2.GaussianBlur(cv2.resize(focus_test, size), (3, 3), 0)

            gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

            ssim_score, _ = ssim(gray_gab, gray_test, full=True)
            diff = cv2.absdiff(gray_gab, gray_test).astype(np.float32) / 255.0
            
            mean_diff = float(np.mean(diff))
            pct_changed = float(np.mean(diff > 0.15))
            edge_change = float(np.mean(cv2.absdiff(cv2.Canny(gray_gab, 70, 180), cv2.Canny(gray_test, 70, 180)) > 0))
            hist_corr = float(cv2.compareHist(cv2.calcHist([gray_gab], [0], None, [64], [0, 256]), cv2.calcHist([gray_test], [0], None, [64], [0, 256]), cv2.HISTCMP_CORREL))

            # 3. Análise de Contexto Micro
            ctx_score, ctx_reason = 0.3, ""
            if full_gab is not None and full_test is not None and box_w > 0:
                h_f, w_f = full_gab.shape[:2]
                expand = max(box_w, box_h)
                ctx_g = full_gab[max(0, box_y-expand):min(h_f, box_y+box_h+expand), max(0, box_x-expand):min(w_f, box_x+box_w+expand)]
                ctx_t = full_test[max(0, box_y-expand):min(h_f, box_y+box_h+expand), max(0, box_x-expand):min(w_f, box_x+box_w+expand)]
                
                if ctx_g.size > 0 and ctx_t.size > 0:
                    if ctx_g.shape != ctx_t.shape:
                         ctx_t = cv2.resize(ctx_t, (ctx_g.shape[1], ctx_g.shape[0]))
                    
                    _, smap = ssim(cv2.cvtColor(cv2.resize(ctx_g, (96, 96)), cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.resize(ctx_t, (96, 96)), cv2.COLOR_BGR2GRAY), full=True)
                    _, dt = cv2.threshold(((1.0 - smap) * 255).astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
                    cnts, _ = cv2.findContours(dt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    is_loc = len(cnts) <= 3
                    ctx_score = 0.6 if is_loc else 0.20
                    base_reason = f"{'concentrada' if is_loc else 'espalhada'}"
                    if is_epicenter:
                        ctx_score = min(1.0, ctx_score + 0.20)
                        ctx_reason = f"Foco Validado | Diferença {base_reason}"
                    else:
                        ctx_reason = f"Diferença {base_reason}"

            # =================================================================
            # 4. VISÃO MACRO (Caça ao Elefante na Sala - Imune a Reflexos)
            # =================================================================
            global_boxes = []
            diff_edges = None 
            
            if full_gab is not None and full_test is not None:
                if full_gab.shape == full_test.shape:
                    fg_gray = cv2.cvtColor(full_gab, cv2.COLOR_BGR2GRAY)
                    ft_gray = cv2.cvtColor(full_test, cv2.COLOR_BGR2GRAY)
                    
                    fg_blur = cv2.GaussianBlur(fg_gray, (7, 7), 0)
                    ft_blur = cv2.GaussianBlur(ft_gray, (7, 7), 0)
                    
                    edges_g = cv2.Canny(fg_blur, 50, 150)
                    edges_t = cv2.Canny(ft_blur, 50, 150)
                    
                    kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    edges_g_dilated = cv2.dilate(edges_g, kernel_tol, iterations=1)
                    
                    diff_edges = cv2.bitwise_xor(edges_t, edges_g_dilated)
                    
                    kernel_macro = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                    thresh_macro = cv2.morphologyEx(diff_edges, cv2.MORPH_CLOSE, kernel_macro)
                    
                    thresh_macro = cv2.morphologyEx(thresh_macro, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                    
                    cnts_macro, _ = cv2.findContours(thresh_macro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in cnts_macro:
                        if cv2.contourArea(c) > 600: 
                            gx, gy, gw, gh = cv2.boundingRect(c)
                            global_boxes.append((gx, gy, gw, gh))

            # 5. Cálculo Score Final Local
            local_score = sum([
                max(0, (0.85 - ssim_score) / 0.85) * 0.35,
                min(1.0, mean_diff / 0.25) * 0.20,
                min(1.0, pct_changed / 0.40) * 0.20,
                min(1.0, edge_change / 0.25) * 0.15,
                max(0, (0.80 - hist_corr) / 0.80) * 0.10
            ])
            local_score = max(0.0, min(1.0, local_score))
            if is_epicenter: local_score = min(1.0, local_score * 1.30)
            
            if len(global_boxes) > 0:
                local_score = max(local_score, 0.95)
                ctx_reason = "DANO ESTRUTURAL MASSIVO DETECTADO"

            return {
                "local_score": float(local_score), "ctx_score": float(ctx_score), "ssim": float(ssim_score),
                "mean_diff": float(mean_diff), "pct_changed": float(pct_changed), "edge_change": float(edge_change),
                "hist_corr": float(hist_corr), "ctx_reason": ctx_reason, "is_epicenter": is_epicenter,
                "global_boxes": global_boxes,
                
                # === EXPORTAÇÃO PARA O DEBUGGER VISUAL DE 3 TELAS ===
                "heat_map_raw": (diff * 255).astype(np.uint8), 
                "macro_edges": diff_edges, 
                "crop_gab": focus_gab,   # Exporta a foto real do foco do gabarito
                "crop_test": focus_test, # Exporta a foto real do foco do teste
                "full_test": full_test  
            }
        except Exception as e:
            print(f"⚠️ Erro no SSIMExpert: {e}")
            return {"local_score": 0, "ctx_score": 0, "ssim": 1, "mean_diff": 0, "pct_changed": 0, "edge_change": 0, "hist_corr": 1, "ctx_reason": "", "is_epicenter": False, "global_boxes": []}