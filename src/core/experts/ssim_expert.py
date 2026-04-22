# src/core/experts/ssim_expert.py
"""
Módulo Especialista em Textura e Manchas.
Analisa a similaridade estrutural (SSIM), diferença de bordas e correlação de histograma.
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
                return {"is_defect": False, "local_score": 0, "reason": "Imagem nula"}

            # 1. Análise Local
            size = (64, 64)
            gab = cv2.GaussianBlur(cv2.resize(crop_gab, size), (3, 3), 0)
            test = cv2.GaussianBlur(cv2.resize(crop_test, size), (3, 3), 0)

            gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

            ssim_score, _ = ssim(gray_gab, gray_test, full=True)
            diff = cv2.absdiff(gray_gab, gray_test).astype(np.float32) / 255.0
            
            mean_diff = float(np.mean(diff))
            pct_changed = float(np.mean(diff > 0.15))
            edge_change = float(np.mean(cv2.absdiff(cv2.Canny(gray_gab, 70, 180), cv2.Canny(gray_test, 70, 180)) > 0))
            hist_corr = float(cv2.compareHist(cv2.calcHist([gray_gab], [0], None, [64], [0, 256]), cv2.calcHist([gray_test], [0], None, [64], [0, 256]), cv2.HISTCMP_CORREL))

            # 2. Epicentro
            is_epicenter = False
            if aoi_epicenters:
                for (ex, ey, ew, eh) in aoi_epicenters:
                    x_right, x_left = min(box_x + box_w, ex + ew), max(box_x, ex)
                    y_bottom, y_top = min(box_y + box_h, ey + eh), max(box_y, ey)
                    if x_right > x_left and y_bottom > y_top:
                        is_epicenter = True
                        break

            # 3. Análise de Contexto (Se full for enviado)
            ctx_score, ctx_reason = 0.3, ""
            if full_gab is not None and full_test is not None and box_w > 0:
                h_f, w_f = full_gab.shape[:2]
                expand = max(box_w, box_h)
                ctx_g = full_gab[max(0, box_y-expand):min(h_f, box_y+box_h+expand), max(0, box_x-expand):min(w_f, box_x+box_w+expand)]
                ctx_t = full_test[max(0, box_y-expand):min(h_f, box_y+box_h+expand), max(0, box_x-expand):min(w_f, box_x+box_w+expand)]
                
                if ctx_g.size > 0 and ctx_t.size > 0:
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

            # 4. Cálculo Score
            local_score = sum([
                max(0, (0.85 - ssim_score) / 0.85) * 0.35,
                min(1.0, mean_diff / 0.25) * 0.20,
                min(1.0, pct_changed / 0.40) * 0.20,
                min(1.0, edge_change / 0.25) * 0.15,
                max(0, (0.80 - hist_corr) / 0.80) * 0.10
            ])
            local_score = max(0.0, min(1.0, local_score))
            if is_epicenter: local_score = min(1.0, local_score * 1.30)

            return {
                "local_score": float(local_score), "ctx_score": float(ctx_score), "ssim": float(ssim_score),
                "mean_diff": float(mean_diff), "pct_changed": float(pct_changed), "edge_change": float(edge_change),
                "hist_corr": float(hist_corr), "ctx_reason": ctx_reason, "is_epicenter": is_epicenter
            }
        except Exception as e:
            print(f"⚠️ Erro no SSIMExpert: {e}")
            return {"local_score": 0, "ctx_score": 0, "ssim": 1, "mean_diff": 0, "pct_changed": 0, "edge_change": 0, "hist_corr": 1, "ctx_reason": "", "is_epicenter": False}