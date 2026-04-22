# src/core/experts/shift_expert.py
"""
Módulo Especialista em Deslocamento (Shift) Híbrido.
Executa três passes:
0. Visão Global (Gross Error): Procura por diferenças espalhafatosas e estruturais na área total.
1. Macro: Varre a imagem inteira com FFT para achar grandes escorregões.
2. Micro: Foca no menor epicentro com filtro Canny para achar micro-escorregões.
"""
import cv2
import numpy as np
import math

class ShiftExpert:
    def __init__(self):
        pass

    def analyze(self, full_gab: np.ndarray, full_test: np.ndarray, global_box_info: dict, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        try:
            if full_gab is None or full_test is None or full_gab.size == 0 or full_test.size == 0:
                return {"is_defect": False, "shift_pixels": 0, "shift_pct": 0, "reason": ""}

            # Tolerância base
            tolerance = 0.08  
            
            if aoi_info:
                val_text = str(aoi_info.get("value", "")).upper()
                cat_text = str(aoi_info.get("category", "")).upper()
                # Tolerância cai pela metade se for da categoria certa
                if "SHIFT" in val_text or "SIFT" in val_text or "SHIFT" in cat_text:
                    tolerance = 0.04  

            gw = global_box_info.get("w", 1)
            gh = global_box_info.get("h", 1)
            
            if gw < 10 or gh < 10:
                return {"is_defect": False, "shift_pixels": 0, "shift_pct": 0, "reason": ""}

            if full_gab.shape != full_test.shape:
                h_gab, w_gab = full_gab.shape[:2]
                full_test = cv2.resize(full_test, (w_gab, h_gab))

            gray_gab_macro = cv2.cvtColor(full_gab, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_test_macro = cv2.cvtColor(full_test, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # =================================================================
            # PASSO 0: VISÃO GLOBAL (Gross Error / Diferenças Espalhafatosas)
            # =================================================================
            # Borramos as imagens para ignorar ruído de câmera e focar em grandes borrões/manchas
            blur_gab = cv2.GaussianBlur(gray_gab_macro, (5, 5), 0)
            blur_test = cv2.GaussianBlur(gray_test_macro, (5, 5), 0)
            
            # Subtração brutal pixel a pixel
            diff_global = cv2.absdiff(blur_gab, blur_test)
            
            # Filtra apenas mudanças fortes (diferença de cor/brilho > 45)
            _, thresh_global = cv2.threshold(diff_global, 45, 255, cv2.THRESH_BINARY)
            
            # Quantos % da imagem total virou uma anomalia gritante?
            gross_diff_pct = float(cv2.countNonZero(thresh_global) / diff_global.size)

            # Se mais de 15% do componente foi brutalmente alterado, condena direto!
            if gross_diff_pct > 0.15:
                reason = f"ANOMALIA MACRO (Estrutural): Diferença visual severa detectada ({gross_diff_pct:.1%})"
                # Passa o erro como "pct_changed" para o Radar Chart ter algo para desenhar!
                return {
                    "is_defect": True,
                    "shift_pixels": 0.0,
                    "shift_pct": gross_diff_pct,
                    "pct_changed": gross_diff_pct, # O radar vai ler isso
                    "dx": 0.0,  # Zero porque não foi um deslocamento, foi uma alteração estrutural
                    "dy": 0.0,
                    "tolerance": tolerance,
                    "reason": reason,
                    "bounding_box": aoi_epicenters[0] if aoi_epicenters and len(aoi_epicenters) > 0 else None
                }

            # =================================================================
            # PASSO 1: ANÁLISE MACRO (Imagem Inteira - Para Grandes Saltos)
            # =================================================================
            h_m, w_m = gray_gab_macro.shape
            hann_macro = cv2.createHanningWindow((w_m, h_m), cv2.CV_32F)
            
            shift_vector_macro, _ = cv2.phaseCorrelate(gray_test_macro, gray_gab_macro, hann_macro)
            dx_m, dy_m = shift_vector_macro

            shift_pixels_macro = math.sqrt(dx_m**2 + dy_m**2)
            maior_dimensao_macro = max(gw, gh)
            shift_pct_macro = shift_pixels_macro / maior_dimensao_macro

            if shift_pct_macro > tolerance:
                reason = f"SHIFT CRITICO (MACRO): {round(shift_pixels_macro, 1)}px ({shift_pct_macro:.1%})"
                return {
                    "is_defect": True,
                    "shift_pixels": round(shift_pixels_macro, 1),
                    "shift_pct": shift_pct_macro,
                    "dx": round(dx_m, 2),
                    "dy": round(dy_m, 2),
                    "tolerance": tolerance,
                    "reason": reason,
                    "bounding_box": aoi_epicenters[0] if aoi_epicenters else None 
                }

            # =================================================================
            # PASSO 2: ANÁLISE MICRO (Foco no Epicentro com Canny - Para Micro-Escorregões)
            # =================================================================
            if aoi_epicenters and len(aoi_epicenters) > 0:
                menor_epicentro = min(aoi_epicenters, key=lambda b: b[2] * b[3])
                ex, ey, ew, eh = menor_epicentro
                epicenter_box = (ex, ey, ew, eh)
                
                comp_w, comp_h = ew, eh
                
                margin_x = max(15, int(ew * 0.3))
                margin_y = max(15, int(eh * 0.3))
                
                x1 = max(0, ex - margin_x)
                y1 = max(0, ey - margin_y)
                x2 = min(w_m, ex + ew + margin_x)
                y2 = min(h_m, ey + eh + margin_y)
                
                roi_gab = full_gab[y1:y2, x1:x2]
                roi_test = full_test[y1:y2, x1:x2]

                if comp_w > 10 and comp_h > 10 and roi_gab.size > 0 and roi_test.size > 0:
                    if roi_gab.shape != roi_test.shape:
                        roi_test = cv2.resize(roi_test, (roi_gab.shape[1], roi_gab.shape[0]))

                    gray_gab_micro = cv2.cvtColor(roi_gab, cv2.COLOR_BGR2GRAY)
                    gray_test_micro = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY)

                    edges_gab = cv2.Canny(gray_gab_micro, 50, 150).astype(np.float32)
                    edges_test = cv2.Canny(gray_test_micro, 50, 150).astype(np.float32)

                    h_mic, w_mic = edges_gab.shape
                    hann_micro = cv2.createHanningWindow((w_mic, h_mic), cv2.CV_32F)
                    
                    shift_vector_micro, _ = cv2.phaseCorrelate(edges_test, edges_gab, hann_micro)
                    dx_mic, dy_mic = shift_vector_micro

                    shift_pixels_micro = math.sqrt(dx_mic**2 + dy_mic**2)
                    maior_dimensao_micro = max(comp_w, comp_h)
                    shift_pct_micro = shift_pixels_micro / maior_dimensao_micro
                    
                    if shift_pct_micro > tolerance:
                        reason = f"SHIFT CRITICO (MICRO): {round(shift_pixels_micro, 1)}px ({shift_pct_micro:.1%})"
                        return {
                            "is_defect": True,
                            "shift_pixels": round(shift_pixels_micro, 1),
                            "shift_pct": shift_pct_micro,
                            "dx": round(dx_mic, 2),
                            "dy": round(dy_mic, 2),
                            "tolerance": tolerance,
                            "reason": reason,
                            "bounding_box": epicenter_box 
                        }
            
            return {
                "is_defect": False,
                "shift_pixels": round(shift_pixels_macro, 1),
                "shift_pct": shift_pct_macro,
                "dx": round(dx_m, 2),
                "dy": round(dy_m, 2),
                "tolerance": tolerance,
                "reason": "",
                "bounding_box": None
            }

        except Exception as e:
            print(f"⚠️ Erro no ShiftExpert: {e}")
            return {"is_defect": False, "shift_pixels": 0, "shift_pct": 0, "reason": ""}