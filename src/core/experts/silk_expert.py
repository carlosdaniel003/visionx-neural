# src/core/experts/silk_expert.py
"""
Módulo Especialista em Silkscreen.
Usa Visão de Chassi (Corte Geométrico Absoluto do CI), Alinhamento sem bordas falsas 
e Threshold Adaptativo para ler marcações a laser.
"""
import cv2
import numpy as np

class SilkExpert:
    def __init__(self):
        pass

    def analyze(self, full_gab: np.ndarray, full_test: np.ndarray, global_box_info: dict = None, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        try:
            if full_gab is None or full_test is None or full_gab.size == 0 or full_test.size == 0:
                return {"is_defect": False, "silk_error_pct": 0, "reason": ""}

            if full_gab.shape != full_test.shape:
                h_gab, w_gab = full_gab.shape[:2]
                full_test = cv2.resize(full_test, (w_gab, h_gab))

            # =================================================================
            # 1. VISÃO DE CHASSI: Arranca a placa verde e os pinos de solda
            # =================================================================
            h_full, w_full = full_gab.shape[:2]
            
            # Corta 22% das laterais e 18% de cima/baixo. 
            # Isso garante que só sobra o miolo de resina preta do CI.
            crop_x = int(w_full * 0.22)
            crop_y = int(h_full * 0.18)
            
            if crop_y >= h_full//2 or crop_x >= w_full//2:
                 return {"is_defect": False, "silk_error_pct": 0, "reason": ""}
                 
            roi_gab = full_gab[crop_y:h_full-crop_y, crop_x:w_full-crop_x]
            roi_test = full_test[crop_y:h_full-crop_y, crop_x:w_full-crop_x]
            offset_x, offset_y = crop_x, crop_y

            gray_gab = cv2.cvtColor(roi_gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY)

            # =================================================================
            # 2. ALINHAMENTO MAGNÉTICO (Com proteção de borda BORDER_REPLICATE)
            # =================================================================
            edges_gab = cv2.Canny(gray_gab, 50, 150)
            edges_test = cv2.Canny(gray_test, 50, 150)
            
            h_m, w_m = edges_gab.shape
            hann = cv2.createHanningWindow((w_m, h_m), cv2.CV_32F)
            
            shift, _ = cv2.phaseCorrelate(edges_test.astype(np.float32), edges_gab.astype(np.float32), hann)
            dx, dy = shift
            
            if abs(dx) < (w_m * 0.25) and abs(dy) < (h_m * 0.25):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                # BORDER_REPLICATE evita a criação de faixas pretas falsas na borda
                gray_test = cv2.warpAffine(gray_test, M, (w_m, h_m), borderMode=cv2.BORDER_REPLICATE)

            # =================================================================
            # 3. EXTRAÇÃO ADAPTATIVA DE TINTA (Texto a Laser / Serigrafia)
            # =================================================================
            blur_gab = cv2.GaussianBlur(gray_gab, (3, 3), 0)
            blur_test = cv2.GaussianBlur(gray_test, (3, 3), 0)

            # Bloco de 15x15 para pegar traços mais grossos de laser
            mask_gab = cv2.adaptiveThreshold(blur_gab, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
            mask_test = cv2.adaptiveThreshold(blur_test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

            # Limpeza de micro-ruídos
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask_gab = cv2.morphologyEx(mask_gab, cv2.MORPH_OPEN, kernel_clean)
            mask_test = cv2.morphologyEx(mask_test, cv2.MORPH_OPEN, kernel_clean)

            # Expansão para engolir pequenas diferenças de iluminação
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            mask_gab_expanded = cv2.dilate(mask_gab, kernel_dilate, iterations=1)

            # Duelo de Serigrafia (XOR)
            diff_mask = cv2.bitwise_xor(mask_test, mask_gab_expanded)
            
            # Corta a borda extrema (5 pixels) de dentro da máscara para evitar restos de alinhamento
            border_ignore = 5
            cv2.rectangle(diff_mask, (0,0), (w_m, border_ignore), 0, -1) # Top
            cv2.rectangle(diff_mask, (0,h_m-border_ignore), (w_m, h_m), 0, -1) # Bottom
            cv2.rectangle(diff_mask, (0,0), (border_ignore, h_m), 0, -1) # Left
            cv2.rectangle(diff_mask, (w_m-border_ignore,0), (w_m, h_m), 0, -1) # Right

            # Efeito Ímã: Agrupa as letras separadas num "Blocão" de erro horizontal
            kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel_group)

            total_pixels = gray_gab.size
            wrong_pixels = cv2.countNonZero(diff_mask)
            error_pct = wrong_pixels / total_pixels

            # Tolerância
            tolerance = 0.03  
            if aoi_info:
                val_text = str(aoi_info.get("value", "")).upper()
                if "SHIFT" in val_text or "SIFT" in val_text:
                    tolerance = 0.08  

            is_critical = error_pct > tolerance
            
            result = {
                "is_defect": False,
                "silk_error_pct": error_pct,
                "pct_changed": error_pct, 
                "dx": round(dx, 2),
                "dy": round(dy, 2),
                "reason": "",
                "bounding_box": None,
                "mask_gab": mask_gab_expanded,
                "mask_test": mask_test,
                "diff_mask": diff_mask
            }
            
            if is_critical:
                contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Pega apenas a MAIOR anomalia agrupada (o bloco de texto)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    if cv2.contourArea(largest_contour) > 20: 
                        cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                        
                        real_x = cx + offset_x
                        real_y = cy + offset_y
                        margin = 8
                        
                        silk_box_coords = (
                            max(0, real_x - margin),
                            max(0, real_y - margin),
                            min(w_full, real_x + cw + margin*2),
                            min(h_full, real_y + ch + margin*2)
                        )
                        
                        result["is_defect"] = True
                        result["bounding_box"] = silk_box_coords
                        
                        if error_pct > 0.15:
                            result["reason"] = f"SERIGRAFIA MASSIVAMENTE ALTERADA ({error_pct:.1%})"
                        else:
                            result["reason"] = f"TEXTO DIFERENTE / CHIP INVERTIDO ({error_pct:.1%})"

            return result
            
        except Exception as e:
            print(f"⚠️ Erro no SilkExpert: {e}")
            return {"is_defect": False, "silk_error_pct": 0, "reason": ""}