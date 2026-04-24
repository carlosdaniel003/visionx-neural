# src/core/experts/silk_expert.py
"""
Módulo Especialista em Silkscreen.
Ajuste Foco Extremo: Agora recebe as coordenadas do epicentro da AOI e realiza
a análise e o "Raio-X" EXCLUSIVAMENTE dentro dessa área menor, eliminando falsos positivos periféricos.
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
            # 1. VISÃO DE CHASSI (FOCO NO EPICENTRO)
            # =================================================================
            h_full, w_full = full_gab.shape[:2]
            offset_x, offset_y = 0, 0
            
            # Se recebemos o epicentro focado do Control Panel, nós usamos ele!
            if aoi_epicenters and len(aoi_epicenters) > 0:
                ex, ey, ew, eh = aoi_epicenters[0]
                
                # Garante que o epicentro não vaze para fora da imagem principal
                y1 = max(0, ey)
                y2 = min(h_full, ey + eh)
                x1 = max(0, ex)
                x2 = min(w_full, ex + ew)
                
                if y2 > y1 and x2 > x1:
                    roi_gab = full_gab[y1:y2, x1:x2]
                    roi_test = full_test[y1:y2, x1:x2]
                    offset_x, offset_y = x1, y1
                else:
                    roi_gab = full_gab
                    roi_test = full_test
            else:
                # Fallback antigo: Corta 22% das laterais e 18% de cima se não houver epicentro
                crop_x = int(w_full * 0.22)
                crop_y = int(h_full * 0.18)
                if crop_y >= h_full//2 or crop_x >= w_full//2:
                    roi_gab, roi_test = full_gab, full_test
                else:
                    roi_gab = full_gab[crop_y:h_full-crop_y, crop_x:w_full-crop_x]
                    roi_test = full_test[crop_y:h_full-crop_y, crop_x:w_full-crop_x]
                    offset_x, offset_y = crop_x, crop_y

            gray_gab = cv2.cvtColor(roi_gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY)

            # =================================================================
            # 2. ALINHAMENTO MAGNÉTICO
            # =================================================================
            edges_gab = cv2.Canny(gray_gab, 50, 150)
            edges_test = cv2.Canny(gray_test, 50, 150)
            
            h_m, w_m = edges_gab.shape
            # Se a ROI for muito minúscula, o HanningWindow dá erro, então protegemos:
            if h_m < 5 or w_m < 5:
                 return {"is_defect": False, "silk_error_pct": 0, "reason": "ROI muito pequena para o Raio-X"}
                 
            hann = cv2.createHanningWindow((w_m, h_m), cv2.CV_32F)
            
            shift, _ = cv2.phaseCorrelate(edges_test.astype(np.float32), edges_gab.astype(np.float32), hann)
            dx, dy = shift
            
            if abs(dx) < (w_m * 0.25) and abs(dy) < (h_m * 0.25):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                gray_test = cv2.warpAffine(gray_test, M, (w_m, h_m), borderMode=cv2.BORDER_REPLICATE)

            # =================================================================
            # 3. EXTRAÇÃO ADAPTATIVA DE TINTA E XOR (A Magia do Raio-X)
            # =================================================================
            blur_gab = cv2.GaussianBlur(gray_gab, (3, 3), 0)
            blur_test = cv2.GaussianBlur(gray_test, (3, 3), 0)

            # Extrai os brancos (tinta/reflexos)
            mask_gab = cv2.adaptiveThreshold(blur_gab, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
            mask_test = cv2.adaptiveThreshold(blur_test, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

            # Limpeza de micro-ruídos e engorda a tinta base (gab) para evitar falso positivo
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask_gab = cv2.morphologyEx(mask_gab, cv2.MORPH_OPEN, kernel_clean)
            mask_test = cv2.morphologyEx(mask_test, cv2.MORPH_OPEN, kernel_clean)

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            mask_gab_expanded = cv2.dilate(mask_gab, kernel_dilate, iterations=1)

            # XOR - Pega APENAS o que tem na Câmera que não tem no Gabarito!
            diff_mask = cv2.bitwise_xor(mask_test, mask_gab_expanded)
            
            # Corta a borda extrema interna da ROI para evitar sujeiras de desalinhamento
            border_ignore = 3
            cv2.rectangle(diff_mask, (0,0), (w_m, border_ignore), 0, -1) 
            cv2.rectangle(diff_mask, (0,h_m-border_ignore), (w_m, h_m), 0, -1) 
            cv2.rectangle(diff_mask, (0,0), (border_ignore, h_m), 0, -1) 
            cv2.rectangle(diff_mask, (w_m-border_ignore,0), (w_m, h_m), 0, -1) 

            # Agrupa os erros no foco
            kernel_group = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel_group)

            total_pixels = gray_gab.size
            wrong_pixels = cv2.countNonZero(diff_mask)
            error_pct = wrong_pixels / total_pixels

            # Tolerância Focada (Pode ser mais restrita agora que tiramos as bordas)
            tolerance = 0.05  
            if aoi_info:
                val_text = str(aoi_info.get("value", "")).upper()
                if "SHIFT" in val_text or "SIFT" in val_text:
                    tolerance = 0.10  

            is_critical = error_pct > tolerance
            
            result = {
                "is_defect": False,
                "silk_error_pct": error_pct,
                "tolerance": tolerance,
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
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    if cv2.contourArea(largest_contour) > 10: 
                        cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                        
                        # Transforma a coordenada local da ROI em coordenada global da imagem para a Box desenhar certo
                        real_x = cx + offset_x
                        real_y = cy + offset_y
                        margin = 5
                        
                        silk_box_coords = (
                            max(0, real_x - margin),
                            max(0, real_y - margin),
                            min(w_full, cw + margin*2), # Width absoluto
                            min(h_full, ch + margin*2)  # Height absoluto
                        )
                        
                        result["is_defect"] = True
                        result["bounding_box"] = silk_box_coords
                        
                        if error_pct > 0.15:
                            result["reason"] = f"ANOMALIA ESTRUTURAL NO FOCO ({error_pct:.1%})"
                        else:
                            result["reason"] = f"DIVERGÊNCIA NO FOCO ({error_pct:.1%})"

            return result
            
        except Exception as e:
            print(f"⚠️ Erro no SilkExpert: {e}")
            return {"is_defect": False, "silk_error_pct": 0, "reason": ""}