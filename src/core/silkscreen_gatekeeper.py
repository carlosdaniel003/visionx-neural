# src/core/silkscreen_gatekeeper.py
"""
Módulo responsável exclusivamente por analisar a máscara de Silkscreen (Textos, Inversões e Nomenclaturas).
Agora usa cruzamento de dados com a AOI (OCR e Epicentros) para validar inversões reais.
"""
import cv2
import numpy as np

class SilkscreenGatekeeper:
    def __init__(self):
        print("🚪 Porteiro 2 (Fiscal de Silkscreen) inicializado.")

    def check_silkscreen_anomaly(self, full_gab: np.ndarray, full_test: np.ndarray, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        try:
            if full_gab is None or full_test is None or full_gab.size == 0 or full_test.size == 0:
                return {"is_critical_silk": False, "silk_error_pct": 0}

            if full_gab.shape != full_test.shape:
                h_gab, w_gab = full_gab.shape[:2]
                full_test = cv2.resize(full_test, (w_gab, h_gab))

            gray_gab = cv2.cvtColor(full_gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(full_test, cv2.COLOR_BGR2GRAY)

            h, w = gray_gab.shape
            crop_y, crop_x = int(h * 0.15), int(w * 0.15)
            
            if crop_y >= h//2 or crop_x >= w//2:
                return {"is_critical_silk": False, "silk_error_pct": 0}
                
            miolo_gab = gray_gab[crop_y:h-crop_y, crop_x:w-crop_x]
            miolo_test = gray_test[crop_y:h-crop_y, crop_x:w-crop_x]

            # ESTRATÉGIA 1.1: ALINHAMENTO MAGNÉTICO USANDO BORDAS
            edges_gab = cv2.Canny(miolo_gab, 50, 150)
            edges_test = cv2.Canny(miolo_test, 50, 150)
            
            h_m, w_m = edges_gab.shape
            hann = cv2.createHanningWindow((w_m, h_m), cv2.CV_32F)
            
            shift, _ = cv2.phaseCorrelate(edges_test.astype(np.float32), edges_gab.astype(np.float32), hann)
            dx, dy = shift
            
            if abs(dx) < (w_m * 0.25) and abs(dy) < (h_m * 0.25):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                miolo_test = cv2.warpAffine(miolo_test, M, (w_m, h_m), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            _, mask_gab = cv2.threshold(miolo_gab, 90, 255, cv2.THRESH_BINARY)
            _, mask_test = cv2.threshold(miolo_test, 90, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_gab_expanded = cv2.dilate(mask_gab, kernel, iterations=1)

            diff_mask = cv2.bitwise_xor(mask_test, mask_gab_expanded)
            diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

            total_pixels = miolo_gab.size
            wrong_pixels = cv2.countNonZero(diff_mask)
            error_pct = wrong_pixels / total_pixels

            # ===================================================
            # ACORDO DE CAVALHEIROS E VALIDAÇÃO CRUZADA (OCR Context)
            # ===================================================
            tolerance = 0.03  # Tolerância normal (3%)
            is_ocr_reverse = False

            if aoi_info:
                val_text = str(aoi_info.get("value", "")).upper()
                
                # 1. Se a AOI avisou que tá deslocado (Shift), a IA fica mais relaxada no Silkscreen
                if "SHIFT" in val_text or "SIFT" in val_text:
                    tolerance = 0.08  # Sobe a tolerancia para 8%, pq o deslocamento borra as letras
                    print(f"📡 Porteiro 2: OCR alertou SHIFT. Tolerância de Silkscreen afrouxada para {tolerance:.0%}")
                
                # 2. Se a AOI avisou que tá invertido (Reverse)
                if "REVERS" in val_text or "WRONG" in val_text:
                    is_ocr_reverse = True
                    print("📡 Porteiro 2: OCR alertou REVERSE. Validando interseccao com epicentro...")

            is_critical = error_pct > tolerance
            result = {
                "is_critical_silk": False,  # Começamos assumindo falso até passar nos filtros
                "silk_error_pct": error_pct
            }
            
            if is_critical:
                contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = 0, 0
                    valid_contours_found = False

                    for cnt in contours:
                        if cv2.contourArea(cnt) > 5:
                            cx, cy, cw, ch = cv2.boundingRect(cnt)
                            min_x = min(min_x, cx)
                            min_y = min(min_y, cy)
                            max_x = max(max_x, cx + cw)
                            max_y = max(max_y, cy + ch)
                            valid_contours_found = True
                    
                    if valid_contours_found:
                        global_w = max_x - min_x
                        global_h = max_y - min_y
                        real_x = min_x + crop_x
                        real_y = min_y + crop_y
                        margin = 5
                        
                        silk_box_coords = (
                            max(0, real_x - margin),
                            max(0, real_y - margin),
                            min(w, global_w + margin*2),
                            min(h, global_h + margin*2)
                        )
                        
                        # --- FILTRO FINAL: A VALIDAÇÃO CRUZADA ---
                        # Se a maquina avisou que era REVERSE, nós cruzamos a nossa caixa com a caixa da maquina.
                        valido_para_barrar = True 
                        
                        if is_ocr_reverse and aoi_epicenters and len(aoi_epicenters) > 0:
                            # Caixa da AOI
                            ax, ay, aw, ah = aoi_epicenters[0]
                            ax2, ay2 = ax + aw, ay + ah
                            
                            # Caixa da Nossa IA
                            sx, sy, sw, sh = silk_box_coords
                            sx2, sy2 = sx + sw, sy + sh
                            
                            # Testa se as caixas se encostam (Intersecção)
                            intersecta = not (sx2 < ax or sx > ax2 or sy2 < ay or sy > ay2)
                            
                            if intersecta:
                                print("✅ CONFIRMADO! Caixa do Silkscreen BATE com o Epicentro da AOI.")
                            else:
                                print("❌ IGNORADO! Caixa do Silkscreen NÃO BATE com o Epicentro da AOI.")
                                valido_para_barrar = False
                                
                        if valido_para_barrar:
                            result["is_critical_silk"] = True
                            result["silk_box"] = silk_box_coords
                            print(f"⚠️ GATEKEEPER 2 ACIONADO: Silkscreen Invertido/Diferente! ({error_pct:.1%} de erro)")

            return result
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular Silkscreen: {e}")
            return {"is_critical_silk": False, "silk_error_pct": 0}