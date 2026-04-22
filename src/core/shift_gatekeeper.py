# src/core/shift_gatekeeper.py
"""
Módulo responsável exclusivamente por calcular o deslocamento físico (Shift) das peças.
Agora integra dados de OCR para ajuste dinâmico de sensibilidade.
"""
import cv2
import numpy as np
import math

class ShiftGatekeeper:
    def __init__(self):
        print("🚪 Porteiro 1 (Deslocamento Físico) inicializado.")

    def check_global_shift(self, full_gab: np.ndarray, full_test: np.ndarray, global_box_info: dict, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        try:
            if full_gab is None or full_test is None or full_gab.size == 0 or full_test.size == 0:
                return {"is_critical_shift": False, "shift_pixels": 0, "shift_pct": 0}

            gw = global_box_info.get("w", 1)
            gh = global_box_info.get("h", 1)
            
            if gw < 10 or gh < 10:
                return {"is_critical_shift": False, "shift_pixels": 0, "shift_pct": 0}

            if full_gab.shape != full_test.shape:
                h_gab, w_gab = full_gab.shape[:2]
                full_test = cv2.resize(full_test, (w_gab, h_gab))

            gray_gab = cv2.cvtColor(full_gab, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_test = cv2.cvtColor(full_test, cv2.COLOR_BGR2GRAY).astype(np.float32)

            h, w = gray_gab.shape
            hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
            
            shift_vector, _ = cv2.phaseCorrelate(gray_test, gray_gab, hann)
            dx, dy = shift_vector

            shift_pixels = math.sqrt(dx**2 + dy**2)
            maior_dimensao = max(gw, gh)
            shift_pct = shift_pixels / maior_dimensao

            # ===================================================
            # ACORDO DE CAVALHEIROS (OCR Context)
            # ===================================================
            tolerance = 0.08  # Tolerância normal (8%)
            
            if aoi_info:
                val_text = str(aoi_info.get("value", "")).upper()
                # Se a AOI avisou que tá deslocado (Shifted ou Sifted), a IA fica mais rigorosa.
                if "SHIFT" in val_text or "SIFT" in val_text:
                    tolerance = 0.04  # Cai para 4% de tolerância
                    print(f"📡 Porteiro 1: OCR alertou SHIFT. Tolerância de deslocamento reduzida para {tolerance:.0%}")
            
            is_critical = shift_pct > tolerance
            
            return {
                "is_critical_shift": is_critical,
                "shift_pixels": round(shift_pixels, 1),
                "shift_pct": shift_pct
            }
        except Exception as e:
            print(f"⚠️ Erro ao calcular deslocamento de fase: {e}")
            return {"is_critical_shift": False, "shift_pixels": 0, "shift_pct": 0}