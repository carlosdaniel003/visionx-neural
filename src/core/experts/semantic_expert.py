# src/core/experts/semantic_expert.py
"""
Módulo Especialista em Extração Semântica (ORB Feature Matching).
Resolve o problema da "Parede Branca": Foca apenas em características estruturais (quinas/bordas)
ignorando completamente variações de iluminação, reflexos e sombras.
Ideal para caçar componentes invertidos, ausentes ou trocados.
"""
import cv2
import numpy as np

class SemanticExpert:
    def __init__(self):
        # Inicia o extrator de características ORB
        self.orb = cv2.ORB_create(nfeatures=500)
        # Inicia o combinador de força bruta focado na distância de Hamming
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def analyze(self, crop_gab: np.ndarray, crop_test: np.ndarray, global_box_info: dict = None, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        try:
            if crop_gab is None or crop_test is None or crop_gab.size == 0 or crop_test.size == 0:
                return {"is_defect": False, "score": 0.0, "reason": "Imagem nula", "bounding_box": None}

            # Foco Extremo: Se a AOI enviou a caixinha verde menor, recorta ela!
            focus_gab = crop_gab
            focus_test = crop_test
            
            if aoi_epicenters:
                for (ex, ey, ew, eh) in aoi_epicenters:
                    if focus_gab.shape[0] >= ey+eh and focus_gab.shape[1] >= ex+ew:
                        focus_gab = crop_gab[ey:ey+eh, ex:ex+ew]
                        focus_test = crop_test[ey:ey+eh, ex:ex+ew]
                        break

            # Garante tamanho seguro
            if focus_gab.size < 100 or focus_test.size < 100:
                focus_gab, focus_test = crop_gab, crop_test

            # 1. Extração de "Conceitos" (Keypoints) em Escala de Cinza
            gray_gab = cv2.cvtColor(focus_gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(focus_test, cv2.COLOR_BGR2GRAY)

            kp_gab, des_gab = self.orb.detectAndCompute(gray_gab, None)
            kp_test, des_test = self.orb.detectAndCompute(gray_test, None)

            # Se o componente for uma "parede perfeitamente lisa" sem textura, o ORB não funciona
            if des_gab is None or des_test is None or len(kp_gab) < 5:
                return {"is_defect": False, "score": 0.0, "reason": "Sem textura estrutural para analisar", "bounding_box": None}

            # 2. Liga os pontos entre Gabarito e Câmera
            matches = self.matcher.match(des_gab, des_test)
            
            # 3. Calcula o Índice de Perda Estrutural (Quantos pontos do gabarito sumiram?)
            match_ratio = len(matches) / len(kp_gab)
            structural_loss = 1.0 - match_ratio

            # 4. Caça ao Ponto Faltante (Onde está o erro?)
            bounding_box = None
            if structural_loss > 0.45: # Se perdeu mais de 45% das características, algo tá errado
                # Pega as coordenadas dos pontos do gabarito que não acharam par na câmera
                matched_gab_idx = {m.queryIdx for m in matches}
                missing_points = [kp_gab[i].pt for i in range(len(kp_gab)) if i not in matched_gab_idx]

                if missing_points:
                    pts = np.array(missing_points, dtype=np.float32)
                    x, y, w, h = cv2.boundingRect(pts)
                    
                    # Adiciona uma margem de segurança na caixa
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(crop_test.shape[1] - x, w + padding * 2)
                    h = min(crop_test.shape[0] - y, h + padding * 2)
                    
                    # Converte as coordenadas focadas de volta para a imagem original
                    if focus_gab is not crop_gab and aoi_epicenters:
                        ex, ey, _, _ = aoi_epicenters[0]
                        x += ex
                        y += ey
                        
                    bounding_box = (int(x), int(y), int(w), int(h))

            # 5. O Veredito
            is_defect = structural_loss > 0.45
            score = min(1.0, max(0.0, structural_loss))
            reason = f"Perda Semântica: {structural_loss:.0%} | Features OK: {len(matches)}/{len(kp_gab)}"

            return {
                "is_defect": is_defect,
                "score": float(score),
                "reason": reason,
                "bounding_box": bounding_box,
                "semantic_loss": float(structural_loss)
            }

        except Exception as e:
            print(f"⚠️ Erro no SemanticExpert: {e}")
            return {"is_defect": False, "score": 0.0, "reason": f"Erro interno: {str(e)}", "bounding_box": None}