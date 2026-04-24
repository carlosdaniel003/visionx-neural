# src/core/experts/semantic_expert.py
"""
Módulo Especialista em Extração Semântica (ORB Feature Matching).
Resolve o problema da "Parede Branca": Foca apenas em características estruturais (quinas/bordas)
ignorando completamente variações de iluminação, reflexos e sombras.
Ajuste Foco Extremo: Agora trabalha única e exclusivamente dentro do epicentro.
Ajuste Debugger: Exporta a imagem com os keypoints desenhados para a interface.
"""
import cv2
import numpy as np

class SemanticExpert:
    def __init__(self):
        # Inicia o extrator de características ORB. 500 pontos focados numa caixinha pequena dão extrema precisão.
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def analyze(self, crop_gab: np.ndarray, crop_test: np.ndarray, global_box_info: dict = None, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        # Prepara imagens padrão em branco pro painel não capotar se algo der errado
        blank_img = np.zeros((50, 50, 3), dtype=np.uint8)
        default_return = {
            "is_defect": False, "score": 0.0, "reason": "Imagem nula", "bounding_box": None,
            "semantic_loss": 0.0, "query_emb": [0,0], "ref_emb": [0,0], 
            "sem_img_gab": blank_img, "sem_img_test": blank_img
        }

        try:
            if crop_gab is None or crop_test is None or crop_gab.size == 0 or crop_test.size == 0:
                return default_return

            # =================================================================
            # FOCO EXTREMO: Isola a caixinha do epicentro
            # =================================================================
            focus_gab = crop_gab
            focus_test = crop_test
            
            if aoi_epicenters:
                for (ex, ey, ew, eh) in aoi_epicenters:
                    if focus_gab.shape[0] >= ey+eh and focus_gab.shape[1] >= ex+ew:
                        focus_gab = crop_gab[ey:ey+eh, ex:ex+ew].copy()
                        focus_test = crop_test[ey:ey+eh, ex:ex+ew].copy()
                        break

            # Garante tamanho seguro
            if focus_gab.size < 50 or focus_test.size < 50:
                focus_gab, focus_test = crop_gab, crop_test

            # Trava de segurança OpenCV
            if focus_gab.shape != focus_test.shape:
                focus_test = cv2.resize(focus_test, (focus_gab.shape[1], focus_gab.shape[0]))

            # =================================================================
            # EXTRAÇÃO DE CONCEITOS ESTRUTURAIS (ORB)
            # =================================================================
            gray_gab = cv2.cvtColor(focus_gab, cv2.COLOR_BGR2GRAY)
            gray_test = cv2.cvtColor(focus_test, cv2.COLOR_BGR2GRAY)

            kp_gab, des_gab = self.orb.detectAndCompute(gray_gab, None)
            kp_test, des_test = self.orb.detectAndCompute(gray_test, None)

            # Preparação das imagens de debug visual para o Painel
            debug_img_gab = focus_gab.copy()
            debug_img_test = focus_test.copy()

            if des_gab is None or len(kp_gab) < 3:
                default_return["reason"] = "Sem textura estrutural no foco"
                default_return["sem_img_gab"] = debug_img_gab
                default_return["sem_img_test"] = debug_img_test
                return default_return

            if des_test is None:
                matches = []
            else:
                matches = self.matcher.match(des_gab, des_test)
            
            # =================================================================
            # CÁLCULO DA PERDA DE ESTRUTURA (O QUE SUMIU DA PEÇA?)
            # =================================================================
            match_ratio = len(matches) / len(kp_gab)
            structural_loss = 1.0 - match_ratio

            bounding_box = None
            matched_gab_idx = {m.queryIdx for m in matches}
            
            # Desenha os pontos na imagem de Debug
            for i, kp in enumerate(kp_gab):
                pt = tuple(map(int, kp.pt))
                if i in matched_gab_idx:
                    # Ponto verde (OK)
                    cv2.circle(debug_img_gab, pt, 2, (0, 255, 0), -1)
                else:
                    # Ponto Vermelho (Sumiu na Câmera)
                    cv2.circle(debug_img_gab, pt, 2, (0, 0, 255), -1)

            if kp_test:
                 for kp in kp_test:
                     cv2.circle(debug_img_test, tuple(map(int, kp.pt)), 2, (255, 100, 0), -1)

            if structural_loss > 0.40: # Corte ajustado
                missing_points = [kp_gab[i].pt for i in range(len(kp_gab)) if i not in matched_gab_idx]

                if missing_points:
                    pts = np.array(missing_points, dtype=np.float32)
                    x, y, w, h = cv2.boundingRect(pts)
                    
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(focus_test.shape[1] - x, w + padding * 2)
                    h = min(focus_test.shape[0] - y, h + padding * 2)
                    
                    # Desenha o quadrado vermelho da falha no debug
                    cv2.rectangle(debug_img_gab, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 1)

                    # Repassa a coordenada Global
                    if focus_gab is not crop_gab and aoi_epicenters:
                        ex, ey, _, _ = aoi_epicenters[0]
                        x += ex
                        y += ey
                        
                    bounding_box = (int(x), int(y), int(w), int(h))

            is_defect = structural_loss > 0.40
            score = min(1.0, max(0.0, structural_loss))
            reason = f"Perda Semântica: {structural_loss:.0%} | Features OK: {len(matches)}/{len(kp_gab)}"

            # Mockando as variáveis 'query_emb' e 'ref_emb' para que o painel de DNA consiga desenhar o status visual
            pseudo_dna_query = [len(kp_test) / 500.0, structural_loss]
            pseudo_dna_ref = [len(kp_gab) / 500.0, 0.0]

            return {
                "is_defect": is_defect,
                "score": float(score),
                "reason": reason,
                "bounding_box": bounding_box,
                "semantic_loss": float(structural_loss),
                "query_emb": pseudo_dna_query, 
                "ref_emb": pseudo_dna_ref,
                "sem_img_gab": debug_img_gab,
                "sem_img_test": debug_img_test
            }

        except Exception as e:
            print(f"⚠️ Erro no SemanticExpert: {e}")
            default_return["reason"] = f"Erro interno: {str(e)}"
            return default_return