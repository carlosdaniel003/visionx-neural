# src/core/experts/semantic_expert.py
"""
Módulo Especialista em Extração Semântica (Embedding Generator).
Muda a abordagem de Keypoints (ORB) para Extração Densa de Características (HOG/Histogramas).
Isso gera um vetor de 128 dimensões (pseudo-embedding) que representa o "DNA" da peça,
permitindo a visualização no formato de "Código de Barras" (Barcode) para checagem estrutural.
"""
import cv2
import numpy as np
from scipy.spatial.distance import cosine

class SemanticExpert:
    def __init__(self):
        # Usaremos uma combinação de Histogramas Espaciais para simular um Embedding de 128 dimensões
        self.embedding_size = 128

    def _generate_pseudo_embedding(self, img: np.ndarray) -> np.ndarray:
        """ Converte uma imagem em um vetor 1D contínuo de 128 posições (DNA) """
        # 1. Padroniza o tamanho para não distorcer a matemática
        img_resized = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 2. Extrai as bordas estruturais pesadas
        edges = cv2.Canny(gray, 50, 150)
        
        # 3. Divide a imagem em um grid 4x4 (16 blocos)
        grid_size = 16
        blocks_edges = []
        blocks_gray = []
        
        for y in range(0, 64, grid_size):
            for x in range(0, 64, grid_size):
                block_e = edges[y:y+grid_size, x:x+grid_size]
                block_g = gray[y:y+grid_size, x:x+grid_size]
                
                # Densidade de bordas por bloco (1 número por bloco = 16 dimensões)
                blocks_edges.append(np.mean(block_e) / 255.0)
                # Brilho médio por bloco (1 número por bloco = 16 dimensões)
                blocks_gray.append(np.mean(block_g) / 255.0)
                
        # 4. Histograma de cores comprimido (3 canais x 32 bins = 96 dimensões)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normaliza
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        
        # Junta tudo (16 + 16 + 32 + 32 + 32 = 128 dimensões!)
        embedding = np.concatenate([
            np.array(blocks_edges), 
            np.array(blocks_gray), 
            hist_h.flatten(), 
            hist_s.flatten(), 
            hist_v.flatten()
        ])
        
        return embedding

    def analyze(self, crop_gab: np.ndarray, crop_test: np.ndarray, global_box_info: dict = None, aoi_info: dict = None, aoi_epicenters: list = None) -> dict:
        default_return = {
            "is_defect": False, "score": 0.0, "reason": "Imagem nula", "bounding_box": None,
            "semantic_loss": 0.0, "query_emb": None, "ref_emb": None 
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

            if focus_gab.size < 50 or focus_test.size < 50:
                focus_gab, focus_test = crop_gab, crop_test

            # =================================================================
            # GERAÇÃO DO DNA (EMBEDDINGS)
            # =================================================================
            emb_gab = self._generate_pseudo_embedding(focus_gab)
            emb_test = self._generate_pseudo_embedding(focus_test)

            # =================================================================
            # CÁLCULO DA DISTÂNCIA DE COSSENO (Quão diferente é a essência?)
            # =================================================================
            # 0.0 = Imagens idênticas. 1.0 = Imagens completamente diferentes.
            semantic_distance = cosine(emb_gab, emb_test)
            
            # Se deu divisão por zero ou erro na matemática
            if np.isnan(semantic_distance): 
                semantic_distance = 0.0

            # Um pequeno amplificador para deixar a pontuação entre 0.0 e 1.0 visível
            semantic_loss = min(1.0, semantic_distance * 2.5) 

            is_defect = semantic_loss > 0.45
            
            return {
                "is_defect": is_defect,
                "score": float(semantic_loss),
                "reason": f"Distância Semântica: {semantic_loss:.0%}",
                "bounding_box": None, # Como é uma análise holística densa, não tem caixa de recorte
                "semantic_loss": float(semantic_loss),
                "ref_emb": emb_gab.tolist(),   # DNA Gabarito
                "query_emb": emb_test.tolist() # DNA Teste (Câmera)
            }

        except Exception as e:
            print(f"⚠️ Erro no SemanticExpert (Embedding Generator): {e}")
            default_return["reason"] = f"Erro interno: {str(e)}"
            return default_return