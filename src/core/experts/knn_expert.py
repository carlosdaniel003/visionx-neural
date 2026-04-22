# src/core/experts/knn_expert.py
"""
Módulo Especialista em K-Nearest Neighbors (Banco de Dados Semântico).
Transforma as imagens em vetores usando MobileNetV2 e compara com a categoria específica salva no HD.
Ajuste: Bloqueio de Fallback para defeitos críticos de posição (Shift/Silk), evitando a mistura de conceitos matemáticos.
"""
import cv2
import numpy as np
import os
import json
import urllib.request
import re
from pathlib import Path
from numpy.linalg import norm
from src.config.settings import settings

class KNNExpert:
    def __init__(self):
        print("🧠 Inicializando K-NN Expert (cv2.dnn)...")
        self.net = self._load_mobilenet_model()
        self.signatures_ok = []
        self.signatures_ng = []
        self._load_all()

    def _load_mobilenet_model(self):
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mobilenetv2-7.onnx"

        if not model_path.exists():
            print(f"⚙️ Modelo não encontrado. Baixando MobileNetV2 (13MB)...")
            url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
            urllib.request.urlretrieve(url, str(model_path))
        return cv2.dnn.readNetFromONNX(str(model_path))

    def _clean_string(self, text: str) -> str:
        if not text: return ""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _load_all(self):
        print("🔍 Varredura Semântica KNN...")
        for folder, sig_list, label_target in [(settings.NORMAL_DIR, self.signatures_ok, "OK"), (settings.ANOMALY_DIR, self.signatures_ng, "NG")]:
            if not folder.exists(): continue
            files = [f for f in folder.rglob("*.json") if f.is_file()]
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                    embedding_list = data.get("analysis", {}).get("embedding", [])
                    aoi_info = data.get("aoi_info", {})
                    part_name = self._clean_string(aoi_info.get("parts", ""))
                    category_name = self._clean_string(aoi_info.get("category", "Unknown"))
                    img_file = data.get("image_file", "")
                    path_to_save = str(f.parent / img_file) if img_file else str(f)
                    if embedding_list:
                        sig_list.append({"part": part_name, "category": category_name, "sig": np.array(embedding_list, dtype=np.float32), "path": path_to_save})
                except Exception:
                    pass

    def reload_memory(self):
        self.signatures_ok = []
        self.signatures_ng = []
        self._load_all()

    def _compute_embedding(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5: return None
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0, size=(224, 224), mean=(0.485*255, 0.456*255, 0.406*255), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward().flatten()

    def analyze(self, full_gab: np.ndarray, full_test: np.ndarray, crop_gab: np.ndarray = None, crop_test: np.ndarray = None, aoi_info: dict = None, top_k: int = 5) -> dict:
        try:
            query_img = full_test if full_test is not None and full_test.size > 0 else crop_test
            part_name = aoi_info.get("parts", "") if aoi_info else ""
            
            # Puxa o nome real para checar a trava de segurança
            raw_category = aoi_info.get("category", "") if aoi_info else ""
            
            target_part = self._clean_string(part_name)
            target_category = self._clean_string(raw_category)

            # Categorias "Físicas/Geométricas" que não podem ser misturadas com manchas e soldas
            strict_categories = ["SHIFTED", "UPSIDEDOWN", "REVERSE"]

            # 1. Tenta achar o Match Perfeito (Mesma Peça + Mesma Categoria)
            valid_ok = [i for i in self.signatures_ok if (not target_part or target_part in i["part"]) and (not target_category or target_category == i.get("category", ""))]
            valid_ng = [i for i in self.signatures_ng if (not target_part or target_part in i["part"]) and (not target_category or target_category == i.get("category", ""))]
            
            total_valid = len(valid_ok) + len(valid_ng)
            
            # 2. Se não achou Match Perfeito, aciona a trava de segurança para o Plano B
            if total_valid == 0:
                # Se for Shifted, Up Side Down ou Reverse, nós DESISTIMOS. 
                # É melhor dizer "Não sei" do que comparar um resistor deslocado com um faltando solda.
                if target_category in strict_categories:
                    print(f"⚠️ K-NN: Fallback bloqueado. Categoria {raw_category} exige similaridade estrita e não possui banco de dados.")
                    valid_ok = []
                    valid_ng = []
                else:
                    # Se for uma manchinha boba, pode comparar com outras manchas do mesmo componente
                    valid_ok = [i for i in self.signatures_ok if not target_part or target_part in i["part"]]
                    valid_ng = [i for i in self.signatures_ng if not target_part or target_part in i["part"]]
                    total_valid = len(valid_ok) + len(valid_ng)
                    
                    # 3. Plano C: Banco Inteiro (Também só liberado para categorias não-estritas)
                    if total_valid == 0:
                        valid_ok, valid_ng = self.signatures_ok, self.signatures_ng
                        total_valid = len(valid_ok) + len(valid_ng)

            query_sig = self._compute_embedding(query_img)
            
            # Se a trava segurou ou a imagem era ruim, retorna vazio
            if total_valid == 0 or query_sig is None:
                return {
                    "has_memory": False, 
                    "vote_defect": 0.5, 
                    "best_similarity": 0.0, 
                    "n_neighbors": 0, 
                    "query_embedding": query_sig.tolist() if query_sig is not None else []
                }

            distances = []
            for item in valid_ok:
                cos_sim = float(np.dot(query_sig, item["sig"]) / (norm(query_sig) * norm(item["sig"])))
                distances.append((float(1.0 - cos_sim), "OK", item["path"]))
            for item in valid_ng:
                cos_sim = float(np.dot(query_sig, item["sig"]) / (norm(query_sig) * norm(item["sig"])))
                distances.append((float(1.0 - cos_sim), "NG", item["path"]))

            distances.sort(key=lambda x: x[0])
            neighbors = distances[:top_k]

            votes_ng = sum(1.0 / max(d, 0.0001) for d, l, _ in neighbors if l == "NG")
            votes_total = sum(1.0 / max(d, 0.0001) for d, _, _ in neighbors)
            vote_defect = votes_ng / votes_total if votes_total > 0 else 0.5
            
            # Penalidade se o banco estiver desbalanceado
            if len(valid_ok) == 0: vote_defect = min(0.60, vote_defect)
            elif len(valid_ng) == 0: vote_defect = max(0.40, vote_defect)

            best_dist, best_label, best_path = neighbors[0]
            best_sim = float(max(0.0, 1.0 - best_dist))

            return {
                "has_memory": True, "vote_defect": float(vote_defect), "best_similarity": best_sim,
                "n_neighbors": int(len(neighbors)), "best_match_path": str(best_path), "best_match_label": str(best_label),
                "query_embedding": query_sig.tolist()
            }
        except Exception as e:
            print(f"⚠️ Erro no KNNExpert: {e}")
            return {"has_memory": False, "vote_defect": 0.5, "best_similarity": 0.0, "n_neighbors": 0, "query_embedding": []}