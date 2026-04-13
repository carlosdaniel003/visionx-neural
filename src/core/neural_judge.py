# src\core\neural_judge.py
"""
Módulo do Juiz Neural v4.4 (Tipagem Segura para JSON + Tolerância a Recortes Finos).
Utiliza a rede convolucional MobileNetV2 nativa via cv2.dnn para extrair
a "assinatura semântica" das imagens e compara usando Distância de Cosseno.
"""
import cv2
import numpy as np
import os
import urllib.request
import re
from pathlib import Path
from numpy.linalg import norm
from skimage.metrics import structural_similarity as ssim

from src.config.settings import settings


class DatasetMemory:
    def __init__(self):
        print("🧠 Inicializando Memória Profunda (cv2.dnn)...")
        self.net = self._load_mobilenet_model()
        self.signatures_ok = [] 
        self.signatures_ng = []
        self._load_all()

    def _load_mobilenet_model(self):
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "mobilenetv2-7.onnx"

        if not model_path.exists():
            print(f"⚙️ Modelo não encontrado em {model_path}. Baixando MobileNetV2 (13MB)...")
            url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
            try:
                urllib.request.urlretrieve(url, str(model_path))
                print("✅ Download concluído!")
            except Exception as e:
                print(f"⚠️ Erro ao baixar o modelo. A rede corporativa pode estar bloqueando.")
                print(f"Baixe manualmente deste link:\n{url}\nE salve como: {model_path}")
                raise e

        return cv2.dnn.readNetFromONNX(str(model_path))

    def _clean_string(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _get_part_from_filename(self, filename: str) -> str:
        base = os.path.basename(filename)
        raw_part = base.split('_')[0] if '_' in base else base.split('.')[0]
        return self._clean_string(raw_part)

    def _load_all(self):
        print("🔍 Iniciando varredura das pastas de dataset...")
        
        for folder, sig_list, label in [
            (settings.NORMAL_DIR, self.signatures_ok, "OK"),
            (settings.ANOMALY_DIR, self.signatures_ng, "NG")
        ]:
            if not folder.exists():
                print(f"⚠️ Aviso: A pasta '{folder}' não foi encontrada.")
                continue
            
            files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            print(f"📂 Pasta [{label}] ({folder}): {len(files)} arquivos de imagem encontrados.")
            
            for f in files:
                try:
                    img_array = np.fromfile(str(f), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"❌ Erro crítico ao ler arquivo {f}: {e}")
                    continue

                if img is None:
                    continue
                
                h, w = img.shape[:2]
                if w > h * 1.5: 
                    img_target = img[:, w//2:] 
                else:
                    img_target = img
                
                sig = self._compute_embedding(img_target)
                part_name = self._get_part_from_filename(str(f))
                
                if sig is not None:
                    sig_list.append({"part": part_name, "sig": sig, "path": str(f)})

        total = len(self.signatures_ok) + len(self.signatures_ng)
        print(f"📚 RESUMO DA MEMÓRIA: {total} Embeddings indexados com sucesso "
              f"({len(self.signatures_ok)} OK, {len(self.signatures_ng)} NG)")

    def _compute_embedding(self, img: np.ndarray) -> np.ndarray:
        # DIMINUÍDO DE 10 PARA 5 PIXELS (Para não descartar as fotos finas detectadas nos logs)
        if img is None or img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
            return None

        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0, size=(224, 224),
                                     mean=(0.485*255, 0.456*255, 0.406*255), 
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds.flatten()

    def query_similar(self, crop_test: np.ndarray, part_name: str = "", top_k: int = 5) -> dict:
        target_part = self._clean_string(part_name)
        
        valid_ok = [item for item in self.signatures_ok if not target_part or target_part in item["part"]]
        valid_ng = [item for item in self.signatures_ng if not target_part or target_part in item["part"]]
        
        total_valid = len(valid_ok) + len(valid_ng)
        
        if total_valid == 0 and (len(self.signatures_ok) > 0 or len(self.signatures_ng) > 0):
            print(f"⚠️ Peça '{target_part}' não tem histórico (ou arquivo foi salvo sem o prefixo). Avaliando contra todo o banco...")
            valid_ok = self.signatures_ok
            valid_ng = self.signatures_ng
            total_valid = len(valid_ok) + len(valid_ng)

        if total_valid == 0:
            return {
                "has_memory": False, "vote_defect": 0.5,
                "best_similarity": 0.0, "n_neighbors": 0,
                "best_match_path": "", "best_match_label": ""
            }

        query_sig = self._compute_embedding(crop_test)
        if query_sig is None:
             return {"has_memory": False, "vote_defect": 0.5, "best_similarity": 0.0, "n_neighbors": 0, "best_match_path": "", "best_match_label": ""}

        distances = []
        for item in valid_ok:
            cos_sim = np.dot(query_sig, item["sig"]) / (norm(query_sig) * norm(item["sig"]))
            distances.append((float(1.0 - cos_sim), "OK", item["path"]))
            
        for item in valid_ng:
            cos_sim = np.dot(query_sig, item["sig"]) / (norm(query_sig) * norm(item["sig"]))
            distances.append((float(1.0 - cos_sim), "NG", item["path"]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:top_k]

        votes_ng = sum(1.0 / max(d, 0.0001) for d, l, _ in neighbors if l == "NG")
        votes_total = sum(1.0 / max(d, 0.0001) for d, _, _ in neighbors)

        vote_defect = votes_ng / votes_total if votes_total > 0 else 0.5
        best_dist, best_label, best_path = neighbors[0]

        return {
            "has_memory": True,
            "vote_defect": float(vote_defect),
            "best_similarity": float(max(0.0, 1.0 - best_dist)),
            "n_neighbors": int(len(neighbors)),
            "best_match_path": str(best_path),
            "best_match_label": str(best_label)
        }


class NeuralJudge:
    def __init__(self):
        print("⚖️ Iniciando Juiz Neural v4.4 (MobileNetV2 + Json Safe)...")
        self.memory = DatasetMemory()

    def reload_memory(self):
        print("🔄 Recarregando memória do dataset...")
        self.memory = DatasetMemory()

    def _analyze_local(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> dict:
        size = (64, 64)
        gab = cv2.resize(crop_gab, size)
        test = cv2.resize(crop_test, size)

        gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        ssim_score, _ = ssim(gray_gab, gray_test, full=True)
        diff = cv2.absdiff(gray_gab, gray_test).astype(np.float32) / 255.0
        
        return {
            "ssim": float(ssim_score),
            "mean_diff": float(np.mean(diff)),
            "pct_changed": float(np.mean(diff > 0.12)),
            "edge_change": float(np.mean(cv2.absdiff(cv2.Canny(gray_gab, 50, 150), cv2.Canny(gray_test, 50, 150)) > 0)),
            "hist_corr": float(cv2.compareHist(
                cv2.calcHist([gray_gab], [0], None, [64], [0, 256]), 
                cv2.calcHist([gray_test], [0], None, [64], [0, 256]), 
                cv2.HISTCMP_CORREL))
        }

    def _analyze_context(self, full_gab: np.ndarray, full_test: np.ndarray,
                         box_x: int, box_y: int, box_w: int, box_h: int) -> dict:
        h_full, w_full = full_gab.shape[:2]
        expand = max(box_w, box_h)
        cx1, cy1 = max(0, box_x - expand), max(0, box_y - expand)
        cx2, cy2 = min(w_full, box_x + box_w + expand), min(h_full, box_y + box_h + expand)

        ctx_gab, ctx_test = full_gab[cy1:cy2, cx1:cx2], full_test[cy1:cy2, cx1:cx2]

        if ctx_gab.size == 0 or ctx_test.size == 0:
            return {"ctx_ssim": 1.0, "is_localized": True, "n_clusters": 0}

        gray_gab = cv2.cvtColor(cv2.resize(ctx_gab, (96, 96)), cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(cv2.resize(ctx_test, (96, 96)), cv2.COLOR_BGR2GRAY)

        _, ctx_ssim_map = ssim(gray_gab, gray_test, full=True)
        _, diff_thresh = cv2.threshold(((1.0 - ctx_ssim_map) * 255).astype(np.uint8), 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return {"ctx_ssim": float(np.mean(ctx_ssim_map)), "n_clusters": int(len(contours)), "is_localized": bool(len(contours) <= 3)}

    def verify_anomaly(self, crop_gab: np.ndarray, crop_test: np.ndarray,
                       part_metadata: str = "",
                       full_gab: np.ndarray = None, full_test: np.ndarray = None,
                       box_x: int = 0, box_y: int = 0,
                       box_w: int = 0, box_h: int = 0) -> dict:
        
        if crop_gab is None or crop_test is None or crop_test.size == 0:
            return {"is_defect": False, "verdict": "IGNORADO", "confidence": 0.0, "reason": "Imagem nula"}

        local = self._analyze_local(crop_gab, crop_test)

        context = None
        if full_gab is not None and full_test is not None and box_w > 0:
            context = self._analyze_context(full_gab, full_test, box_x, box_y, box_w, box_h)

        db_result = self.memory.query_similar(crop_test, part_name=part_metadata)

        local_score = sum([
            max(0, (0.92 - local["ssim"]) / 0.92) * 0.35,
            min(1.0, local["mean_diff"] / 0.15) * 0.20,
            min(1.0, local["pct_changed"] / 0.25) * 0.20,
            min(1.0, local["edge_change"] / 0.15) * 0.15,
            max(0, (0.95 - local["hist_corr"]) / 0.95) * 0.10
        ])
        local_score = max(0.0, min(1.0, local_score))

        ctx_score, ctx_reason = 0.5, ""
        if context:
            ctx_score = 0.7 if context["is_localized"] else 0.25
            ctx_reason = f"Diferença {'concentrada' if context['is_localized'] else 'espalhada'}"

        db_score, db_reason = 0.5, ""
        if db_result["has_memory"]:
            db_score = db_result["vote_defect"]
            db_reason = f"Dataset: {db_score:.0%} de ser defeito (Sim. máx: {db_result['best_similarity']:.0%})"

        if db_result["has_memory"]:
            final_score = local_score * 0.50 + ctx_score * 0.20 + db_score * 0.30
        else:
            final_score = local_score * 0.65 + ctx_score * 0.35

        # FORÇANDO CONVERSÃO PURA DO PYTHON PARA O JSON NÃO QUEBRAR
        is_defect = bool(final_score > 0.45)
        confidence = float(min(0.99, 0.50 + abs(final_score - 0.45) * 2.5))
        conf_percent = int(confidence * 100)

        reasons = [f"SSIM={local['ssim']:.2f}", f"Δpix={local['pct_changed']:.0%}"]
        if ctx_reason: reasons.append(ctx_reason)
        if db_reason: reasons.append(db_reason)

        return {
            "is_defect": is_defect,
            "confidence": confidence,
            "score_text": f"{conf_percent}%",
            "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
            "reason": " | ".join(reasons),
            "detail": {
                "local_score": float(local_score), "ctx_score": float(ctx_score), "db_score": float(db_score), "final_score": float(final_score),
                "db_best_sim": float(db_result["best_similarity"]),
                "db_best_path": str(db_result["best_match_path"])
            }
        }