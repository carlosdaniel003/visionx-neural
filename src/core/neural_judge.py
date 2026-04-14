# src/core/neural_judge.py
"""
Módulo do Juiz Neural v5.2 — Correspondência Visual + Epicentros + Pesos Dinâmicos.

MUDANÇAS FUNDAMENTAIS:
═══════════════════════════════
1. query_similar recebe a IMAGEM NG COMPLETA (não crop de anomalia)
   → Compara "maçã com maçã" → mesma imagem retorna ~100%

2. Embedding cache em JSON (memory_cache.json)
   → Inicialização instantânea quando o cache existe

3. Não faz mais split de imagem pareada (w > h * 1.5)
   → Imagens agora são salvas individualmente

4. EPICENTER BOOST (v5.2): A IA agora recebe a "Atenção da AOI".
   Se o defeito matemático encostar no quadrado verde da AOI, ganha bônus.

5. PODER DE VETO (v5.1): Se a similaridade do banco for >95%, 
   o banco dita 90% do Score Final.
"""
import cv2
import numpy as np
import os
import json
import urllib.request
import re
from pathlib import Path
from numpy.linalg import norm
from skimage.metrics import structural_similarity as ssim

from src.config.settings import settings

CACHE_PATH = settings.DATASET_DIR / "memory_cache.json"


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

    # =================================================================
    # CACHE DE EMBEDDINGS EM JSON
    # =================================================================

    def _load_cache(self) -> dict:
        """Carrega o cache de embeddings do disco."""
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📦 Cache de embeddings carregado: {len(data)} entradas")
                return data
            except Exception as e:
                print(f"⚠️ Erro ao ler cache: {e}")
        return {}

    def _save_cache(self, cache: dict):
        """Salva o cache de embeddings no disco."""
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"💾 Cache de embeddings salvo: {len(cache)} entradas")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")

    # =================================================================
    # CARREGAMENTO
    # =================================================================

    def _load_all(self):
        print("🔍 Iniciando varredura das pastas de dataset...")

        cache = self._load_cache()
        new_cache = {}
        cache_hits = 0
        cache_misses = 0

        for folder, sig_list, label in [
            (settings.NORMAL_DIR, self.signatures_ok, "OK"),
            (settings.ANOMALY_DIR, self.signatures_ng, "NG")
        ]:
            if not folder.exists():
                print(f"⚠️ Aviso: A pasta '{folder}' não foi encontrada.")
                continue

            files = [f for f in folder.rglob("*")
                     if f.is_file()
                     and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                     and '_sample' not in f.stem]  # Ignora as fotos _sample de referência

            print(f"📂 Pasta [{label}] ({folder}): {len(files)} arquivos de imagem encontrados.")

            for f in files:
                file_key = str(f)
                file_mtime = str(f.stat().st_mtime)

                # Verifica se tem no cache e se não foi modificado
                if file_key in cache and cache[file_key].get("mtime") == file_mtime:
                    # Cache hit — carrega embedding do JSON
                    sig = np.array(cache[file_key]["embedding"], dtype=np.float32)
                    part_name = cache[file_key].get("part", "")
                    cache_hits += 1
                else:
                    # Cache miss — precisa computar
                    try:
                        img_array = np.fromfile(str(f), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    except Exception as e:
                        print(f"❌ Erro ao ler arquivo {f}: {e}")
                        continue

                    if img is None:
                        continue

                    sig = self._compute_embedding(img)
                    part_name = self._get_part_from_filename(str(f))
                    cache_misses += 1

                if sig is not None:
                    sig_list.append({
                        "part": part_name,
                        "sig": sig,
                        "path": str(f)
                    })
                    # Atualiza cache
                    new_cache[file_key] = {
                        "mtime": file_mtime,
                        "part": part_name,
                        "embedding": sig.tolist(),
                        "label": label
                    }

        # Salva cache atualizado
        if new_cache:
            self._save_cache(new_cache)

        total = len(self.signatures_ok) + len(self.signatures_ng)
        print(f"📚 RESUMO DA MEMÓRIA: {total} Embeddings indexados "
              f"({len(self.signatures_ok)} OK, {len(self.signatures_ng)} NG)")
        print(f"   Cache: {cache_hits} hits, {cache_misses} recalculados")

    def _compute_embedding(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
            return None

        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0/255.0, size=(224, 224),
            mean=(0.485*255, 0.456*255, 0.406*255),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds.flatten()

    # =================================================================
    # CONSULTA
    # =================================================================

    def query_similar(self, query_image: np.ndarray, part_name: str = "",
                      top_k: int = 5) -> dict:
        target_part = self._clean_string(part_name)

        valid_ok = [item for item in self.signatures_ok
                    if not target_part or target_part in item["part"]]
        valid_ng = [item for item in self.signatures_ng
                    if not target_part or target_part in item["part"]]

        total_valid = len(valid_ok) + len(valid_ng)

        if total_valid == 0 and (len(self.signatures_ok) > 0 or len(self.signatures_ng) > 0):
            print(f"⚠️ Peça '{target_part}' sem histórico. Avaliando contra todo o banco...")
            valid_ok = self.signatures_ok
            valid_ng = self.signatures_ng
            total_valid = len(valid_ok) + len(valid_ng)

        if total_valid == 0:
            return {
                "has_memory": False, "vote_defect": 0.5,
                "best_similarity": 0.0, "n_neighbors": 0,
                "best_match_path": "", "best_match_label": ""
            }

        query_sig = self._compute_embedding(query_image)
        if query_sig is None:
            return {
                "has_memory": False, "vote_defect": 0.5,
                "best_similarity": 0.0, "n_neighbors": 0,
                "best_match_path": "", "best_match_label": ""
            }

        distances = []
        for item in valid_ok:
            cos_sim = float(np.dot(query_sig, item["sig"]) /
                            (norm(query_sig) * norm(item["sig"])))
            distances.append((float(1.0 - cos_sim), "OK", item["path"]))

        for item in valid_ng:
            cos_sim = float(np.dot(query_sig, item["sig"]) /
                            (norm(query_sig) * norm(item["sig"])))
            distances.append((float(1.0 - cos_sim), "NG", item["path"]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:top_k]

        votes_ng = sum(1.0 / max(d, 0.0001) for d, l, _ in neighbors if l == "NG")
        votes_total = sum(1.0 / max(d, 0.0001) for d, _, _ in neighbors)

        vote_defect = votes_ng / votes_total if votes_total > 0 else 0.5
        best_dist, best_label, best_path = neighbors[0]

        best_sim = float(max(0.0, 1.0 - best_dist))
        print(f"🔎 Query similar: best_sim={best_sim:.3f} "
              f"label={best_label} path={os.path.basename(best_path)}")

        return {
            "has_memory": True,
            "vote_defect": float(vote_defect),
            "best_similarity": best_sim,
            "n_neighbors": int(len(neighbors)),
            "best_match_path": str(best_path),
            "best_match_label": str(best_label)
        }


class NeuralJudge:
    def __init__(self):
        print("⚖️ Iniciando Juiz Neural v5.2 (Foco de Epicentro + Pesos Dinâmicos)...")
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
            "edge_change": float(np.mean(
                cv2.absdiff(
                    cv2.Canny(gray_gab, 50, 150),
                    cv2.Canny(gray_test, 50, 150)) > 0)),
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

        ctx_gab = full_gab[cy1:cy2, cx1:cx2]
        ctx_test = full_test[cy1:cy2, cx1:cx2]

        if ctx_gab.size == 0 or ctx_test.size == 0:
            return {"ctx_ssim": 1.0, "is_localized": True, "n_clusters": 0}

        gray_gab = cv2.cvtColor(cv2.resize(ctx_gab, (96, 96)), cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(cv2.resize(ctx_test, (96, 96)), cv2.COLOR_BGR2GRAY)

        _, ctx_ssim_map = ssim(gray_gab, gray_test, full=True)
        _, diff_thresh = cv2.threshold(
            ((1.0 - ctx_ssim_map) * 255).astype(np.uint8), 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return {
            "ctx_ssim": float(np.mean(ctx_ssim_map)),
            "n_clusters": int(len(contours)),
            "is_localized": bool(len(contours) <= 3)
        }

    def verify_anomaly(self, crop_gab: np.ndarray, crop_test: np.ndarray,
                       part_metadata: str = "",
                       full_gab: np.ndarray = None, full_test: np.ndarray = None,
                       box_x: int = 0, box_y: int = 0,
                       box_w: int = 0, box_h: int = 0,
                       aoi_epicenters: list = None) -> dict:

        if crop_gab is None or crop_test is None or crop_test.size == 0:
            return {
                "is_defect": False, "verdict": "IGNORADO",
                "confidence": 0.0, "reason": "Imagem nula"
            }

        local = self._analyze_local(crop_gab, crop_test)

        context = None
        if full_gab is not None and full_test is not None and box_w > 0:
            context = self._analyze_context(
                full_gab, full_test, box_x, box_y, box_w, box_h)

        # =========================================================
        # MUDANÇA v5.2: VERIFICAÇÃO DE EPICENTRO (ATENÇÃO DA AOI)
        # =========================================================
        is_epicenter = False
        if aoi_epicenters:
            for (ex, ey, ew, eh) in aoi_epicenters:
                # Verifica se o defeito bate no quadrado verde menor da AOI
                x_left = max(box_x, ex)
                y_top = max(box_y, ey)
                x_right = min(box_x + box_w, ex + ew)
                y_bottom = min(box_y + box_h, ey + eh)
                
                if x_right > x_left and y_bottom > y_top:
                    is_epicenter = True
                    break

        query_img = full_test if full_test is not None else crop_test
        db_result = self.memory.query_similar(query_img, part_name=part_metadata)

        local_score = sum([
            max(0, (0.92 - local["ssim"]) / 0.92) * 0.35,
            min(1.0, local["mean_diff"] / 0.15) * 0.20,
            min(1.0, local["pct_changed"] / 0.25) * 0.20,
            min(1.0, local["edge_change"] / 0.15) * 0.15,
            max(0, (0.95 - local["hist_corr"]) / 0.95) * 0.10
        ])
        local_score = max(0.0, min(1.0, local_score))

        # BÔNUS DE EPICENTRO! Se a IA concordar com a máquina física, aumenta o alarme local.
        if is_epicenter:
            local_score = min(1.0, local_score * 1.30)

        ctx_score, ctx_reason = 0.5, ""
        if context:
            ctx_score = 0.7 if context["is_localized"] else 0.25
            base_reason = f"{'concentrada' if context['is_localized'] else 'espalhada'}"
            if is_epicenter:
                ctx_score = min(1.0, ctx_score + 0.20)
                ctx_reason = f"Foco Validado (Epicentro AOI) | Diferença {base_reason}"
            else:
                ctx_reason = f"Diferença {base_reason}"

        db_score, db_reason = 0.5, ""
        if db_result["has_memory"]:
            db_score = db_result["vote_defect"]
            db_reason = (f"Dataset: {db_score:.0%} de ser defeito "
                        f"(Sim. máx: {db_result['best_similarity']:.0%})")

        # =========================================================
        # MUDANÇA v5.1: PESOS DINÂMICOS (PODER DE VETO DA MEMÓRIA)
        # =========================================================
        if db_result["has_memory"]:
            best_sim = db_result["best_similarity"]
            db_score = db_result["vote_defect"]

            # NÍVEL 1: Correspondência Exata (Poder de Veto)
            # Se a foto for >= 95% idêntica ao histórico, a memória domina a decisão.
            if best_sim >= 0.95:
                final_score = local_score * 0.05 + ctx_score * 0.05 + db_score * 0.90
            
            # NÍVEL 2: Correspondência Forte
            # Se for muito parecida (> 85%), o banco tem peso majoritário (60%)
            elif best_sim >= 0.85:
                final_score = local_score * 0.25 + ctx_score * 0.15 + db_score * 0.60
            
            # NÍVEL 3: Correspondência Fraca (Pesos Padrões)
            # Se achou algo no banco, mas não é tão igual, divide a atenção.
            else:
                final_score = local_score * 0.50 + ctx_score * 0.20 + db_score * 0.30
                
        else:
            # Sem histórico: A IA se vira apenas com a visão computacional
            final_score = local_score * 0.65 + ctx_score * 0.35

        is_defect = bool(final_score > 0.45)
        confidence = float(min(0.99, 0.50 + abs(final_score - 0.45) * 2.5))
        conf_percent = int(confidence * 100)

        reasons = [f"SSIM={local['ssim']:.2f}", f"Δpix={local['pct_changed']:.0%}"]
        if ctx_reason:
            reasons.append(ctx_reason)
        if db_reason:
            reasons.append(db_reason)

        return {
            "is_defect": is_defect,
            "confidence": confidence,
            "score_text": f"{conf_percent}%",
            "verdict": "DEFEITO REAL" if is_defect else "FALHA FALSA",
            "reason": " | ".join(reasons),
            "detail": {
                "local_score": float(local_score),
                "ctx_score": float(ctx_score),
                "db_score": float(db_score),
                "final_score": float(final_score),
                "ssim": float(local["ssim"]),
                "pct_changed": float(local["pct_changed"]),
                "edge_change": float(local["edge_change"]),
                "hist_corr": float(local["hist_corr"]),
                "db_best_sim": float(db_result["best_similarity"]),
                "db_best_path": str(db_result["best_match_path"]),
                "db_best_label": str(db_result["best_match_label"]),
                "db_has_memory": bool(db_result["has_memory"]),
                "db_neighbors": int(db_result["n_neighbors"]),
                "db_vote": float(db_result["vote_defect"]),
            }
        }