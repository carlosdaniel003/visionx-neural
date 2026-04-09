"""
Módulo do Juiz Neural v3 (Análise Semântica + Consulta ao Banco de Dados).
NÃO adivinha nem chuta — calcula a diferença estrutural real entre as imagens
e consulta o histórico de curadoria humana para embasar a decisão.

Três camadas de análise:
  1. Análise Local: compara o recorte suspeito (quadrado menor) pixel a pixel.
  2. Análise de Contexto: olha a região expandida ao redor (quadro maior).
  3. Consulta ao Dataset: busca os pares mais similares já curados pelo humano
     e usa o conhecimento histórico como "voto de desempate".
"""
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

from src.config.settings import settings


class DatasetMemory:
    """
    Memória do sistema: carrega todas as imagens curadas do dataset
    e extrai uma 'assinatura visual' de cada uma para busca por similaridade.
    """
    def __init__(self):
        self.signatures_ok = []   # Lista de (assinatura, filepath)
        self.signatures_ng = []
        self._load_all()

    def _load_all(self):
        """Carrega e indexa todas as imagens do dataset ao iniciar."""
        for folder, sig_list, label in [
            (settings.NORMAL_DIR, self.signatures_ok, "OK"),
            (settings.ANOMALY_DIR, self.signatures_ng, "NG")
        ]:
            if not folder.exists():
                continue
            files = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                sig = self._compute_signature(img)
                if sig is not None:
                    sig_list.append((sig, str(f)))

        total = len(self.signatures_ok) + len(self.signatures_ng)
        print(f"📚 Memória do Dataset: {total} amostras indexadas "
              f"({len(self.signatures_ok)} OK, {len(self.signatures_ng)} NG)")

    def _compute_signature(self, pair_img: np.ndarray) -> np.ndarray:
        """
        Calcula uma assinatura compacta de uma imagem-par (sample|ng lado a lado).
        Usa histograma multi-canal + estatísticas de diferença.
        """
        h, w = pair_img.shape[:2]
        if w < 20 or h < 10:
            return None

        mid = w // 2
        left = pair_img[:, :mid]   # Sample
        right = pair_img[:, mid:]  # NG/Teste

        size = (48, 48)
        left_r = cv2.resize(left, size)
        right_r = cv2.resize(right, size)

        # Diferença absoluta
        diff = cv2.absdiff(left_r, right_r)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        sig = []
        # Histograma da diferença (16 bins)
        hist = cv2.calcHist([cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)],
                            [0], None, [16], [0, 256])
        cv2.normalize(hist, hist)
        sig.extend(hist.flatten().tolist())

        # Estatísticas
        sig.append(float(np.mean(gray_diff)))
        sig.append(float(np.std(gray_diff)))
        sig.append(float(np.percentile(gray_diff, 95)))
        sig.append(float(np.mean(gray_diff > 0.1)))

        return np.array(sig, dtype=np.float32)

    def query_similar(self, crop_gab: np.ndarray, crop_test: np.ndarray, top_k: int = 5) -> dict:
        """
        Consulta o banco de dados: encontra as amostras curadas mais similares
        ao par atual e retorna o 'voto' do histórico humano.

        Retorna:
            {
                "has_memory": True/False,
                "vote_defect": float (0.0 a 1.0 — proporção de vizinhos NG),
                "best_similarity": float,
                "n_neighbors": int
            }
        """
        total = len(self.signatures_ok) + len(self.signatures_ng)
        if total == 0:
            return {"has_memory": False, "vote_defect": 0.5, "best_similarity": 0.0, "n_neighbors": 0}

        # Monta a assinatura do par atual
        size = (48, 48)
        gab_r = cv2.resize(crop_gab, size)
        test_r = cv2.resize(crop_test, size)
        diff = cv2.absdiff(gab_r, test_r)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        query_sig = []
        hist = cv2.calcHist([cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)],
                            [0], None, [16], [0, 256])
        cv2.normalize(hist, hist)
        query_sig.extend(hist.flatten().tolist())
        query_sig.append(float(np.mean(gray_diff)))
        query_sig.append(float(np.std(gray_diff)))
        query_sig.append(float(np.percentile(gray_diff, 95)))
        query_sig.append(float(np.mean(gray_diff > 0.1)))
        query_sig = np.array(query_sig, dtype=np.float32)

        # Calcula distância para todos os vizinhos
        distances = []
        for sig, path in self.signatures_ok:
            dist = np.linalg.norm(query_sig - sig)
            distances.append((dist, "OK", path))
        for sig, path in self.signatures_ng:
            dist = np.linalg.norm(query_sig - sig)
            distances.append((dist, "NG", path))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:top_k]

        if not neighbors:
            return {"has_memory": False, "vote_defect": 0.5, "best_similarity": 0.0, "n_neighbors": 0}

        # Voto ponderado por distância inversa (vizinhos mais próximos pesam mais)
        votes_ng = 0.0
        votes_total = 0.0
        for dist, label, _ in neighbors:
            weight = 1.0 / max(dist, 0.001)  # Peso inversamente proporcional à distância
            if label == "NG":
                votes_ng += weight
            votes_total += weight

        vote_defect = votes_ng / votes_total if votes_total > 0 else 0.5

        # Similaridade do melhor vizinho (0-1, quanto menor a distância, mais similar)
        best_dist = neighbors[0][0]
        best_similarity = max(0.0, 1.0 - (best_dist / 5.0))  # Normaliza grosseiramente

        return {
            "has_memory": True,
            "vote_defect": vote_defect,
            "best_similarity": best_similarity,
            "n_neighbors": len(neighbors)
        }


class NeuralJudge:
    def __init__(self):
        print("🧠 Iniciando Juiz Neural v3 (Análise Semântica + Memória de Dataset)...")
        self.memory = DatasetMemory()

    def reload_memory(self):
        """Recarrega o banco de dados (chamar após salvar novas amostras)."""
        print("🔄 Recarregando memória do dataset...")
        self.memory = DatasetMemory()

    def _analyze_local(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> dict:
        """
        CAMADA 1 — Análise Local (quadrado menor):
        Comparação pixel-a-pixel do recorte suspeito com o equivalente do padrão.
        Usa SSIM (qualidade estrutural) + diferença absoluta + análise de bordas.
        """
        size = (64, 64)
        gab = cv2.resize(crop_gab, size)
        test = cv2.resize(crop_test, size)

        gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        # 1. SSIM — índice de similaridade estrutural (0.0 = nada parecido, 1.0 = idêntico)
        ssim_score, ssim_map = ssim(gray_gab, gray_test, full=True)

        # 2. Diferença absoluta
        diff = cv2.absdiff(gray_gab, gray_test).astype(np.float32) / 255.0
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))
        pct_changed = float(np.mean(diff > 0.12))  # % de pixels que mudaram >12%

        # 3. Análise de bordas (Canny) — defeitos reais geralmente alteram bordas
        edges_gab = cv2.Canny(gray_gab, 50, 150)
        edges_test = cv2.Canny(gray_test, 50, 150)
        edge_diff = cv2.absdiff(edges_gab, edges_test)
        edge_change = float(np.mean(edge_diff > 0))

        # 4. Correlação de histograma
        hist_gab = cv2.calcHist([gray_gab], [0], None, [64], [0, 256])
        hist_test = cv2.calcHist([gray_test], [0], None, [64], [0, 256])
        cv2.normalize(hist_gab, hist_gab)
        cv2.normalize(hist_test, hist_test)
        hist_corr = float(cv2.compareHist(hist_gab, hist_test, cv2.HISTCMP_CORREL))

        return {
            "ssim": ssim_score,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "pct_changed": pct_changed,
            "edge_change": edge_change,
            "hist_corr": hist_corr
        }

    def _analyze_context(self, full_gab: np.ndarray, full_test: np.ndarray,
                          box_x: int, box_y: int, box_w: int, box_h: int) -> dict:
        """
        CAMADA 2 — Análise de Contexto (quadro maior):
        Olha uma região expandida ao redor do defeito suspeito para entender
        se a diferença é localizada (provável defeito) ou espalhada (provável ruído/alinhamento).
        """
        h_full, w_full = full_gab.shape[:2]

        # Expande a região 3x ao redor do suspeito
        expand = max(box_w, box_h)
        cx1 = max(0, box_x - expand)
        cy1 = max(0, box_y - expand)
        cx2 = min(w_full, box_x + box_w + expand)
        cy2 = min(h_full, box_y + box_h + expand)

        ctx_gab = full_gab[cy1:cy2, cx1:cx2]
        ctx_test = full_test[cy1:cy2, cx1:cx2]

        if ctx_gab.size == 0 or ctx_test.size == 0:
            return {"ctx_ssim": 1.0, "is_localized": True}

        size = (96, 96)
        ctx_gab_r = cv2.resize(ctx_gab, size)
        ctx_test_r = cv2.resize(ctx_test, size)

        gray_gab = cv2.cvtColor(ctx_gab_r, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(ctx_test_r, cv2.COLOR_BGR2GRAY)

        ctx_ssim, ctx_ssim_map = ssim(gray_gab, gray_test, full=True)

        # Analisa se a diferença é LOCALIZADA (defeito real) ou ESPALHADA (ruído)
        diff_map = (1.0 - ctx_ssim_map)
        diff_map_norm = (diff_map * 255).astype(np.uint8)
        _, diff_thresh = cv2.threshold(diff_map_norm, 80, 255, cv2.THRESH_BINARY)

        # Conta quantos "clusters" de diferença existem
        contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_clusters = len(contours)

        # Se tem poucos clusters concentrados → localizado → mais provável ser defeito real
        # Se tem muitos clusters espalhados → ruído/desalinhamento → falso positivo
        is_localized = n_clusters <= 3

        return {
            "ctx_ssim": ctx_ssim,
            "n_clusters": n_clusters,
            "is_localized": is_localized
        }

    def verify_anomaly(self, crop_gab: np.ndarray, crop_test: np.ndarray,
                        full_gab: np.ndarray = None, full_test: np.ndarray = None,
                        box_x: int = 0, box_y: int = 0,
                        box_w: int = 0, box_h: int = 0) -> dict:
        """
        Veredito final: combina as 3 camadas de análise.
        NÃO adivinha — cada decisão é calculada e justificada.
        """
        # Proteção contra recortes minúsculos/vazios
        if (crop_gab is None or crop_test is None or
                crop_gab.size == 0 or crop_test.size == 0 or
                crop_gab.shape[0] < 5 or crop_gab.shape[1] < 5):
            return {
                "is_defect": False,
                "confidence": 0.99,
                "score_text": "99%",
                "verdict": "IGNORADO",
                "reason": "Recorte muito pequeno para análise"
            }

        # === CAMADA 1: Análise Local ===
        local = self._analyze_local(crop_gab, crop_test)

        # === CAMADA 2: Análise de Contexto (se disponível) ===
        context = None
        if full_gab is not None and full_test is not None and box_w > 0:
            context = self._analyze_context(full_gab, full_test, box_x, box_y, box_w, box_h)

        # === CAMADA 3: Consulta ao Banco de Dados ===
        db_result = self.memory.query_similar(crop_gab, crop_test)

        # === FUSÃO INTELIGENTE: Decisão baseada em evidências ===
        #
        # Score de anomalia: 0.0 = certeza que é OK, 1.0 = certeza que é defeito
        #
        # Peso de cada camada:
        #   Local (SSIM + diff + bordas): 50% — é a evidência direta
        #   Contexto (localização):       20% — confirma se é concentrado ou espalhado
        #   Dataset (memória humana):     30% — experiência passada do operador

        # --- Score Local ---
        # SSIM < 0.85 = diferença significativa; mean_diff > 0.08 = mudança visível
        local_score = 0.0
        local_score += max(0, (0.92 - local["ssim"]) / 0.92) * 0.35       # SSIM contribui 35%
        local_score += min(1.0, local["mean_diff"] / 0.15) * 0.20          # Diff média 20%
        local_score += min(1.0, local["pct_changed"] / 0.25) * 0.20        # % pixels mudados 20%
        local_score += min(1.0, local["edge_change"] / 0.15) * 0.15        # Mudança de bordas 15%
        local_score += max(0, (0.95 - local["hist_corr"]) / 0.95) * 0.10   # Histograma 10%
        local_score = max(0.0, min(1.0, local_score))

        # --- Score de Contexto ---
        ctx_score = 0.5  # Neutro se não tiver contexto
        ctx_reason = ""
        if context is not None:
            if context["is_localized"]:
                # Diferença localizada: reforça que pode ser defeito real
                ctx_score = 0.7
                ctx_reason = f"diferença concentrada ({context['n_clusters']} região(ões))"
            else:
                # Diferença espalhada: sugere ruído ou desalinhamento
                ctx_score = 0.25
                ctx_reason = f"diferença espalhada ({context['n_clusters']} regiões)"

        # --- Score do Dataset (voto dos vizinhos) ---
        db_score = 0.5  # Neutro se não tiver memória
        db_reason = ""
        if db_result["has_memory"]:
            db_score = db_result["vote_defect"]
            n = db_result["n_neighbors"]
            db_reason = f"dataset votou {db_score:.0%} defeito ({n} vizinhos)"

        # --- Fusão Ponderada ---
        if db_result["has_memory"]:
            final_score = local_score * 0.50 + ctx_score * 0.20 + db_score * 0.30
        else:
            # Sem memória: local pesa mais
            final_score = local_score * 0.65 + ctx_score * 0.35

        # --- Decisão ---
        threshold = 0.45
        is_defect = final_score > threshold

        # --- Confiança: quão longe da fronteira estamos ---
        distance_from_threshold = abs(final_score - threshold)
        confidence = min(0.99, 0.50 + distance_from_threshold * 2.5)
        conf_percent = max(50, min(99, int(confidence * 100)))

        # --- Justificativa legível ---
        if is_defect:
            verdict = "DEFEITO REAL"
        else:
            verdict = "FALHA FALSA"

        reasons = []
        reasons.append(f"SSIM={local['ssim']:.2f}")
        reasons.append(f"Δpixels={local['pct_changed']:.0%}")
        reasons.append(f"Δbordas={local['edge_change']:.0%}")
        if ctx_reason:
            reasons.append(ctx_reason)
        if db_reason:
            reasons.append(db_reason)

        return {
            "is_defect": is_defect,
            "confidence": confidence,
            "score_text": f"{conf_percent}%",
            "verdict": verdict,
            "reason": " | ".join(reasons),
            # Dados detalhados para o painel
            "detail": {
                "local_score": local_score,
                "ctx_score": ctx_score,
                "db_score": db_score,
                "final_score": final_score,
                "ssim": local["ssim"],
                "mean_diff": local["mean_diff"],
                "pct_changed": local["pct_changed"],
                "edge_change": local["edge_change"],
                "hist_corr": local["hist_corr"],
                "db_has_memory": db_result["has_memory"],
                "db_vote": db_result["vote_defect"],
                "db_neighbors": db_result["n_neighbors"],
                "db_best_sim": db_result["best_similarity"]
            }
        }