"""
Módulo do Juiz Neural v2 (ONNX Runtime - Sem PyTorch).
Usa um modelo treinado com seu dataset local para verificar anomalias.
Fallback inteligente: se não houver modelo treinado, usa análise por histograma.
"""
import cv2
import numpy as np
from pathlib import Path

# Tenta importar ONNX Runtime (leve, sem DLL problemática)
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "visionx_judge.onnx"


class NeuralJudge:
    def __init__(self):
        self.session = None
        self.mode = "FALLBACK"

        if HAS_ONNX and MODEL_PATH.exists():
            try:
                self.session = ort.InferenceSession(str(MODEL_PATH))
                self.input_name = self.session.get_inputs()[0].name
                self.mode = "ONNX"
                print("🧠 Juiz Neural: Modelo ONNX customizado carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar modelo ONNX: {e}")
                self.session = None

        if self.session is None:
            print("🧠 Juiz Neural: Modo FALLBACK (Histograma + Estrutural). Treine o modelo para melhorar!")

    def _preprocess_pair(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> np.ndarray:
        """
        Pré-processa um par de imagens para o formato esperado pelo modelo ONNX.
        Gera um vetor de features comparativas entre as duas imagens.
        """
        size = (64, 64)
        gab = cv2.resize(crop_gab, size)
        test = cv2.resize(crop_test, size)

        # Feature 1: Diferença absoluta normalizada (canal por canal)
        diff = cv2.absdiff(gab, test).astype(np.float32) / 255.0

        # Feature 2: Diferença em escala de cinza
        gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_diff = np.abs(gray_gab - gray_test)

        # Estatísticas globais do diff
        features = []

        # Média e desvio padrão da diferença por canal BGR
        for c in range(3):
            features.append(float(np.mean(diff[:, :, c])))
            features.append(float(np.std(diff[:, :, c])))

        # Média e desvio da diferença em cinza
        features.append(float(np.mean(gray_diff)))
        features.append(float(np.std(gray_diff)))

        # Percentual de pixels com diferença significativa (> 15%)
        features.append(float(np.mean(gray_diff > 0.15)))

        # Diferença de histograma (correlação) por canal
        for c in range(3):
            hist_gab = cv2.calcHist([gab], [c], None, [32], [0, 256])
            hist_test = cv2.calcHist([test], [c], None, [32], [0, 256])
            cv2.normalize(hist_gab, hist_gab)
            cv2.normalize(hist_test, hist_test)
            corr = cv2.compareHist(hist_gab, hist_test, cv2.HISTCMP_CORREL)
            features.append(float(corr))

        # Diferença de histograma em cinza
        hist_g_gab = cv2.calcHist([cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        hist_g_test = cv2.calcHist([cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
        cv2.normalize(hist_g_gab, hist_g_gab)
        cv2.normalize(hist_g_test, hist_g_test)
        features.append(float(cv2.compareHist(hist_g_gab, hist_g_test, cv2.HISTCMP_CORREL)))

        # Diferença de energia (Laplaciano — detecta mudanças de textura/borda)
        lap_gab = cv2.Laplacian(cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        lap_test = cv2.Laplacian(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        features.append(float(abs(lap_gab - lap_test) / max(lap_gab, lap_test, 1.0)))

        return np.array(features, dtype=np.float32).reshape(1, -1)

    def _fallback_analysis(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> dict:
        """
        Análise sem modelo treinado: usa histograma + diferença estrutural.
        Mais conservadora (prefere dizer que É defeito na dúvida).
        """
        size = (64, 64)
        gab = cv2.resize(crop_gab, size)
        test = cv2.resize(crop_test, size)

        # 1. Correlação de histograma em cinza
        gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        hist_gab = cv2.calcHist([gray_gab], [0], None, [64], [0, 256])
        hist_test = cv2.calcHist([gray_test], [0], None, [64], [0, 256])
        cv2.normalize(hist_gab, hist_gab)
        cv2.normalize(hist_test, hist_test)
        hist_corr = cv2.compareHist(hist_gab, hist_test, cv2.HISTCMP_CORREL)

        # 2. Percentual de pixels diferentes
        diff = cv2.absdiff(gray_gab, gray_test)
        pct_diff = float(np.mean(diff > 35))  # pixels com diff > 35 de 255

        # 3. Diferença média normalizada
        mean_diff = float(np.mean(diff)) / 255.0

        # Decisão: combinação ponderada
        # hist_corr perto de 1.0 = muito similar, perto de 0 = diferente
        # Se a correlação for alta E poucos pixels mudaram → provavelmente falso positivo
        score = (1.0 - hist_corr) * 0.4 + pct_diff * 0.35 + mean_diff * 0.25

        threshold = 0.12  # Calibrado para ser conservador
        is_defect = score > threshold

        # Mapeamento de confiança
        if is_defect:
            confidence = min(0.99, 0.5 + score * 2.5)
        else:
            confidence = min(0.99, 0.5 + (threshold - score) * 4.0)

        conf_percent = max(50, min(99, int(confidence * 100)))

        return {
            "is_defect": is_defect,
            "confidence": confidence,
            "score_text": f"{conf_percent}%"
        }

    def verify_anomaly(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> dict:
        """
        Verifica se uma anomalia detectada pela matemática é real ou falso-positivo.
        """
        # Proteção contra recortes minúsculos/vazios
        if crop_gab.size == 0 or crop_test.size == 0 or crop_gab.shape[0] < 5 or crop_gab.shape[1] < 5:
            return {"is_defect": False, "confidence": 0.99, "score_text": "99%"}

        # Se temos o modelo ONNX treinado, usa ele
        if self.mode == "ONNX" and self.session is not None:
            try:
                features = self._preprocess_pair(crop_gab, crop_test)
                # O modelo retorna [probabilidade_OK, probabilidade_NG]
                outputs = self.session.run(None, {self.input_name: features})
                probabilities = outputs[0][0]

                is_defect = bool(probabilities[1] > probabilities[0])
                confidence = float(max(probabilities))
                conf_percent = max(50, min(99, int(confidence * 100)))

                return {
                    "is_defect": is_defect,
                    "confidence": confidence,
                    "score_text": f"{conf_percent}%"
                }
            except Exception as e:
                print(f"⚠️ Erro na inferência ONNX, usando fallback: {e}")

        # Fallback: análise por histograma (funciona sem modelo treinado)
        return self._fallback_analysis(crop_gab, crop_test)