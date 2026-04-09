"""
Script de Treinamento do Juiz Neural VisionX.
Treina um classificador usando as imagens curadas do dataset local.

Uso:
    python -m src.core.train_model

Pré-requisito:
    pip install scikit-learn Pillow
    
    Ter imagens em:
    - public/dataset/nao_anomalia/   (imagens rotuladas como OK)
    - public/dataset/anomalia/       (imagens rotuladas como NG / Defeito Real)
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "public" / "dataset"
ANOMALY_DIR = DATASET_DIR / "anomalia"
NORMAL_DIR = DATASET_DIR / "nao_anomalia"
MODEL_OUTPUT = BASE_DIR / "models" / "visionx_judge.pkl"


def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extrai o mesmo vetor de features que o NeuralJudge usa na inferência.
    
    IMPORTANTE: Como as imagens salvas pelo DatasetManager são PARES 
    (sample + ng lado a lado via hstack), precisamos separar as duas metades.
    """
    h, w = img_bgr.shape[:2]
    mid = w // 2
    
    crop_gab = img_bgr[:, :mid]    # Metade esquerda = Sample (Padrão)
    crop_test = img_bgr[:, mid:]    # Metade direita = NG (Teste)
    
    size = (64, 64)
    gab = cv2.resize(crop_gab, size)
    test = cv2.resize(crop_test, size)
    
    diff = cv2.absdiff(gab, test).astype(np.float32) / 255.0
    
    gray_gab = cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_diff = np.abs(gray_gab - gray_test)
    
    features = []
    
    # Média e desvio por canal BGR
    for c in range(3):
        features.append(float(np.mean(diff[:, :, c])))
        features.append(float(np.std(diff[:, :, c])))
    
    # Média e desvio em cinza
    features.append(float(np.mean(gray_diff)))
    features.append(float(np.std(gray_diff)))
    
    # Percentual de pixels com diferença > 15%
    features.append(float(np.mean(gray_diff > 0.15)))
    
    # Correlação de histograma por canal
    for c in range(3):
        hist_gab = cv2.calcHist([gab], [c], None, [32], [0, 256])
        hist_test = cv2.calcHist([test], [c], None, [32], [0, 256])
        cv2.normalize(hist_gab, hist_gab)
        cv2.normalize(hist_test, hist_test)
        corr = cv2.compareHist(hist_gab, hist_test, cv2.HISTCMP_CORREL)
        features.append(float(corr))
    
    # Histograma em cinza
    hist_g_gab = cv2.calcHist([cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
    hist_g_test = cv2.calcHist([cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)], [0], None, [32], [0, 256])
    cv2.normalize(hist_g_gab, hist_g_gab)
    cv2.normalize(hist_g_test, hist_g_test)
    features.append(float(cv2.compareHist(hist_g_gab, hist_g_test, cv2.HISTCMP_CORREL)))
    
    # Diferença de energia (Laplaciano)
    lap_gab = cv2.Laplacian(cv2.cvtColor(gab, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    lap_test = cv2.Laplacian(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    features.append(float(abs(lap_gab - lap_test) / max(lap_gab, lap_test, 1.0)))
    
    return np.array(features, dtype=np.float32)


def load_dataset():
    """Carrega todas as imagens do dataset e extrai features."""
    X = []
    y = []
    
    # Classe 0 = OK (não é defeito real, falha falsa)
    ok_files = list(NORMAL_DIR.glob("*.png")) + list(NORMAL_DIR.glob("*.jpg"))
    print(f"📂 Imagens OK (nao_anomalia): {len(ok_files)}")
    for f in ok_files:
        img = cv2.imread(str(f))
        if img is not None:
            feat = extract_features_from_image(img)
            X.append(feat)
            y.append(0)  # OK
    
    # Classe 1 = NG (defeito real confirmado)
    ng_files = list(ANOMALY_DIR.glob("*.png")) + list(ANOMALY_DIR.glob("*.jpg"))
    print(f"📂 Imagens NG (anomalia): {len(ng_files)}")
    for f in ng_files:
        img = cv2.imread(str(f))
        if img is not None:
            feat = extract_features_from_image(img)
            X.append(feat)
            y.append(1)  # NG
    
    return np.array(X), np.array(y)


def train():
    print("=" * 60)
    print("  VisionX Neural - Treinamento do Juiz Neural")
    print("=" * 60)
    
    X, y = load_dataset()
    
    total = len(y)
    if total < 10:
        print(f"\n❌ Dataset muito pequeno ({total} imagens).")
        print("   Mínimo recomendado: 10 imagens (5 OK + 5 NG)")
        print("   Use o botão 'Salvar como Falha Falsa (OK)' e 'Confirmar Defeito Real (NG)' no app.")
        return
    
    n_ok = int(np.sum(y == 0))
    n_ng = int(np.sum(y == 1))
    print(f"\n📊 Dataset: {total} amostras ({n_ok} OK, {n_ng} NG)")
    
    if n_ok == 0 or n_ng == 0:
        print("❌ Precisa ter pelo menos 1 imagem de cada classe (OK e NG).")
        return
    
    # Treina com Gradient Boosting (excelente para features tabulares, leve, sem GPU)
    print("\n🔧 Treinando modelo GradientBoosting...")
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42
    )
    
    # Cross-validation se tiver amostras suficientes
    if total >= 20:
        n_splits = min(5, n_ok, n_ng)
        if n_splits >= 2:
            scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
            print(f"📈 Cross-Validation ({n_splits}-fold): {scores.mean():.1%} ± {scores.std():.1%}")
    
    # Treina com tudo para o modelo final
    model.fit(X, y)
    
    # Relatório no dataset completo
    y_pred = model.predict(X)
    print("\n📋 Relatório de Treino:")
    print(classification_report(y, y_pred, target_names=["OK (Falha Falsa)", "NG (Defeito Real)"]))
    
    # Salva o modelo
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(MODEL_OUTPUT), 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Modelo salvo em: {MODEL_OUTPUT}")
    
    # Exporta automaticamente para ONNX
    print("\n🔄 Exportando para ONNX...")
    export_to_onnx(model, X.shape[1])
    

def export_to_onnx(model, n_features: int):
    """Converte o modelo sklearn para ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        initial_type = [('input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type,
                                     options={id(model): {'zipmap': False}})
        
        onnx_path = MODEL_OUTPUT.parent / "visionx_judge.onnx"
        with open(str(onnx_path), 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ Modelo ONNX salvo em: {onnx_path}")
        
    except ImportError:
        print("⚠️ Para exportar ONNX, instale: pip install skl2onnx")
        print("   O modelo .pkl foi salvo e pode ser usado diretamente.")


if __name__ == "__main__":
    train()