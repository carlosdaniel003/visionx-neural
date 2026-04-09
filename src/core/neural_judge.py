"""
Módulo do Juiz Neural (Rede Siamesa Zero-Shot).
Utiliza um backbone profundo para verificar se uma anomalia matemática é real ou falso-positivo,
retornando o grau de confiança (Confidence Score).
"""
import torch
import numpy as np
from torchvision import models, transforms
import warnings

# Oculta avisos padrão do PyTorch
warnings.filterwarnings("ignore")

class NeuralJudge:
    def __init__(self):
        print("🧠 Iniciando Cérebro Neural (MobileNetV2) com Confidence Scoring...")
        # Carrega a IA pré-treinada
        weights = models.MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        
        # Removemos o "classificador". Ficamos apenas com os "Olhos" da IA (Feature Extractor)
        self.backbone = base_model.features
        self.backbone.eval() # Modo de inferência
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def verify_anomaly(self, crop_gab: np.ndarray, crop_test: np.ndarray) -> dict:
        """
        Calcula a distância semântica e devolve um dicionário com o veredito e a confiança.
        """
        # Se a imagem for demasiado pequena, o filtro matemático errou. Retorna Falha Falsa com alta confiança.
        if crop_gab.size == 0 or crop_test.size == 0 or crop_gab.shape[0] < 5 or crop_gab.shape[1] < 5:
            return {"is_defect": False, "confidence": 0.99, "score_text": "99%"}

        with torch.no_grad():
            tensor_gab = self.preprocess(crop_gab).unsqueeze(0)
            tensor_test = self.preprocess(crop_test).unsqueeze(0)

            feat_gab = self.backbone(tensor_gab).flatten()
            feat_test = self.backbone(tensor_test).flatten()

            # Similaridade de Cosseno (1.0 = idêntico, 0.0 = totalmente diferente)
            cos_sim = torch.nn.functional.cosine_similarity(feat_gab.unsqueeze(0), feat_test.unsqueeze(0)).item()
            
            # --- LÓGICA DE CONFIDENCE SCORE ---
            # O nosso limiar de decisão: abaixo de 0.92 consideramos um defeito estrutural.
            threshold = 0.92
            
            if cos_sim < threshold:
                is_real_defect = True
                # Quanto MENOR a similaridade (mais longe do 0.92 para baixo), MAIOR a confiança de ser Defeito.
                # Mapeia o intervalo [0.0, 0.92] para [100%, 50%]
                confidence = 1.0 - (cos_sim / threshold) * 0.5 
            else:
                is_real_defect = False
                # Quanto MAIOR a similaridade (mais longe do 0.92 para cima), MAIOR a confiança de ser Falha Falsa (OK).
                # Mapeia o intervalo [0.92, 1.0] para [50%, 100%]
                range_ok = 1.0 - threshold
                confidence = 0.5 + ((cos_sim - threshold) / range_ok) * 0.5

            # Formata para percentagem legível (ex: 94%)
            conf_percent = int(confidence * 100)
            # Trava o valor entre 50% e 99% para nunca dizer "100% de certeza" (nenhuma IA deve ter 100%)
            conf_percent = max(50, min(99, conf_percent)) 

            return {
                "is_defect": is_real_defect,
                "confidence": confidence,
                "score_text": f"{conf_percent}%"
            }