# src\scripts\train_semantic.py
"""
Módulo de Treinamento Semântico (Fine-Tuning).
Treina a rede MobileNetV2 usando as imagens coletadas pela fábrica.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from src.config.settings import settings
import os

def train_model():
    print("🧠 Iniciando a Academia de Treinamento VisionX Neural...")
    
    # 1. Configuração de Pastas
    data_dir = str(settings.DATASET_DIR)
    
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"❌ ERRO: A pasta {data_dir} não existe ou está vazia.")
        return

    # 2. A MÁGICA: Data Augmentation (Multiplicador de Imagens)
    # Isso transforma suas 10 fotos em milhares de variações na memória
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Dá um zoom aleatório de até 20%
        transforms.RandomHorizontalFlip(), # Espelha horizontalmente
        transforms.RandomVerticalFlip(),   # Espelha verticalmente
        transforms.RandomRotation(15),     # Gira até 15 graus
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Altera levemente a luz e sombra da AOI
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Carregando o Dataset
    try:
        # O ImageFolder lê os nomes das pastas (anomalia / nao_anomalia) e transforma em Classes!
        dataset = datasets.ImageFolder(data_dir, train_transforms)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        class_names = dataset.classes
        print(f"📦 Dataset carregado! Encontradas {len(dataset)} imagens nas categorias: {class_names}")
    except Exception as e:
        print(f"❌ ERRO ao ler imagens: {e}")
        print("Certifique-se de que as pastas 'anomalia' e 'nao_anomalia' existem dentro de public/dataset/")
        return

    # Verifica se tem GPU (Placa de vídeo NVIDIA) ou se vai usar CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Processamento de Treino rodando em: {device}")

    # 4. Preparando o Cérebro (MobileNetV2)
    print("📥 Baixando/Carregando arquitetura MobileNetV2...")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Congela as primeiras camadas (Elas já sabem ver bordas, linhas e cores perfeitamente)
    for param in model.features.parameters():
        param.requires_grad = False

    # Substitui a última camada (que tinha 1000 categorias de animais/objetos) 
    # por apenas 2 categorias: (0: anomalia, 1: nao_anomalia)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Função de erro e Otimizador (O professor que corrige a rede)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 5. O Loop de Treinamento (As "Aulas")
    epochs = 20 # 20 passadas completas pelas imagens
    print("\n🚀 Iniciando o Treinamento...\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Zera a memória do professor

            outputs = model(inputs) # A IA tenta adivinhar
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) # O professor calcula o quão errado ela foi

            loss.backward() # Estuda o erro
            optimizer.step() # Ajusta os neurônios

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = corrects.double() / len(dataset)

        print(f"Época {epoch+1}/{epochs} | Margem de Erro: {epoch_loss:.4f} | Acertos (Acurácia): {epoch_acc:.4f}")

    # 6. Salva a mente da IA no disco
    save_path = str(settings.BASE_DIR / "visionx_neural_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Treinamento Concluído! Cérebro salvo em: {save_path}")

if __name__ == "__main__":
    train_model()