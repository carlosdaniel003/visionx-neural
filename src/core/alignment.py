"""
Módulo de Alinhamento Geométrico.
Responsável por corrigir rotações, escalas e perspectivas entre duas imagens usando ORB e Homografia.
"""
import cv2
import numpy as np

def align_images(img_gabarito: np.ndarray, img_teste: np.ndarray, max_features: int = 5000) -> np.ndarray:
    """
    Alinha a imagem de teste sobre o gabarito.
    
    :param img_gabarito: A imagem perfeita (padrão).
    :param img_teste: A imagem recortada da tela ao vivo.
    :param max_features: Quantidade máxima de pontos-chave a procurar.
    :return: A imagem de teste deformada/alinhada para bater com o gabarito.
    """
    # 1. Converte para escala de cinza para o algoritmo ORB
    gray_gab = cv2.cvtColor(img_gabarito, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_teste, cv2.COLOR_BGR2GRAY)

    # 2. Inicializa o ORB e extrai os pontos-chave (Keypoints e Descritores)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_gab, des_gab = orb.detectAndCompute(gray_gab, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    # Proteção: Se a imagem for uma tela branca/preta sem detalhes, aborta.
    if des_gab is None or des_test is None:
        return img_teste

    # 3. Compara os pontos usando Força Bruta (Brute-Force Hamming)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_test, des_gab) # Mapeia do teste PARA o gabarito
    
    # 4. Ordena pela menor distância (melhores matches primeiro)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Fica apenas com os 15% melhores matches (Filtro de Elite)
    keep = int(len(matches) * 0.15)
    good_matches = matches[:keep]

    # Precisamos de pelo menos 4 pontos para a matriz de perspectiva, 10 por segurança
    if len(good_matches) < 10:
        return img_teste

    # 6. Extrai as coordenadas XY dos matches de elite
    pts_test = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_gab = np.float32([kp_gab[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 7. Calcula a matriz matemática da deformação (Homografia com filtro de ruído RANSAC)
    matrix, mask = cv2.findHomography(pts_test, pts_gab, cv2.RANSAC, 5.0)

    # 8. Aplica a matriz de perspectiva para 'esticar' a imagem de teste
    if matrix is not None:
        h, w = img_gabarito.shape[:2]
        aligned_test = cv2.warpPerspective(img_teste, matrix, (w, h))
        return aligned_test

    return img_teste