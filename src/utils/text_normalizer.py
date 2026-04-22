# src/utils/text_normalizer.py
"""
Módulo de Normalização de Texto OCR via Fuzzy Matching.
Garante que os defeitos lidos na máquina sejam categorizados perfeitamente.
"""
import re
import difflib

# As 9 Categorias Oficiais
CATEGORIES = [
    "Bridge", "Dust", "Little Solder", "Missing", 
    "Much Adhesive", "No solder", "Reverse", 
    "Shifted", "Up Side Down"
]

# Mapa de segurança para os erros de OCR mais bizarros e frequentes
ALIASES = {
    "MUSING": "Missing",
    "MISSMG": "Missing",
    "SHUFTED": "Shifted",
    "STFTED": "Shifted",
    "STUFTED": "Shifted",
    "SIFTED": "Shifted",
    "UP SIDE DOAN": "Up Side Down",
    "UP SIDE DOWM": "Up Side Down"
}

def normalize_aoi_text(ocr_text: str):
    """
    Recebe um texto sujo (ex: '0 <= 82 <= 10 Sifted')
    Retorna a categoria limpa ('Shifted') e a string reconstruída ('0 <= 82 <= 10 Shifted').
    """
    if not ocr_text or ocr_text == "-":
        return "Unknown", ocr_text

    # 1. Limpeza pesada: Remove números e sinais matemáticos/pontuação
    text_only = re.sub(r'[\d\<\>\=\~\?\|\(\)\[\]\{\}\_\-\,]', ' ', ocr_text)
    
    # 2. Remove espaços extras que sobraram
    text_only = ' '.join(text_only.split())
    
    if not text_only:
        return "Unknown", ocr_text

    upper_text = text_only.upper()
    best_match = "Unknown"

    # 3. Testa o Dicionário de Apelidos (Aliases rápidos)
    if upper_text in ALIASES:
        best_match = ALIASES[upper_text]
    else:
        # 4. Fuzzy Matching (Similaridade Matemática)
        # Tenta achar uma categoria que seja pelo menos 60% igual à palavra lida
        matches = difflib.get_close_matches(text_only, CATEGORIES, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
        else:
            # Fallback ignorando maiúsculas e minúsculas
            upper_categories = [c.upper() for c in CATEGORIES]
            matches_upper = difflib.get_close_matches(upper_text, upper_categories, n=1, cutoff=0.5)
            if matches_upper:
                idx = upper_categories.index(matches_upper[0])
                best_match = CATEGORIES[idx]

    # 5. Reconstrói a string original trocando a sujeira pelo acerto
    if best_match != "Unknown" and text_only:
        try:
            # Substitui a palavra lida pelo match oficial, preservando todo o resto (números e sinais)
            normalized_value = re.sub(re.escape(text_only), best_match, ocr_text, flags=re.IGNORECASE)
        except Exception:
            normalized_value = ocr_text
    else:
        normalized_value = ocr_text

    return best_match, normalized_value