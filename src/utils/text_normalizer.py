# src/utils/text_normalizer.py
"""
Módulo de Normalização de Texto OCR via Fuzzy Matching.
Ajuste: Tokenização inteligente. Agora a IA avalia palavra por palavra e ignora ruídos numéricos.
Ajuste 2: Rastreabilidade. Se falhar, exibe no Painel exatamente quais palavras ele testou e falhou.
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
    "SH1FTED": "Shifted", # Comum quando OCR lê 'i' como '1'
    "UP SIDE DOAN": "Up Side Down",
    "UP SIDE DOWM": "Up Side Down"
}

def normalize_aoi_text(ocr_text: str):
    """
    Recebe um texto sujo (ex: '0 <= 82 <= 10 Sifted')
    Retorna a categoria limpa ('Shifted') e a string reconstruída ('0 <= 82 <= 10 Shifted').
    Se falhar, retorna 'Unknown - Tentou: [palavras]'.
    """
    if not ocr_text or ocr_text.strip() == "-" or ocr_text.strip() == "":
        return "Unknown", ocr_text

    # Substitui caracteres especiais (não-alfanuméricos) por espaço para separar "tokens"
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', ocr_text)
    tokens = clean_text.split()
    
    best_match = "Unknown"
    
    # ESTRATÉGIA 1: Verifica Aliases nas combinações de Tokens
    upper_full = " ".join(tokens).upper()
    for alias, correct_cat in ALIASES.items():
        if alias in upper_full:
            best_match = correct_cat
            break
            
    # Variável para rastreabilidade do que a IA testou
    valid_words = []

    # ESTRATÉGIA 2: Fuzzy Matching Progressivo (Se o Alias não funcionou)
    if best_match == "Unknown":
        upper_categories = [c.upper() for c in CATEGORIES]
        
        # Filtra os tokens: só avalia "palavras" que tenham pelo menos 3 letras e que não sejam apenas números
        valid_words = [t.upper() for t in tokens if len(t) >= 3 and not t.isdigit()]
        
        for word in valid_words:
            # 2.1 Tenta achar na palavra isolada (ex: "Shifted")
            matches = difflib.get_close_matches(word, upper_categories, n=1, cutoff=0.6)
            if matches:
                idx = upper_categories.index(matches[0])
                best_match = CATEGORIES[idx]
                break
                
        # 2.2 Tenta achar em blocos de 2 e 3 palavras (para pegar "Little Solder" e "Up Side Down")
        if best_match == "Unknown" and len(valid_words) >= 2:
            for i in range(len(valid_words) - 1):
                bigram = f"{valid_words[i]} {valid_words[i+1]}"
                matches = difflib.get_close_matches(bigram, upper_categories, n=1, cutoff=0.65)
                if matches:
                    idx = upper_categories.index(matches[0])
                    best_match = CATEGORIES[idx]
                    break
                    
        if best_match == "Unknown" and len(valid_words) >= 3:
            for i in range(len(valid_words) - 2):
                trigram = f"{valid_words[i]} {valid_words[i+1]} {valid_words[i+2]}"
                matches = difflib.get_close_matches(trigram, upper_categories, n=1, cutoff=0.65)
                if matches:
                    idx = upper_categories.index(matches[0])
                    best_match = CATEGORIES[idx]
                    break

    # =======================================================
    # RECONSTRUÇÃO DA STRING ORIGINAL E RASTREABILIDADE
    # =======================================================
    normalized_value = ocr_text
    if best_match != "Unknown":
        # Se achou uma categoria válida, a gente reconstrói o texto do OCR trocando a parte defeituosa
        # Cria uma regex que busca a palavra de forma tolerante (ignorando números e lixo no meio)
        # Ex: Para trocar "Sifted", a gente varre a string e troca pela palavra oficial
        
        if best_match == "Up Side Down":
             normalized_value = re.sub(r'(?i)(up.{0,3}side.{0,3}dow[nm])', best_match, ocr_text)
        elif best_match == "Little Solder":
             normalized_value = re.sub(r'(?i)(little.{0,3}solder)', best_match, ocr_text)
        elif best_match == "Much Adhesive":
             normalized_value = re.sub(r'(?i)(much.{0,3}adhesive)', best_match, ocr_text)
        elif best_match == "No solder":
             normalized_value = re.sub(r'(?i)(no.{0,3}solder)', best_match, ocr_text)
        else:
             # Para palavras únicas (Shifted, Missing, Bridge, Reverse, Dust)
             # Buscamos a palavra no OCR que mais se parece com o best_match e a trocamos
             for word in ocr_text.replace("<", " ").replace(">", " ").replace("=", " ").split():
                 if len(word) >= 3 and not word.isdigit():
                     if difflib.get_close_matches(word.upper(), [best_match.upper()], n=1, cutoff=0.5):
                         normalized_value = ocr_text.replace(word, best_match)
                         break
    else:
        # Se chegou aqui como Unknown, nós devolvemos o que ele testou para podermos debugar!
        tested_terms = ", ".join(valid_words) if valid_words else "Nenhuma palavra válida"
        best_match = f"Unknown - Testou: [{tested_terms}]"

    return best_match, normalized_value