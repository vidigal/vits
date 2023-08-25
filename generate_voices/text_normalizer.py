# Faz alterações no texto, por exemplo, retirar *
def text_normalize(text):
    normalized_text = text
    # Pontuações
    normalized_text = normalized_text.replace("*", "")
    normalized_text = normalized_text.replace(":", ".")
    normalized_text = normalized_text.replace(".", ". ")

    # Palavras
    normalized_text = normalized_text.replace("bitcoin", "biticonhen")
    normalized_text = normalized_text.replace("Bitcoin", "biticonhen")
    
    normalized_text = normalized_text.replace("moeda", "moéda")
    normalized_text = normalized_text.replace("Moeda", "Moéda")

    return normalized_text