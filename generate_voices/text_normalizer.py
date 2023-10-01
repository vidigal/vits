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

    normalized_text = normalized_text.replace("Mindfulness", "maidifulnés")
    normalized_text = normalized_text.replace("mindfulness", "maidifulnés")

    normalized_text = normalized_text.replace("mouse", "mauze")
    normalized_text = normalized_text.replace("Mouse", "mauze")

    normalized_text = normalized_text.replace("iphone", "aifone")
    normalized_text = normalized_text.replace("iPhone", "aifone")

    normalized_text = normalized_text.replace("webcam", "uébicam")
    normalized_text = normalized_text.replace("Webcam", "uébicam")

    normalized_text = normalized_text.replace("gadgets", "guedijetiz")
    normalized_text = normalized_text.replace("Gadgets", "guedijetiz")

    normalized_text = normalized_text.replace("youtube", "iuu tube")
    normalized_text = normalized_text.replace("Youtube", "iuu tube")

    return normalized_text