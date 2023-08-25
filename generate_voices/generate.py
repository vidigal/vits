import json
from inference import inference_vits
import soundfile as sf

INPUT_FILE_PATH = './input/prompts.json'
OUTPUT_PATH = './output'
SAMPLE_RATE = 22050

# Carrega arquivo JSON com os prompts
def load_prompts_json():
    with open(INPUT_FILE_PATH) as file:
        return json.load(file)
    
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

# realiza inferência e retorna o áudio
def inference(text):
    return inference_vits(text)


# Execução inicia aqui
prompts_json = load_prompts_json()
for prompt in prompts_json['prompts']:
    title = prompt['title'] 
    text = prompt['text']

    print(f"Gerando audio para: {title}")
    text_normalized = text_normalize(text)
    generated_audio = inference(text_normalized)

    sf.write(f"{OUTPUT_PATH}/{title}.wav", generated_audio, SAMPLE_RATE)