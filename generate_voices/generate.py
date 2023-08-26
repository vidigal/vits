import inference
import text_normalizer
import generator_utils
import soundfile as sf
import improve_audio

INPUT_FILE_PATH = './input/prompts.json'
OUTPUT_PATH = './output'
SAMPLE_RATE = 22050

# Execução inicia aqui
prompts_json = generator_utils.load_prompts_json(INPUT_FILE_PATH)
for prompt in prompts_json['prompts']:
    title = prompt['title'] 
    text = prompt['text']

    print(f"Gerando audio para: {title}")
    text_normalized = text_normalizer.text_normalize(text)
    generated_audio = inference.inference_vits(text_normalized)

    output_file_path = f"{OUTPUT_PATH}/{title}.wav"
    sf.write(output_file_path, generated_audio, SAMPLE_RATE)

    generated_audio = improve_audio.add_change_pitch_and_background_music(f"{OUTPUT_PATH}/{title}.wav")
