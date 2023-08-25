import generator_utils
import random
from pydub import AudioSegment


BACKGROUND_AUDIOS_PATH = './background_audios'

def add_background_music(voice_audio_file_path):
    voice_audio = AudioSegment.from_file(voice_audio_file_path)
        
    background_audios_list = generator_utils.list_files_from_directory(BACKGROUND_AUDIOS_PATH)
    selected_background_file_name = random.choice(background_audios_list)
    print(f"Música de fundo selecionada: {selected_background_file_name}")
    selected_background_sound = AudioSegment.from_file(f"{BACKGROUND_AUDIOS_PATH}/{selected_background_file_name}")
    
    selected_background_sound = selected_background_sound - 30

    time_filled = 0
    combined_audio = voice_audio
    while time_filled < len(voice_audio):
        combined_audio = combined_audio.overlay(selected_background_sound, position=time_filled)
        time_filled += len(selected_background_sound)


    combined_audio.export("output/bbbb.wav")
