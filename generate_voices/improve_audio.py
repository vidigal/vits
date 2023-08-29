import generator_utils
import random
from pydub import AudioSegment


BACKGROUND_AUDIOS_PATH = './background_audios'

def add_change_pitch_and_background_music(voice_audio_file_path):
    voice_audio = AudioSegment.from_file(voice_audio_file_path)
    
    voice_audio = change_pitch(voice_audio)

    background_audios_list = generator_utils.list_files_from_directory(BACKGROUND_AUDIOS_PATH)
    selected_background_file_name = random.choice(background_audios_list)
    print(f"Música de fundo selecionada: {selected_background_file_name}")
    selected_background_sound = AudioSegment.from_file(f"{BACKGROUND_AUDIOS_PATH}/{selected_background_file_name}")
    
    selected_background_sound = selected_background_sound - 23

    time_filled = 0
    combined_audio = voice_audio
    while time_filled < len(voice_audio):
        combined_audio = combined_audio.overlay(selected_background_sound, position=time_filled)
        time_filled += len(selected_background_sound)


    combined_audio.export(voice_audio_file_path)

def change_pitch(voice_audio):
    octaves = -0.09
    new_sample_rate = int(voice_audio.frame_rate * (2.0 ** octaves))
    lowpitch_sound = voice_audio._spawn(voice_audio.raw_data, overrides={'frame_rate': new_sample_rate})
    return lowpitch_sound
