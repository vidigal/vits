import nltk
import numpy as np
import re

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import soundfile as sf


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def split_text(text):
    text_splited = re.split(r'(?<=[\.\!\?\:])\s*', text)
    return text_splited
    

for i in ["G_226000"]:
    CONFIG_FILE = "./configs/ljs_base.json"
    MODEL_PATH = f"./logs/prime/{i}.pth"
    FILE_WAV_OUTPUT = f"./output/{i}.wav"

    hps = utils.get_hparams_from_file(CONFIG_FILE)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(MODEL_PATH, net_g, None)


    # frase1 = "Inscreva-se no canal pára solicitar seu próprio Mantra."
    # silencio = np.zeros(int(0.5 * hps.data.sampling_rate))
    # frase2 = "Eu Sou 1 campeão. Eu. Sou 1 campeão. Eu Sou 1. campeão."
    # frase3 = "Eu Sou 1 campeão. Eu. Sou 1 campeão. Eu Sou 1. campeão."
    
    frase = f"""
Esta é a voz {i} que está falando.
Havia um homem chamado Daniel que sempre teve grandes sonhos e ambições. Desde jovem, ele sonhava em alcançar o sucesso e a prosperidade, não apenas para si mesmo, mas também para sua família e comunidade. Ele cresceu em um bairro modesto, onde a falta de recursos muitas vezes limitava as oportunidades para muitos.
Determinado a transformar sua vida, Daniel mergulhou de cabeça em seus estudos e desenvolveu habilidades empreendedoras desde cedo. Ele começou a trabalhar enquanto ainda estava na escola, economizando cada centavo para investir em seu futuro. Com determinação e perseverança, ele conseguiu uma bolsa de estudos para a faculdade, onde estudou administração de empresas.
Depois de se formar, Daniel começou a trabalhar em uma pequena startup. Sua ética de trabalho inabalável e sua capacidade de enxergar oportunidades onde os outros viam obstáculos logo chamaram a atenção de seus colegas e superiores. Com o tempo, ele subiu nas fileiras da empresa e, eventualmente, teve a chance de lançar sua própria ideia de negócio.
Sua startup cresceu rapidamente, tornando-se um sucesso surpreendente. Daniel havia conseguido transformar sua visão em realidade. Com dedicação, inovação e a capacidade de tomar decisões audaciosas, ele não apenas acumulou riqueza, mas também criou empregos para muitas pessoas em sua comunidade.
No entanto, o verdadeiro segredo da felicidade de Daniel não estava apenas em sua conquista financeira. Ele nunca esqueceu suas raízes humildes e sempre se lembrou das dificuldades que enfrentou no caminho para o sucesso. Ele usou sua posição para apoiar organizações beneficentes locais, oferecendo bolsas de estudo e programas de capacitação para jovens que se encontravam em situações semelhantes à que ele já estivera.
Além disso, Daniel cultivou relacionamentos significativos com sua família e amigos. Ele reservava tempo para passar com seus entes queridos, valorizando cada momento compartilhado. Sua atenção aos detalhes e sua capacidade de escutar os outros o tornaram alguém querido por todos que o conheciam.
À medida que os anos passaram, Daniel continuou a prosperar nos negócios, mas nunca perdeu de vista o que era realmente importante em sua vida. Sua riqueza permitiu-lhe viajar, explorar novas culturas e experimentar diferentes formas de enriquecer sua vida através do conhecimento e das experiências.
Assim, Daniel se tornou um exemplo de como a riqueza pode ser uma ferramenta para a realização de sonhos e para o bem-estar dos outros. Sua jornada de superação e sucesso não apenas o tornou rico em termos financeiros, mas também o enriqueceu em sabedoria, compaixão e felicidade duradoura. E, acima de tudo, ele encontrou alegria em compartilhar seu sucesso com os outros e em ver as sementes do seu trabalho árduo florescerem em oportunidades para os mais necessitados.
"""
    
    stn_tst = get_text(frase, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.cpu().float().numpy()

    # text_to_speech = ""
    # for i in range(100):
    #     print(text_to_speech)
    #     text_to_speech += frase

    #sentences = nltk.sent_tokenize(TEXT_TO_SPEACH) 

    #sentences = split_text(TEXT_TO_SPEACH)
   
    #silence = np.zeros(int(0.5 * hps.data.sampling_rate))  # quarter second of silence
    #audio_concatenado = []
    #for text in sentences:
     #   print(text)
    
    
    
    # stn_tst1 = get_text(frase1, hps)
    # stn_tst2 = get_text(frase2, hps)
    # stn_tst3 = get_text(frase3, hps)
    # with torch.no_grad():
    #     x_tst = stn_tst1.cuda().unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst1.size(0)]).cuda()
    #     audio1 = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.888, noise_scale_w=0.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()

    #     x_tst = stn_tst2.cuda().unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst2.size(0)]).cuda()
    #     audio2 = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.5)[0][0,0].data.cpu().float().numpy()

    #     x_tst = stn_tst3.cuda().unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst3.size(0)]).cuda()
    #     audio3 = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.0, length_scale=1.5)[0][0,0].data.cpu().float().numpy()
        #audio_concatenado += [audio]
        #audio_concatenado.append(audio)
        #audio_concatenado.append(silence)

    # print(f"Tipo do audio1 {type(audio1)}")
    # print(f"Tipo do silencio {type(silencio)}")

   # pieces = np.concatenate((audio1, silencio, audio2, silencio, audio3))

    
    #audio_final = np.concatenate(audio_concatenado)
    sf.write(FILE_WAV_OUTPUT, audio, hps.data.sampling_rate)