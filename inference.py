## TODO: retirar os imports não utilizados

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

CONFIG_FILE = "./configs/ljs_base.json"
MODEL_PATH = f"./logs/prime/{i}.pth"
FILE_WAV_OUTPUT = f"./output/{i}.wav"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def split_text(text):
    text_splited = re.split(r'(?<=[\.\!\?\:])\s*', text)
    return text_splited
    

for i in ["G_257000"]:
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
Dez frases que irão mudar a sua vida.
I. Cada dia é uma chance de brilhar.
II. Sorria, a vida é cheia de possibilidades.
III. Pequenos passos levam a grandes conquistas.
IV. A positividade transforma desafios em oportunidades.
V. Seja gentil: isso espalha alegria ao redor.
VI. Acredite em você mesmo e alcance o impossível.
VII. O otimismo é a chave para o sucesso.
VIII. Supere as adversidades com força e coragem.
IX. Suas ações inspiram outros a serem melhores.
X. O futuro é promissor para quem mantém positividade.

Conhecimento é o que vai diferenciar você da manada. Inscreva-se para aprimorar-se com mais conteúdos como este.
"""
    
    stn_tst = get_text(frase, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.2)[0][0,0].data.cpu().float().numpy()

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