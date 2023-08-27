import sys
sys.path.insert(1, '../')

import numpy as np
import re

import matplotlib.pyplot as plt
import IPython.display as ipd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

CONFIG_FILE = "../configs/ljs_base.json"
MODEL_PATH = f"../logs/prime/G_350000.pth"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def split_text(text):
    text_splited = re.split(r'(?<=[\.\!\?\:])\s*', text)
    return text_splited
    
def inference_vits(text):
    hps = utils.get_hparams_from_file(CONFIG_FILE)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()

    _ = utils.load_checkpoint(MODEL_PATH, net_g, None)

    silencio = np.zeros(int(0.8 * hps.data.sampling_rate))
    frase_inscreva_se = "Se você se inscrever no canal, você vai se tornar um milionário. Você não tem nada a perder."


    text_frases =  [e+"." for e in text.split(".") if e]

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        # Gerar audio principal
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio_principal = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.cpu().float().numpy()
        
        #Gerar frase final
        stn_tst = get_text(frase_inscreva_se, hps)
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio_inscreva_se = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.cpu().float().numpy()
    
    audio_final = np.concatenate((audio_principal, silencio, audio_inscreva_se))
    
    return audio_final