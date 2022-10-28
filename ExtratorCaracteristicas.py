import numpy as np
from sklearn import preprocessing
from python_speech_features import mfcc, delta
from spafe.features.bfcc import bfcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc
from spafe.features.msrcc import msrcc
from spafe.features.rplp import rplp
from librosa import stft


def extract_mfcc(audio, rate):
    mfcc_feature  = mfcc(sig = audio, fs = rate)
    mfcc_feature  = preprocessing.scale(mfcc_feature)
    deltas        = delta(mfcc_feature, 2)
    double_deltas = delta(deltas, 2)
    
    combined      = np.hstack((mfcc_feature, deltas, double_deltas))
    return combined

def extract_gfcc(audio, rate):
    gfcc_feature = gfcc(sig = audio, fs = rate)
    gfcc_feature = preprocessing.scale(gfcc_feature)
    return gfcc_feature

def extract_lpcc(audio, rate):
    lpcc_feature = lpcc(sig = audio, fs = rate)
    lpcc_feature = preprocessing.scale(lpcc_feature)
    return lpcc_feature
    
def extract_bfcc(audio, rate):
    bfcc_feature = bfcc(sig = audio, fs = rate)
    bfcc_feature = preprocessing.scale(bfcc_feature)
    return bfcc_feature

def extract_lfcc(audio, rate):
    lfcc_feature = lfcc(sig = audio, fs = rate)
    lfcc_feature = preprocessing.scale(lfcc_feature)
    return lfcc_feature

def extract_msrcc(audio, rate):
    msrcc_feature = msrcc(sig = audio, fs = rate)
    msrcc_feature = preprocessing.scale(msrcc_feature)
    return msrcc_feature

def extract_rplp(audio, rate):
    rplp_feature = rplp(sig = audio, fs = rate)
    rplp_feature = preprocessing.scale(rplp_feature)
    return rplp_feature 

def extract_stft(audio, rate):
    ft_feature = stft(y=audio, n_fft=512)
    ft_feature = np.abs(ft_feature).T
    ft_feature = preprocessing.scale(ft_feature)
    return ft_feature