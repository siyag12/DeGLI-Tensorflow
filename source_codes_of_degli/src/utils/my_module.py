# -*- coding: utf-8 -*-
###############################################################################
import numpy as np


def zero_pad(x, win_len, shift):
    lf  = len(x)
    T   = int(np.ceil(lf-win_len)/shift) + 1
    lf2 = win_len + T*shift
    x   = np.concatenate((x,np.zeros(lf2+shift-lf,)), axis=0)
    return x


def normalize_1d(signal, maxval=(2.**15-1.)/2**15):
    max_data = np.max(np.abs(signal))
    signal   = signal / max_data * maxval
    return signal