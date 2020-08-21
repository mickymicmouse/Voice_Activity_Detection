# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:29:47 2020

@author: seungjun
"""

import os
import shutil


path = r'/home/itm1/seungjun/VAD/Entropy_fuzzy_under_2_ef30/result/noisy'



lists = os.listdir(path)
for i in range(len(lists)):
    if "30_" in lists[i]:
        src = os.path.join(path, lists[i])
        dest = r"/home/itm1/seungjun/speech-denoising-wavenet/data/NSDTSEA_VAD/under_2/noisy_testset_wav"
        shutil.copy(src, dest)