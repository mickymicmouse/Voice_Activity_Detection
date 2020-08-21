# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:26:28 2020

@author: seungjun
"""

import librosa
from utility_vad import *
import pandas as pd
import numpy as np
import shutil as sh
import os
import matplotlib.pyplot as plt


path = r"C:\\VAD\\sample"
types = ['under_2','between','over_10']
types_2 = ['noisy_testset_wav','clean_testset_wav']


total=[]
for i in range(len(types)):

    sound_list = os.listdir(os.path.join(path, types[i], types_2[0]))
    length = []
    for j in range(len(sound_list)):
        
        y,sr = librosa.load(os.path.join(path,types[i],types_2[0], sound_list[j]), sr=16000)
        length.append(len(y))
        print(j)
    total.append(length)
    
df = pd.DataFrame(total)
df = df.T
df.columns=types

df.to_csv(r"C:\\VAD\\t_sample_info.csv")

under_2 = [x/16000 for x in total[0]]
between = [x/16000 for x in total[1]]
over_10 = [x/16000 for x in total[2]]

sec = np.mean(under_2)
sec1 = np.mean(between)
sec2 = np.mean(over_10)

std = np.std(under_2)
std1 = np.std(between)
std2 = np.std(over_10)

f=open("mean_std.txt", 'w')
f.write('under 2 sample mean is {:2.4} and std is {:2.5}'.format(sec,std)+"\n")
f.write('between sample mean is {:2.4} and std is {:2.5}'.format(sec1,std1)+"\n")
f.write('over 10 sample mean is {:2.4} and std is {:2.5}'.format(sec2,std2))
f.close()
    
    