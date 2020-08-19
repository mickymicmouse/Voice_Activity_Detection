# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 20:10:40 2020

@author: seungjun
"""


import librosa
from utility_vad import *
import pandas as pd
import numpy as np
import shutil as sh
import os
#파일 샘플링

noisy_path = r"E:\NSDTSEA\noisy_testset_wav"
clean_path = r'E:\NSDTSEA\clean_testset_wav'
sample = pd.read_csv(r"C:\\VAD\\snr_result_upgrade.csv", encoding = 'utf-8')

under_2 = sample['filename'][sample['snr']<=2]
df1 = sample['filename'][sample['snr']>2]
df2 = sample['filename'][sample['snr']<=10]
over_10 = sample['filename'][sample['snr']>10]
between = pd.merge(df1, df2, on = "filename", how = "inner")

for i in under_2:
    sh.copy(os.path.join(noisy_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\under_2\\noisy_testset_wav",i+".wav"))
    sh.copy(os.path.join(clean_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\under_2\\clean_testset_wav",i+".wav"))

for i in between['filename']:
    sh.copy(os.path.join(noisy_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\between\\noisy_testset_wav",i+".wav"))
    sh.copy(os.path.join(clean_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\between\\clean_testset_wav",i+".wav"))
    
for i in over_10:
    sh.copy(os.path.join(noisy_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\over_10\\noisy_testset_wav",i+".wav"))
    sh.copy(os.path.join(clean_path,i+".wav"), os.path.join(r"C:\\VAD\\sample\\over_10\\clean_testset_wav",i+".wav"))