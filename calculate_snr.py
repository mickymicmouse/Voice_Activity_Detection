# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:10:47 2020

@author: seungjun
"""

#snr 추출 하여 excel 파일로 제작

import librosa
from utility_vad import *
import pandas as pd
import numpy as np

clean_path = r"E:\NSDTSEA\clean_testset_wav"
noisy_path = r"E:\NSDTSEA\noisy_testset_wav"

clean_list = voice_get(clean_path)
noisy_list = voice_get(noisy_path)

result=[]
for i in range(len(clean_list)):
    
    clean,_=librosa.load(clean_list[i], sr = 16000)
    noisy,_=librosa.load(noisy_list[i], sr = 16000)
    noise = noisy-clean
    mean_clean = np.mean(np.square(clean))
    mean_noise = np.mean(np.square(noise))
    snr_before = mean_clean/mean_noise
    snr_before_db = 10* np.log10(snr_before)

    file_name = clean_list[i].split("\\")[-1].split('.')[0]
    
    second = len(clean)/16000
    
    result.append([file_name, snr_before_db, second, mean_clean, mean_noise])
    print(file_name + " complete, snr is "+str(snr_before_db))

df = pd.DataFrame(result, columns = ['filename','snr','time','mean_clean','mean_noise'])
df.to_csv(r"snr_result_upgrade.csv",encoding='utf-8')


#디노이즈된 snr 값 구하
residual_noise = signal - noise_reduced_signal; 
snr_after = mean( signal .^ 2 ) / mean( residual_noise .^ 2 ); 
snr_after_db = 10 * log10( snr_after )