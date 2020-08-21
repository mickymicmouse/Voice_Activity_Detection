# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:58:31 2020

@author: seungjun
"""


import os

file2 = ['Energy_fuzzy.py','Energy_based.py','Entropy_based.py','Fuzzy_based.py']
file = ['Entropy_fuzzy.py']
thres = ['over_10','between']

for i in range(len(file)):
    for j in range(len(thres)):
        operation = "python "+file[i]+" --noise_path /home/itm1/seungjun/VAD/sample/"+thres[j]+"/noisy_testset_wav --entropy_th 30 --clean_path /home/itm1/seungjun/VAD/sample/"+thres[j]+"/clean_testset_wav --start_th 10 --finish_th 110 --interval 10 --checkpoint /home/itm1/seungjun/VAD/"+str(file[i].split('.')[0])+"_"+thres[j]+"_ef30"
        os.system(operation)
