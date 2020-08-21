# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:08:54 2020

@author: seungjun
"""

import os

file = ['Energy_fuzzy.py','Energy_based.py','Entropy_based.py','fuzzy_based.py']
file_entropy = ['Entropy_fuzzy.py']
thres = ['under_2','over_10','between']

for i in file_entropy:
    for j in thres:
        operation = "python "+i+" --noise_path C:\\VAD\\sample\\"+j+"\\noisy_testset_wav --clean_path C:\\VAD\\sample\\"+j+"\\clean_testset_wav --start_th 90 --finish_th 100 --interval 1 --checkpoint C:\\VAD\\"+str(i.split('.')[0])+"_"+j+"_detail"
        os.system(operation)
    