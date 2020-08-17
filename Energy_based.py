# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:58:47 2020

@author: seungjun
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:15:03 2020

@author: seungjun
"""
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import skfuzzy
import numpy.matlib
from sklearn.metrics import accuracy_score
import sklearn.metrics as sk
import time
import argparse
os.chdir(r"C:\\VAD")
from utility_vad import *
import json

def energy_options():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--noise_path', default = r'C:\\Users\\seungjun\\Desktop\\noisy_testset_wav', type = str)
    parser.add_argument('--clean_path', default = r'C:\\Users\\seungjun\\Desktop\\clean_testset_wav', type = str)
    parser.add_argument('--checkpoint', default = r'C:\\VAD\\energy_fuzzy_checkpoint')
    parser.add_argument('--clean_th', default = 0.01, type = float, help='clean threshold')
    parser.add_argument('--sr', default = 16000, type = int)
    parser.add_argument('--start_th', default = 10, type = int)
    parser.add_argument('--finish_th', default = 110, type = int)
    parser.add_argument('--ener_interval', default = 100, type = int)


    args = parser.parse_args()
    
    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)

    with open(os.path.join(args.checkpoint,'configuration.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)  
    return args

def main(opt):
    args = opt
    start=time.time()
    envoice_path=(args.clean_path)
    ennoise_path=(args.noise_path)
    
    envo=voice_get(envoice_path)
    enno=voice_get(ennoise_path)

    #energy-based filtering
    times=[]
    result_final=[]
    
    for k in range(len(envo)):
        
        env,sr=librosa.load(envo[k],sr=16000)
        enn,sr=librosa.load(enno[k],sr=16000)
        
        speech=[]
        for i in range(len(env)):
            if np.abs(env[i])>args.clean_th:
                speech.append(1)
            else:
                speech.append(0)
        speech_smo=[]
        for i in range(len(speech)):
            if i<200:
                sur=np.count_nonzero(speech[:i+201])
                if sur>(i+201)/2:
                    speech_smo.append(1)
                else:
                    speech_smo.append(0)
            elif len(speech)-i<200:
                sur = np.count_nonzero(speech[i-200:])
                if sur>(len(speech)-i+201)/2:
                    speech_smo.append(1)
                else:
                    speech_smo.append(0)
            else:
                sur = np.count_nonzero(speech[i-200:i+201])
                if sur>200.5:
                    speech_smo.append(1)
                else:
                    speech_smo.append(0)
    
        # 정답 label 도출
    
        print(str(envo[k].split("\\")[-1])+" data is loaded")
        result_name = envo[k].split("\\")[-1]+".txt"
        f=open(os.path.join(args.checkpoint,result_name),'w', encoding='utf-8')
        for h in range(args.start_th,args.finish_th,10):
            lab, ener =cal_ener(enn, args.ener_interval, h)
            count=[]
            
            for i in range(len(lab)):
                if i<200:
                    sur=np.count_nonzero(lab[:i+201])
                    if sur>(i+201)/2:
                        count.append(1)
                    else:
                        count.append(0)
                elif len(lab)-i<200:
                    sur = np.count_nonzero(lab[i-200:])
                    if sur>(len(lab)-i+201)/2:
                        count.append(1)
                    else:
                        count.append(0)
                else:
                    sur = np.count_nonzero(lab[i-200:i+201])
                    if sur>200.5:
                        count.append(1)
                    else:
                        count.append(0)


    
        #정확도 검증 
            result_prob=[]
            prob=acc(speech_smo, count)
            result_prob.append(prob) 
            result_final.append(result_prob)
            f.write(str(h)+" ")
            f.write(str(prob[0])+" ")
            f.write(str(prob[1])+" ")
            f.write(str(prob[2])+" ")
            f.write(str(prob[3])+" ")
            f.writelines()
            
            file_name = "vad_"+str(h)+"_"+envo[k].split("\\")[-1]+".png"
            save_image2(count, env, args.checkpoint, file_name, title = "vad label & audio signal", xlabel="time", ylabel = "Amplitude & label")
        file_name2 = "energy_"+envo[k].split("\\")[-1]+".png"
        save_image1(src=ener, dest = args.checkpoint, file_name=file_name2,title = "Energy", xlabel = "time", ylabel = "Energy")
        f.close()
        
if __name__=='__main__':
    args = energy_options()
    main(args)

