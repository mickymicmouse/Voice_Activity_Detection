# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:47:44 2020

@author: seungjun
"""

import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sklearn.metrics as sk
import time
import skfuzzy
import numpy.matlib
import pandas as pd
def voice_get(sound_path):
    sound_folder=os.listdir(sound_path)
    sound_list=[]

    for i in sound_folder:
        path=sound_path+'\\'+i
        sound_list.append(path)

    

    return sound_list


def entropy_zero(wav,l):
    pro=[]
    wav=wav.tolist()
    pro=np.abs(wav)/np.max(wav)
    for i in range(len(pro)):
        if pro[i]==0:
            pro[i]=0.0000000001
    inner=[]
      
    for i in range(len(pro)):
        inner.append(pro[i]*np.log(pro[i]))
        
    ent=[]  
    for i in range(len(inner)):
        if i <l//2:
            en=-np.sum(inner[:i+l//2])
            ent.append(en)
        elif i>len(inner)-l//2:
            en=-np.sum(inner[i-l//2:])
            ent.append(en)
        else:
            en=-np.sum(inner[i-l//2:i+l//2])
            ent.append(en)
    
    entropy=(ent-np.mean(ent))/np.std(ent)    
    original=wav.copy()
    novo=[]
    label=[]
    for i in range(len(entropy)):
        if entropy[i]<=0:
            original[i]=0
            label.append(0)
        else:
            label.append(1)

    return original, entropy, label

def makepredict(label, ent, noise):
    predict_non=[]
    zeros=[]
    for i in range(len(label)):
        if label[i] ==0:
            zeros.append(i)
            b=ent[i],noise[i]**2
            predict_non.append(b)
            
    return predict_non,zeros



def fuzzy(predict_non, noise, label, zeros,thres):
    data_pu=np.array(predict_non).T
    final=skfuzzy.cmeans(data_pu,2,2,0.000000001,2000) 

    labels=label.copy()
    clustering=final[1]
    if final[0][0,0]>final[0][1,0]:
        n=0
    else:
        n=1
    for i in range(len(zeros)):
        
        if clustering[n,i]>=thres:
            labels[zeros[i]]=1
    
    noise_1=noise.copy()
    for i in range(len(label)):
        if labels[i]==0:
            noise_1[i]=0
    nono=[]
    for i in noise_1:
        if i!=0:
            nono.append(i)
    nono=np.array(nono)
    return nono, labels


def acc(answer, pred):
    res=[]
    a=accuracy_score(answer, pred)
    b=sk.f1_score(answer, pred)
    c=sk.precision_score(answer,pred)
    d=sk.recall_score(answer,pred)
    print("accuracy is "+str(a)+ "\n" +"f1_score is "+str(b)+"\n"+"precision is "+str(c)+"\n"+"recall is "+str(d)+"\n" )
    res.append(a)
    res.append(b)
    res.append(c)
    res.append(d)
    return res


envoice_path=(r'E:\NSDTSEA\clean_testset_wav')
ennoise_path=(r'E:\NSDTSEA\noisy_testset_wav')
#english

envo=voice_get(envoice_path)
enno=voice_get(ennoise_path)

times=[]
result_final=[]
start=time.time()
sound_fold=os.listdir(r'E:\NSDTSEA\noisy_testset_wav')
for k in range(len(envo)):
    
    env,sr=librosa.load(envo[k],sr=16000)
    enn,sr=librosa.load(enno[k],sr=16000)
    
    speech=[]
    
    for i in range(len(env)):
        if np.abs(env[i])>0.015:
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
    
    ab,ent_en,label_en=entropy_zero(enn,5)
    predict_non,zeros = makepredict(label_en, ent_en, enn)
    vad, vad_label=fuzzy(predict_non, enn, label_en, zeros, 90)
   
    count=[]
    for i in range(len(vad_label)):
        if i<200:
            sur=np.count_nonzero(vad_label[:i+201])
            if sur>(i+201)/2:
                count.append(1)
            else:
                count.append(0)
        elif len(vad_label)-i<200:
            sur = np.count_nonzero(vad_label[i-200:])
            if sur>(len(vad_label)-i+201)/2:
                count.append(1)
            else:
                count.append(0)
        else:
            sur = np.count_nonzero(vad_label[i-200:i+201])
            if sur>200.5:
                count.append(1)
            else:
                count.append(0)


    for i in range(len(count)):
        if count[i] == 0:
            env[i]=-1000
            enn[i]=-1000
        
    env=env[env!=-1000]
    enn=enn[enn!=-1000]
    librosa.output.write_wav(r'E:\NSDTSEA_vad'+'\\noisy_testset_wav\\'+sound_fold[k],enn,sr=16000)
    librosa.output.write_wav(r'E:\NSDTSEA_vad'+'\\clean_testset_wav\\'+sound_fold[k],env,sr=16000)

    #정확도 검증 
    result_prob=[]
    prob=acc(speech_smo, count)
    result_prob.append(prob) 
    result_final.append(result_prob)
    print(str(sound_fold[k])+" is finished, number is "+str(k))
    
    
finish=time.time()
times=finish-start
results=pd.DataFrame(result_final)
results.to_csv(r"result_vad.csv", encoding='euc-kr')

