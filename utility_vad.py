# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:32:45 2020

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

# 폴더 안에 있는 사운드 파일 받아오기
def voice_get(sound_path):
    

    
    sound_folder=os.listdir(sound_path)
    sound_list=[]

    for i in sound_folder:
        path=sound_path+'\\'+i
        sound_list.append(path)
        
    return sound_list


def entropy_zero(wav,l,h):
    
    """    
    
    엔트로피 구하기
    input) Wav = 음성 파일, l = 구간, h = 문턱값 변수
    output) original = 비음성 구간이 0으로 변환된 음성 파일, entropy = 엔트로피 값, label - 엔트로피로 결정된 음성 또는 비음성
    
    """
    
    pro=[]
    wav=wav.tolist()
    pro=np.abs(wav)/np.max(wav)
    for i in range(len(pro)):
        if pro[i]==0:
            pro[i]=0.00000000001
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
    #표준화
    entropy=(ent-np.mean(ent))/np.std(ent)    
    original=wav.copy()
    
    label=[]
    thr = (1-(h/100))*max(entropy)+(h/100)*min(entropy)
    for i in range(len(entropy)):
        if entropy[i]<=thr:
            original[i]=0
            label.append(0)
        else:
            label.append(1)
    print(thr)
    return original, entropy, label



def cal_ener(sound, p, th):
    
    """
    에너지 계산
    input) sound - 음성파일, p - 구간, th - 문턱값 변수
    output) lab - 에너지로 계산된 라벨, energ - 에너지 
    
    """
    
    ener=[]
    for i in range(len(sound)):
        
        energy=np.square(sound[i])
        ener.append(energy)
        
    energ=[]  
    for i in range(len(ener)):
        if i <p//2:
            en=np.sqrt((1/len(ener[:i+p//2]))*np.sum(ener[:i+p//2]))
            energ.append(en)
        elif i>len(ener)-p//2:
            en=np.sqrt((1/len(ener[i-p//2:]))*np.sum(ener[i-p//2:]))
            energ.append(en)
        else:
            en=np.sqrt((1/len(ener[i-p//2:i+p//2]))*np.sum(ener[i-p//2:i+p//2]))
            energ.append(en)
    
    lab=[]
    thr = (1-(th/100))*max(energ)+(th/100)*min(energ)
    for i in range(len(energ)):
            
        if energ[i]>thr:
            lab.append(1)
            
        else:
            lab.append(0)
            
    return lab, energ

def makepredict(label, ent, noise):
    
    """
    fuzzy clustering 변수 만들기
    input) label - 1차 분류 방식으로 분류한 라벨 (에너지, 엔트로피 등), ent - 엔트로피 계산된 값, noise - 노이즈 포함된 음성 파일
    output) predict_non - 2차원 배열의 엔트로피값과 노이즈값이 포함, zeros - 노이즈 포함 음성 파일의 값이 0인 부분의 인덱스
    
    """
    
    predict_non=[]
    zeros=[]
    for i in range(len(label)):
        if label[i] ==0:
            zeros.append(i)
            b=ent[i],noise[i]**2
            predict_non.append(b)
            
    return predict_non,zeros




def fuzzy(predict_non, noise, label, zeros, thres):
    
    """
    퍼지 클러스터링 진행
    비지도 학습이지만 엔트로피 특성 상 음성 파일이 높은 부분에 몰린 특성을 사용하여 두 군집 중 높은 값을 가진 부분을 음성 구간으로 특정하고 분류 진행
    input) predict_non - 2차원 배열의 엔트로피값과 노이즈값이 포함, noise - 노이즈 포함된 음성 파일, label - 1차 분류 방식으로 분류한 라벨 (에너지, 엔트로피 등),
    zeros - 노이즈 포함 음성 파일의 값이 0인 부분의 인덱스 ,thres - 문턱값 변수 기준값
    output) nono - 퍼지클러스터링 이후 Vad된 음성 파일, label - 결과값
    """
    
    data_pu=np.array(predict_non).T
    final=skfuzzy.cmeans(data_pu,2,2,0.00000000001,2000) 

    plt.figure(figsize=(12,6))
    plt.grid()
    plt.title("fuzzy clustering")
    plt.xlabel("Entropy value")
    plt.ylabel("Amplitude")
    plt.scatter(data_pu[0],data_pu[1],color="R", label="sample")
    plt.scatter(final[0][:,0],final[0][:,1],color="Black", label="center")
    plt.legend()
    plt.show()
    
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
    plt.figure(figsize=(20,2))
    plt.grid()
    plt.title("vad label")
    plt.plot(labels)

    plt.show()
    nono=np.array(nono)
    return nono, labels


def acc(answer, pred):
    """
    결과값 계산
    """
    
    res=[]
    accu=accuracy_score(answer, pred)
    f1_sc=sk.f1_score(answer, pred)
    preci=sk.precision_score(answer,pred)
    reca=sk.recall_score(answer,pred)
    print("accuracy is "+str(accu)+ "\n" +"f1_score is "+str(f1_sc)+"\n"+"precision is "+str(preci)+"\n"+"recall is "+str(reca)+"\n" )
    res.append(accu)
    res.append(f1_sc)
    res.append(preci)
    res.append(reca)
    
    return res
def save_image1(src,dest,file_name, title, xlabel, ylabel):
    plt.figure(figsize=(12,2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.plot(src)
    plt.savefig(os.path.join(dest,file_name))

def save_image2(src,src2,dest,file_name, title, xlabel, ylabel):
    plt.figure(figsize=(12,2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.plot(src)
    plt.plot(src2)
    plt.savefig(os.path.join(dest,file_name))
