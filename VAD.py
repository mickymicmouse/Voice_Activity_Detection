# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 01:29:02 2020

@author: seungjun
"""

import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
import sklearn.metrics as sk

def voice_get(sound_path):
    sound_folder=os.listdir(sound_path)
    sound_list=[]

    for i in sound_folder:
        path=sound_path+'\\'+i
        sound_list.append(path)

    

    return sound_list



def th_vad(wav, th):
    inte=[]
    wav=wav.tolist()
    for i in range(len(wav)):
        if wav[i]<th and wav[i]>-th:
            inte.append(i)
        
    inte.reverse()
    for i in range(len(inte)):
        wav.pop(inte[i])
    return wav

def th_zero(wav, th):
    inte=[]
    wav=wav.tolist()
    for i in range(len(wav)):
        if wav[i]<th and wav[i]>-th:
            wav[i]=0
        
    return wav




def entropy_vad(wav,l):
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
    
    novo=[]
    for i in range(len(entropy)):
        if entropy[i]<=0:
            novo.append(i)
    novo.reverse()
    original=wav
    for i in range(len(novo)):
        original.pop(novo[i])
    original=np.array(original)
    plt.plot(original)
    return original


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
    plt.figure(figsize=(20,2))
    plt.grid()
    plt.title("entropy")
    plt.plot(entropy)

    plt.show()
    return original, entropy, label




#퍼지클러스터링
import skfuzzy
import numpy.matlib

#--------------------------------

def makepredict(label, ent, noise):
    predict_non=[]
    zeros=[]
    for i in range(len(label)):
        if label[i] ==0:
            zeros.append(i)
            b=ent[i],noise[i]**2
            predict_non.append(b)
            
    return predict_non,zeros
#--------------------------------
#0인 구간 검출 및 데이터 합병
#0이라고 예측한 위치 구간과 엔트로피 값을 가져오기
def allmake(label, ent, noise):
    predict_non=[]
    zeros=[]
    for i in range(len(label)):
        if label[i] < 3:
            zeros.append(i)
            b=ent[i],noise[i]**2
            predict_non.append(b)
            
    return predict_non,zeros

    
def makezero(label):
    zero=[]
    for i in range(len(label)):
        if label[i]==0:
            zero.append(i)
    return zero

    
def fuzzy(predict_non, noise, label, zeros,thres):
    data_pu=np.array(predict_non).T
    final=skfuzzy.cmeans(data_pu,2,2,error=0.00000000001,maxiter=2000) 

    plt.figure(figsize=(12,6))
    plt.grid()
    plt.title("fuzzy clustering")
    plt.xlabel("Entropy value")
    plt.ylabel("Amplitude")
    plt.scatter(data_pu[0],data_pu[1],color="red", label="sample")
    plt.scatter(final[0][:,0],final[0][:,1],color="black", label="center")
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
#퍼지 클러스터링실시 predict_non=> makepredict로 나온 리턴값
#noise-> 노이즈 파일
#label-> 엔트로피 vad이후로 나온 라벨
#zeros-> makepredict로 나온 리턴
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





envoice_path=(r'C:\Users\seungjun\Desktop\clean_testset_wav')
ennoise_path=(r'C:\Users\seungjun\Desktop\noisy_testset_wav')
#english

envo=voice_get(envoice_path)
enno=voice_get(ennoise_path)

env,sr=librosa.load(envo[k],sr=16000)
enn,sr=librosa.load(enno[k],sr=16000)

plt.figure(figsize=(12,4))
plt.ylabel("Energy")
plt.xlabel("time")
plt.title("Energy")
plt.xticks(fontsize=13)
plt.grid()
plt.plot(env,color='orange')


plt.show()
##########################################################
#entropy+fuzzy 분할 하는 그림
enn_entropy=entropy_vad(enn,5)
ab,ent_en,label_en=entropy_zero(enn,5)
predict_non, zeros=makepredict(label_en, ent_en, enn)
vad,vad_label=fuzzy(predict_non, enn, label_en, zeros, 0.9)
 
count=[]
for i in range(len(label_en)):
    if i<200:
        sur=np.count_nonzero(label_en[:i+201])
        if sur>(i+201)/2:
            count.append(1)
        else:
            count.append(0)
    elif len(vad_label)-i<200:
        sur = np.count_nonzero(label_en[i-200:])
        if sur>(len(label_en)-i+201)/2:
            count.append(1)
        else:
            count.append(0)
    else:
        sur = np.count_nonzero(label_en[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)

predict_non, zeros=makepredict(label_en, ent_en, enn)
data_pu=np.array(predict_non).T
cntr, u ,u0, d, jm, p ,fpc=skfuzzy.cmeans(data_pu,2,2,error=0.000000001,maxiter=2000) 
#cntr=center , u= final partition u0 = initial partition d = distance jm = history p = iter
cluster_membership=[]
if cntr[0,0]>cntr[1,0]:
    n=0
else:
    n=1

for i in range(len(u[1])):
    if (u[n,i]>0.9):
        cluster_membership.append(1)
    else:
        cluster_membership.append(0)
colors=['orange','blue']
cluster_membership=np.array(cluster_membership)

plt.figure(figsize=(12,6))
plt.grid()
plt.title("fuzzy clustering")
plt.xlabel("Entropy value")
plt.ylabel("Amplitude")
for j in range(0,2,1):
    plt.scatter(data_pu[0][cluster_membership==j],data_pu[1][cluster_membership==j],color=colors[j], label=labels[j])
plt.scatter(cntr[:,0],cntr[:,1],color="black", label="center")
plt.legend()
plt.show()
##############################################################
enn_entropy=entropy_vad(enn,5)
ab,ent_en,label_en=entropy_zero(enn,5)
predict_non, zeros=makepredict(label_en, ent_en, enn)
vad,vad_label=fuzzy(predict_non, enn, label_en, zeros, 0.9)



predict_non, zeros=allmake(label_en, ent_en, enn)

data_pu=np.array(predict_non).T
cntr, u ,u0, d, jm, p ,fpc=skfuzzy.cmeans(data_pu,2,2,error=0.000000001,maxiter=2000) 
#cntr=center , u= final partition u0 = initial partition d = distance jm = history p = iter
cluster_membership=[]
if cntr[0,0]>cntr[1,0]:
    n=0
else:
    n=1

for i in range(len(u[1])):
    if (u[n,i]>0.5):
        cluster_membership.append(1)
    else:
        cluster_membership.append(0)
colors=['orange','blue']
cluster_membership=np.array(cluster_membership)


 
count=[]
for i in range(len(cluster_membership)):
    if i<200:
        sur=np.count_nonzero(cluster_membership[:i+201])
        if sur>(i+201)/2:
            count.append(1)
        else:
            count.append(0)
    elif len(vad_label)-i<200:
        sur = np.count_nonzero(cluster_membership[i-200:])
        if sur>(len(vad_label)-i+201)/2:
            count.append(1)
        else:
            count.append(0)
    else:
        sur = np.count_nonzero(cluster_membership[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)
            

count=[]
for i in range(len(label_en)):
    if i<200:
        sur=np.count_nonzero(label_en[:i+201])
        if sur>(i+201)/2:
            count.append(1)
        else:
            count.append(0)
    elif len(vad_label)-i<200:
        sur = np.count_nonzero(label_en[i-200:])
        if sur>(len(vad_label)-i+201)/2:
            count.append(1)
        else:
            count.append(0)
    else:
        sur = np.count_nonzero(label_en[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)
            
 
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
labels=["non-speech","speech"]

count=np.array(count)
plt.figure(figsize=(12,6))
plt.grid()
plt.title("fuzzy clustering")
plt.xlabel("Entropy value")
plt.ylabel("Amplitude")
for j in range(1,-1,-1):
    plt.scatter(data_pu[0][count==j],data_pu[1][count==j],color=colors[j], label=labels[j])
plt.scatter(cntr[:,0],cntr[:,1],color="black", label="center")
plt.legend()
plt.show()



# vad 이후에 대한 분포
label_en=np.array(label_en)
plt.figure(figsize=(12,6))
plt.grid()
plt.title("fuzzy clustering")
plt.xlabel("Entropy value")
plt.ylabel("Amplitude")
for j in range(1,-1,-1):
    plt.scatter(data_pu[0][label_en==j],data_pu[1][label_en==j],color=colors[j], label=labels[j])
#plt.scatter(cntr[:,0],cntr[:,1],color="black", label="center")
plt.legend()
plt.show()

#정답에 대한 분포
speech_smo=np.array(speech_smo)
plt.figure(figsize=(12,6))
plt.grid()
plt.title("fuzzy clustering")
plt.xlabel("Entropy value")
plt.ylabel("Amplitude")
for j in range(1,-1,-1):
    plt.scatter(data_pu[0][speech_smo==j],data_pu[1][speech_smo==j],color=colors[j], label=labels[j])

plt.legend()
plt.show()

count=np.array(count)
count_en=np.array(count_en)
label_en=np.array(label_en)
plt.figure(figsize=(12,6))
plt.grid()
plt.title("fuzzy clustering")
plt.xlabel("Entropy value")
plt.ylabel("Amplitude")
for j in range(1,-1,-1):
    plt.scatter(data_pu[0][count_en==j],data_pu[1][count_en==j],color=colors[j], label="sample")
plt.scatter(cntr[:,0],cntr[:,1],color="black", label="center")
plt.legend()
plt.show()
#################################################################################################

plt.figure(figsize=(12,4))
plt.ylabel("Entropy")
plt.xlabel("time")
plt.title("Entropy signal")
plt.xticks(fontsize=13)
plt.grid()
plt.plot(ent_en, color='black')
plt.show()






#vad label plot
    plt.figure(figsize=(12,2))
    plt.title("vad label & audio signal")
    plt.xlabel("time")
    plt.ylabel("Amplitude & label")
    plt.grid()
    plt.plot(count)
    plt.plot(enn)
    plt.show()

    #vad 파일 추출 
    vad_file=[]
    for i in range(len(enn)):
        if count[i]!=0:
            vad_file.append(enn[i])
    vad_file=np.array(vad_file)
    librosa.output.write_wav(r"E:\vad_final\noise_"+str(k)+".wav", vad_file, sr=16000)
    
    vad_file=[]
    for i in range(len(env)):
        if count[i]!=0:
            vad_file.append(env[i])
    vad_file=np.array(vad_file)
    librosa.output.write_wav(r"E:\vad_final\clean\clean_"+str(k)+".wav", vad_file, sr=16000)
    
    
envoice_path=(r'C:\Users\seungjun\Desktop\original_clean_testset_wav')
ennoise_path=(r'C:\Users\seungjun\Desktop\original_noisy_testset_wav')
#english

envo=voice_get(envoice_path)
enno=voice_get(ennoise_path)
    


#-----------------------------------------------------0.9~0.99차이 비교-------------------
result_final=[]
for k in range(len(envo)):
    env,sr=librosa.load(envo[k],sr=16000)
    enn,sr=librosa.load(enno[k],sr=16000)
    #voice file에서 추출 0.01
    speech=[]
    for i in range(len(env)):
        if np.abs(env[i])>0.015:
            speech.append(1)
        else:
            speech.append(0)
    #smoothing
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
    

    plt.figure(figsize=(12,2))
    plt.title("vad label & audio signal")
    plt.xlabel("chunk")
    plt.ylabel("Amplitude & label")
    plt.grid()
    plt.plot(speech_smo)
    plt.plot(env)
    plt.show()

    enn_entropy=entropy_vad(enn,5)
    ab,ent_en,label_en=entropy_zero(enn,5)
    
    predict_non,zeros = makepredict(label_en, ent_en, enn)
    result_prob=[]
    #fuzzy clustering을 할건데 확률 값을 기준으로 threshold값 선정 0.9 ~ 0.99
    for h in range(10,110,10):
        print(h)
        vad, vad_label=fuzzy(predict_non, enn, label_en, zeros, h/100)
    
        count=[]
        for i in range(len(vad_label)):
            sur=np.count_nonzero(vad_label[i-200:i+201])
            if sur>200.5:
                count.append(1)
            else:
                count.append(0)
    
    #정확도 검증 

        prob=acc(speech_smo, count)
        eff=(len(enn)-np.count_nonzero(count))/len(enn)
        prob.append(eff)
        result_prob.append(prob)
    #threshold값을 이용한 vad 
    np.array(result_prob)
    result_final.append(result_prob)
    
result_final_array=np.array(result_final)


    thres=[]
    mean=np.mean(np.abs(enn))
    for i in range(len(enn)):
        if np.abs(enn[i])>mean:
            thres.append(1)
        else:
            thres.append(0)
    
    prob_th = acc(speech_smo, thres)
    effi=(len(enn)-np.count_nonzero(thres))/len(enn)
    prob_th.append(effi)
    result_prob.append(prob_th)
    
    
    count_en=[]
    for i in range(len(label_en)):
        sur=np.count_nonzero(label_en[i-200:i+201])
        if sur>200.5:
            count_en.append(1)
        else:
            count_en.append(0)
    
    result_en=[]
    en=acc(speech_smo, count_en)
    ef=(len(enn)-np.count_nonzero(count_en))/len(enn)
    en.append(ef)
    result_prob.append(en)
    result_final.append(result_prob)
    
result_final_array=np.array(result_final)

#-------------------------------------------------------------------------------------------

result_final=[]
for k in range(len(envo)):
    env,sr=librosa.load(envo[k],sr=16000)
    enn,sr=librosa.load(enno[k],sr=16000)
    
    speech=[]
    
    for i in range(len(env)):
        if np.abs(env[i])>0.01:
            speech.append(1)
        else:
            speech.append(0)
    speech_smo=[]
    for i in range(len(speech)):
        sur=np.count_nonzero(speech[i-200:i+201])
        if sur>200.5:
            speech_smo.append(1)
        else:
            speech_smo.append(0)
    
    thres=[]
    mean=np.mean(np.abs(enn))
    for i in range(len(enn)):
        if np.abs(enn[i])>mean:
            thres.append(1)
        else:
            thres.append(0)
    predict_non_thres=[]
    for i in range(len(thres)):
        if thres[i]==0:
            predict_non_thres.append(enn[i])
    

    ab,ent_en,label_en=entropy_zero(enn,5)
    
    predict_non,zeros = makepredict(label_en, ent_en, enn)
    result_prob=[]
    th_zero=makezero(thres)
    
    
    vad, vad_label=fuzzy(predict_non, enn, label_en, zeros, 0.9)
    
    count=[]
    for i in range(len(vad_label)):
        sur=np.count_nonzero(vad_label[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)

    
    
    #정확도 검증 


    prob=acc(speech_smo, count)
    result_prob.append(prob)
    prob=acc(speech_smo, label_en)
    result_prob.append(prob)
    prob=acc(speech_smo, thres)
    result_prob.append(prob)
    
    predict_non, zeros=makepredict(thres, ent_en, enn)
    vad, vad_label=fuzzy(predict_non, enn, thres, th_zero,0.9)
    count=[]
    for i in range(len(vad_label)):
        sur=np.count_nonzero(vad_label[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)
    prob = acc(speech_smo,count)
    result_prob.append(prob)
    result_final.append(result_prob)
    

results=np.array(result_final)






    plt.figure(figsize=(12,2))
    plt.title("vad label & audio signal")
    plt.xlabel("time")
    plt.ylabel("Amplitude & label")
    plt.grid()
    plt.plot(count)
    plt.plot(env)
    plt.show()


#평균값--------------------------------------
for i in range(0,4):
    
    pro=np.mean(results[:,i,0])
    print(pro)
#그래프----------------------------------------
plt.figure(figsize=(10,5))
plt.xlabel("file name")
plt.ylabel("Recall")
plt.title("Recall value of VAD algorithm")
plt.xticks(range(0,10,1))
plt.grid()
plt.plot(results[:,0,3], label="Used VAD")
plt.plot(results[:,2,3], label="threshold VAD")
plt.plot(results[:,1,3],label="Entropy VAD")

plt.legend()
plt.show()    
    
recall_mean=[]
f1mean=[]
accuracy_mean=[]
precision_mean=[]
ef=[]
for i in range(10):
    recall_mean.append(np.mean(result_final_array[:,i,3]))
    f1mean.append(np.mean(result_final_array[:,i,1]))
    accuracy_mean.append(np.mean(result_final_array[:,i,0]))
    precision_mean.append(np.mean(result_final_array[:,i,2]))
    ef.append(np.mean(result_final_array[:,i,4]))

plt.figure(figsize=(10,5))
plt.xlabel("threshold")

plt.ylabel("value")
plt.title("f1 & accuracy mean value of VAD algorithm")

plt.plot(f1mean[0:9], label="f1 score")
plt.plot(accuracy_mean[0:9], label="accuracy")
plt.plot(ef[0:10],label="버려짐")
plt.xticks(ticks=range(0,10,1),labels=range(10,100,10))
plt.xticks()
plt.grid()
plt.legend()

plt.show()


plt.figure(figsize=(10,5))
plt.xlabel("threshold")
plt.ylabel("value")
plt.title("Recall value of used VAD algorithm")
plt.xticks(range(90,100,1))
plt.grid()
plt.plot(result_final_array[], label="Used VAD")
plt.legend()
plt.show()

result_vad_array=pd.DataFrame(result_vad)
result_final_array=pd.DataFrame(result_final)
result_vad_array.to_csv(r"result_vad.csv", encoding='euc-kr')
result_final_array.to_csv(r"result_threshold.csv", encoding='euc-kr')

import pandas as pd
result_final_array=pd.read_csv(r"result_threshold.csv", encoding='euc-kr')
result_final_array=np.array(result_final_array)

for i in range(len(vad_label)):
    if vad_label[i] == 0:
        env[i]=0
env=env[env!=0]

#clean voice vad
librosa.output.write_wav('english_vad1.wav',en_vad,sr=16000)
librosa.output.write_wav('english_vad_clean.wav', env, sr=16000)
deno,sr=librosa.load(r'C:\Users\seungjun\Downloads\english_vad_denoised1.wav', sr=16000)
deno=deno[deno!=0]
librosa.output.write_wav('english_vad2.wav',deno,sr=16000)
plt.plot(deno)
plt.plot(env)
