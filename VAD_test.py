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

# 폴더 안에 있는 사운드 파일 받아오기
def voice_get(sound_path):
    sound_folder=os.listdir(sound_path)
    sound_list=[]

    for i in sound_folder:
        path=sound_path+'\\'+i
        sound_list.append(path)
        
    return sound_list


def entropy_zero(wav,l,h):
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
            
    return lab

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


envoice_path=(r'C:\Users\seungjun\Desktop\clean_testset_wav')
ennoise_path=(r'C:\Users\seungjun\Desktop\noisy_testset_wav')
#english

envo=voice_get(envoice_path)
enno=voice_get(ennoise_path)

#energy-based filtering
times=[]
result_final=[]
start=time.time()
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

    print(str(k)+" data is loaded")
    for h in range(10,100,10):
        
        lab=cal_ener(enn, 100 ,h)
                             
        
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



        plt.figure(figsize=(12,4))
        plt.plot(count)
        plt.plot(env)
        plt.show()

    #정확도 검증 
        result_prob=[]
    
        prob=acc(speech_smo, count)
        result_prob.append(prob) 
        result_final.append(result_prob)


    plt.figure(figsize=(12,2))
    plt.title("vad label & audio signal")
    plt.xlabel("time")
    plt.ylabel("Amplitude & label")
    plt.grid()
    plt.plot(count)
    plt.plot(env)
    plt.show()
    
    plt.figure(figsize=(12,4))
    plt.ylabel("Energy")
    plt.xlabel("time")
    plt.title("Energy")
    plt.grid()
    plt.plot(ener,color='black')
    plt.show()
    
#energy-based filtering with fuzzy
times=[]
result_final=[]
start=time.time()
for k in range(len(envo)):
    
    env,sr=librosa.load(envo[k],sr=16000)
    enn,sr=librosa.load(enno[k],sr=16000)
    print(str(k)+" data is loaded")
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

    lab=cal_ener(enn,50,90)
    ab,ent_en,label_en=entropy_zero(enn,5,50)
    predict_non, zeros = makepredict(lab, ent_en, enn)
    for h in range(10,100,10):
        vad, vad_label=fuzzy(predict_non, enn, lab, zeros, h/100)
        print("threshold is "+str(h/100))
        count=[]
        for i in range(len(vad_label)):
            if i<200:
                sur=np.count_nonzero(vad_label[:i+201])
                if sur>(i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            elif len(lab)-i<200:
                sur = np.count_nonzero(vad_label[i-200:])
                if sur>(len(lab)-i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            else:
                sur = np.count_nonzero(vad_label[i-200:i+201])
                if sur>200.5:
                    count.append(1)
                else:
                    count.append(0)

        plt.figure(figsize=(12,4))
        plt.plot(env)
        plt.plot(count)
        plt.show()

    #정확도 검증 
        result_prob=[]
    
        prob=acc(speech_smo, count)
        result_prob.append(prob) 
        result_final.append(result_prob)
    

ex1=result_final.copy()
ex2=result_final.copy()
ex3=result_final.copy()
ex4=result_final.copy()


#결과 

total=[]
for i in range(9):
    lis=[]
    for j in range(0,90,9):
        lis.append(result_final[j+i])
    total.append(lis)
total=np.array(total)



mean_f1=[]
mean_acc=[]
mean_recall=[]
mean_pre=[]

for i in range(9):
    thres_f1=[]
    thres_acc=[]
    thres_recall=[]
    thres_pre=[]
    for j in range(10):
        
        thres_f1.append(total[i][j][0][1])
        thres_acc.append(total[i][j][0][0])
        thres_recall.append(total[i][j][0][3])
        thres_pre.append(total[i][j][0][2])
        f1=np.mean(thres_f1)
        ac=np.mean(thres_acc)
        recall=np.mean(thres_recall)
        pre=np.mean(thres_pre)
    mean_f1.append(f1)
    mean_acc.append(ac)
    mean_recall.append(recall)
    mean_pre.append(pre)



plt.figure(figsize=(12,4))
plt.plot(mean_recall[0:],label="recall")
plt.plot(mean_f1[0:9],label="f1 score")
plt.plot(mean_pre[0:9],label="precision")
plt.legend()
plt.grid()
plt.xticks(ticks=([0,1,2,3,4,5,6,7,8]),labels=(10,20,30,40,50,60,70,80,90))
plt.title("Energy-based filtering with fuzzy clustering")
plt.xlabel("threshold")
plt.ylabel("score")
plt.show()






#entropy
times=[]
result_final=[]
start=time.time()
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
        sur=np.count_nonzero(speech[i-200:i+201])
        if sur>200.5:
            speech_smo.append(1)
        else:
            speech_smo.append(0)
    


    for h in range(10,100,10):
        print(h)
        ab,ent_en,label_en=entropy_zero(enn,5,h)
        count=[]
        for i in range(len(label_en)):
            if i<200:
                sur=np.count_nonzero(label_en[:i+201])
                if sur>(i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            elif len(label_en)-i<200:
                sur = np.count_nonzero(label_en[i-200:])
                if sur>(len(lab)-i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            else:
                sur = np.count_nonzero(label_en[i-200:i+201])
                if sur>200.5:
                    count.append(1)
                else:
                    count.append(0)

        
        plt.figure(figsize=(12,4))
        plt.plot(env)
        plt.plot(count)
        plt.show()
    
    
    

    #정확도 검증 
        result_prob=[]
    
        prob=acc(speech_smo, count)
        result_prob.append(prob) 
        result_final.append(result_prob)
    
    


#entropy + fuzzy

times=[]
result_final=[]
start=time.time()
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
        sur=np.count_nonzero(speech[i-200:i+201])
        if sur>200.5:
            speech_smo.append(1)
        else:
            speech_smo.append(0)
    
    ab,ent_en,label_en=entropy_zero(enn,5,60)
    predict_non,zeros = makepredict(label_en, ent_en, enn)

    for h in range(10,100,10):
        print(h)
        vad, vad_label=fuzzy(predict_non, enn, label_en, zeros, h/100)
        count=[]
        for i in range(len(vad_label)):
            if i<200:
                sur=np.count_nonzero(vad_label[:i+201])
                if sur>(i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            elif len(lab)-i<200:
                sur = np.count_nonzero(vad_label[i-200:])
                if sur>(len(lab)-i+201)/2:
                    count.append(1)
                else:
                    count.append(0)
            else:
                sur = np.count_nonzero(vad_label[i-200:i+201])
                if sur>200.5:
                    count.append(1)
                else:
                    count.append(0)

        
        plt.figure(figsize=(12,4))
        plt.plot(env)
        plt.plot(count)
        plt.show()
    
    
    

    #정확도 검증 
        result_prob=[]
    
        prob=acc(speech_smo, count)
        result_prob.append(prob) 
        result_final.append(result_prob)
    
    
    
finish=time.time()
times=finish-start
results=np.array(result_final)
re=result_final.copy()
res=result_final.copy()
result_final=res
plt.figure(figsize=(12,4))
plt.plot(count)
plt.plot(env)


ex_entropy=result_final
withfuzzy=result_final
result_final=ex_entropy
energy



total=[]
for i in range(9):
    lis=[]
    for j in range(0,90,9):
        lis.append(result_final[j+i])
    total.append(lis)
total=np.array(total)



mean_f1=[]
mean_acc=[]
mean_recall=[]
mean_pre=[]

for i in range(9):
    thres_f1=[]
    thres_acc=[]
    thres_recall=[]
    thres_pre=[]
    for j in range(10):
        
        thres_f1.append(total[i][j][0][1])
        thres_acc.append(total[i][j][0][0])
        thres_recall.append(total[i][j][0][3])
        thres_pre.append(total[i][j][0][2])
        f1=np.mean(thres_f1)
        ac=np.mean(thres_acc)
        recall=np.mean(thres_recall)
        pre=np.mean(thres_pre)
    mean_f1.append(f1)
    mean_acc.append(ac)
    mean_recall.append(recall)
    mean_pre.append(pre)



plt.figure(figsize=(12,4))
plt.plot(mean_recall[0:],label="recall")
plt.plot(mean_f1[0:9],label="f1 score")
plt.plot(mean_pre[0:9],label="precision")
plt.legend()
plt.grid()
plt.xticks(ticks=([0,1,2,3,4,5,6,7,8]),labels=(10,20,30,40,50,60,70,80,90))
plt.title("Energy-based filtering with fuzzy clustering")
plt.xlabel("threshold")
plt.ylabel("score")
plt.show()





0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9


ent50=result_final
entfuzzy50=result_final
ener50=result_final
result_final=ent50
result_final=entfuzzy50





##시간 측정용
start=time.time()
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
       
    ab,ent_en,label_en=entropy_zero(enn,5)
    
    predict_non,zeros = makepredict(label_en, ent_en, enn)
    result_prob=[]

    
    
    vad, vad_label=fuzzy(predict_non, enn, label_en, zeros, 0.9)
    
    count=[]
    for i in range(len(vad_label)):
        sur=np.count_nonzero(vad_label[i-200:i+201])
        if sur>200.5:
            count.append(1)
        else:
            count.append(0)

    
    
    #정확도 검증 

    prob=[]
    prob=acc(speech_smo, count)
    result_prob.append(prob)
finish=time.time()
eftimes=finish-start
results=np.array(result_final)


