# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:18:32 2020

@author: seungjun
"""

import librosa
from utility_vad import *
import pandas as pd
import numpy as np
import shutil as sh
import os
import matplotlib.pyplot as plt



path = input()

ins = path.split("_")
if len(ins)>=3:
    
    result_path = os.path.join("/home/itm1/seungjun/VAD",path)
    
    file_list = os.listdir(result_path)
    file_list_txt = [file for file in file_list if file.endswith(".txt")]
    
    final=[]
    for i in range(len(file_list_txt)):
        new_result=[]
        f=open(os.path.join(result_path,file_list_txt[i]),"r")
        result=f.readlines()
        for j in range(len(result)):
            
            k=result[j].split(" ")[:7]
            new_result.append(k)
        final+=new_result
    
    
    df = pd.DataFrame(final, columns = ['thres','accuracy','f1_score','precision','recall','realseg','nonseg'])
    df['f1_score']=pd.to_numeric(df['f1_score'])
    df['accuracy']=pd.to_numeric(df['accuracy'])
    df['recall']=pd.to_numeric(df['recall'])
    df['precision']=pd.to_numeric(df['precision'])
    df['thres']=pd.to_numeric(df['thres'])
    df['realseg']=pd.to_numeric(df['realseg'])
    df['nonseg']=pd.to_numeric(df['nonseg'])
    
    acc_list=[]
    f1_list=[]
    recall_list=[]
    pre_list=[]
    real_seg=[]
    non_seg=[]
    
    
    if "detail" in result_path:
        for i in range(90,100,1):
            
            f1_list.append(df['f1_score'][df['thres']==i].mean())
            pre_list.append(df['precision'][df['thres']==i].mean())
            recall_list.append(df['recall'][df['thres']==i].mean())
            acc_list.append(df['accuracy'][df['thres']==i].mean())
            
            
        plt.figure(figsize=(10,5))
        plt.plot(f1_list, color = "orange", label = "f1 score")
        plt.plot(recall_list, color = 'blue', label = 'recall')
        plt.plot(pre_list, color = 'green', label = 'precision')
        plt.legend()
        plt.grid()
        plt.xlabel("Threshold (%)")
        plt.ylabel("Score")
        plt.xticks(range(0,10,1), range(90,100,1))
        plt.savefig(os.path.join(result_path, "result.png"))
        plt.show()
        
        f=open(os.path.join(result_path, "result.txt"), 'w')
        f.write("f1 score is "+ str(f1_list)+"\n")
        f.write("precision is "+ str(pre_list)+"\n")
        f.write("recall is "+ str(recall_list)+"\n")
        f.write("accuracy is "+ str(acc_list)+"\n")
        f.close()
        
        
        
    else:
        
        for i in range(10,110,10):
            
            f1_list.append(df['f1_score'][df['thres']==i].mean())
            pre_list.append(df['precision'][df['thres']==i].mean())
            recall_list.append(df['recall'][df['thres']==i].mean())
            acc_list.append(df['accuracy'][df['thres']==i].mean())
            real_seg.append(df['realseg'][df['thres']==i].mean())
            non_seg.append(df['nonseg'][df['thres']==i].mean())
            
        plt.figure(figsize=(10,5))
        plt.plot(f1_list, color = "orange", label = "f1 score")
        plt.plot(recall_list, color = 'blue', label = 'recall')
        plt.plot(pre_list, color = 'green', label = 'precision')
        plt.legend()
        plt.grid()
        plt.xlabel("Threshold (%)")
        plt.ylabel("Score")
        plt.xticks(range(0,10,1), range(10,110,10))
        plt.savefig(os.path.join(result_path, "result.png"))
        plt.show()

        f=open(os.path.join(result_path, "result.txt"), 'w')
        f.write("f1 score is "+ str(f1_list)+"\n")
        f.write("precision is "+ str(pre_list)+"\n")
        f.write("recall is "+ str(recall_list)+"\n")
        f.write("accuracy is "+ str(acc_list)+"\n")
        f.write("real is "+ str(real_seg)+"\n")
        f.write("non is "+ str(non_seg)+"\n")
        f.close()
        
else:
    new_path=[]
    types = ['under_2','over_10','between']
    for s in range(len(types)):
        new_path.append(path+"_"+types[s])
    final=[]
    for k in range(len(new_path)):
    
        result_path = os.path.join("/home/itm1/seungjun/VAD",new_path[k])
    
        file_list = os.listdir(result_path)
        file_list_txt = [file for file in file_list if file.endswith(".txt")]
        
        
        for i in range(len(file_list_txt)):
            new_result=[]
            f=open(os.path.join(result_path,file_list_txt[i]),"r")
            result=f.readlines()
            for j in range(len(result)):
                
                k=result[j].split(" ")[:5]
                new_result.append(k)
            final+=new_result
    
    
    df = pd.DataFrame(final, columns = ['thres','accuracy','f1_score','precision','recall'])
    df['f1_score']=pd.to_numeric(df['f1_score'])
    df['accuracy']=pd.to_numeric(df['accuracy'])
    df['recall']=pd.to_numeric(df['recall'])
    df['precision']=pd.to_numeric(df['precision'])
    df['thres']=pd.to_numeric(df['thres'])
    
    acc_list=[]
    f1_list=[]
    recall_list=[]
    pre_list=[]
    
    if "detail" in result_path:
        for i in range(90,100,1):
            
            f1_list.append(df['f1_score'][df['thres']==i].mean())
            pre_list.append(df['precision'][df['thres']==i].mean())
            recall_list.append(df['recall'][df['thres']==i].mean())
            acc_list.append(df['accuracy'][df['thres']==i].mean())
            
            
        plt.figure(figsize=(10,5))
        plt.plot(f1_list, color = "orange", label = "f1 score")
        plt.plot(recall_list, color = 'blue', label = 'recall')
        plt.plot(pre_list, color = 'green', label = 'precision')
        plt.legend()
        plt.grid()
        plt.xlabel("Threshold (%)")
        plt.ylabel("Score")
        plt.xticks(range(0,10,1), range(90,100,1))
        plt.savefig(os.path.join("/home/itm1/seungjun/VAD", str(path)+"_result.png"))
        plt.show()

        f=open(os.path.join("/home/itm1/seungjun/VAD", str(path)+"_result.txt"), 'w')
        f.write("f1 score is "+ str(f1_list)+"\n")
        f.write("precision is "+ str(pre_list)+"\n")
        f.write("recall is "+ str(recall_list)+"\n")
        f.write("accuracy is "+ str(acc_list)+"\n")
        f.close()
        
    else:
        
        for i in range(10,110,10):
            
            f1_list.append(df['f1_score'][df['thres']==i].mean())
            pre_list.append(df['precision'][df['thres']==i].mean())
            recall_list.append(df['recall'][df['thres']==i].mean())
            acc_list.append(df['accuracy'][df['thres']==i].mean())
            
            
        plt.figure(figsize=(10,5))
        plt.plot(f1_list, color = "orange", label = "f1 score")
        plt.plot(recall_list, color = 'blue', label = 'recall')
        plt.plot(pre_list, color = 'green', label = 'precision')
        plt.legend()
        plt.grid()
        plt.xlabel("Threshold (%)")
        plt.ylabel("Score")
        plt.xticks(range(0,10,1), range(10,110,10))
        plt.savefig(os.path.join("/home/itm1/seungjun/VAD", str(path)+"_result.png"))
        plt.show()

        f=open(os.path.join("/home/itm1/seungjun/VAD", str(path)+"_result.txt"), 'w')
        f.write("f1 score is "+ str(f1_list)+"\n")
        f.write("precision is "+ str(pre_list)+"\n")
        f.write("recall is "+ str(recall_list)+"\n")
        f.write("accuracy is "+ str(acc_list)+"\n")
        f.close()