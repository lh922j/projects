import pandas as pd 
import os, random 
import numpy as np 
import torch

from ctgan import CTGAN
from sklearn.cluster import KMeans
from autogluon.tabular import TabularPredictor
import time

# 전처리 시작
def main():
    print("데이터를 선택합니다.")
    print("원하시는 번호를 입력해주세요.WnWn")
    print("1. MPI_거리")
    print("2. MPL편의")
    print("3. PREC_거리")
    print("4. PREC_편의")
    key= int(input("\n\n숫자만 입력하세요: "))
    
    switch_dict={
        1:"MPI_거리",
        2:"MPI_편의",
        3:"PREC_거리",
        4:"PREC_편의",
    }
    result = switch_dict.get(key,"잘못 입력하셨습니다.")
    print(f"\n\n{key}. {result}를 선택하셨습니다.")
    return key, result
# 회귀 시작
def main2():
    print("데이터의 종류를 선택합니다.")
    print("\n\n원하시는 번호를 입력해주세요.")
    print("1. RAW")
    print("2. AUG")
    print("3. 종료")
    key= int(input("\n\n숫자만 입력하세요: "))
    switch_dict={
        1: "df_raw",
        2: "df_all",
        3:'회귀 종료'
    }
    result = switch_dict.get(key, "잘못 입력하셨습니다.")
    if key ==1 or key==2:
        print(f"\n\n{result}의 성능평가를 시작합니다.")
    else:
        print("\n\n종료합니다.")
    return key

# 시드고정
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark=False 
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] =str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
# 풍향 군집화
def WD(x):
    kmeans = KMeans(n_clusters=8, random_state=42)
    pred = kmeans.fit_predict(x.reshape(-1,1))
    ndf['풍향']=pred
    df_wd = pd.get_dummies(ndf['풍향'], prefix='풍향')
    wd_list=['풍향_0', '풍향_1', '풍향_2', '풍향_3', '풍향_4', '풍향_5', '풍향_6', '풍향_7']
    return df_wd, wd_list

# 거리오차
def REP(rep, key):
    if key==1:
        n=8
        r_list= ['거리_0', '거리_1', '거리_2', '거리_3','거리_4','거리_5', '거리_6', '거리_7']
    else :
        n=10
        r_list=['거리_0', '거리_1', '거리_2', '거리_3', '거리_4','거리_5','거리_6','거리_7','거리_8','거리_9']
        kmeans = KMeans(n_clusters=n, random_state=42)
        pred = kmeans.fit_predict(rep.reshape(-1,1))
        ndf['거리오차']=pred
        rep = pd.get_dummies(ndf['거리오차'], prefix='거리')
        df_c=['DDH', 'FFG', 'PCC', 'PKG', 'PKM', '풍속', '풍향_0','풍향_1', '풍향_2', '풍향_3', '풍향_4','풍향_5', '풍향_6', '풍향_7', '파 고', 'ER', '거리오차']
        return rep, r_list, df_c, n

# 편의오차
def DEP(dep, key):
    if key==2:
        n=8
        r_list=['편의_0', '편의_1', '편의_2', '편의_3', '편의_4', '편의_5', '편의_6', '편의_7']
    else :
        n=8
        r_list=['편의_0', '편의_1', '편의_2', '편의_3', '편의_4', '편의_5', '편의_6', '편의_7']
        kmeans = KMeans(n_clusters=n, random_state=42)
        pred = kmeans.fit_predict(dep.reshape(-1,1))
        ndf['편의오차']=pred
        dep = pd.get_dummies(ndf['편의오차'], prefix='편의')
        df_c=['DDH', 'FFG', 'PCC', 'PKG', 'PKM', '풍속', '풍향_0', '풍향_1', '풍향_2', '풍향_3', '풍향 4', '풍향_5', '풍향_6', '풍향_7', '파 고', 'ER', '편의오차']
        return dep, r_list, df_c, n
# 풍향 개수 맞추기
def wd_match(key):
    # MPI_거리
    wd_num=51
    if key == 1:
        m=df_raw['풍향_0'].value_counts().values.tolist()
        df_0 = df_aug[(df_aug['wd']=='풍향_0')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_1'].value_counts().values.tolist()
        df_1 = df_aug[(df_aug['wd']=='풍향_1')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_2'].value_counts().values.tolist()
        df_2 = df_aug[(df_aug['wd']=='풍향_2')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_3'].value_counts().values.tolist()
        df_3 = df_aug[(df_aug['wd']=='풍향_3')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_4'].value_counts().values.tolist()
        df_4 = df_aug[(df_aug['wd']=='풍향_4')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_5'].value_counts().values.tolist()
        df_5 = df_aug[(df_aug['wd']=='풍향_5')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_6'].value_counts().values.tolist()
        df_6 = df_aug[(df_aug['wd']=='풍향_6')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_7'].value_counts().values.tolist()
        df_7 = df_aug[(df_aug['wd']=='풍향_7')].sample(n=wd_num-m[1],replace=True,random_state=42)
    # MPI_편의
    elif key == 2:
        m=df_raw['풍향_0'].value_counts().values.tolist()
        df_0 = df_aug[(df_aug['wd']=='풍향_0')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_1'].value_counts().values.tolist()
        df_1 = df_aug[(df_aug['wd']=='풍향_1')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_2'].value_counts().values.tolist()
        df_2 = df_aug[(df_aug['wd']=='풍향_2')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_3'].value_counts().values.tolist()
        df_3 = df_aug[(df_aug['wd']=='풍향_3')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_4'].value_counts().values.tolist()
        df_4 = df_aug[(df_aug['wd']=='풍향_4')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_5'].value_counts().values.tolist()
        df_5 = df_aug[(df_aug['wd']=='풍향_5')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_6'].value_counts().values.tolist()
        df_6 = df_aug[(df_aug['wd']=='풍향_6')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_7'].value_counts().values.tolist()
        df_7 = df_aug[(df_aug['wd']=='풍향_7')].sample(n=wd_num-m[1],replace=True,random_state=42)
    # PREC_거리
    elif key == 3:
        m=df_raw['풍향_0'].value_counts().values.tolist()
        df_0 = df_aug[(df_aug['wd']=='풍향_0')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_1'].value_counts().values.tolist()
        df_1 = df_aug[(df_aug['wd']=='풍향_1')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_2'].value_counts().values.tolist()
        df_2 = df_aug[(df_aug['wd']=='풍향_2')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_3'].value_counts().values.tolist()
        df_3 = df_aug[(df_aug['wd']=='풍향_3')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_4'].value_counts().values.tolist()
        df_4 = df_aug[(df_aug['wd']=='풍향_4')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_5'].value_counts().values.tolist()
        df_5 = df_aug[(df_aug['wd']=='풍향_5')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_6'].value_counts().values.tolist()
        df_6 = df_aug[(df_aug['wd']=='풍향_6')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_7'].value_counts().values.tolist()
        df_7 = df_aug[(df_aug['wd']=='풍향_7')].sample(n=wd_num-m[1],replace=True,random_state=42)
    # PREC_편의
    else :
        m=df_raw['풍향_0'].value_counts().values.tolist()
        df_0 = df_aug[(df_aug['wd']=='풍향_0')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_1'].value_counts().values.tolist()
        df_1 = df_aug[(df_aug['wd']=='풍향_1')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_2'].value_counts().values.tolist()
        df_2 = df_aug[(df_aug['wd']=='풍향_2')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_3'].value_counts().values.tolist()
        df_3 = df_aug[(df_aug['wd']=='풍향_3')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_4'].value_counts().values.tolist()
        df_4 = df_aug[(df_aug['wd']=='풍향_4')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_5'].value_counts().values.tolist()
        df_5 = df_aug[(df_aug['wd']=='풍향_5')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_6'].value_counts().values.tolist()
        df_6 = df_aug[(df_aug['wd']=='풍향_6')].sample(n=wd_num-m[1],replace=True,random_state=42)
        m=df_raw['풍향_7'].value_counts().values.tolist()
        df_7 = df_aug[(df_aug['wd']=='풍향_7')].sample(n=wd_num-m[1],replace=True,random_state=42)
    # RAW대비 개수 맞춰준 데이터 생성
    df_a= pd.concat([df_0,df_1,df_2,df_3,df_4,df_5,df_6,df_7], axis=0)
    return df_a

## 산출물 함수
def ST(x):
    if x==0 :
        st= pd.DataFrame({'DDH': [0],'FFG' :[0], 'PCC' : [0], 'PKG': [0], 'PKM': [1]})
    elif x==1 :
        st= pd.DataFrame({'DDH': [0],'FFG' :[0], 'PCC' : [0], 'PKG': [1], 'PKM': [0]})
    elif x==2 :
        st= pd.DataFrame({'DDH': [0],'FFG' :[0], 'PCC' : [1], 'PKG': [0], 'PKM': [0]})
    elif x==3 :
        st= pd.DataFrame({'DDH': [0],'FFG' :[1], 'PCC' : [0], 'PKG': [0], 'PKM': [0]})
    else :
        st= pd.DataFrame({'DDH': [1],'FFG' :[0], 'PCC' : [0], 'PKG': [1], 'PKM': [0]})
    return st
# 풍속
def WS(x):
    if x==0:
        ws= pd.DataFrame({'풍속':[5]})
    elif x==1:
        ws= pd.DataFrame({'풍속':[8]})
    elif x==2:
        ws= pd.DataFrame({'풍속':[10]})
    return ws
# 파고
def WV(x):
    if x==0:
        wv= pd.DataFrame({'파 고':[0.5]})
    elif x==1:
        wv= pd.DataFrame({'파 고':[1.0]})
    else:
        wv= pd.DataFrame({'파 고':[1.5]})
    return wv
# 풍향
def WDD(x):
    if x==2:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[1], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[0]})
    elif x==4:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[0], '풍향_3' :[0], '풍향_4':[1], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[0]})
    elif x==7:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[0], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[1]})
    elif x==0:
        wd= pd.DataFrame({'풍향_0':[1], '풍향_1':[0], '풍향_2':[1], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[0]})
    elif x==5:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[0], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[1], '풍향_6' :[0], '풍향_7' :[0]})
    elif x==3:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[0], '풍향_3' :[1], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[0]})
    elif x==6:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[0], '풍향_2':[0], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[1], '풍향_7' :[0]})
    else:
        wd= pd.DataFrame({'풍향_0':[0], '풍향_1':[1], '풍향_2':[0], '풍향_3' :[0], '풍향_4':[0], '풍향_5' :[0], '풍향_6' :[0], '풍향_7' :[0]})
    return wd
# 오차범위
def DA(x, j): 
    global da, df_all
    if x=='MPI_거리':
        for i in range(j):
            if i==0:
                da=pd.DataFrame({f'{x[-2:]}_0' : [1], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==1:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [1], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==2:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [1], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==3:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [1], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==4:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [1], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==5:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [1], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==6:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [1], f'{x[-2:]}_7': [0]})
            elif i==7:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [1]})
            max = da.idxmax(axis=1)
            e_r=pd.DataFrame()
            e_r['ER']=max
            da=e_r['ER']
            new_row = pd.concat([gt,wd,ws,wv,da], axis=1)
            df_all=pd.concat([df_all,pd.DataFrame(new_row, index=[0])])
    elif x == 'MPI_편의':
        for i in range(j):
            if i==0:
                da=pd.DataFrame({f'{x[-2:]}_0' : [1], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==1:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [1], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==2:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [1], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==3:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [1], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==4:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [1], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==5:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [1], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==6:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [1], f'{x[-2:]}_7': [0]})
            elif i==7:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [1]})
            max = da.idxmax(axis=1)
            e_r=pd.DataFrame()
            e_r['ER']=max
            da=e_r['ER']
            new_row = pd.concat([gt,wd,ws,wv,da], axis=1)
            df_all=pd.concat([df_all,pd.DataFrame(new_row, index=[0])])
    elif x=='PREC_거리':
        for i in range(j):
            if i==0:
                da=pd.DataFrame({f'{x[-2:]}_0' : [1], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==1:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [1], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==2:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [1], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==3:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [1], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==4:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [1], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==5:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [1], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==6:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [1], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==7:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [1], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [0]})
            elif i==8:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [1], f'{x[-2:]}_9': [0]})    
            elif i==9:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0], f'{x[-2:]}_8': [0], f'{x[-2:]}_9': [1]})
            max = da.idxmax(axis=1)
            e_r=pd.DataFrame()
            e_r['ER']=max
            da=e_r['ER']
            new_row = pd.concat([gt,wd,ws,wv,da], axis=1)
            df_all=pd.concat([df_all,pd.DataFrame(new_row, index=[0])])
    else :
        for i in range(i):
            if i==0:
                da=pd.DataFrame({f'{x[-2:]}_0' : [1], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==1:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [1], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==2:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [1], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==3:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [1], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==4:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [1], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==5:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [1], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [0]})
            elif i==6:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [1], f'{x[-2:]}_7': [0]})
            elif i==7:
                da=pd.DataFrame({f'{x[-2:]}_0' : [0], f'{x[-2:]}_1' : [0], f'{x[-2:]}_2' : [0], f'{x[-2:]}_3' : [0], f'{x[-2:]}_4' : [0], f'{x[-2:]}_5': [0], f'{x[-2:]}_6': [0], f'{x[-2:]}_7': [1]})
            max = da.idxmax(axis=1)
            e_r=pd.DataFrame()
            e_r['ER']=max
            da=e_r['ER']
            new_row = pd.concat([gt,wd,ws,wv,da], axis=1)
            df_all=pd.concat([df_all,pd.DataFrame(new_row, index=[0])])
    return da
# 시드고정
seed_everything(42)
start = time.time()
# 시작
while(1): 
    key, result = 0, 0
    if __name__ == "__main__":
        key, result = main()
    # 데이터 불러오기
    df_raw = pd.read_csv(f'{result}.csv',index_col=None)
    print('\n\n데이터 증강을 시작합니다.')
    # 데이터 증강
    d_c=df_raw.columns
    ctgan=CTGAN(epochs=50)
    ctgan.fit(df_raw, d_c)
    df_fake=ctgan.sample(1000)
    print(df_fake.describe())
    print(df_raw.describe())
    df_new = pd.concat([df_raw,df_fake], axis=0, ignore_index=True)
    print(df_new.info())
    print(df_new)

    # 증강데이터 사용
    df = df_new
    print(df.describe())

    # 가데이터프레임
    ndf = pd.DataFrame()
    
    # 함형
    df_st = pd.get_dummies(df['함 형'])
    print(df_st.columns)
    # 풍향
    wd = df['풍향'].to_numpy()
    df_wd, wd_list=WD(wd)
    # 풍속
    ws = df['풍속']
    # 파고
    pago = df['파 고']
    # DA
    if key == 1 or key == 3:
        rep = df['거리오차'].to_numpy()
        da, r_list, df_c, cluster_num = REP(rep, key)
        target = df['거리오차']
    else :
        dep = df['편의오차'].to_numpy()
        da, r_list, df_c, cluster_num = DEP(dep, key)
        target = df['편의오차']
    # 학습데이터 생성
    df_a=pd.concat([df_st, ws, df_wd, pago, da, target], axis=1)
    print(df_a.info())
    
    df=df_a.astype(float)
    # 전달정확도 개수 확인
    df['ER']=0
    for x in r_list:
         for i, j in enumerate(df[x]):
            if j == 1.0:
                df['ER'][i]=x
    # 풍향 개수 확인
    df['wd']=0
    for x in wd_list:
         for i, j in enumerate(df[x]):
            if j == 1.0:
                df['wd'][i]=x


    # RAW 데이터와 AUG데이터 분리
    df_raw=df[:203]
    df_aug=df[203:]
    print('풍향 칼럼별 개수')
    print('\n\nraw')
    print(df_raw['wd'].value_counts())
    print('\n\naug')
    print(df_aug['wd'].value_counts())
    # 풍향 개수 맞추기
    df_a = wd_match(key)

    # 데이터 합치기
    df_all=pd.concat([df_a,df_raw], axis=0)
    print('\n\n풍향 칼럼별 총합 개수 확인')
    print(df_all['wd'].value_counts())
    print('\n\n오차범주 확인')
    print(df_all['ER'].value_counts())

    df_all=df_all[df_c]
    df_raw=df_raw[df_c]
    df_all.to_csv(f'{result}_train.csv', encoding='utf-8',index=False)
    df_raw.to_csv(f'{result}_raw.csv', encoding='utf-8',index=False)
    print()
    print(f'\n\n{len(df_all)}개의 전처리가 완료되었습니다.')

    # 회귀시작
    while(1):
        if __name__ == "__main__":
            reg_k = main2()
            if reg_k==3:
                ('\n\n종료합니다.')
                break
        print(f'{result} 회귀를 시작합니다.\n\n')
        print('평가지표를 선택해주세요. :')
        print('1. RMSE')
        print('2. MAE')
        print('3. R2')
        z_list=['rmse', 'mae', 'r2']
        z=int(input('숫자를 입력해주세요. : '))
        

        if reg_k == 1:
            df_reg = pd.read_csv(f'{result}_raw.csv', index_col=None)
            if key == 1 or key == 3:
                reg = TabularPredictor(label='거리오차', eval_metric=z_list[z-1]).fit(df_reg,presets='best_quality', num_bag_folds=8)
                ld_board=reg.leaderboard(df_reg, silent=True)
                print(ld_board)
            elif key == 2 or key == 4:
                reg = TabularPredictor(label='편의오차', eval_metric=z_list[z-1]).fit(df_reg, presets='best_quality', num_bag_folds=8)
                ld_board=reg.leaderboard(df_reg, silent=True)
                print(ld_board)
            print('\n\n성능평가 완료')
        elif reg_k == 2 :
            df_reg = pd.read_csv(f'{result}_train.csv',index_col=None)
            if key == 1 or key == 3:
                reg = TabularPredictor(label='거리오차', eval_metric=z_list[z-1]).fit(df_reg, presets='best_quality', num_bag_folds=8)
                ld_board=reg.leaderboard(df_reg, silent=True)
                print(ld_board)
            elif key == 2 or key == 4:
                reg = TabularPredictor(label='편의오차', eval_metric=z_list[z-1]).fit(df_reg, presets='best_quality', num_bag_folds=8)
                ld_board=reg.leaderboard(df_reg, silent=True)
                print(ld_board)
            print('\n\n성능평가 완료')
        break
    if reg_k==1 or reg_k==3:
         ('\n\n종료합니다.')
         break
    d_all=pd.DataFrame()
    da = pd.DataFrame()
    for i in range(0,5):
        gt=ST(i)
        for j in range(0,3): 
                ws = WS(j)
                for k in range(0,8):
                    wd=WDD(k)
                    for l in range(0,3): 
                        wv=WV(l)
                        da=DA(result, cluster_num)
    predictions= reg.predict(df_all)
    df_all['DA'] = predictions

    gt_1=df_all[['DDH', 'FFG', 'PCC', 'PKG', 'PKM']]
    max = gt_1.idxmax(axis=1)
    df_all['함형'] = max

    wd_1=df_all[['풍향_0', '풍향_1', '풍향_2','풍향_3', '풍향_4', '풍향_5', '풍향_6', '풍향_7']]
    max = wd_1.idxmax(axis=1)
    df_all['풍향']=max
    df_sum = df_all[['함형','풍향', '풍속','파 고', 'ER', 'DA']]
    print(df_sum.info())

    df_sum.to_csv(f'{result}_pred.csv', encoding='utf-8',index=False)
    print(f"{result} 무기효과 산출 완료!")
    break
end = time.time()
print(f"\n\n소요시간: {end-start:0.2f}s")