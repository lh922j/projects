import pandas as pd 
import numpy as np
# cumulative_frequency 기준
csv_1 = ['MPI_거리_pred.csv','MPI_편의_pred.csv','PREC_거리_pred.csv','PREC_편의_pred.csv']
csv_2 = ['MPI_거리_raw.csv','MPI_편의_raw.csv','PREC_거리_raw.csv','PREC_편의_raw.csv']

st= ['PKM', 'PKG', 'PCC', 'FFG', 'DDH']
ws=[5, 8, 10] 
wv=[0.5, 1.0, 1.5]

# 각각의 전달정확도 담을 튜플
DA_list=[[],[],[],[]]

for c in csv_1:
    mpi_r=pd.read_csv(c, index_col=None)
    for n in range(4): 
        if c==csv_1[n]:
            # raw data 불러오기
            cluster_value = pd.read_csv(csv_2[n], index_col=None)
            for i in st:
                if i == 'DDH' : 
                    for s in ws:
                        for v in wv:
                            val_sum=[]
                            temp_val = 0
                            all_da, da_n=0,0
                            #RAW에서 최빈수 산출
                            temp = cluster_value[(cluster_value[i]==1) & (cluster_value['파 고']==v) & (cluster_value['풍속']==s)]
                            val_counts=temp['da'].value_counts()
                            idx = val_counts.index.tolist()
                            val = val_counts.values.tolist()
                            
                            for j in range(len(val)):
                                temp = mpi_r[(mpi_r['함형']==i) & (mpi_r['파 고']==v) & (mpi_r['풍속']==s) & (mpi_r['da']==idx[j])]
                                temp_val=round(temp['DA'].mean(),2) * val[j]
                                val_sum.append(temp_val)
                            if val != []:
                                all_da=sum(val_sum)
                                da_n=sum(val)
                                mean_da=round(all_da/da_n,3)
                            else :
                                mean_da = np.nan
                                DA_list[n].append(mean_da)
                elif i == 'FFG':
                    for s in ws:
                        for v in wv:
                            val_sum=[]
                            temp_val = 0
                            all_da, da_n=0,0
                            #RAW에서 최빈수 산출
                            temp = cluster_value[(cluster_value[i]==1) & (cluster_value['파 고']==v) & (cluster_value['풍속']==s)]
                            val_counts=temp['da'].value_counts()
                            idx = val_counts.index.tolist()
                            val = val_counts.values.tolist()
                            
                            for j in range(len(val)):
                                temp = mpi_r[(mpi_r['함형']==i) & (mpi_r['파 고']==v) & (mpi_r['풍속']==s) & (mpi_r['da']==idx[j])]
                                temp_val=round(temp['DA'].mean(),2) * val[j]
                                val_sum.append(temp_val)
                            if val != []:
                                all_da=sum(val_sum)
                                da_n=sum(val)
                                mean_da=round(all_da/da_n,3)
                            else :
                                mean_da = np.nan
                                DA_list[n].append(mean_da)
                elif i == 'PCC':
                    for s in ws:
                        for v in wv:
                            val_sum=[]
                            temp_val = 0
                            all_da, da_n=0,0
                            #RAW에서 최빈수 산출
                            temp = cluster_value[(cluster_value[i]==1) & (cluster_value['파 고']==v) & (cluster_value['풍속']==s)]
                            val_counts=temp['da'].value_counts()
                            idx = val_counts.index.tolist()
                            val = val_counts.values.tolist()
                            
                            for j in range(len(val)):
                                temp = mpi_r[(mpi_r['함형']==i) & (mpi_r['파 고']==v) & (mpi_r['풍속']==s) & (mpi_r['da']==idx[j])]
                                temp_val=round(temp['DA'].mean(),2) * val[j]
                                val_sum.append(temp_val)
                            if val != []:
                                all_da=sum(val_sum)
                                da_n=sum(val)
                                mean_da=round(all_da/da_n,3)
                            else :
                                mean_da = np.nan
                                DA_list[n].append(mean_da)

                elif i == 'PKG':
                    for s in ws:
                        for v in wv:
                            val_sum=[]
                            temp_val = 0
                            all_da, da_n=0,0
                            #RAW에서 최빈수 산출
                            temp = cluster_value[(cluster_value[i]==1) & (cluster_value['파 고']==v) & (cluster_value['풍속']==s)]
                            val_counts=temp['da'].value_counts()
                            idx = val_counts.index.tolist()
                            val = val_counts.values.tolist()
                            
                            for j in range(len(val)):
                                temp = mpi_r[(mpi_r['함형']==i) & (mpi_r['파 고']==v) & (mpi_r['풍속']==s) & (mpi_r['da']==idx[j])]
                                temp_val=round(temp['DA'].mean(),2) * val[j]
                                val_sum.append(temp_val)
                            if val != []:
                                all_da=sum(val_sum)
                                da_n=sum(val)
                                mean_da=round(all_da/da_n,3)
                            else :
                                mean_da = np.nan
                                DA_list[n].append(mean_da)
                else :
                    for s in ws:
                        for v in wv:
                            val_sum=[]
                            temp_val = 0
                            all_da, da_n=0,0
                            #RAW에서 최빈수 산출
                            temp = cluster_value[(cluster_value[i]==1) & (cluster_value['파 고']==v) & (cluster_value['풍속']==s)]
                            val_counts=temp['da'].value_counts()
                            idx = val_counts.index.tolist()
                            val = val_counts.values.tolist()
                            
                            for j in range(len(val)):
                                temp = mpi_r[(mpi_r['함형']==i) & (mpi_r['파 고']==v) & (mpi_r['풍속']==s) & (mpi_r['da']==idx[j])]
                                temp_val=round(temp['DA'].mean(),2) * val[j]
                                val_sum.append(temp_val)
                            if val != []:
                                all_da=sum(val_sum)
                                da_n=sum(val)
                                mean_da=round(all_da/da_n,3)
                            else :
                                mean_da = np.nan
                                DA_list[n].append(mean_da)
                                
# 풍속, 파고별 데이터 생성
alle=mpi_r[['함형','풍속','파 고']]
alle=alle.rename(columns={'함형':'S.T','풍속':'Ws','파 고': 'Wv'})
alle=alle.drop_duplicates(keep='last') 
alle=alle.reset_index(drop=True)
                          
alle['mpi_r']=DA_list[0]
alle['mpi_d']=DA_list[1] 
alle['prec_r']=DA_list[2]
alle['prec_d']=DA_list[3]

# 파고 1 데이터
alle=alle[(alle['Wv']==1.0)]
print(alle.reset_index(drop=True))
alle.to_csv('result.csv', encoding='utf-8', index=False)