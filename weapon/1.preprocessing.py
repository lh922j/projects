# 전처리
import pandas as pd 
import numpy as np 
from scipy.stats import chi2

## 이상치 제거
def mahalanobis(x, data):
    x_minus_mu = x - np.mean(data, axis=0)
    cov=np.cov(data,rowvar=False)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()
def outlier_detection(df):
    range_q1, range_q3 = np.percentile(df['거리오차'],[25,75])
    def_q1, def_q3 = np.percentile(df['표준편의'],[25,75])
    range_iqr = range_q3 - range_q1
    def_iqr = def_q3 - def_q1
    range_lower = range_q1 - (range_iqr*1.5)
    range_upper = range_q3 + (range_iqr*1.5)
    def_lower = def_q1 - (def_iqr*1.5)
    def_upper = def_q3 + (def_iqr*1.5)
    
    df_x = df[['거리오차','표준편의']].reset_index(drop=True)
    
    df_x['mahala'] = mahalanobis(x=df_x, data=df_x)
    df_x['p_value'] = 1 - chi2.cdf(df_x['mahala'],1),
    df_x['Outlier_maha'] = ["outlier" if x < 0.01 else "normal" for x in df_x['p_value']]
    df_x['거리오차_이상치']=["outlier" if (x< range_lower) or (x> range_upper) else "normal" for x in df_x['거리오차']]
    df_x['표준편의_이상치']=["outlier" if (x< def_lower) or (x> def_upper) else "normal" for x in df_x['표준편의']]

    return df_x[['거리오차','표준편의','거리오차_이상치','표준편의_이상치','Outlier_maha']]
                 
def outlier(x):
    temp_df = x[['거리오차', '표준편의']]
    temp_df_outlier = outlier_detection(temp_df)
    x = x.iloc[temp_df_outlier[(temp_df_outlier['Outlier_maha']=='normal')].index,:]
    x = x.drop('편의오차',axis=1)
    x = x.rename(columns={'표준편의 : 편의오차'})
    return x

## 풍향 숫자화
def EtoN(x):
    if isinstance(x, str):
        char_mapping = {'N': 22.4, 'NE' : 67.4, 'E' : 112.4, 'SE': 157.4, 'S':202.4,'SW':247.4, 'W':292.4, 'NW' :337.4} 
        return char_mapping.get(x,x)
    else:
        return x

# 데이터는 비문처리(군사자료는 비문!)
df = pd.read_csv('함포.csv',index_col=None)

# 격파사격(SALVO)만 사용
df=df[df['사격종류']=='RF']
df=df[df['사격방법']=='SALVO']
a=df['편의오차'].astype(str)
b=df['거리오차']

# 편의, 거리오차 전처리
for i in range(len(a)):
    if '~' in a.iloc[i]:
        a.iloc[i]='9999'
    if a.iloc[i]=='-':
        a.iloc[i]='9999'
a=a.replace(to_replace=r'R', value='',regex=True)
a=a.replace(to_replace=r'L', value='-', regex=True)
a=a.replace(to_replace=r'HIT', value='0', regex=True) 
a=a.replace(to_replace=r'H', value='0', regex=True)
a=a.replace(to_replace=r'관측불가', value='9999', regex=True)

for i in range(len(b)):
    if b.iloc[i]=='-':
        b.iloc[i]='9999'
b=b.replace(to_replace=r',', value='', regex=True)
b=b.replace(to_replace=r'Hit', value='0', regex=True)
b=b.replace(to_replace=r'HIT', value='0', regex=True)
b=b.replace(to_replace=r'H', value='0', regex=True)
b=b.replace(to_replace=r'관측불가', value='9999', regex=True)
a.astype(float) 
b.astype(float)
df['편의오차']=a 
df['거리오차']=b 

# 사격거리 결측치 제거
df=df[df['사격거리'].isna()!=True]
# 풍향속 전처리
df[['풍향','풍속']]=df['풍향속'].str.split('-', expand=True)
df=df.drop(['풍향속'],axis= 1)
df=df[df['풍향']!='NW/NE']
df['풍향']=df['풍향'].map(EtoN)
df['풍향']=df['풍향'].astype(float)

# 파고 전처리
df=df[(df['파고']!='0.5~1') & (df['파고']!='1~2') & (df['파고']!='1~1.5') & (df['파고']!='1.5~2') & (df['파고']!='1.4') & (df['파고']!='0.2') & (df['파고']!='-')] 

# 변수 선정 및 전처리
df=df[['연 도','월 일','함 형','함포종류','풍향','풍속','파 고','사격거리','거리오차','편의오차','지정사거리']]
df[['거리오차','편의오차','사격거리']] = df[['거리오차''편의오차','사격거리']].astype('float')
df['풍속']= df['풍속'].astype('float')
df = df[(df['풍속']<=15)]
df127 = df[(df['함포종류']=='5inch')]
df76 = df[(df['함포종류']=='76mm')]
df40 = df[(df['함포종류']=='40mm')]
df = pd.concat([df127,df76,df40],axis=0)

# 이상치 제거
df = df[(df['거리오차']!=9999) & (df['편의오차']!=9999)]
# 표준편의 변환
df['표준편의']= round(df['지정사거리']/df['사격거리']*df['편의오차'],2)
df127 = df[df['함포종류']=='5inch']
df76 = df[df['함포종류']=='76mm']
df40 = df[df['함포종류']=='40mm']

# 표준편의 기준 결측치 제거
df40 = df40.dropna(subset=['표준편의'])

# 이상치 제거
df127 = outlier(df127)
df40 = outlier(df40)
df76 = outlier(df76)

# 127mm, 76mm 데이터 결합
df127['거리오차'] = df127['거리오차'].astype('float')
df76['거리오차']= df76['거리오차'].astype('float')
df = pd.concat([df127,df76], axis=0)

# 함형별 묶기
df_all = df[(df['함 형']=='DDH')|(df['함 형']=='FFG')|(df['함 형']=='PCC')|(df['함 형']=='PKG')]
df_40 = df40[df40['함 형']=='PKM'] 

# 함형별 데이터, 종합
df = pd.concat([df_all, df_40],axis=0)

# 전달정확도 데이터 2가지로 나눠서 구성 (MPI : df1, PREC : df2)
df[['거리오차','편의오차','사격거리']] = df[['거리오차', '편의오차' '사격거리']].astype('float')
df1 = df.groupby(['연 도','월 일','함 형', '함포종류','풍향','풍속','파 고'])[['거리오차','편의오차']].mean()
df1 = df1.rename(columns = {'거리오차':'MPI_Y','편의오차':'MPI_X'})
df2 = df.groupby(['연 도','월 일','함 형', '함포종류','풍향','풍속','파 고'])[['거리오차','편의오차']].std()
df2 = df2.rename(columns = {'거리오차':'PREC_Y','편의오차':'PREC_X'})
df1.reset_index(inplace=True)
df2.reset_index(inplace=True)

# 데이터 종합 후 전처리
df1 = df1[['MPI_Y','MPI_X']]
DA = pd.concat([df1,df2], axis=1)
DA = DA.dropna()
                 
df1 = DA[['함 형','풍향','풍속','파 고','MPI_Y', 'MPI_X']]
df2 = DA[['함 형','풍향','풍속','파고','PREC_Y','PREC_X']]

df1 = df1.rename(columns= {'MPI_Y':'거리오차', 'MPI_X':'편의오차'})
df2 = df2.rename(columns= {'PREC_Y':'거리오차', 'PREC_X':'편의오차'}) 

# 최종 Feature 선정
df1 = df1[['함 형','풍향','풍속','파 고','거리오차']]
df2 = df2[['함 형','풍향','풍속','파 고','편의오차']]

# 전달정확도 데이터셋(MPI_거리,MPL편의, PREC_거리, PREC_편의)
df_y1 = df1[['함 형','풍향','풍속','파 고','거리오차']]
df_x1 = df1[['함 형','풍향','풍속','파 고','편의오차']]
df_y2 = df2[['함 형','풍향','풍속','파 고','거리오차']]
df_x2 = df2[['함 형','풍향','풍속','파 고','편의오차']]
                 
# 전달정확도 데이터 저장
df_y1.to_csv('MPI_거리_raw.csv',index=False)
df_x1.to_csv('MPI_편의_raw.csv',index=False)
df_y2.to_csv('PREC_거리_raw.csv',index=False)
df_x2.to_csv('PREC_편의_raw.csv',index=False)
print('전처리 완료!')