import numpy as np
from numpy import *

##各種パラメータの記入

n_dic_B   = 6
n_dic_kBT = 8

parameter_0 = {'n_dic':n_dic_B*n_dic_kBT,'N':100,'V':1,'t':1,'a':1,'u':3,'gu':1,'n_q':50,'n_B':16,'n_kBT':12}
qs_start,qs_end     = 0,0.1     #(np.pi/a)
Bs_start,Bs_end     = 0.0,0.15  #np.linspace(0,0.08,n1)
kBTs_start,kBTs_end = 1e-5,0.05


qs   = np.linspace(0,0.1,parameter_0['n_q'])
Bs   = np.linspace(0.06,0.12,parameter_0['n_B']*n_dic_B)
kBTs = np.linspace(1e-5,0.05,parameter_0['n_kBT']*n_dic_kBT)



##各種パラメータデータの作成
parameter=[]
k=0
for i in range(n_dic_B):
    for j in range(n_dic_kBT):
        parameter.append([])
        parameter[k].append(k)
        parameter[k].append(parameter_0)
        parameter[k].append(qs[0:parameter_0['n_q']])
        parameter[k].append(Bs[0+i*parameter_0['n_B']:(i+1)*parameter_0['n_B']])
        parameter[k].append(kBTs[0+j*parameter_0['n_kBT']:(j+1)*parameter_0['n_kBT']])
        k = k+1

print(parameter)

##各種パラメータの保存
import pickle
import os

# ディレクトリが存在しない場合は作成する
path_file1 = os.getcwd() + '/dir_data'
os.makedirs(path_file1, exist_ok=True)

# gap計算用にファイルを作成する
path_file = os.getcwd() + '/dir_data/parameter_gap.pkl'
with open(path_file, 'wb') as f:
    pickle.dump(parameter, f, -1)

# max_q用にファイルを作成する
path_file = os.getcwd() + '/dir_data/parameter_q.pkl'
with open(path_file, 'wb') as f:
    pickle.dump(parameter, f, -1)

# max_q用にファイルを作成する
path_file = os.getcwd() + '/dir_data/parameter.pkl'
with open(path_file, 'wb') as f:
    pickle.dump(parameter_0, f, -1)