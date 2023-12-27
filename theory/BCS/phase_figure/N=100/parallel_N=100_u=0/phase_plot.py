import pickle
import os
import numpy as np
from numpy import *
from time import time
import matplotlib.pyplot as plt


##パラメータファイルの読み込み
#コード01a
path_file = os.getcwd() + '/dir_data/parameter.pkl'
with open(path_file, 'rb') as f:
    f = pickle.load(f)
print(f)
print(f['N'])

# ##パラメータの入力
# N, V, t, a, u, gu, n0, n1, n2 =f['N'], f['V'], f['t'], f['a'], f['u'], f['gu'], f['n_q'], f['n_B'], f['n_kBT']    # 7.525 #9.21

q_max_0 = []
for turn in range(f['n_dic']):
    #max_qのデータを読み取り
    #コード01a
    path_file = os.getcwd() + '/dir_data/max_q_'+str(turn)+'.pkl'
    with open(path_file, 'rb') as f:
        max_q_search_result = pickle.load(f)
    q_max_1  = max_q_search_result[0]
    qs       = max_q_search_result[1]
    Bs       = max_q_search_result[2]
    kBTs     = max_q_search_result[3]
    print(q_max_1)
    # q=0をBCSへq＝＼0をFFLOへと色分けする
    for h in range(len(kBTs)):
        for i in range(len(Bs)):
            figure = plt.scatter(kBTs[h], Bs[i], 3, c = q_max_1[h][i], cmap='viridis' ,vmin=0, vmax=len(qs)) # c = q_max[h][i]
    print(turn)
c= plt.colorbar()
plt.savefig("FFLO_BCS_phase_figure.png")
plt.show()


# # q=0をBCSへq＝＼0をFFLOへと色分けする
# for h in range(n2):
#     for i in range(n1):
#             figure = plt.scatter(kBTs[h], Bs[i], 5, c = q_max_1[h][i], cmap='viridis' ,vmin=0, vmax=n0) # c = q_max[h][i]
# c= plt.colorbar()
# plt.savefig("FFLO_BCS_phase_figure.png")
# plt.show()
