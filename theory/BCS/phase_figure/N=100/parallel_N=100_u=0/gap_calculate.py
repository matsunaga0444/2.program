import pickle
import os
import numpy as np
from numpy import *
from time import time

initial = time()

##パラメータファイルの読み込み
#コード01a
path_file = os.getcwd() + '/dir_data/parameter_gap.pkl'
with open(path_file, 'rb') as f:
    f = pickle.load(f)

##パラメータの入力
turn = f[0][0]
N, V, t, a, u, gu, n0, n1, n2 =f[0][1]['N'], f[0][1]['V'], f[0][1]['t'], f[0][1]['a'], f[0][1]['u'], f[0][1]['gu'], f[0][1]['n_q'], f[0][1]['n_B'], f[0][1]['n_kBT']    # 7.525 #9.21
qs   = f[0][2] 
Bs   = f[0][3] 
kBTs = f[0][4]

print(f[0][0])
del f[0]
l = f

with open(path_file, 'wb') as f:
    pickle.dump(l, f, -1)

end = time()

print('parameter_gapの更新にかかった時間は'+ str(end - initial) )


## gap_eq の計算に必要なモジュールの作成

def e_k_spin(k1, k2, q, y, B): 
    return 2*t*(np.cos(a*(k1+q/2))+np.cos(a*(k2))) - u + y * 1/2 * gu * B

def e_k_s(k1, k2, q, B):
    return (e_k_spin(k1, k2, q, 1, B) + e_k_spin(-1*k1, k2, q, -1, B))/2

def e_k_a(k1, k2, q, B):
    return (e_k_spin(k1, k2, q, 1, B) - e_k_spin(-1*k1, k2, q, -1, B))/2

def E_k_q(k1, k2, gap, q, B):
    return np.sqrt(e_k_s(k1, k2, q, B)**2 + gap**2)

def E_k_q_s(k1, k2, gap, q, y, B):
    return E_k_q(k1, k2, gap, q, B) + y * e_k_a(k1, k2, q, B)

def Fermi(beta, E):
    return 1 / (np.exp(beta*E) + 1 )

def func(k1, k2, gap, q, B): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1, B))-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1, B)))/(2*E_k_q(k1, k2, gap, q, B))

def rhs(gap, q, B):
    k1 = -1 * np.pi/a + 2 * arange(N) * np.pi / (a * N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q, B)
    return (V / (N**2)) * sum(f)


initial = time()

##ギャップの逐次計算
# 各温度、各磁場において、重心運動量qごとにギャップを記録する
# ans[h][i][j][0] で各重心運動量,各磁場、各温度において記録する
ans = []
for h in range(n2):
    ans0 = []
    for i in range(n1):
        ans1 = []
        for j in range(n0): # それぞれの温度で秩序パラメータを計算
            beta, d0 = 1/kBTs[h], 100.0
            for k in range(1000): # 収束するまで最大1000回ループ
                d1 = rhs(d0, qs[j], Bs[i]) 
                if abs(d1-d0) < 1e-10: break # 収束チェック
                d0 = d1
            ans1.append(d0)
        ans0.append(ans1)
    ans.append(ans0)
ans = np.array(ans)

end = time()

print('自己無撞着を計算するのにかかった時間は'+ str(end - initial) )

gap_calculate_result = [ans, qs, Bs, kBTs]

##ギャップのデータを保存
path_file = os.getcwd() + '/dir_data/gap_'+str(turn)+'.pkl'

with open(path_file, 'wb') as f:
    pickle.dump(gap_calculate_result, f, -1)

