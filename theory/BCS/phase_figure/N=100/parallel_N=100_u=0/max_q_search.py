import pickle
import os
import numpy as np
from numpy import *
from time import time

##パラメータファイルの読み込み
#コード01a
path_file = os.getcwd() + '/dir_data/parameter_q.pkl'
with open(path_file, 'rb') as f:
    f = pickle.load(f)


##パラメータの入力
turn = f[0][0]
N, V, t, a, u, gu, n0, n1, n2 =f[0][1]['N'], f[0][1]['V'], f[0][1]['t'], f[0][1]['a'], f[0][1]['u'], f[0][1]['gu'], f[0][1]['n_q'], f[0][1]['n_B'], f[0][1]['n_kBT']    # 7.525 #9.21
qs   = np.linspace(0,0.1,n0)  #(np.pi/a)
Bs   = np.linspace(0.1,0.15,n1)     #np.linspace(0,0.08,n1)
kBTs = np.linspace(1e-5,0.05,n2)
print(f[0][0])

del f[0]
l = f

with open(path_file, 'wb') as f:
    pickle.dump(l, f, -1)



# gapの値のファイルを読み取る
#コード01a
path_file = "/Users/matsunagahibiki/Documents/#0.resesarch/#1.プログラム/20230621/phase_figure/N=100/parallel_N=100_u=0" + '/dir_data/gap_'+str(turn)+'.pkl'
with open(path_file, 'rb') as f:
     gap_calculate_result= pickle.load(f)

ans  = gap_calculate_result[0]
qs   = gap_calculate_result[1]
Bs   = gap_calculate_result[2]
kBTs = gap_calculate_result[3]

initial = time()

# 重心運動量qごとにもっともギャップの立ちやすいqを取り出す

ans_h = []
ans_i = []
q_max_1 = []
for h in range(n2):
    q_max_2 = []
    ans_h = ans[h]
    for i in range(n1):
        ans_i = ans_h[i]
        ans_l = ans_i.tolist()
        max_q = float(max(ans_i))
        if max_q < 0.0001: 
            max_index = 0
        else:
            max_index = ans_l.index(max_q)
        # print(ans_i)
        # print(max_q)
        q_max_2.append(max_index)
    q_max_1.append(q_max_2)
q_max_1 = np.array(q_max_1)

end = time()

print('max_qにかかった時間は'+ str(end - initial) )

max_q_search_result = [q_max_1, qs, Bs, kBTs]

# max_qのデータを保存
path_file = "/Users/matsunagahibiki/Documents/#0.resesarch/#1.プログラム/20230621/phase_figure/N=100/parallel_N=100_u=0" + '/dir_data/max_q_'+str(turn)+'.pkl'

with open(path_file, 'wb') as f:
    pickle.dump(max_q_search_result, f, -1)

