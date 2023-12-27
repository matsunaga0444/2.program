

##パラメータファイルの読み込み
#コード01a
import os
path_file = os.getcwd() + '\\dir_data\\input_data_01a.txt' 
with open(path_file, 'r', encoding = 'utf-8') as f:
    s = f.read()
    print(s)


path_file = os.getcwd() + '\\dir_data\\data_json_07a.txt'
#json形式のテキストファイルpath_fileをファイルオブジェクトfとして読み込む
with open(path_file, 'r') as f:
    #ファイルオブジェクトfとして読み込んだjson形式の
    #テキストファイルpath_fileをデシリアライズし、x_listに代入










##パラメータの入力
N, V, t, a, u, gu, n0, n1, n2 =100, 1, 1, 1, 3, 1, 90, 50, 50    # 7.525 #9.21
qs   = np.linspace(0,0.1,n0)  #(np.pi/a)
Bs   = np.linspace(0.1,0.15,n1)     #np.linspace(0,0.08,n1)
kBTs = np.linspace(1e-5,0.05,n2)

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

print('自己無撞着に計算するのにかかった時間は'+ str(end-initial) )








##ギャップのデータを保存c


#コード09a
import pickle
import os
s_list = [1, 2, 3]

path_file = os.getcwd() + '\\dir_data\\data_pickle_09a.pkl'
#書き込み用ファイルpath_fileをファイルオブジェクトfとして開く
with open(path_file, 'wb') as f:
    #オブジェクトx_listを、pickle化し、
    #ファイルオブジェクトfにs_listを追加し、ファイルpath_fileに出力
    pickle.dump(s_list, f, -1)

path_file = os.getcwd() + '\\dir_data\\data_pickle_09a.pkl'
#pickleデータフォーマットのファイルpath_fileを
#ファイルオブジェクトfとして読み込む
with open(path_file, "rb") as f:
    #ファイルオブジェクトfとして読み込んだpickleデータフォーマットの
    #ファイルpath_fileをオブジェクトに再構築し、x_listに代入








