##各種パラメータ
N, V, t, a, u, gu, n0, n1, n2 =100, 1, 1, 1, 3, 1, 90, 50, 50    # 7.525 #9.21
qs   = np.linspace(0,0.1,n0)  #(np.pi/a)
Bs   = np.linspace(0.1,0.15,n1)     #np.linspace(0,0.08,n1)
kBTs = np.linspace(1e-5,0.05,n2)



##各種パラメータの保存

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