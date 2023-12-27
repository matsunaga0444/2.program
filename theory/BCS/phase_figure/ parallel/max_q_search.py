# gapの値のファイルを読み取る






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
print(q_max_1)



# max_qのデータを保存
import os
path_file = os.getcwd() + '\\dir_data\\output_data_02a.txt'
with open(path_file, 'w', encoding = 'utf-8') as f:
  f.write('計算結果\n')
  f.write('鉛直変位：{:10.6f}mm\n'.format(3.25))
  f.write('this ')
  #f.write('year:', 2020)   #エラー
  f.write('year: 2020\n')

import os
path_file = os.getcwd() + '\\dir_data\\output_data_02b.txt'
with open(path_file, 'w', encoding = 'utf-8') as f:
    print('計算結果', file = f)
    print('鉛直変位：{:10.6f}mm'.format(3.25), file = f)
    print('this ', end = '', file = f)
    print('year:', 2020, file = f)

