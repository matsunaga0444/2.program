#!/bin/bash
#PBS -l nodes=1:ppn=1


n=48  # 実行回数を設定する


#パラメーターの入力
cd $PBS_O_WORKDIR
python gap_parameter.py 


# #ギャップ計算の繰り返し
cd $PBS_O_WORKDIR
i=0   # カウンタ変数を初期化する
while [ $i -lt $n ]
do
    python gap_calculate.py &
    i=$((i+1))  # カウンタ変数をインクリメントする
    sleep 0.1
done
wait

sleep 7000

#ギャップ計算結果の層図データの作成の繰り返し
cd $PBS_O_WORKDIR
i=0   # カウンタ変数を初期化する
while [ $i -lt $n ]
do
    python max_q_search.py &
    i=$((i+1))  # カウンタ変数をインクリメントする
    sleep 0.05

done
wait

sleep 20

#層図の作成
python phase_plot.py

