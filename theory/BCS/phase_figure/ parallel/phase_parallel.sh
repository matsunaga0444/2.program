#!/bin/bash
#PBS -l nodes=1:ppn=1
cd $PBS_O_WORKDIR



#パラメーターの入力
python gap_parameter.py &


wait

#ギャップ計算の繰り返し

for ~~
  python gap_calculate.py &



wait

#ギャップ計算結果の層図データの作成の繰り返し

for ~~
  python max_q_search.py &


wait

#層図の作成
python plot_phase.py

wait

#ポストプロセス
python plot_phase.py 

