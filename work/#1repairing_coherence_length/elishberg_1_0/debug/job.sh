#!/bin/bash
#PBS -N eliashberg
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -q GroupA

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

export PATHPYTHON3=/home/matsunaga/anaconda3/bin/python

$PATHPYTHON3 main_faeb.py