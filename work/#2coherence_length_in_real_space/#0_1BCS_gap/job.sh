#!/bin/bash
#PBS -N eliashberg
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -q GroupA

cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

export PATHPYTHON3=/home/matsunaga/anaconda3/bin/python

# $PATHPYTHON3 FFLO_1D_mu_0.0_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_1D_mu_0.1_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_1D_mu_0.2_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_1D_mu_0.3_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_1D_mu_0.4_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.0_N_600.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.0_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.1_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.2_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.3_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_2D_mu_0.4_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_3D_mu_0.0_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_3D_mu_0.1_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_3D_mu_0.2_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_3D_mu_0.3_N_200.py
# sleep 1
#$PATHPYTHON3 FFLO_3D_mu_0.4_N_200.py
