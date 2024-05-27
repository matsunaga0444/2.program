from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_3D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =30, 1, 1 , 0, 1, 100, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
dq = 0.001
ini_gap = 100
qs   = linspace(0,0.005,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.006,0.05,n2)

gap = []
gap    = Gap_equation_SC_BCS_3D.scf_3D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)


########################################################################################################################
#plot the figure of comparing free energy to extended GL
print(gap[:,0,0,0])
plt.scatter(qs, gap[:,0,0,0], 5)
plt.savefig("q_gap_3D_kBT_0.006.png")
plt.clf()

###################################
##output
file = open("q_gap_3D_kBT_0.006", "w")
for i in range(n0):
    file.write(str(qs[i]) + " " + str(gap[i,0,0,0]) + " "  + "\n")
file.close()