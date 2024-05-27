from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D

##パラメータの調整
n = 100
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 1, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
dq = 0.001
ini_gap = 100
Ns   = linspace(10,N,n) 
qs   = linspace(0,0.005,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.02,n2)

gap = []
for i in range(n):
    gap_1    = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, Ns[i], t, qs, mu, gu, Bs, V)
    gap.append(gap_1[0][0][0][0])
gap = array(gap)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
print(gap)
plt.scatter(Ns, gap, 5)
plt.savefig("1D.png")
plt.clf()

###################################
##output
file = open("1D", "w")
for i in range(n0):
    file.write(str(Ns[i]) + " " + str(gap[i]) + " "  + "\n")
file.close()