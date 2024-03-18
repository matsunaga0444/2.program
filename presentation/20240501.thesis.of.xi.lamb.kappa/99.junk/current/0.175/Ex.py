from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_1D
from SC_penetration_depth import penetration_depth_SC_1D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_kappa import kappa_1D


##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 2, 1, 0.1, 0, 100, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.2
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.175,0.195,n2)

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)

j_q = []
for i in range(n0):
    j = penetration_depth_SC_1D.free_energy_to_current(t, mu, gu, Bs[0], 1/kBTs[0], V, N, ans[i,0,0,0], dq, qs[i])
    j_q.append(j)
j_q = array(j_q)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(qs* pi, j_q, 5)
plt.savefig("j_q_GL_Ex.png")
plt.clf()

###################################
##output
file = open("j_q_GL_Ex", "w")
for i in range(n0):
    file.write(str(qs[i]*pi) + " " + str(j_q[i]) + " "  + "\n")
file.close()



