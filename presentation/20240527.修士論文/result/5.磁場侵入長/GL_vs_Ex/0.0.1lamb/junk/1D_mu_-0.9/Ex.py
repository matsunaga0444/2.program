from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_coherence_length  import coherence_length_1D
from SC_penetration_depth import penetration_depth_SC_1D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =2000, 2, 1 , -0.9, 1, 1, 1, 100, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.20,n2)

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)

lamb = []
for i in range(n2):
    lamb_Ex_GL = penetration_depth_SC_1D.penetration_depth_from_extended_GL_1D(t, mu, gu, Bs[0], 1/kBTs[i], V, N, ans[0,0,i,0], dq)
    lamb.append(lamb_Ex_GL)
lamb = array(lamb)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(kBTs, lamb, 5)
plt.savefig("lamb_Ex_GL_2000.png")
plt.clf()

###################################
##output
file = open("lamb_Ex_GL_2000", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(lamb[i]) + " "  + "\n")
file.close()
