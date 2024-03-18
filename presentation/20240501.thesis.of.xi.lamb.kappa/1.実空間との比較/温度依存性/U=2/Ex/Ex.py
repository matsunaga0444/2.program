from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_1D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 2, 1 , 0, 1, 1, 1, 100, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.83,n2)

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)

xi = []
for i in range(n2):
    xi_Ex_GL    = coherence_length_1D.coherence_length_from_extended_GL_theory_1D(t, mu, gu, Bs[0], kBTs[i], V, N, dq, ans[0,0,i,0])
    xi.append(xi_Ex_GL)
xi = array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(kBTs, xi, 5)
plt.savefig("xi_Ex_GL.png")
plt.clf()

###################################
##output
file = open("xi_Ex_GL", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
