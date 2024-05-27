from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_3D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_3D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =50, 2, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.2,n2)

n_mu = 100
mus = linspace(-6,6,n_mu)

xi = []
for i in range(n_mu):
    ans = Gap_equation_SC_BCS_3D.scf_3D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mus[i], gu, Bs, V)
    xi_Ex_GL    = coherence_length_3D.coherence_length_from_extended_GL_theory_3D(t, mus[i], gu, Bs[0], kBTs[0], V, N, dq, ans[0,0,0,0])
    xi.append(xi_Ex_GL)
xi = array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(mus, xi, 5)
plt.savefig("xi_Ex_GL.png")
plt.clf()

###################################
##output
file = open("xi_Ex_GL", "w")
for i in range(100):
    file.write(str(mus[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
