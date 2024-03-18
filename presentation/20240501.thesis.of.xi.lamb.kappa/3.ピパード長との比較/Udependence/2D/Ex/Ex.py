from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_2D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_2D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =100, 2, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.2,n2)

n_U = 100
Us = linspace(1,3,n_U)

xi = []
for i in range(n_U):
    ans = Gap_equation_SC_BCS_2D.scf_2D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, Us[i])
    xi_Ex_GL    = coherence_length_2D.coherence_length_from_extended_GL_theory_2D(t, mu, gu, Bs[0], kBTs[0], Us[i], N, dq, ans[0,0,0,0])
    xi.append(xi_Ex_GL)
xi = array(xi)


########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(Us, xi, 5)
plt.savefig("xi_Ex_GL.png")
plt.clf()

###################################
##output
file = open("xi_Ex_GL", "w")
for i in range(100):
    file.write(str(Us[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
