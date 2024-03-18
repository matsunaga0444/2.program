from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_3D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_2D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,1.1,n2)
dE  = 0.1

# ans = Gap_equation_SC_BCS_2D.scf_2D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)
# print(ans)

# xi = []
# for i in range(n2):
#     xi_Ex_GL    = coherence_length_2D.coherence_length_from_extended_GL_theory_2D(t, mu, gu, Bs[0], kBTs[i], V, N, dq, ans[0,0,i,0])
#     print(xi_Ex_GL)
#     xi.append(xi_Ex_GL)
# xi = array(xi)

xi = []
for i in range(n2):
    xi_Ex_GL    = coherence_length_3D.pippard(N, t, qs[0], mu, 0, gu, V, kBTs[0], Bs[0], dE, ini_gap, nscf)
    xi.append(xi_Ex_GL)
xi = array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
print(kBTs, xi)
plt.scatter(kBTs, xi, 5)
plt.savefig("xi_Ex_GL.png")
plt.clf()

###################################
##output
file = open("xi_Ex_GL", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
