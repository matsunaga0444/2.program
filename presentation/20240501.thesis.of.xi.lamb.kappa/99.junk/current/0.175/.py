from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_coherence_length import coherence_length_1D
from SC_penetration_depth import penetration_depth_SC_1D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_kappa import kappa_1D

##パラメータの調整
h = pi
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 2, 1, 0.1, 0, 100, 1, 1, 10000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.2
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.175,0.2,n2)
ini_gap = 100

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)

ans_j   = []
for i in range(n0):
    beta = 1/kBTs[0]
    j   = penetration_depth_SC_1D.current_1D(t, N, ans[i,0,0,0], qs[i], mu, gu, Bs[0], beta)
    ans_j.append(j)
ans_j   = array(ans_j)

xi = coherence_length_1D.gap_to_xi_1D(ans[0,0,0,0], ans[1,0,0,0], qs[1])


########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(qs * pi, ans_j, 5)
plt.axvline(x=1/(sqrt(3)*xi), color = 'green')
plt.savefig("j_q_GL.png")
plt.clf()

###################################
##output
file = open("j_q_GL", "w")
file.write("##kBT-lamb" + "\n")
for i in range(n0):
    file.write(str(qs[i]* pi) + " " + str(ans_j[i]) + " "  + "\n")
file.close()