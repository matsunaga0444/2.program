from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2.program/my_module")
# sys.path.append("../../../../../my_module")
from SC_coherence_length import coherence_length_2D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_2D

##パラメータの調整
n = 10
N, V, t, mu, gu, n0, n1, n2, nscf =500, 1, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
dq = 0.001
ini_gap = 100
Ns   = linspace(10,N,n) 
qs   = linspace(0,0.005,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.02,n2)


ans = []
for i in range(n):
    gap = Gap_equation_SC_BCS_2D.scf_2D(n0, n1, n2, kBTs, ini_gap, nscf, Ns[i], t, qs, mu, gu, Bs, V)
    x    = coherence_length_2D.coherence_length_from_extended_GL_theory_2D(t, mu, gu, Bs[0], kBTs[0], V, Ns[i], dq, gap[0,0,0,0])
    ans.append(x)
ans = array(ans)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(Ns, ans, 5)
plt.savefig("2D.png")
plt.clf()

###################################
##output
file = open("2D", "w")
for i in range(n):
    file.write(str(Ns[i]) + " " + str(ans[i]) + " "  + "\n")
file.close()