from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_free_energy import free_energy_SC_BCS_1D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 2, 1 , 0, 100, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs    = linspace(-1*wide_q,wide_q,n0)         #(np.pi/a)
Bs    = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs  = linspace(0.10,0.193,n2)

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)
print(ans[0][0][0][0])
psi_s = linspace(0-0.1,0+0.1,100)

xi = []
for i in range(100):
    xi_GL  =  free_energy_SC_BCS_1D.d2df_q_1D(t, mu, gu, Bs[0], 1/kBTs[0], V, N, psi_s[i], dq) 
    xi.append(xi_GL)
xi = array(xi)


########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(psi_s, xi, 5)
plt.savefig("g_2_psi.png")
plt.clf()


###################################
##output
file = open("g_2_psi", "w")
for i in range(100):
    file.write(str(psi_s[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
