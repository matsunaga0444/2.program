from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_free_energy import free_energy_SC_BCS_1D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =1000, 1, 1 , 0, 1, 1, 1, 30, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
dq = 0.001
dd = 0.001
ini_gap = 100
qs   = linspace(0,0.005,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.001,0.02,n2)

ans = []
for i in range(n2):
    dc   =   Gap_equation_SC_BCS_1D.scf_1D_simple(kBTs[i], ini_gap, nscf, N, t, qs[0], mu, gu, Bs[0], V)
    x    =   free_energy_SC_BCS_1D.dda_1D(t, mu, gu, Bs[0], 1/kBTs[i], V, N, dc[0], dd)
    ans.append(x)
ans = array(ans)


# def bpp_1D(t, mu, gu, B, beta, V, N, dc, dq):
#     return (1/2)*d2df_q_1D(t, mu, gu, B, beta, V, N, dc, dq)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
print(ans)
plt.scatter(kBTs, ans, 5)
plt.savefig("1D.png")
plt.clf()

###################################
##output
file = open("1D", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(ans[i]) + " "  + "\n")
file.close()