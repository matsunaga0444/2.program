from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_free_energy import free_energy_SC_BCS_1D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =100, 2, 1 , 0, 100, 100, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.03
qs   = linspace(-1*wide_q,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.19,0.2,n2)

xi = []
for i in range(n0):
    xi_GL  = free_energy_SC_BCS_1D.free_energy_1D(t, mu, gu, Bs[0], 1/kBTs[0], V, N, qs[i], 0.00)
    xi.append(xi_GL)
xi = array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
print(shape(qs) , shape(xi))
plt.scatter(qs, xi, 5)
plt.savefig("fq-f0.png")
plt.clf()

###################################
##output
file = open("fq-f0", "w")
for i in range(n0):
    file.write(str(qs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
