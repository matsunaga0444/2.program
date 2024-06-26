from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2.program/my_module")
# sys.path.append("../../../../../my_module")
from SC_penetration_depth import penetration_depth_SC_3D
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_3D

##パラメータの調整
n = 20
N, V, t, mu, gu, n0, n1, n2, nscf =300, 1, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
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
    gap = Gap_equation_SC_BCS_3D.scf_3D(n0, n1, n2, kBTs, ini_gap, nscf, Ns[i], t, qs, mu, gu, Bs, V)
    x    = penetration_depth_SC_3D.penetration_depth_from_extended_GL_3D(t, mu, gu, Bs[0], 1/kBTs[0], V, N, gap[0,0,0,0], dq)
    ans.append(x)
ans = array(ans)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(Ns, ans, 5)
plt.savefig("3D.png")
plt.clf()

###################################
##output
file = open("3D", "w")
for i in range(n):
    file.write(str(Ns[i]) + " " + str(ans[i]) + " "  + "\n")
file.close()