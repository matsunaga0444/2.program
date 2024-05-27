from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_kappa import kappa_3D
from SC_coherence_length import coherence_length_3D
from SC_penetration_depth import penetration_depth_SC_3D

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =50, 2, 1 , 0.1, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 100
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.106,0.2,n2)

xi = []
for i in range(n2):
    xi = coherence_length_3D.coherence_length_from_GL_theory_3D(dq, kBTs[i], ini_gap, nscf, N, t, mu, gu, Bs[0], V)
    #print("xi",  kBTs[i], xi )
    #print("lam", kBTs[i], penetration_depth_SC_3D.penetration_depth_GL(xi, dq, kBTs[i], ini_gap, nscf, N, t, mu, gu, Bs[0], V))
    xi_GL    = kappa_3D.kappa_from_GL_3D(t, mu, gu, Bs[0], kBTs[i], V, N, ini_gap, nscf, dq)
    #print(xi_GL)
    xi.append(xi_GL)
xi = array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(kBTs, xi, 5)
plt.savefig("xi_GL.png")
plt.clf()

###################################
##output
file = open("xi_GL", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()
