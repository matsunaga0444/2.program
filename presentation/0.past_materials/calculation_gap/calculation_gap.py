from numpy import *
import sys
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
import Gap_equation_SC_BCS

N, V, t, mu, gu  = 1000, 2, 1 , 0, 1
n0, n1, n2, nscf = 1, 1, 100, 100000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
qs   = linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.01,0.2,n2)

ans = Gap_equation_SC_BCS.scf_1D(n0, n1, n2, kBTs, 100, nscf, \
                           N, t, qs, mu, gu, Bs, V)

print(ans)


###################################
##output
file = open("kBT-gap", "w")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(ans[0,0,i,0]) + " "  + "\n")
file.close()
