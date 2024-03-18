from numpy import *
import matplotlib.pyplot as plt
import sys 
sys.path.append("/Users/matsunagahibiki/Documents/#0.resesarch/2program/my_module")
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_BCS_H_to_gap import H_to_gap

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =10, 2, 1 , 0.1, 1, 2, 1, 100, 1  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
dq = 0.001
ini_gap = 10
qs   = linspace(0.01,wide_q,n0)         #(np.pi/a)
Bs   = linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = linspace(0.01,0.2,n2)


a = Gap_equation_SC_BCS_1D.scf_1D_simple(kBTs[0], ini_gap, nscf, N, t, qs[0], mu, gu, Bs[0], V)
b = H_to_gap.scf_1D_simple_finite_q(kBTs[0], ini_gap, nscf, N, t, qs[0], mu, gu, Bs[0], V)

print(a)
print(b)

