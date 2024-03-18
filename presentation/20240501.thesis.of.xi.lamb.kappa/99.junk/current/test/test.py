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
kBTs = linspace(0.18,0.2,n2)
ini_gap = 100

ans = Gap_equation_SC_BCS_1D.scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, qs, mu, gu, Bs, V)

xi = coherence_length_1D.gap_to_xi_1D(ans[0,0,0,0], ans[1,0,0,0], qs[1])

print(max_current(t, N, ans[0,0,0,0], xi, mu, gu, Bs[0], 1/kBTs[0]))
print(free_energy_to_max_current(nscf, xi, t, mu, gu, Bs[0], 1/kBTs[0], V, N, 0.001))



