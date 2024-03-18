from numpy import *
from basic import electron_band, fermi

###############################################################################
## q = 0の時

def gap_to_E(gap, N, t, mu, gu, B):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    H11 = diag(electron_band.e_k_s_1D(t, k1, 0, mu, 0, gu, B))
    H12 = -1 * gap * eye(N)
    H21 = -1 * gap * eye(N)
    H22 = diag(-1 * electron_band.e_k_s_1D(t, -1 * k1, 0, mu, 0, gu, B))
    H = block([[H11, H12], [H21, H22]])
    eig = linalg.eig(H)
    E = eig[0]
    print(E)
    return E

def E_to_gap(gap, E, V, beta):
    N = len(E)
    abs_E = abs(E)
    g= V/N * sum(gap/(2*abs_E)*tanh(beta * abs_E/2))
    return g

def scf_1D_simple(kBT, ini_gap, nscf, N, t, q, mu, gu, B, V):
    beta, d0 = 1/kBT, ini_gap
    for k in range(nscf): # 収束するまで最大1000回ループ
        E  = gap_to_E(d0, N, t, q, mu, gu, B)
        d1 = E_to_gap(d0, E, V, beta)
        if abs(d1-d0) < 1e-10: break # 収束チェック
        d0 = d1
    ans = [d0, abs(d1-d0), k]
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]

###############################################################################
## q 有限の時

def gap_to_E_finite_q(gap, N, t, q, mu, gu, B):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    H11 = diag(electron_band.e_k_s_1D(t, k1, q, mu, 0, gu, B))
    H12 = -1 * gap * eye(N)
    H21 = -1 * gap * eye(N)
    H22 = diag(-1 * electron_band.e_k_s_1D(t, -1 * k1, q, mu, 0, gu, B))
    H = block([[H11, H12], [H21, H22]])
    eig = linalg.eig(H)
    E = array(eig[0])
    E = block([[E[0:N], -1 * E[N:2*N]]])
    print((E))
    return E

def E_to_gap_finite_q(gap, E_q, E, V, beta):
    N = len(E)
    abs_E = abs(E)
    f = gap/(2*E)*(1-fermi.Fermi(beta, E_q))/2
    g= V/N * sum(f)
    return g

def scf_1D_simple_finite_q(kBT, ini_gap, nscf, N, t, q, mu, gu, B, V):
    beta, d0 = 1/kBT, ini_gap
    for k in range(nscf): # 収束するまで最大1000回ループ
        E    = gap_to_E(d0, N, t, mu, gu, B)
        E_q  = gap_to_E_finite_q(d0, N, t, q, mu, gu, B)
        d1   = E_to_gap_finite_q(d0, E_q, E, V, beta)
        if abs(d1-d0) < 1e-10: break # 収束チェック
        d0 = d1
    ans = [d0, abs(d1-d0), k]
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]
