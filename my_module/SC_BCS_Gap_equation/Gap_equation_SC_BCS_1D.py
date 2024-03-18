from numpy import *
from basic import electron_band, fermi

def e_k_s_1D(t, k1, q, mu, gu, B):
    return (electron_band.e_k_s_1D(t, k1, q, mu, 1, gu, B) + electron_band.e_k_s_1D(t, -1*k1, q, mu, -1, gu, B))/2

def e_k_a_1D(t, k1, q, mu, gu, B):
    return (electron_band.e_k_s_1D(t, k1, q, mu, 1, gu, B) - electron_band.e_k_s_1D(t, -1*k1, q, mu, -1, gu, B))/2

def E_k_q_1D(t, k1, q, mu, gu, B, gap):
    return sqrt(e_k_s_1D(t, k1, q, mu, gu, B)**2 + gap**2)

def E_k_q_s_1D(t, k1, q, mu, y, gu, B, gap):
    return E_k_q_1D(t, k1, q, mu, gu, B, gap) + y * e_k_a_1D(t, k1, q, mu, gu, B)

def func_1D(beta,t, k1, q, mu, gu, B, gap): 
    return gap*(1-fermi.Fermi(beta, E_k_q_s_1D(t, k1, q, mu, -1, gu, B, gap))-fermi.Fermi(beta, E_k_q_s_1D(t, k1, q, mu, 1, gu, B, gap)))/(2*E_k_q_1D(t, k1, q, mu, gu, B, gap))

def rhs_1D(N, beta, t, q, mu, gu, B, gap, V):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    f = func_1D(beta,t, k1, q, mu, gu, B, gap)
    return (V / (N)) * sum(f)

def scf_1D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, q, mu, gu, B, V):
    ans = []
    for h in range(n0):
        ans0 = []
        for i in range(n1):
            ans1 = []
            for j in range(n2): # それぞれの温度で秩序パラメータを計算
                beta, d0 = 1/kBTs[j], ini_gap
                for k in range(nscf): # 収束するまで最大1000回ループ
                    d1 = rhs_1D(N, beta, t, q[h], mu, gu, B[i], d0, V)
                    if abs(d1-d0) < 1e-10: break # 収束チェック
                    d0 = d1
                ans1.append([d0, abs(d1-d0), k])
            ans0.append(ans1)
        ans.append(ans0)
    ans = array(ans)
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]

def scf_1D_simple(kBT, ini_gap, nscf, N, t, q, mu, gu, B, V):
    beta, d0 = 1/kBT, ini_gap
    for k in range(nscf): # 収束するまで最大1000回ループ
        d1 = rhs_1D(N, beta, t, q, mu, gu, B, d0, V)
        if abs(d1-d0) < 1e-10: break # 収束チェック
        d0 = d1
    ans = [d0, abs(d1-d0), k]
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]
