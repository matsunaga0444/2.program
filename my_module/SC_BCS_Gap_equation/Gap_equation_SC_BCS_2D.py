from numpy import *
from basic import electron_band, fermi

def e_k_s_2D(t, k1, k2, q, mu, gu, B):
    return (electron_band.e_k_s_2D(t, k1, k2, q, mu, 1, gu, B) + electron_band.e_k_s_2D(t, -1*k1, -1*k2, q, mu, -1, gu, B))/2

def e_k_a_2D(t, k1, k2, q, mu, gu, B):
    return (electron_band.e_k_s_2D(t, k1, k2, q, mu, 1, gu, B) - electron_band.e_k_s_2D(t, -1*k1, -1*k2, q, mu, -1, gu, B))/2

def E_k_q_2D(t, k1, k2, q, mu, gu, B, gap):
    return sqrt(e_k_s_2D(t, k1, k2, q, mu, gu, B)**2 + gap**2)

def E_k_q_s_2D(t, k1, k2, q, mu, y, gu, B, gap):
    return E_k_q_2D(t, k1, k2, q, mu, gu, B, gap) + y * e_k_a_2D(t, k1, k2, q, mu, gu, B)

def func_2D(beta,t, k1, k2, q, mu, gu, B, gap): 
    return gap*(1-fermi.Fermi(beta, E_k_q_s_2D(t, k1, k2, q, mu, -1, gu, B, gap))-fermi.Fermi(beta, E_k_q_s_2D(t, k1, k2, q, mu, 1, gu, B, gap)))/(2*E_k_q_2D(t, k1, k2, q, mu, gu, B, gap))

def rhs_2D(N, beta, t, q, mu, gu, B, gap, V):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func_2D(beta,t, kx, ky, q, mu, gu, B, gap)
    return (V / (N**2)) * sum(f)

def scf_2D(n0, n1, n2, kBTs, ini_gap, nscf, N, t, q, mu, gu, B, V):
    ans = []
    for h in range(n0):
        ans0 = []
        for i in range(n1):
            ans1 = []
            for j in range(n2): # それぞれの温度で秩序パラメータを計算
                beta, d0 = 1/kBTs[j], ini_gap
                for k in range(nscf): # 収束するまで最大1000回ループ
                    d1 = rhs_2D(N, beta, t, q[h], mu, gu, B[i], d0, V)
                    if abs(d1-d0) < 1e-10: break # 収束チェック
                    d0 = d1
                ans1.append([d0, abs(d1-d0), k])
            ans0.append(ans1)
        ans.append(ans0)
    ans = array(ans)
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]

def scf_2D_simple(kBTs, ini_gap, nscf, N, t, q, mu, gu, B, V):
    beta, d0 = 1/kBTs, ini_gap
    for k in range(nscf): # 収束するまで最大1000回ループ
        d1 = rhs_2D(N, beta, t, q, mu, gu, B, d0, V)
        if abs(d1-d0) < 1e-10: break # 収束チェック
        d0 = d1
    ans = [d0, abs(d1-d0), k]
    return ans #   [q][B][kBTs][d0, abs(d1-d0), k]
