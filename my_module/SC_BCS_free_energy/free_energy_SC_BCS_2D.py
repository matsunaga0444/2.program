from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_2D

###################################################################################################################
#free energy_2D
def F1_2D(t, mu, gu, B, beta, N, qs, ans_q):
    y = -1 + 2 * arange(2)
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, ky, y = meshgrid(k1, k1, y, indexing='ij')
    g = log(1+exp(-1*beta*Gap_equation_SC_BCS_2D.E_k_q_s_2D(t, kx, ky, qs, mu, y, gu, B, ans_q)))   
    return -1*(1/beta) * sum(g)

def F0_2D(t, mu, gu, B, N, qs, ans_q):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = electron_band.e_k_s_2D(t, -1*kx, -1*ky, qs, mu, -1, gu, B) - Gap_equation_SC_BCS_2D.E_k_q_s_2D(t, kx, ky, qs, mu, -1, gu, B, ans_q)
    return sum(f)

def Fc_2D(N, ans_q, V):
    return(N**2)*(ans_q**2)/V

def Fn_2D(t, mu, gu, beta, N, V):
    return F1_2D(t, mu, gu, 0, beta, N, 0, 0) + F0_2D(t, mu, gu, 0, N, 0, 0) + Fc_2D(N, 0, V)

def free_energy_2D(t, mu, gu, B, beta, V, N, qs,ans_q):
    return (F1_2D(t, mu, gu, B, beta, N, qs, ans_q) + F0_2D(t, mu, gu, B, N, qs, ans_q) + Fc_2D(N, ans_q, V) -Fn_2D(t, mu, gu, beta, N, V)) / N**2


###################################################################################################################
#a, b

def d2f_2D(t, mu, gu, B, beta, V, N, dc, dd):
    d1 = dc + dd
    d2 = dc + 2 * dd
    dfj1  = (free_energy_2D(t, mu, gu, B, beta, V, N, 0, d1) - free_energy_2D(t, mu, gu, B, beta, V, N, 0, dc))/(d1-dc)
    dfj2  = (free_energy_2D(t, mu, gu, B, beta, V, N, 0, d2) - free_energy_2D(t, mu, gu, B, beta, V, N, 0, d1))/(d2-d1)
    return (dfj2-dfj1)/(d2-d1)

def d2df_q_2D(t, mu, gu, B, beta, V, N, dc, dq):
    qs = linspace(0.0,2*dq,3) 
    ddf_q1 = ((free_energy_2D(t, mu, gu, B, beta, V, N, qs[1], dc))-(free_energy_2D(t, mu, gu, B, beta, V, N, qs[0], dc)))/((qs[1]-qs[0])*pi)
    ddf_q2 = ((free_energy_2D(t, mu, gu, B, beta, V, N, qs[2], dc))-(free_energy_2D(t, mu, gu, B, beta, V, N, qs[1], dc)))/((qs[2]-qs[1])*pi)
    return (ddf_q2 - ddf_q1)/((qs[1]-qs[0])*pi)

def dda_2D(t, mu, gu, B, beta, V, N, dc, dd):
    return (1/2)*d2f_2D(t, mu, gu, B, beta, V, N, dc, dd)

def bpp_2D(t, mu, gu, B, beta, V, N, dc, dq):
    return (1/2)*d2df_q_2D(t, mu, gu, B, beta, V, N, dc, dq)

