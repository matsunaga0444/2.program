from numpy import *
from basic import electron_band_free, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D_free

###################################################################################################################
#free energy_1D
def F1_1D(t, mu, gu, B, beta, N, qs, ans_q):
    y = -1 + 2 * arange(2)
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, y = meshgrid(k1, y, indexing='ij')
    g = log(1+exp(-1*beta*Gap_equation_SC_BCS_1D_free.E_k_q_s_1D(t, kx, qs, mu, y, gu, B, ans_q)))   
    return -1*(1/beta) * sum(g)

def F0_1D(t, mu, gu, B, N, qs, ans_q):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx = k1
    f = electron_band_free.e_k_s_1D(t, -1*k1, qs, mu, -1, gu, B) - Gap_equation_SC_BCS_1D_free.E_k_q_s_1D(t, kx, qs, mu, -1, gu, B, ans_q)
    return sum(f)


def abs_F0_1D(t, mu, gu, B, N, qs, ans_q):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx = k1
    f = abs(electron_band_free.e_k_s_1D(t, -1*k1, qs, mu, -1, gu, B)) - Gap_equation_SC_BCS_1D_free.E_k_q_s_1D(t, kx, qs, mu, -1, gu, B, ans_q)
    return sum(f)

def Fc_1D(N, ans_q, V):
    return(N)*(ans_q**2)/V

def Fn_1D(t, mu, gu, beta, N, V):
    return F1_1D(t, mu, gu, 0, beta, N, 0, 0) + F0_1D(t, mu, gu, 0, N, 0, 0) + Fc_1D(N, 0, V)

def free_energy_1D(t, mu, gu, B, beta, V, N, qs,ans_q):
    return (F1_1D(t, mu, gu, B, beta, N, qs, ans_q) + F0_1D(t, mu, gu, B, N, qs, ans_q) + Fc_1D(N, ans_q, V) -Fn_1D(t, mu, gu, beta, N, V)) / N

def abs_free_energy_1D(t, mu, gu, B, beta, V, N, qs,ans_q):
    return (F1_1D(t, mu, gu, B, beta, N, qs, ans_q) + abs_F0_1D(t, mu, gu, B, N, qs, ans_q) + Fc_1D(N, ans_q, V) )/ N

###################################################################################################################
#a,b

def d2f_1D(t, mu, gu, B, beta, V, N, dc, dd):
    d1 = dc + dd
    d2 = dc + 2 * dd
    dfj1  = (free_energy_1D(t, mu, gu, B, beta, V, N, 0, d1) - free_energy_1D(t, mu, gu, B, beta, V, N, 0, dc))/(d1-dc)
    dfj2  = (free_energy_1D(t, mu, gu, B, beta, V, N, 0, d2) - free_energy_1D(t, mu, gu, B, beta, V, N, 0, d1))/(d2-d1)
    return (dfj2-dfj1)/(d2-d1)

def d2df_q_1D(t, mu, gu, B, beta, V, N, dc, dq):
    qs = linspace(-1*dq,dq,3) 
    ddf_q1 = ((free_energy_1D(t, mu, gu, B, beta, V, N, qs[1], dc))-(free_energy_1D(t, mu, gu, B, beta, V, N, qs[0], dc)))/((qs[1]-qs[0])*pi)
    ddf_q2 = ((free_energy_1D(t, mu, gu, B, beta, V, N, qs[2], dc))-(free_energy_1D(t, mu, gu, B, beta, V, N, qs[1], dc)))/((qs[2]-qs[1])*pi)
    return (ddf_q2 - ddf_q1)/((qs[1]-qs[0])*pi)


def dda_1D(t, mu, gu, B, beta, V, N, dc, dd):
    return (1/2)*d2f_1D(t, mu, gu, B, beta, V, N, dc, dd)

def bpp_1D(t, mu, gu, B, beta, V, N, dc, dq):
    return (1/2)*d2df_q_1D(t, mu, gu, B, beta, V, N, dc, dq)



