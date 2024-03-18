from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D

###################################################################################################################
#free energy_1D
def F1_1D(t, mu, gu, B, beta, N, qs, ans_q):
    y = -1 + 2 * arange(2)
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, y = meshgrid(k1, y, indexing='ij')
    g = log(1+exp(-1*beta*Gap_equation_SC_BCS_1D.E_k_q_s_1D(t, kx, qs, mu, y, gu, B, ans_q)))   
    return -1*(1/beta) * sum(g)

def F0_1D(t, mu, gu, B, N, qs, ans_q):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx = k1
    f = electron_band.e_k_s_1D(t, -1*k1, qs, mu, -1, gu, B) - Gap_equation_SC_BCS_1D.E_k_q_s_1D(t, kx, qs, mu, -1, gu, B, ans_q)
    return sum(f)

def abs_F0_1D(t, mu, gu, B, N, qs, ans_q):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx = k1
    f = abs(electron_band.e_k_s_1D(t, -1*k1, qs, mu, -1, gu, B)) - Gap_equation_SC_BCS_1D.E_k_q_s_1D(t, kx, qs, mu, -1, gu, B, ans_q)
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

def d4f_1D(t, mu, gu, B, beta, V, N, dc, dd):
    ds = linspace(dc, dc+4*dd, 5)
    dfs = []
    for i in range(4):
        ans   = (free_energy_1D(t, mu, gu, B, beta, V, N, 0, ds[i+1]) - free_energy_1D(t, mu, gu, B, beta, V, N, 0, ds[i]))/(ds[i+1]-ds[i])
        dfs.append(ans)
    dfs = array(dfs)
    d2fs  = (dfs[1:4]-dfs[0:3])/(ds[2:5]-ds[1:4])
    d3fs  = (d2fs[1:3]-d2fs[0:2])/(ds[3:5]-ds[2:4])
    d4fs  = (d3fs[1]-d3fs[0])/(ds[4]-ds[3])
    return d4fs

def d6f_1D(t, mu, gu, B, beta, V, N, dc, dd):
    ds = linspace(dc, dc+6*dd, 7)
    dfs = []
    for i in range(6):
        ans   = (free_energy_1D(t, mu, gu, B, beta, V, N, 0, ds[i+1]) - free_energy_1D(t, mu, gu, B, beta, V, N, 0, ds[i]))/(ds[i+1]-ds[i])
        dfs.append(ans)
    dfs = array(dfs)
    d2fs  = (dfs[1:6]-dfs[0:5])/(ds[2:7]-ds[1:6])
    d3fs  = (d2fs[1:5]-d2fs[0:4])/(ds[3:7]-ds[2:6])
    d4fs  = (d3fs[1:4]-d3fs[0:3])/(ds[4:7]-ds[3:6])
    d5fs  = (d4fs[1:3]-d4fs[0:2])/(ds[5:7]-ds[4:6])
    d6fs  = (d5fs[1]-d5fs[0])/(ds[6]-ds[5])
    return d6fs

def d2df_q_1D(t, mu, gu, B, beta, V, N, dc, dq):
    qs = linspace(-1*dq,dq,3) 
    ddf_q1 = ((free_energy_1D(t, mu, gu, B, beta, V, N, qs[1], dc))-(free_energy_1D(t, mu, gu, B, beta, V, N, qs[0], dc)))/((qs[1]-qs[0])*pi)
    ddf_q2 = ((free_energy_1D(t, mu, gu, B, beta, V, N, qs[2], dc))-(free_energy_1D(t, mu, gu, B, beta, V, N, qs[1], dc)))/((qs[2]-qs[1])*pi)
    return (ddf_q2 - ddf_q1)/((qs[1]-qs[0])*pi)

def d4df_q_1D(t, mu, gu, B, beta, V, N, dc, dq):
    qs = linspace(-1*dq,dq,5) 
    ddf_qs = []
    for i in range(4):
        ans   = (free_energy_1D(t, mu, gu, B, beta, V, N, qs[i+1], dc) - free_energy_1D(t, mu, gu, B, beta, V, N, qs[i], dc))/(qs[i+1]-qs[i])
        ddf_qs.append(ans)
    d2f_qs  = (ddf_qs[1:4]-ddf_qs[0:3])/(qs[2:5]-qs[1:4])
    d3f_qs  = (d2f_qs[1:3]-d2f_qs[0:2])/(qs[3:5]-qs[2:4])
    d4f_qs  = (d3f_qs[1]-d3f_qs[0])/(qs[4]-qs[3])
    return d4f_qs



def dda_1D(t, mu, gu, B, beta, V, N, dc, dd):
    return (1/2)*d2f_1D(t, mu, gu, B, beta, V, N, dc, dd)

def bpp_1D(t, mu, gu, B, beta, V, N, dc, dq):
    return (1/2)*d2df_q_1D(t, mu, gu, B, beta, V, N, dc, dq)



