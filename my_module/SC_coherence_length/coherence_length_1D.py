from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_BCS_free_energy import free_energy_SC_BCS_1D


############################################################################################
###from GL
def gap_to_xi_1D(gap_0, gap_q, dq):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))

def coherence_length_from_GL_theory_1D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    qs = linspace(0.0, dq, 2) 
    ans_0 = Gap_equation_SC_BCS_1D.scf_1D_simple(kBT, ini_gap, nscf, N, t, qs[0], mu, gu, B, V)
    gap_0 = ans_0[0]
    ans_q = Gap_equation_SC_BCS_1D.scf_1D_simple(kBT, ini_gap, nscf, N, t, qs[1], mu, gu, B, V)
    gap_q = ans_q[0]
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))



def coherence_length_from_extended_GL_theory_1D(t, mu, gu, B, kBT, V, N, dq, dc):
    return sqrt(2 * free_energy_SC_BCS_1D.bpp_1D(t, mu, gu, B, 1/kBT, V, N, dc, dq)\
                /(dc**2 * free_energy_SC_BCS_1D.dda_1D(t, mu, gu, B, 1/kBT, V, N, dc, dq)))


############################################################################################
### Pippard　length

def max_gradient(kx):
    return sqrt(sin(kx)**2)

def vf_weight(dE, max_gradient):
    return max_gradient/dE

def sum_vf_weight(vf_weights):
    return sum(vf_weights)

def vf_s(kx):
    return sin(kx)

def vf(N, t, mu, y, gu, B, dE):
    list_k = []
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    list_k = [k for k in k1 if abs(electron_band.e_k_s_1D(t, k, 0, mu, y, gu, B)) < dE]
    sum_v_f_weight = sum_vf_weight(vf_weight(dE, max_gradient(list_k)))
    sum_gradient  = sum(abs(sin(list_k)) * vf_weight(dE, max_gradient(list_k)))
    return  sum_gradient / sum_v_f_weight

def pippard(N, t, mu, y, gu, V, kBT, B, dE, ini_gap, nscf):
    delta = Gap_equation_SC_BCS_1D.scf_1D_simple(kBT, ini_gap, nscf, N, t, 0, mu, gu, B, V)
    v_f = vf(N, t, mu, y, gu, B, dE)
    return v_f/delta[0]
