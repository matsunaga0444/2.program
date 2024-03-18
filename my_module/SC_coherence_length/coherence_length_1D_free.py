from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D_free
from SC_BCS_free_energy import free_energy_SC_BCS_1D_free


############################################################################################
###from GL
def gap_to_xi_1D(gap_0, gap_q, dq):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))

def coherence_length_from_GL_theory_1D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    qs = linspace(0.0, dq, 2) 
    ans_0 = Gap_equation_SC_BCS_1D_free.scf_1D_simple(kBT, ini_gap, nscf, N, t, qs[0], mu, gu, B, V)
    gap_0 = ans_0[0]
    ans_q = Gap_equation_SC_BCS_1D_free.scf_1D_simple(kBT, ini_gap, nscf, N, t, qs[1], mu, gu, B, V)
    gap_q = ans_q[0]
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))



def coherence_length_from_extended_GL_theory_1D(t, mu, gu, B, kBT, V, N, dq, dc):
    return sqrt(2 * free_energy_SC_BCS_1D_free.bpp_1D(t, mu, gu, B, 1/kBT, V, N, dc, dq)\
                /(dc**2 * free_energy_SC_BCS_1D_free.dda_1D(t, mu, gu, B, 1/kBT, V, N, dc, dq)))

