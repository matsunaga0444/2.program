from numpy import *
from SC_coherence_length import coherence_length_2D
from SC_penetration_depth import penetration_depth_SC_2D

def kappa_from_Ex_GL_2D(t, mu, gu, B, kBT, V, N, dc, dq):
    return penetration_depth_SC_2D.penetration_depth_from_extended_GL_2D(t, mu, gu, B, 1/kBT, V, N, dc, dq) /  \
           coherence_length_2D.coherence_length_from_extended_GL_theory_2D(t, mu, gu, B, kBT, V, N, dq, dc)

def kappa_from_GL_2D(t, mu, gu, B, kBT, V, N, ini_gap, nscf, dq):
    xi = coherence_length_2D.coherence_length_from_GL_theory_2D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V)
    return penetration_depth_SC_2D.penetration_depth_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) / xi