from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_3D
from SC_BCS_free_energy import free_energy_SC_BCS_3D
from SC_coherence_length import coherence_length_3D


############################################################################################
###from extended GL

def penetration_depth_from_extended_GL_3D(t, mu, gu, B, beta, V, N, dc, dq):
    return sqrt(1/(32*pi*free_energy_SC_BCS_3D.bpp_3D(t, mu, gu, B, beta, V, N, dc, dq)))

def free_energy_to_current(t, mu, gu, B, beta, V, N, dc, dq, q, dc_q):
    return 4*(free_energy_SC_BCS_3D.bpp_3D(t, mu, gu, B, beta, V, N, dc, dq)/dc**2) * dc_q **2 * q * pi

def free_energy_to_max_current(nscf, xi, t, mu, gu, B, beta, V, N, dq):
    max_q = 1/(sqrt(3)*(xi))
    dc_q  = Gap_equation_SC_BCS_3D.scf_3D_simple(1/beta, 100, nscf, N, t, max_q/pi, mu, gu, B, V)
    return 4*free_energy_SC_BCS_3D.bpp_3D(t, mu, gu, B, beta, V, N, dc_q[0], dq) * max_q/pi

def penetration_depth_Extended_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    lamb = 1 / sqrt(12 * sqrt(3) * pi \
            * coherence_length_3D.coherence_length_from_GL_theory_3D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) \
            *  free_energy_to_max_current(nscf, xi, t, mu, gu, B, 1/kBT, V, N, dq))
    return lamb

############################################################################################
###from GL
def v_k_3D(t, k1):
    return -2 * t * (sin(k1))

def G_f_3D(t, k1, k2, k3, gap, q, mu, gu, B, beta):
    a = ( -1 * Gap_equation_SC_BCS_3D.e_k_s_3D(t, k1, k2, k3, q, mu, gu, B) \
         + Gap_equation_SC_BCS_3D.E_k_q_3D(t, k1, k2, k3, q, mu, gu, B, gap))/ \
            (2 * Gap_equation_SC_BCS_3D.E_k_q_3D(t, k1, k2, k3, q, mu, gu, B, gap))
    b = (Gap_equation_SC_BCS_3D.e_k_s_3D(t, k1, k2, k3, q, mu, gu, B) \
         + Gap_equation_SC_BCS_3D.E_k_q_3D(t, k1, k2, k3, q, mu, gu, B, gap))/ \
            (2 * Gap_equation_SC_BCS_3D.E_k_q_3D(t, k1, k2, k3, q, mu, gu, B, gap))
    return (a*(1-fermi.Fermi(beta, Gap_equation_SC_BCS_3D.E_k_q_s_3D(t, k1, k2, k3, q, mu, -1, gu, B, gap)))) \
           +(b*fermi.Fermi(beta, Gap_equation_SC_BCS_3D.E_k_q_s_3D(t, k1, k2, k3, q, mu, 1, gu, B, gap)))

def current_3D(t, n_k, gap, qs, mu, gu, B, beta):
    k1 = -1 * pi + 2 * arange(n_k) * pi / (n_k)
    kx, ky, kz = meshgrid(k1, k1, k1, indexing='ij')
    j_k = v_k_3D(t, kx) * G_f_3D(t, kx-(qs/2)*pi, ky, kz, gap, qs, mu, gu, B, beta)
    return 2 * sum(j_k)/n_k**3

def max_current(nscf, N, t, n_k, xi, mu, gu, B, beta, V):
    max_q = 1/(sqrt(3)*(xi))
    dc_q  = Gap_equation_SC_BCS_3D.scf_3D_simple(1/beta, 100, nscf, N, t, max_q/pi, mu, gu, B, V)
    return current_3D(t, n_k, dc_q[0], max_q/pi, mu, gu, B, beta)

def penetration_depth_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    lamb = 1 / sqrt(12 * sqrt(3) * pi \
            * coherence_length_3D.coherence_length_from_GL_theory_3D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) \
            * max_current(nscf, N, t, N, xi, mu, gu, B, 1/kBT, V))
    return lamb

############################################################################################
###linear response theory

def penetration_depth_linear_3D(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, ky, kz = meshgrid(k1, k1, k1, indexing='ij')
    e_k = electron_band.e_k_s_3D(t, kx, ky, kz, 0, mu, 0, 0, 0)
    dfE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    dfxi = fermi.derivative_Fermi_function(beta, e_k)
    #f = ((kx)**2+(ky)**2+(kz)**2)/3 * (dfE-dfxi)
    f = (-2*t*sin(kx))**2 * (dfE-dfxi)
    return 1/sqrt(2*pi*sum(f)/(N**3))

