from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_1D
from SC_BCS_free_energy import free_energy_SC_BCS_1D
from SC_coherence_length import coherence_length_1D

############################################################################################
###from GL
def v_k_1D(t, k1):
    return -2 * t * (sin(k1))

def G_f_1D(t, k1, gap, q, mu, gu, B, beta):
    a = ( -1 * Gap_equation_SC_BCS_1D.e_k_s_1D(t, k1, q, mu, gu, B) \
         + Gap_equation_SC_BCS_1D.E_k_q_1D(t, k1, q, mu, gu, B, gap))/ \
            (2 * Gap_equation_SC_BCS_1D.E_k_q_1D(t, k1, q, mu, gu, B, gap))
    b = (Gap_equation_SC_BCS_1D.e_k_s_1D(t, k1, q, mu, gu, B) \
         + Gap_equation_SC_BCS_1D.E_k_q_1D(t, k1, q, mu, gu, B, gap))/ \
            (2 * Gap_equation_SC_BCS_1D.E_k_q_1D(t, k1, q, mu, gu, B, gap))
    return (a*(1-fermi.Fermi(beta, Gap_equation_SC_BCS_1D.E_k_q_s_1D(t, k1, q, mu, -1, gu, B, gap)))) \
           +(b*fermi.Fermi(beta, Gap_equation_SC_BCS_1D.E_k_q_s_1D(t, k1, q, mu, 1, gu, B, gap)))

def current_1D(t, n_k, dc_q, qs, mu, gu, B, beta):
    k1 = -1 * pi + 2 * arange(n_k) * pi / (n_k)
    j_k = v_k_1D(t,k1) * G_f_1D(t, k1-(qs/2)*pi, dc_q, qs, mu, gu, B, beta)
    return 2 * sum(j_k)/n_k

def current_1D_test(t, n_k, dc_q, qs, mu, gu, B, beta):
    k1 = -1 * pi + 2 * arange(n_k) * pi / (n_k)
    j_k = v_k_1D(t, k1-(qs/2)*pi) * G_f_1D(t, k1-(qs/2)*pi, dc_q, qs, mu, gu, B, beta)
    return 2 * sum(j_k)/n_k

def max_current(nscf, N, t, n_k, xi, mu, gu, B, beta, V):
    max_q = 1/(sqrt(3)*(xi))
    dc_q  = Gap_equation_SC_BCS_1D.scf_1D_simple(1/beta, 100, nscf, N, t, max_q/pi, mu, gu, B, V)
    return current_1D(t, n_k, dc_q[0], max_q/pi, mu, gu, B, beta)

# def penetration_depth_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V, gap):
#     lamb = 1 / sqrt(6 * sqrt(3) * pi * 2 \
#             * coherence_length_1D.coherence_length_from_GL_theory_1D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) \
#             * max_current(t, N, gap, xi, mu, gu, B, 1/kBT))
#     return lamb

def penetration_depth_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    lamb = 1 / sqrt(12 * sqrt(3) * pi \
            * coherence_length_1D.coherence_length_from_GL_theory_1D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) \
            * max_current(nscf, N, t, N, xi, mu, gu, B, 1/kBT, V))
    return lamb

############################################################################################
###from extended GL

def penetration_depth_from_extended_GL_1D(t, mu, gu, B, beta, V, N, dc, dq):
    return sqrt(1/(32*pi*free_energy_SC_BCS_1D.bpp_1D(t, mu, gu, B, beta, V, N, dc, dq)))

def free_energy_to_current(t, mu, gu, B, beta, V, N, dc, dq, q, dc_q):
    return 4*(free_energy_SC_BCS_1D.bpp_1D(t, mu, gu, B, beta, V, N, dc, dq)/dc**2) * dc_q **2 * q * pi

def free_energy_to_max_current(nscf, xi, t, mu, gu, B, beta, V, N, dq):
    max_q = 1/(sqrt(3)*(xi))
    dc_q  = Gap_equation_SC_BCS_1D.scf_1D_simple(1/beta, 100, nscf, N, t, max_q/pi, mu, gu, B, V)
    return 4*free_energy_SC_BCS_1D.bpp_1D(t, mu, gu, B, beta, V, N, dc_q[0], dq) * max_q/pi

def penetration_depth_Extended_GL(xi, dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    lamb = 1 / sqrt(12 * sqrt(3) * pi \
            * coherence_length_1D.coherence_length_from_GL_theory_1D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V) \
            *  free_energy_to_max_current(nscf, xi, t, mu, gu, B, 1/kBT, V, N, dq))
    return lamb


############################################################################################
###linear_response_theory

def penetration_depth_linear_1D(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    e_k = electron_band.e_k_s_1D(t, k1, 0, mu, 0, 0, 0)
    dfE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    dfxi = fermi.derivative_Fermi_function(beta, e_k)
    #f = (k1)**2 * (dfE-dfxi)
    f = (-2*t*sin(k1))**2 * (dfE-dfxi)
    # return 1/sqrt(sum(f))
    return 1/sqrt(2**2*pi*sum(f)/(N))

def penetration_depth_linear_1D_test(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    e_k = electron_band.e_k_s_1D(t, k1, 0, mu, 0, 0, 0)
    dfE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    dfxi = fermi.derivative_Fermi_function(beta, e_k)
    #f = (k1)**2 * (dfE-dfxi)
    f = (k1)**2 * (dfE-dfxi)
    # return 1/sqrt(sum(f))
    return 1/sqrt(2**2*2*pi*sum(f)/(N))

def penetration_depth_linear_1D_test2(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    e_k = electron_band.e_k_s_1D(t, k1, 0, mu, 0, 0, 0)
    dfE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    f = (-2*t*sin(k1))**2 * (dfE-e_k)
    return 1/sqrt(2*pi*sum(f)/(N))

def penetration_depth_linear_1D_test3(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    e_k = electron_band.e_k_s_1D(t, k1, 0, mu, 0, 0, 0)
    dfE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    dfxi = fermi.derivative_Fermi_function(beta, e_k)
    f = (-2*t*sin(k1))**2 * (dfE-dfxi)
    return 1/sqrt(4*pi*sum(f)/(N))

def penetration_depth_linear_1D_test4(t, mu, beta, N, gap):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    e_k = electron_band.e_k_s_1D(t, k1, 0, mu, 0, 0, 0)
    # fE  = fermi.Fermi(beta, sqrt(e_k**2+gap**2))
    # fxi = fermi.Fermi(beta, e_k)
    fE  = fermi.derivative_Fermi_function(beta, sqrt(e_k**2+gap**2))
    fxi = fermi.derivative_Fermi_function(beta, e_k)
    f = (1/(-2*cos(k1))) * (fE-fxi)
    f = 1 * (1/(2*cos(k1))) * (fE-fxi)
    print(sqrt(sum(f)/(N)))
    return 1/sqrt(sum(f)/(N))
