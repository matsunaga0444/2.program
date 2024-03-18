from numpy import *
from basic import electron_band, fermi
from SC_BCS_Gap_equation import Gap_equation_SC_BCS_3D
from SC_BCS_free_energy import free_energy_SC_BCS_3D

def gap_to_xi_3D(gap_0, gap_q, dq):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))

def coherence_length_from_GL_theory_3D(dq, kBT, ini_gap, nscf, N, t, mu, gu, B, V):
    qs = linspace(0.0, dq, 2) 
    ans_0 = Gap_equation_SC_BCS_3D.scf_3D_simple(kBT, ini_gap, nscf, N, t, qs[0], mu, gu, B, V)
    gap_0 = ans_0[0]
    ans_q = Gap_equation_SC_BCS_3D.scf_3D_simple(kBT, ini_gap, nscf, N, t, qs[1], mu, gu, B, V)
    gap_q = ans_q[0]
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*dq**2*pi**2))

def coherence_length_from_extended_GL_theory_3D(t, mu, gu, B, kBT, V, N, dq, dc):
    return sqrt(2 * free_energy_SC_BCS_3D.bpp_3D(t, mu, gu, B, 1/kBT, V, N, dc, dq)\
                /(dc**2 * free_energy_SC_BCS_3D.dda_3D(t, mu, gu, B, 1/kBT, V, N, dc, dq)))

############################################################################################
### Pippard　length

def max_gradient(kx, ky, kz):
    return sqrt(sin(kx)**2 + sin(ky)**2 + sin(kz)**2)

def vf_weight(dE, max_gradient):
    return max_gradient/dE

def sum_vf_weight(vf_weights):
    return sum(vf_weights)

def vf_s(kx, ky, kz):
    return sin(kx)

def vf(N, t, mu, y, gu, B, dE):
    k1 = -1 * pi + 2 * arange(N) * pi / (N)
    kx, ky, kz = meshgrid(k1, k1, k1, indexing='ij') 
    kx = kx.reshape(N**3,1)
    ky = ky.reshape(N**3,1)
    kz = kz.reshape(N**3,1)
    k = hstack((kx,ky,kz))
    k = k.tolist()
    #l_k = [x for x in k if abs(electron_band.e_k_s_3D(t, x[0], x[1], x[2], 0, mu, y, gu, B)) < dE]
    l_k = [x for x in k if abs(electron_band.e_k_s_3D(t, x[0], x[1], x[2], 0, mu, y, gu, B)) < dE]
    l_k = array(l_k)
    sum_v_f_weight = sum_vf_weight(vf_weight(dE, max_gradient(l_k[:,0], l_k[:,1], l_k[:,2])))
    sum_gradient  = sum(abs(sin(l_k[:,0])) * vf_weight(dE, max_gradient(l_k[:,0], l_k[:,1], l_k[:,2])))
    return  sum_gradient / sum_v_f_weight

def pippard(N, t, mu, y, gu, V, kBT, B, dE, ini_gap, nscf):
    delta = Gap_equation_SC_BCS_3D.scf_3D_simple(kBT, ini_gap, nscf, N, t, 0, mu, gu, B, V)
    v_f = vf(N, t, mu, y, gu, B, dE)
    return v_f/delta[0]
