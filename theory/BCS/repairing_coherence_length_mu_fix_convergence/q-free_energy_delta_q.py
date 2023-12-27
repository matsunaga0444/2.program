import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整

n_N = 10
Ns = 500 * (arange(n_N) /n_N)
V, t, mu, gu, n0, n1, n2, nscf =1, 1 , 0, 1, 2, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
qs   = np.linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = np.linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = np.linspace(0.01,1,n2)

###################################################################################################################
## gap_eq をdef

def e_k_spin(k1, k2, k3, q, y, B): 
#    return 2*t*(np.cos((k1+(q/2)*np.pi))) - mu + y * 1/2 * gu * B
    return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))) - mu + y * 1/2 * gu * B
#    return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))+np.cos((k3))) - mu + y * 1/2 * gu * B

def e_k_s(k1, k2, k3, q, B):
    return (e_k_spin(k1, k2, k3, q, 1, B) + e_k_spin(-1*k1, -1*k2, -1*k3, q, -1, B))/2

def e_k_a(k1, k2, k3, q, B):
    return (e_k_spin(k1, k2, k3, q, 1, B) - e_k_spin(-1*k1, -1*k2, -1*k3, q, -1, B))/2

def E_k_q(k1, k2, k3, gap, q, B):
    return np.sqrt(e_k_s(k1, k2, k3, q, B)**2 + gap**2)

def E_k_q_s(k1, k2, k3, gap, q, y, B):
    return E_k_q(k1, k2, k3, gap, q, B) + y * e_k_a(k1, k2, k3, q, B)

def Fermi(beta, E):
    #return  1 / (np.exp(beta*E) + 1 )
    return (1 - np.tanh(beta*E/2)) /2

def func(k1, k2, k3, gap, q, B): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, k3, gap, q, -1, B))-Fermi(beta, E_k_q_s(k1, k2, k3, gap, q, 1, B)))/(2*E_k_q(k1, k2, k3, gap, q, B))

# def rhs(gap, q, B):
#     k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
#     kx, ky, kz = meshgrid(k1, k1, k1, indexing='ij')
#     f = func(kx, ky, kz, gap, q, B)
#     return (V / (N**3)) * sum(f)

def rhs(gap, q, B):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    kz = 1
    f = func(kx, ky, kz, gap, q, B)
    return (V / (N**2)) * sum(f)

###################################################################################################################
#free energy の定義

def Fn():
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    y  = -1 + 2 * arange(2)
    kx, ky, y = meshgrid(k1, k1, y, indexing='ij')
    kz = 1
    f_n = np.log(1+np.exp(-1*beta*e_k_spin(kx, ky, kz, 0, y, 0)))
    #f_n = np.log(-2 / np.tanh(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)/2)-1)
    print("fn",-1*1/beta*np.sum(f_n))
    return -1*1/beta*np.sum(f_n)
    
def F1(h):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    y  = -1 + 2 * arange(2)
    kx, ky, y = meshgrid(k1, k1, y, indexing='ij')
    kz = 1
    f_1 = log(1+exp(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)))
    #f_1 = np.log(-2 / np.tanh(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)/2)-1)
    print("f1",-1*1/beta*np.sum(f_1))
    return -1*1/beta*np.sum(f_1)

def F0(h):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    kz = 1
    f_0 = e_k_spin(-1*kx, -1*ky, -1*kz, qs[h], -1, B) - E_k_q_s(kx, ky, kz, ans_q[h], qs[h], -1, B)
    print("f0",sum(f_0))
    return sum(f_0)

def Fc(h):
    print("fc",(N**2)*(ans_q[h]**2)/V)
    return(N**2)*(ans_q[h]**2)/V

def free_energy(h):              #vn0 = (v / n^2) * n0
    print("f",F1(h) + F0(h) + Fc(h) - Fn())
    return F1(h) + F0(h) + Fc(h) - Fn()#  


###################################################################################################################
## def for finding the coefficients of extended GL free energy
def calculate_a_1(x_1, y_1):
    return 2 * y_1/x_1
def calculate_a(x_1, y_1, T_1, T, Tc):
    p = (1-T/Tc)/(1-T_1/Tc)
    return 2 * y_1/x_1 * p
def calculate_b(x_1, y_1):
    return -1 * y_1/(x_1)**2
def calculate_d(x_1, y_1, x_2, y_2, q):
    a = 2 * y_1/x_1
    d = 2 * y_2/x_2
    return (d-a)/q**2
def calculate_e(x_1, y_1, x_2, y_2, q):
    b = -1 * y_1/(x_1)**2
    e = -1 * y_2/(x_2)**2
    return (e-b)/q**2

###################################################################################################################
## def for calculating coherence length
def c_l_puterbation_GL(a, d):
    return sqrt(-1*(d/a))
def c_l_puterbation_extended_GL(a, b, d, e):
    return sqrt(-1*(d/a) + (e/(2*b)) )
def coherence_length_GL_formulation(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))


###################################################################################################################
##ギャップの逐次計算
ans_c_l = []
for i_N in range(n_N):
    ans = []
    for h in range(n0):
        ans0 = []
        for i in range(n1):
            ans1 = []
            for j in range(n2): # それぞれの温度で秩序パラメータを計算
                beta, d0, N = 1/kBTs[j], 100.0, Ns[i_N]
                for k in range(nscf): # 収束するまで最大1000回ループ
                    d1 = rhs(d0, qs[h], Bs[i]) 
                    if abs(d1-d0) < 1e-10: break # 収束チェック
                    d0 = d1
                ans1.append([d0, abs(d1-d0), k])
            ans0.append(ans1)
        ans.append(ans0)
    ans = np.array(ans)

    ###################################################################################################################
    #free_energy_parameter
    ans_q= ans[:,0,0,0]

    ###################################################################################################################
    #free energy の計算
    kBT = kBTs[0]
    beta = 1/kBT
    B = Bs[0]

    ans_F = []
    for h in range(n0):
        ans_F.append(free_energy(h))
    ans_F = np.array(ans_F)

    ###################################################################################################################
    ## find the coefficients of extended GL free energy
    a_1 = calculate_a_1(ans[0,0,0,0], ans_F[0])
    b = calculate_b(ans[0,0,0,0], ans_F[0])
    d = calculate_d(ans[0,0,0,0], ans_F[0], ans[1:,0,0,0], ans_F[1:], qs[1:])
    e = calculate_e(ans[0,0,0,0], ans_F[0], ans[1:,0,0,0], ans_F[1:], qs[1:])
    c_l = c_l_puterbation_extended_GL(a_1, b, d, e)
    ans_c_l.append(c_l)
ans_c_l = array(ans_c_l)

###################################
##plot

plt.scatter(Ns, ans_c_l)
plt.savefig("test.png")
plt.clf


###################################
##output

file = open("c_l", "w")
for i_N in range(n_N):
    file.write(str(Ns[i_N]) + " " + str(ans_c_l[i_N,0])+  "\n")
file.close()