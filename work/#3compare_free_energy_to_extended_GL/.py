import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N = 2000
V, t, mu, gu, n0, n1, n2, n3, nscf =1.5, 1 , 0, 1, 5, 1, 1, 100, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
qs     = np.linspace(0.0,0.001,n0)         #(np.pi/a)
Bs     = np.linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs   = np.linspace(0.001,1,n2)
deltas = np.linspace(0, 0.2, n3)

#index of q by which calculating extended GL
i_q = 2

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, k3, q, y, B): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))) - mu + y * 1/2 * gu * B

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

def rhs(gap, q, B):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f = func(kx, ky, kz, gap, q, B)
    return (V / (N)) * sum(f)

###################################################################################################################
#free energy の定義
def Fn():
    return F1(0,0) + F0(0,0) + Fc(0)

def F1(qs,ans_q):
    y = -1 + 2 * arange(2)
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx, y = meshgrid(k1, y, indexing='ij')
    ky, kz = 1, 1
    g = log(1+exp(-1*beta*E_k_q_s(kx, ky, kz, ans_q, qs, y, Bs[0])))
    return -1*(1/beta) * sum(g)

def F0(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f = e_k_spin(-1*kx, -1*ky, -1*kz, qs,-1, Bs[0]) - E_k_q_s(kx, ky, kz, ans_q, qs, -1, Bs[0])
    return sum(f)

def Fc(ans_q):
    return(N)*(ans_q**2)/V

def free_energy(qs,ans_q):
    return F1(qs,ans_q) + F0(qs,ans_q) + Fc(ans_q) -Fn()#  

###################################################################################################################
## def for finding the coefficients of extended GL free energy
def calculate_a_1(x_1, y_1):
    return 2 * y_1/(x_1)**2
def calculate_a(x_1, y_1, T_1, T, Tc):
    p = (1-T/Tc)/(1-T_1/Tc)
    return 2 * y_1/(x_1)**2 * p
def calculate_b(x_1, y_1):
    return -1 * y_1/(x_1)**4
def calculate_d(x_1, y_1, x_2, y_2, q):
    a = 2 * y_1/(x_1)**2
    d = 2 * y_2/(x_2)**2
    return (d-a)/(q*pi)**2
def calculate_e(x_1, y_1, x_2, y_2, q):
    b = -1 * y_1/(x_1)**4
    e = -1 * y_2/(x_2)**4
    return (e-b)/(q*pi)**2

###################################################################################################################
## def for finding the coefficients of extended GL free energy
def extende_GL(a, b, d, e, q, delta):
    return a*delta**2 + b*delta**4 \
                      + d*q**2*delta**2 + e*q**2*delta**4


###################################################################################################################
##ギャップの逐次計算
ans= []
for h in range(n0):
    ans0 = []
    for i in range(n1):
        ans1 = []
        for j in range(n2): # それぞれの温度で秩序パラメータを計算
            beta, d0= 1/kBTs[j], 100.0
            for k in range(nscf): # 収束するまで最大1000回ループ
                d1 = rhs(d0, qs[h], Bs[i]) 
                if abs(d1-d0) < 1e-10: break # 収束チェック
                d0 = d1
            ans1.append([d0, abs(d1-d0), k])
        ans0.append(ans1)
    ans.append(ans0)
ans = np.array(ans)

###################################################################################################################
#free energy の計算
kBT = kBTs[0]
beta = 1/kBT
B = Bs[0]
ans_F = []
ans_F.append(free_energy(qs[0], ans[0,0,0,0]))
ans_F.append(free_energy(qs[i_q], ans[i_q,0,0,0]))
ans_F = np.array(ans_F)

###################################################################################################################
## find the coefficients of extended GL free energy
a_1 = calculate_a_1(ans[0,0,0,0], ans_F[0])
b = calculate_b(ans[0,0,0,0], ans_F[0])
d = calculate_d(ans[0,0,0,0], ans_F[0], ans[i_q,0,0,0], ans_F[1], qs[i_q])
e = calculate_e(ans[0,0,0,0], ans_F[0], ans[i_q,0,0,0], ans_F[1], qs[i_q])

###################################################################################################################
##calculate_freeenergy
f, extended_GL = [], []
for i in range(n0):
    f_1, extended_GL_1 = [], []
    for j in range(n3):
        f_temp = free_energy(qs[i],deltas[j])
        f_1.append(f_temp)
        extended_GL_temp = extende_GL(a_1, b, d, e, qs[i],deltas[j])
        extended_GL_1.append(extended_GL_temp)
    f.append(f_1)
    extended_GL.append(extended_GL_1)
f = np.array(f)
extended_GL = np.array(extended_GL)


########################################################################################################################
#plot the figure of comparing free energy to extended GL
for i in range(n0):
    plt.scatter(deltas, f[i,:], 5, label=r"$F_\Delta$")
    plt.scatter(deltas, extended_GL[i,:], 5, label="extended GL free energy")
    plt.legend()
    plt.savefig("figure/"+ str(qs[i]) + ".png")
    plt.clf()

###################################
##output
for i in range(n0):
    file = open("output/"+ str(qs[i]), "w")
    file.write("##delta---free_energy---extended_GL" + "\n")
    for j in range (n3):
        file.write(str(deltas[j]) + " " + str(f[i,j]) + " " + str(extended_GL[i,j]) + "\n")
    file.close()
