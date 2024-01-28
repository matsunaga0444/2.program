import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N = 500
V, t, mu, gu, n0, n1, n2, nscf =2.0, 1 , 0, 1, 1, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
qs     = np.linspace(0.0,0.001,n0)         #(np.pi/a)
Bs     = np.linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs   = np.linspace(0.001,1,n2)

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
    return (F1(qs,ans_q) + F0(qs,ans_q) + Fc(ans_q) -Fn())/N#  

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

###################################
##output
file = open("output/kBT-\Delta", "w")
file.write("##kBT-\Delta" + "\n")
for j in range (n2):
    file.write(str(kBTs[j]) + " " + str(ans[0][0][j][0]) + "\n")
file.close()
