import numpy as np
from numpy import *
import matplotlib.pyplot as plt

###################################################################################################################
##パラメータの調整
N = 1000
V, t, mu, gu, n1, n2, nscf =2.0, 1 , 0, 1, 1, 100, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
d_delta = 0.0001
Bs     = np.linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs   = np.linspace(0.001,0.2,n2)


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

def F0_0(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f = e_k_spin(-1*kx, -1*ky, -1*kz, qs,-1, Bs[0])
    return sum(f)

def F0_1(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f = E_k_q_s(kx, ky, kz, ans_q, qs, -1, Bs[0])
    return sum(f)

def Fc(ans_q):
    return(N)*(ans_q**2)/V

def free_energy(qs,ans_q):
    return (F1(qs,ans_q) + F0(qs,ans_q) + Fc(ans_q) -Fn()) / N#  


###################################################################################################################
##ギャップの逐次計算
delta_c= []
for i in range(n1):
    ans1 = []
    for j in range(n2): # それぞれの温度で秩序パラメータを計算
        beta, d0= 1/kBTs[j], 100.0
        for k in range(nscf): # 収束するまで最大1000回ループ
            d1 = rhs(d0, 0, Bs[i]) 
            if abs(d1-d0) < 1e-10: break # 収束チェック
            d0 = d1
        ans1.append([d0, abs(d1-d0), k])
    delta_c.append(ans1)
delta_c = np.array(delta_c)

xi=[]
for i in range(n2):
    ###################################################################################################################
    ##F''(\Delta,0)
    beta= 1/kBTs[i]
    dc = delta_c[0][i][0]
    d1 = dc + d_delta
    d2 = dc + 2 * d_delta
    qs = np.linspace(0.0,0.001,3) 

    dfj1  = (free_energy(0,d1) - free_energy(0,dc))/(d1-dc)
    dfj2  = (free_energy(0,d2) - free_energy(0,d1))/(d2-d1)
    d2f =   (dfj2-dfj1)/(d2-d1)
    

    ###################################################################################################################
    ##F''(\Delta_c, q)-F(\Delta_c,0)

    ddf_q1 = ((free_energy(qs[1],dc))-(free_energy(qs[0],dc)))/((qs[1]-qs[0])*pi)
    ddf_q2 = ((free_energy(qs[2],dc))-(free_energy(qs[1],dc)))/((qs[2]-qs[1])*pi)
    dddf_q = (ddf_q2 - ddf_q1)/((qs[1]-qs[0])*pi)

    ###################################################################################################################
    ##\xi
    xi.append(sqrt(dddf_q/(dc**2 * (1/2) *d2f)))
xi = np.array(xi)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
plt.scatter(kBTs, xi, 5)
plt.savefig("figure/q-DeltaF.png")
plt.clf()

###################################
##output
file = open("output/q-\DeltaF", "w")
file.write("##q-\DeltaF" + "\n")
for i in range(n2):
    file.write(str(kBTs[i]) + " " + str(xi[i]) + " "  + "\n")
file.close()



