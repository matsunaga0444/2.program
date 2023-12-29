import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N = 5000
V, t, mu, gu, n0, n1, n2, n3, nscf =1, 1 , 0, 1, 2, 1, 1, 100, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
qs     = np.linspace(0.00,0.03,n0)         #(np.pi/a)
Bs     = np.linspace(0.0,0.0,n1)            #np.linspace(0,0.08,n1)
kBTs   = np.linspace(0.001,1,n2)            
deltas = np.linspace(0.0, 0.15, n3)         
beta = 1/kBTs[0]                            

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
    return (F1(qs,ans_q) + F0(qs,ans_q) + Fc(ans_q) -Fn()) / N#  

###################################################################################################################
##calculate_freeenergy
f, extended_GL = [], []
for j in range(n3):
    f_temp = (F0(qs[1],deltas[j]) - F0(qs[0],deltas[j]))/N
    f.append(f_temp)
f = np.array(f)
extended_GL = np.array(extended_GL)

########################################################################################################################
#plot the figure of comparing free energy to extended GL
for i in range(n0):
    plt.scatter(deltas, f[:], 5)
    plt.savefig("figure/delta.dependent.f0q-f00.png")
    plt.clf()

###################################
##output
file = open("output/delta.dependent.f0q-f00", "w")
file.write("##delta---f0q-f00" + "\n")
for j in range (n3):
    file.write(str(deltas[j]) + " " + str(f[j]) + " "  + "\n")
file.close()