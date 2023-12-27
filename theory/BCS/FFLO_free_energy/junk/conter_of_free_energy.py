import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time

##パラメータの調整
N, V, t, a, u, gu, n0, n1, n2, n3 =100, 1, 1, 1, 0, 1, 1, 1, 1, 100   # 7.525 #9.21
qs   = np.linspace(0.05,0.1,n0)     #(np.pi/a)
Bs   = np.linspace(0.06,0.1,n1)     #np.linspace(0,0.08,n1)
kBTs = np.linspace(0.001,0.05,n2)
delta_q = np.linspace(-0.1,0.1,n3)    

## gap_eq をdef
def e_k_spin(k1, k2, q, y, B): 
    return 2*t*(np.cos(a*(k1+q/2))+np.cos(a*(k2))) - u + y * 1/2 * gu * B
def e_k_s(k1, k2, q, B):
    return (e_k_spin(k1, k2, q, 1, B) + e_k_spin(-1*k1, k2, q, -1, B))/2
def e_k_a(k1, k2, q, B):
    return (e_k_spin(k1, k2, q, 1, B) - e_k_spin(-1*k1, k2, q, -1, B))/2
def E_k_q(k1, k2, gap, q, B):
    return np.sqrt(e_k_s(k1, k2, q, B)**2 + gap**2)
def E_k_q_s(k1, k2, gap, q, y, B):
    return E_k_q(k1, k2, gap, q, B) + y * e_k_a(k1, k2, q, B)
def Fermi(beta, E):
    return 1 / (np.exp(beta*E) + 1 )
def func(k1, k2, gap, q, B): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1, B))-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1, B)))/(2*E_k_q(k1, k2, gap, q, B))
def rhs(gap, q, B):
    k1 = -1 * np.pi/a + 2 * arange(N) * np.pi / (a * N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q, B)
    return (V / (N**2)) * sum(f)

#free energy の定義
def F1(i, h):
    sum = 0
    for n1 in range(N):
        k1 = -1 * np.pi/a + 2 * n1 * np.pi / (a * N)
        for n2 in range(N):
            k2 = -1 * np.pi/a + 2 * n2 * np.pi / (a * N)
            for y in range(-1,1):
                sum = sum + np.log(1+np.exp(-1*beta*E_k_q_s(k1, k2, delta_q[h], qs[i], y, B)))
    return -1*1/beta*sum
def F0(i, h):
    sum = 0
    for n1 in range(N):
        k1 = -1 * np.pi/a + 2 * n1 * np.pi / (a * N)
        for n2 in range(N):
            k2 = -1 * np.pi/a + 2 * n2 * np.pi / (a * N)
            sum = sum + e_k_spin(-1*k1, -1*k2, qs[i], -1, B) - E_k_q_s(k1, k2, delta_q[h], qs[i], -1, B)
    return sum
def Fc(h):
    return(N**2)*(delta_q[h]**2)/V
def free_energy(h):              #vn0 = (v / n^2) * n0
    return F1(h) + F0(h) + Fc(h) #  

#free energy の計算
kBT = kBTs[0]
beta = 1/kBT
B = Bs[0]
ans_F0 = []
for i in range(n0):
    for h in range(n3):
        ans = F0(i,h)
        ans_F0.append(ans)
ans0 = np.array(ans_F0)

ans_F1 = []
for i in range(n0):
    for h in range(n3):
        ans = F1(i,h)
        ans_F1.append(ans)
ans1 = np.array(ans_F1)

ans_FC = []
for i in range(n0):
    for h in range(n3):
        ans = Fc(h)
        ans_FC.append(ans)
ansC = np.array(ans_FC)

ans_F = []
for i in range(n0):
    for h in range(n3):
        ans = free_energy(i,h)
        ans_F.append(ans)
ans = np.array(ans_F)


#描画
plt.scatter(delta_q, ans1)
plt.savefig("delta_q_ans1.png")
plt.clf()

plt.scatter(delta_q, ans0)
plt.savefig("delta_q_ans0.png")
plt.clf()

plt.scatter(delta_q, ansC)
plt.savefig("delta_q_ansC.png")
plt.clf()

plt.scatter(delta_q, ans)
plt.savefig("delta_q_ans.png")
plt.clf()

