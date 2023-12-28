import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整(1D, mu_0.0)
#N, V, t, mu, gu, n0, n1, n2, nscf =600, 1, 1 , 0.0, 1, 1, 1, 100, 10000  #for test
N, V, t, mu, gu, n0, n1, n2, nscf =50, 1.5, 1 , 0.0, 1, 1, 1, 10, 10000  #q-gap_in_each_kBT
qs   = np.linspace(0.0,0.01,n0)    
Bs   = np.linspace(0.0,0.1,n1)     
kBTs = np.linspace(0.001,0.07,n2)

###################################################################################################################
## gap_eq をdef

def e_k_spin(k1, k2, k3, q, y, B): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))) - mu + y * 1/2 * gu * B
#    return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))) - mu + y * 1/2 * gu * B
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

def rhs(gap, q, B):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f = func(kx, ky, kz, gap, q, B)
    return (V / N) * sum(f)

###################################################################################################################
##ギャップの逐次計算
ans = []
for h in range(n0):
    ans0 = []
    for i in range(n1):
        ans1 = []
        for j in range(n2): # それぞれの温度で秩序パラメータを計算
            beta, d0 = 1/kBTs[j], 100.0
            for k in range(nscf): # 収束するまで最大1000回ループ
                d1 = rhs(d0, qs[h], Bs[i]) 
                if abs(d1-d0) < 1e-10: break # 収束チェック
                d0 = d1
            ans1.append([d0, abs(d1-d0), k])
        ans0.append(ans1)
    ans.append(ans0)
ans = np.array(ans)

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("./output/kBT-q-gap-iter_1D_mu_0.0" ,"w")
file.write("# kBT-q-gap-iter_scf")
for h in range(n0):
    for i in range(n1):
        for j in range(n2):  
                file.write(str(kBTs[j]) + " " + str(qs[h]) + " " + str(ans[h][i][j][0]) + " " + str(ans[h][i][j][2]) + " " +  "\n")

###################################################################################################################
##figure
#q-gap_in_each_kBT
for j in range(n0):    
    figure = plt.scatter(kBTs, ans[j,0,:,0], 5, c=ones(n2)*kBTs[j],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-gap_1D_mu_0.0" + ".png")
plt.clf() 

#q-iter_in_each_kBT
for j in range(n0):  
    figure = plt.scatter(kBTs, ans[j,0,:,2], 5, c=ones(n2)*kBTs[j],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-iter_1D_mu_0.0"  + ".png")
plt.clf() 

###################################################################################################################