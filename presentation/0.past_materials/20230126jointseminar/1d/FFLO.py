import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整(1d)
N, V, t, mu, gu, n0, n1, n2, nscf =100, 2, 1, 0, 1, 20, 1, 1, 2000  #kBT-gap_in_each_q
#N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 1, 1, 10, 2000  #q-gap_in_each_kBT
qs   = np.linspace(0,0.11,n0)        
Bs   = np.linspace(0.0,0.1,n1)     
kBTs = np.linspace(0.15,0.1,n2)

# ###################################################################################################################
# ##パラメータの調整(2d)
# N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 3, 1, 3, 2000  #kBT-gap_in_each_q
# #N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 100, 1, 10, 2000  #q-gap_in_each_kBT
# qs   = np.linspace(0.0,0.08,n0)        
# Bs   = np.linspace(0.0,0.1,n1)     
# kBTs = np.linspace(0.0001,0.08,n2)

# ###################################################################################################################
# ##パラメータの調整(3d)
# N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 3, 1, 3, 2000  #kBT-gap_in_each_q
# #N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 100, 1, 10, 2000  #q-gap_in_each_kBT
# qs   = np.linspace(0.0,0.08,n0)        
# Bs   = np.linspace(0.0,0.1,n1)     
# kBTs = np.linspace(0.0001,0.08,n2)

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
    kx, ky, kz = meshgrid(k1, k1, k1, indexing='ij')
    f = func(kx, ky, kz, gap, q, B)
    return (V / (N**3)) * sum(f)


###################################################################################################################
##ギャップの逐次計算
time_scf_start = time()

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

time_scf_finish = time()
time_scf = time_scf_finish - time_scf_start 
print("time_scf : " + str(time_scf//3600) + "時間" + str((time_scf%3600)//60) +"分" + str(time_scf%60) + "秒")

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("./output/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + "_nscf_" + str(nscf)  ,"w")
for j in range(n2):   
    for i in range(n1):
        for h in range(n0):
                file.write(str(kBTs[j]) + " " + str(qs[h]) + " " + str(ans[h][i][j][0]) + " " + str(ans[h][i][j][2]) + " " +  "\n")

###################################################################################################################
# ##figure
# #kBT-gap_in_each_q
# for h in range(n0):  
#     for i in range(n1):
#         figure = plt.scatter(kBTs, ans[h][i][:,0], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
# c= plt.colorbar()
# plt.tick_params(labelsize=18)
# plt.legend()
# plt.savefig("figure/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
# plt.clf() 

# #kBT-iter_in_each_q
# for h in range(n0):   
#     for i in range(n1):
#         figure = plt.scatter(kBTs, ans[h][i][:,2], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
# c= plt.colorbar()
# plt.savefig("figure/kBT-iter_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
# plt.clf() 

#q-gap_in_each_kBT
for j in range(n2):    
    figure = plt.scatter(qs, ans[:,0,j,0], 5, c=ones(n0)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
#c= plt.colorbar()
plt.tick_params(labelsize=18)
plt.savefig("figure/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

# #q-iter_in_each_kBT
# for j in range(n2):  
#     for i in range(n1):
#             figure = plt.scatter(qs, ans[:,0,j,2], 5, c=ones(n0)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
# c= plt.colorbar()
# plt.savefig("figure/q-iter_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
# plt.clf() 

###################################################################################################################