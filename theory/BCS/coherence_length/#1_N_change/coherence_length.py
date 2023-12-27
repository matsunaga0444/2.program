import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time

###################################################################################################################
##パラメータの調整
N, V, t, mu =1000, 1, 1, 0
n_q, n_kBT, n_scf =2, 100, 100 # 7.525 #9.21
qs   = linspace(0.0,0.001,n_q)            #(np.pi/a)
kBTs = linspace(0.0001,0.1,n_kBT)      

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, q): 
    return 2*t*(cos((k1+q/2)*pi)+cos((k2)*pi)) - mu
    #return 2*t*(np.cos((k1+(q/2)))+np.cos((k2))) - mu + y * 1/2 * gu * B

def e_k_s(k1, k2, q):
    return (e_k_spin(k1, k2, q) + e_k_spin(-1*k1, -1*k2, q))/2

def e_k_a(k1, k2, q):
    return (e_k_spin(k1, k2, q) - e_k_spin(-1*k1, -1*k2, q))/2

def E_k_q(k1, k2, gap, q):
    return sqrt(e_k_s(k1, k2, q)**2 + gap**2)

def E_k_q_s(k1, k2, gap, q, y):
    return E_k_q(k1, k2, gap, q) + y * e_k_a(k1, k2, q)

def Fermi(beta, E):
    #return  1 / (exp(beta*E) + 1 )
    return (1 - np.tanh(beta*E/2)) /2

def func(k1, k2, gap, q): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1))-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1)))/(2*E_k_q(k1, k2, gap, q))

def rhs(gap, q):
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q)
    return (V / (N**2)) * sum(f)

def coherence_length(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))

###################################################################################################################
##ギャップの逐次計算
time_scf_start = time()
ans = []
for h in range(n_q):
    ans1 = []
    for j in range(n_kBT): # それぞれの温度で秩序パラメータを計算
        beta, d0 = 1/kBTs[j], 100.0
        for k in range(n_scf): # 収束するまで最大1000回ループ
            d1 = rhs(d0, qs[h]) 
            if abs(d1-d0) < 1e-10: break # 収束チェック
            d0 = d1
        ans1.append([d0, abs(d1-d0), k])
    ans.append(ans1)
ans = np.array(ans)

time_scf_finish = time()
time_scf = time_scf_finish - time_scf_start 
print("time_scf : " + str(time_scf//3600) + "時間" + str((time_scf%3600)//60) +"分" + str(time_scf%60) + "秒")

###################################################################################################################
##coherence_lengthの計算
ans_c = []
for h in range(n_q):
    ans1 = []
    for j in range(n_kBT): # それぞれの温度で秩序パラメータを計算
        ans1.append(coherence_length(ans[h][j][0], ans[0][j][0], qs[h]) )
    ans_c.append(ans1)
ans_c = np.array(ans_c)

###################################################################################################################
##output-kBT-q-coherence_length
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) ,"w")
for j in range(n_q):    
    for h in range(n_kBT):
        file.write(str(kBTs[h]) + " " + str(qs[j]) + " " + str(ans_c[j][h]) + " " +  "\n")
file.close()

###################################################################################################################
##kBT-coherence_length_in_each_q
for j in range(n_q):    
    ans_q = []
    for h in range(n_kBT):
        figure = plt.scatter(kBTs[h], ans_c[j][h], 5, c=qs[j],  cmap='viridis' ,vmin=qs[1], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-coherence_in_each_momentum(q)" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.show()

###################################################################################################################