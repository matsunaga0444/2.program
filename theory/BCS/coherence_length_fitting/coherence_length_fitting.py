import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time

###################################################################################################################
##パラメータの調整
N, V, t, mu =100, 1, 1, 0
n_q, n_kBT, n_scf =10, 100, 100 # 7.525 #9.21
qs   = linspace(0.0,0.005,n_q)            #(np.pi/a)
kBTs = linspace(0.0001,0.05,n_kBT)  
sarch_kBTs = linspace(0.0,0.1,2)   
n_search = 100
error    = 1e-10
check_gap = 1e-6

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
##search_Tc
#search_TC
time_binary_search_start = time()

ans_kBTs = []
for i_q in range(n_q):
    kBT_a = sarch_kBTs[0]
    kBT_b = sarch_kBTs[1]
    ans1 = []
    for j in range(n_search):
        kBT = (kBT_a+kBT_b)/2
        beta, d0 = 1/kBT, 100.0
        for k in range(1000): # 収束するまで最大1000回ループ
            d1 = rhs(d0, qs[h]) 
            if abs(d1-d0) < error: break # 収束チェック
            d0 = d1
        ans1.append([kBT, d0, abs(d1-d0), k])
        if d0 < check_gap: 
            kBT_b =  (kBT_a + kBT_b) /2
        if d0 > check_gap: 
            kBT_a =  (kBT_a + kBT_b) /2   
    ans_kBTs.append(kBT_a)
ans_kBTs = np.array(ans_kBTs)

time_binary_search_finish = time()
###################################################################################################################
##
#output-kBT-q-coherence_length
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) ,"w")
for j in range(n_q):    
    for h in range(n_kBT):
        file.write(str(kBTs[h]) + " " + str(qs[j]) + " " + str(ans_c[j][h]) + " " +  "\n")
file.close()

#output-kBT-q-coherence_length
file = open("./output/kBT_gap_error_iter" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_check_gap_" + str(check_gap) + "_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_n_serach_" + str(n_search)  ,"w") 
for i_q in range(n_q):
        file.write(str("qは") + " " + str(qs[i_q]) + " " + str("Tcは") + " " + str(ans_kBTs[i_q]) + " " +  "\n")
file.close()

###################################################################################################################
##kBT-coherence_length_in_each_q
for j in range(n_q):    
    ans_q = []
    for h in range(n_kBT):
        figure = plt.scatter(kBTs[h], ans_c[j][h], 5, c=qs[j],  cmap='viridis' ,vmin=qs[1], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-coherence_in_each_momentum(q)" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

###################################################################################################################
##kBT-fitting_coherence_length_in_each_q
for i_kBT in range(n_kBT):    
    ans_q = []
    for i_q in range(n_q):
        figure = plt.scatter(kBTs[i_kBT], ans_c[i_q][i_kBT] * sqrt(1-kBTs[i_kBT]/ans_kBTs[i_q]), 5, c=qs[i_q],  cmap='viridis' ,vmin=qs[1], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-fitting_coherence_in_each_momentum(q)" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" +str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.show()

print(time())

print(ans_c[0][0][0])
