import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time

##パラメータの調整
N, V, t, mu, gu, n0, n1, n2 =10, 1, 1, 0, 1, 2, 1, 100  # 7.525 #9.21
qs   = np.linspace(0,0.02,n0)      #(np.pi/a)
Bs   = np.linspace(0.0,0.1,n1)     #np.linspace(0,0.08,n1)
kBTs = np.linspace(0.001,0.02,n2)  

## gap_eq をdef
def e_k_spin(k1, k2, q, y, B): 
    return 2*t*(np.cos((k1+q/2))+np.cos((k2))) - mu + y * 1/2 * gu * B

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
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q, B)
    return (V / (N**2)) * sum(f)

def coherence_length(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))

def penetration_depth(gap_0):
    return sqrt(m/4*mu_0*(e**2)*(gap_0**2))

def critical_field_1(c_length, p_depth):
    return h/(2*e*4*pi*(p_depth**2))*log(p_depth/c_length)

def critical_field_2(c_length):
    return h/(2*e*2*pi*(c_length)**2)



time_scf_start = time()

##ギャップの逐次計算
ans = []
for h in range(n0):
    ans0 = []
    for i in range(n1):
        ans1 = []
        for j in range(n2): # それぞれの温度で秩序パラメータを計算
            beta, d0 = 1/kBTs[j], 100.0
            for k in range(1000): # 収束するまで最大1000回ループ
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
time_cal_coherence_start = time()

##coherence_lengthの計算
ans_c = []
for h in range(n0):
    ans0 = []
    for i in range(n1):
        ans1 = []
        for j in range(n2): # それぞれの温度で秩序パラメータを計算
            ans1.append([coherence_length(ans[h][i][j][0], ans[0][i][j][0], qs[h]),qs[h],kBTs[j]] )
        ans0.append(ans1)
    ans_c.append(ans0)
ans_c = np.array(ans_c)

##penetration_depthの計算
m, mu_0, e  = 1, 1, 1
ans_p    = penetration_depth(ans[0,0,:,0])
ans_Hc_1 = critical_field_1(ans_c[1,0,:,0], ans_p)
ans_Hc_2 = critical_field_2(ans_c[1,0,:,0])

print(ans[0,0,:,0])
print(ans_p)
print(ans_Hc_1)
print(ans_Hc_2)


########################################################################################################################################################################
##output-kBT-q-ans_p
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) ,"w")
for j in range(n2):    
    file.write(str(kBTs[j]) + " " + str(qs[1]) + " " + str(ans_p[j]) + " " +  "\n")
file.close()

##output-kBT-q-ans_Hc_1
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) ,"w")
for j in range(n2):    
    file.write(str(kBTs[j]) + " " + str(qs[1]) + " " + str(ans_Hc_1[j]) + " " +  "\n")
file.close()

##output-kBT-q-ans_Hc_2
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) ,"w")
for j in range(n2):    
    file.write(str(kBTs[j]) + " " + str(qs[1]) + " " + str(ans_Hc_2[j]) + " " +  "\n")
file.close()


########################################################################################################################################################################
##kBT-penetration_depth_in_each_q
figure = plt.scatter(kBTs, ans_p, 5, c=qs[1]*ones(n2),  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-ans_p" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.show()

##kBT-penetration_depth_in_each_q
figure = plt.scatter(kBTs, ans_Hc_1, 5, c=qs[1]*ones(n2),  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-ans_Hc_1" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.show()

##kBT-penetration_depth_in_each_q
figure = plt.scatter(kBTs, ans_Hc_2, 5, c=qs[1]*ones(n2),  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-ans_Hc_2" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.show()

