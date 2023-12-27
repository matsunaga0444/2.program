import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
#N, V, t, mu, gu, n0, n1, n2, nscf =10, 1, 1 , 0, 1, 10, 1, 100, 10000  #kBT-gap_in_each_q
N, V, t, mu, gu  = 10, 1, 1 , 0, 1
n0, n1, n2, nscf = 1, 1, 1, 10000  #q-gap_in_each_kBT
qs   = np.linspace(0.08,0.08,n0)        
Bs   = np.linspace(0.0,0.1,n1)     
kBTs = np.linspace(0.001,0.08,n2)

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, q): 
    return 2*t*(np.cos((k1+q/2)*np.pi)+np.cos((k2)*np.pi)) - mu
    #return 2*t*(np.cos((k1+(q/2)))+np.cos((k2))) - mu + y * 1/2 * gu * B

def e_k_s(k1, k2, q):
    return (e_k_spin(k1, k2, q) + e_k_spin(-1*k1, -1*k2, q))/2

def e_k_a(k1, k2, q):
    return (e_k_spin(k1, k2, q) - e_k_spin(-1*k1, -1*k2, q))/2

def E_k_q(k1, k2, gap, q):
    return np.sqrt(e_k_s(k1, k2, q)**2 + gap**2)

def E_k_q_s(k1, k2, gap, q, y):
    return E_k_q(k1, k2, gap, q) + y * e_k_a(k1, k2, q)

def Fermi(beta, E):
    #return  1 / (np.exp(beta*E) + 1 )
    return (1 - np.tanh(beta*E/2)) /2

def func(k1, k2, gap, q): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1))-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1)))/(2*E_k_q(k1, k2, gap, q))

def rhs(gap, q):
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q)
    return (V / (N**2)) * sum(f)

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
                d1 = rhs(d0, qs[h]) 
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
##figure
#kBT-gap_in_each_q
for h in range(n0):  
    for i in range(n1):
        plt.scatter(kBTs, ans[h][i][:,0], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("test")
plt.show
#plt.savefig("figure/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

#kBT-iter_in_each_q
for h in range(n0):   
    for i in range(n1):
        figure = plt.scatter(kBTs, ans[h][i][:,2], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
#plt.savefig("figure/kBT-iter_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

#q-gap_in_each_kBT
for j in range(n2):    
    figure = plt.scatter(qs, ans[:,0,j,0], 5, c=ones(n0)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
#plt.savefig("figure/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

#q-iter_in_each_kBT
for j in range(n2):  
    for i in range(n1):
            figure = plt.scatter(qs, ans[:,0,j,2], 5, c=ones(n0)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
#plt.savefig("figure/q-iter_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

###################################################################################################################