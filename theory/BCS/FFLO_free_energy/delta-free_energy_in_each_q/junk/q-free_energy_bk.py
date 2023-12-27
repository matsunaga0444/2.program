import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N, V, t, mu =100, 1, 1, 0
n_q, n_kBT, n_delta, n_scf =1, 1, 10, 2000  # 7.525 #9.21
qs   = np.linspace(0.0,1,n_q)            #(np.pi/a)
kBTs = np.linspace(0.0001,0.04,n_kBT)      
ans_q=  np.linspace(0.0001,0.01,n_delta)   

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
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q))-Fermi(beta, E_k_q_s(k1, k2, gap, q)))/(2*E_k_q(k1, k2, gap, q))

def rhs(gap, q):
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q)
    return (V / (N**2)) * sum(f)

###################################################################################################################
#free energy の定義
def F1(h, i):
    sum = 0
    for n1 in range(N):
        k1 = -1 * np.pi + 2 * n1 * np.pi / (N)
        for n2 in range(N):
            k2 = -1 * np.pi + 2 * n2 * np.pi / (N)
            for y in range(-1,1):
                #sum = sum + np.log(1+np.exp(-1*beta*E_k_q_s(k1, k2, ans_q[h], qs[h])))
                sum = sum + np.log(-2 / np.tanh(-1*beta*E_k_q_s(k1, k2, ans_q[i], qs[h], y)/2)-1)
    return -1*1/beta*sum

def F0(h, i):
    sum = 0
    for n1 in range(N):
        k1 = -1 * np.pi + 2 * n1 * np.pi / (N)
        for n2 in range(N):
            k2 = -1 * np.pi + 2 * n2 * np.pi / (N)
            sum = sum + e_k_spin(-1*k1, -1*k2, qs[h]) - E_k_q_s(k1, k2, ans_q[i], qs[h], -1)
    return sum

def Fc(i):
    return(N**2)*(ans_q[i]**2)/V

def free_energy(h, i):              #vn0 = (v / n^2) * n0
    return F1(h, i) + F0(h, i) + Fc(i) #  

###################################################################################################################
#free energy の計算
kBT = kBTs[0]
beta = 1/kBT
ans_F0 = []
for i in range(n_delta):
    for h in range(n_q):
        ans = F0(h, i)
        ans_F0.append(ans)
ans0 = np.array(ans_F0)

ans_F1 = []
for i in range(n_delta):
    for h in range(n_q):
        ans = F1(h, i)
        ans_F1.append(ans)
ans1 = np.array(ans_F1)

ans_FC = []
for i in range(n_delta):
    ans = Fc(i)
    ans_FC.append(ans)
ansC = np.array(ans_FC)

ans_F = []
for i in range(n_delta):
    for h in range(n_q):
        ans = free_energy(h, i)
        ans_F.append(ans)
ans = np.array(ans_F)
print(ans_q, ans, ans1, ans0, ansC)

###################################################################################################################
#描画
plt.scatter(ans_q, ans1)
plt.clf()

plt.scatter(ans_q, ans0)
plt.clf()

plt.scatter(ans_q, ansC)
plt.clf()

#kBT-iter_in_each_q
for i in range():
    plt.scatter(ans_q, ans, 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1] )
c= plt.colorbar()
plt.savefig("test.png")
#plt.savefig("figure/q-free_energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

for h in range(n0):  
    for i in range(n1):
        figure = plt.scatter(kBTs, ans[h][i][:,0], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.legend()
plt.savefig("figure/Nchange/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.show()