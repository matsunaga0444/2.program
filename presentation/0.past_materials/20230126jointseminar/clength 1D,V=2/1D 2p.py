import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N, V, t, mu, gu, n0, n1, n2, nscf =500, 2, 1 , 0, 1, 2, 1, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 2
wide_q = 0.001
qs   = np.linspace(0,wide_q,n0)         #(np.pi/a)
Bs   = np.linspace(0.0,0.0,n1)          #np.linspace(0,0.08,n1)
kBTs = np.linspace(0.01,1,n2)

###################################################################################################################
## gap_eq をdef

def e_k_spin(k1, k2, k3, q, y, B): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))) - mu + y * 1/2 * gu * B
#   return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))) - mu + y * 1/2 * gu * B
#   return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))+np.cos((k3))) - mu + y * 1/2 * gu * B

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

print(ans)
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
file.close()

###################################################################################################################
##figure
# #kBT-gap_in_each_q
# for h in range(n0):  
#     for i in range(n1):
#         figure = plt.scatter(kBTs, ans[h][i][:,0], 5, c=ones(n2)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
# c= plt.colorbar()
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
c= plt.colorbar()
plt.savefig("figure/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 

#q-iter_in_each_kBT
for j in range(n2):  
    for i in range(n1):
            figure = plt.scatter(qs, ans[:,0,j,2], 5, c=ones(n0)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-iter_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 


###################################################################################################################
##find Tc


ans1 = []
for j in range(n_search):
    kBT = (kBT_a+kBT_b)/2
    beta, d0 = 1/kBT, 100.0
    for k in range(1000): # 収束するまで最大1000回ループ
        d1 = rhs(d0, qs[0], 0) 
        if abs(d1-d0) < error: break # 収束チェック
        d0 = d1
    ans1.append([kBT, d0, abs(d1-d0), k])
    if d0 < check_gap: 
        kBT_b =  (kBT_a + kBT_b) /2
    if d0 > check_gap: 
        kBT_a =  (kBT_a + kBT_b) /2   
    if  abs(kBT_a-kBT_b) <  error: break
Tc = (kBT_a + kBT_b)/2

###################################################################################################################
#free_energy_parameter
ans_q= ans[:,0,0,0]
###################################################################################################################
#free energy の定義

def Fn():
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    y  = -1 + 2 * arange(2)
    kx, y = meshgrid(k1, y, indexing='ij')
    ky,kz = 1, 1
    f_n = np.log(1+np.exp(-1*beta*e_k_spin(kx, ky, kz, 0, y, 0)))
    #f_n = np.log(-2 / np.tanh(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)/2)-1)
    print("fn",-1*1/beta*np.sum(f_n))
    return -1*1/beta*np.sum(f_n)
    
def F1(h):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    y  = -1 + 2 * arange(2)
    kx, y = meshgrid(k1, y, indexing='ij')
    ky, kz = 1, 1
    f_1 = log(1+exp(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)))
    #f_1 = np.log(-2 / np.tanh(-1*beta*E_k_q_s(kx, ky, kz, ans_q[h], qs[h], y, B)/2)-1)
    print("f1",-1*1/beta*np.sum(f_1))
    return -1*1/beta*np.sum(f_1)

def F0(h):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx = k1
    ky, kz = 1, 1
    f_0 = e_k_spin(-1*kx, -1*ky, -1*kz, qs[h], -1, B) - E_k_q_s(kx, ky, kz, ans_q[h], qs[h], -1, B)
    print("f0",sum(f_0))
    return sum(f_0)

def Fc(h):
    return(N)*(ans_q[h]**2)/V

def free_energy(h):              #vn0 = (v / n^2) * n0
    print("f",F1(h) + F0(h) + Fc(h) - Fn())
    return F1(h) + F0(h) + Fc(h) - Fn()#  

###################################################################################################################
#free energy の計算
kBT = kBTs[0]
beta = 1/kBT
B = Bs[0]

ans_F = []
for h in range(n0):
    ans_F.append(free_energy(h))
ans_F = np.array(ans_F)

###################################################################################################################
#描画
#kBT-iter_in_each_q
plt.scatter(qs, ans_F)
plt.savefig("test.png")
#plt.savefig("figure/q-free_energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.clf() 



###################################################################################################################
## def for finding the coefficients of extended GL free energy
def calculate_a_1(x_1, y_1):
    return 2 * y_1/x_1
def calculate_a(x_1, y_1, T_1, T, Tc):
    p = (1-T/Tc)/(1-T_1/Tc)
    return 2 * y_1/x_1 * p
def calculate_b(x_1, y_1):
    return -1 * y_1/(x_1)**2
def calculate_d(x_1, y_1, x_2, y_2, q):
    a = 2 * y_1/x_1
    d = 2 * y_2/x_2
    return (d-a)/q**2
def calculate_e(x_1, y_1, x_2, y_2, q):
    b = -1 * y_1/(x_1)**2
    e = -1 * y_2/(x_2)**2
    return (e-b)/q**2

###################################################################################################################
## find the coefficients of extended GL free energy
n2 = 100
#kBTs  = np.linspace(0, Tc, n2)
kBTs  = np.linspace(0, 0.175, n2)
print(ans[0][0][0][0])
a_0 = calculate_a_1(ans[0,0,0,0], ans_F[0])
a = calculate_a(ans[0,0,0,0], ans_F[0], kBTs[0], kBTs, Tc)
b = calculate_b(ans[0,0,0,0], ans_F[0])
d = calculate_d(ans[0,0,0,0], ans_F[0], ans[1:,0,0,0], ans_F[1:], qs[1:])
e = calculate_e(ans[0,0,0,0], ans_F[0], ans[1:,0,0,0], ans_F[1:], qs[1:])
print(a,d)

###################################################################################################################
## def for calculating coherence length
def c_l_puterbation_GL(a, d):
    return sqrt(-1*(d/a))
def c_l_puterbation_extended_GL(a, b, d, e):
    return sqrt(-1*(d[:,None]/a[None,:]) + (e[:,None]/(2*b)) )
def coherence_length_GL_formulation(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))
def c_l_puterbation_extended_GL_square(a, b, d, e):
    return -1*(d[:,None]/a[None,:]) + (e[:,None]/(2*b)) 
def coherence_length_GL_formulation_square(gap_q, gap_0, q):
    return (gap_0**2 - gap_q**2) / (gap_0**2*q**2)

###################################################################################################################
## calculate T-gap with q = 0 or qs[1]
ans_a = []
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
    ans_a.append(ans0)
ans_a = np.array(ans_a)

###################################################################################################################
#plot comopared_T-each_c_l
for i_q in range(n0-1):
    plt.scatter(kBTs, coherence_length_GL_formulation_square(ans_a[(i_q+1),0,:,0], ans_a[0,0,:,0], qs[(i_q+1)]),5 ,marker= "o" ,label =r'$\xi$ in conventional method')
    plt.scatter(kBTs, c_l_puterbation_extended_GL_square(a, b, d[i_q]*ones(1), e[i_q]*ones(1)), 5,marker= "o" ,label = r'$\xi$ in modified method')
    #plt.scatter(kBTs, c_l_puterbation_GL(a, d[i_q]), 5,marker= "o" ,label = 'conventional coherence length')
    plt.axhline(y=0, color = 'green')
plt.legend(fontsize=16)
#plt.legend()
plt.savefig("figure/compared_T-each_c_l_N_10_mu_0.png")
plt.show()
plt.clf()

for i_q in range(n0-1):
    c_l_1 = coherence_length_GL_formulation(ans_a[(i_q+1),0,:,0], ans_a[0,0,:,0], qs[(i_q+1)])
    c_l_2 = c_l_puterbation_extended_GL(a, b, d[i_q]*ones(1), e[i_q]*ones(1))
    c_l_3 = c_l_puterbation_GL(a, d[i_q])

###################################################################################################################
##output  
# kBT-c_l_Niklas-c_l_matsuanga-c_l_GL
file = open("./output/kBT-c_l_Niklas-c_l_matsuanga-c_l_GL_N_10_mu_0"  ,"w")
for i_q in range(n0-1):
    for j in range(n2):   
        file.write(str(kBTs[j]) + " " + str(c_l_1[j])\
                        + " " + str(c_l_2[i_q, j]) \
                            + " " + str(c_l_3[j]) + " " +  "\n")
file.close()


