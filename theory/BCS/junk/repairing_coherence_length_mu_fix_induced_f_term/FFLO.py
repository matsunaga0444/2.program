import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

# ###################################################################################################################
# ##パラメータの調整
#N, V, t, mu, gu, n0, n1, n2, nscf =1000, 1, 1 , 1.1, 1, 10, 1, 10, 10000  #test
#N, V, t, mu, gu, n0, n1, n2, nscf =1000, 1, 1 , 1.1, 1, 10, 1, 100, 10000  #kBT-gap_in_each_q
N, V, t, mu = 10, 1, 1 ,0
n_q, n_kBT, n_scf, check_delta = 3, 100, 10000, 1e-7  #q-gap_in_each_kBT
qs   = np.linspace(0.0,0.008,n_q)            
kBTs = np.linspace(1e-10,0.08,n_kBT)

# plt_parameter
f_x_c, n_delta = 2.2, 1000

###################################################################################################################
## gap_eq をdef

def e_k_spin(k1, k2, q): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))) - mu
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
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1))-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1)))/(2*E_k_q(k1, k2, gap, q))

def rhs(gap, q):
    k1 = -1 * np.pi + 2 * arange(N) * np.pi / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q)
    return (V / (N**2)) * sum(f)

def calculate_Tc(T_1, T_2, delta_1, delta_2):
    return (T_2*delta_1**2-T_1*delta_2**2)/(delta_1**2-delta_2**2)

def calculate_b(T_1, T_2, delta_1, delta_2):
    return (T_2-T_1)/(2*(delta_1**2-delta_2**2))

def calculate_d(q_1, q_2, delta_1, delta_2, T, Tc, b):
    a = (T-Tc)/Tc
    B = q_1**2*delta_1**2*(-2*b*delta_2**2-a)-q_2**2*delta_2**2*(-2*b*delta_1**2-a)
    return B/(-1*q_1**2*q_2**2*(delta_2**2-delta_1**2))

def calculate_e(q_1, q_2, delta_1, delta_2, T, Tc, b):
    a = (T-Tc)/Tc
    A = q_1**2*(-2*b*delta_2**2-a)-q_2**2*(-2*b*delta_1**2-a)
    return A/(2*q_1**2*q_2**2*(delta_2**2-delta_1**2))

def free_energy(T, Tc, b, d, e, q, delta):    #vn0 = (v / n^2) * n0
#def free_energy(T, Tc, b, c, d, e, q, delta):    #vn0 = (v / n^2) * n0
    a = (T-Tc)/Tc           
#    return  a * (delta**2) + b * (delta**4) + c * (delta**6)\
#             + d * (q**2)*(delta**2) + e * (q**2)*(delta**4)
    return (delta**2)*((a + d * (q**2)) + (b+ e * (q**2))*(delta**2))

def free_energy_q(T, Tc, b, d, e, q, delta):    #vn0 = (v / n^2) * n0
    a = (T-Tc)/Tc           
    return (delta**2)*((d * (q**2)) + (e * (q**2))*(delta**2))

def solution_free_energy_4(a, b, d, e, q):              #vn0 = (v / n^2) * n0
    return  sqrt(-1 * (a + d * (q**2))/(b + e * (q**2)))

def solution_free_energy(a, b, d, q):              #vn0 = (v / n^2) * n0
    return  sqrt(-1 * (a + d * (q**2))/(b))

def minimum_free_energy(a, b, d, e, q):              #vn0 = (v / n^2) * n0
    return  -1 *  (b+ e * (q**2)) * ((a + d * (q**2))/(b + e * (q**2)))**2 /4

def coherence_length_GL_formulation(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))

# def c_l_puterbation_beyond_GL(a, b, c, d, e):
#     phi_0 = sqrt(-1*b/3*c)
#     alpha = 4*b* phi_0**2 + 6*c* phi_0**4
#     A = -1*(d+e*phi_0*2)/alpha
#     return sqrt(A)

def c_l_puterbation_beyond_GL(a, b, d, e):
    return sqrt(-1*(d/a) + (e/(2*b)) )

# def c_l_puterbation_beyond_GL(a, b, c, d, e):
#     phi_0 = sqrt(-1*b/3*c)
#     alpha = 4*b* phi_0**2 + 6*c* phi_0**4
#     A = -1*(d+e*phi_0*2)/alpha
#     return sqrt(A)

def c_l_puterbation_GL(a, d):
    return sqrt(-1*(d/a))

###################################################################################################################
##ギャップの逐次計算

ans, T_1 = [], 0
for h in range(n_q):
    ans1, delta_1  = [], 0
    for j in range(n_kBT): # それぞれの温度で秩序パラメータを計算
        beta, d0 = 1/kBTs[j], 100.0
        for k in range(n_scf): # 収束するまで最大1000回ループ
            d1 = rhs(d0, qs[h]) 
            if abs(d1-d0) < 1e-10: break # 収束チェック
            d0 = d1
        if h == 0 and d0 > check_delta: 
            delta_2 = delta_1
            delta_1 = d0
            T_2 = T_1
            T_1 = kBTs[j]
            print(T_1, T_2, delta_1, delta_2)
        ans1.append([d0, abs(d1-d0), k])
    ans.append(ans1)
ans = np.array(ans)

#T_1, T_2, delta_1, delta_2 = kBTs[0], kBTs[1], ans[0,0,0], ans[0,1,0]
q_1, q_2, delta_q_1, delta_q_2, T = qs[1], qs[2], ans[1,0,0], ans[2,0,0], kBTs[0]
print(T_1, T_2, delta_1, delta_2)
Tc = calculate_Tc(T_1, T_2, delta_1, delta_2)
print(T, Tc)
a = (T-Tc)/Tc
#b  = calculate_b(T_1, T_2, delta_1, delta_2)
b = -1*a/(2*ans[0,0,0]**2)
#c = -1*b/(2*ans[0,0,0]**2)

print(q_1, q_2, delta_q_1, delta_q_2, T, Tc, b)
d  = calculate_d(q_1, q_2, delta_q_1, delta_q_2, T, Tc, b)
e  = calculate_e(q_1, q_2, delta_q_1, delta_q_2, T, Tc, b)
print(a, b, d, e)

ans_xsi = []
for i_kBT in range(n_kBT):
    a = (kBTs[i_kBT]-Tc)/Tc
    xsi = c_l_puterbation_beyond_GL(a, b, d, e)
    #xsi = c_l_puterbation_beyond_GL(a, b, c, d, e)
    ans_xsi.append([xsi])
ans_xsi = np.array(ans_xsi)

###################################################################################################################
##output 
# kBT-q-gap-iter
file = open("./output/kBT-q-gap-iter" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("# kBT-q-gap-iter ")
for j in range(n_kBT):   
    for h in range(n_q):
        file.write(str(kBTs[j]) + " " + str(qs[h]) + " " + str(ans[h][j][0]) + " " + str(ans[h][j][2]) + " " +  "\n")

# kBT-c_l
file = open("./output/kBT-c_l" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("# kBT-c_l ")
for j in range(n_kBT):   
    file.write(str(kBTs[j]) + " " + str(ans_xsi[j][0]) + " " +  "\n")

# Tc-b-d-e
file = open("./output/Tc-b-d-e" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("# Tc-b-d-e")
file.write(str(Tc) + " " + str(b) + " " + str(d) + " " + str(e) + " " +  "\n")

###################################################################################################################
##figure

#kBT-c_l_with_Tc
figure = plt.scatter(kBTs, ans_xsi[:,0])
plt.savefig("figure/kBT-c_l_with_Tc" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#free-energy
x = ans[0,0,0] * f_x_c * (-1 + 2 * arange(n_delta) / n_delta)
for i_q in range(2):
#                               free_energy(T, Tc, b, d, e, q, delta)
    figure = plt.scatter(x, free_energy(kBTs[0], Tc, b, d, e, qs[i_q], x) , 5, c=ones(n_delta)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[1])
    figure = plt.scatter(x, free_energy_q(kBTs[0], Tc, b, d, e, qs[i_q], x) , 5, c=ones(n_delta)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[1])
c= plt.colorbar()
plt.savefig("figure/free-energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#kBT-gap_in_each_q
for h in range(n_q):  
    figure = plt.scatter(kBTs, ans[h][:,0], 5, c=ones(n_kBT)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#kBT-iter_in_each_q
for h in range(n_q):
    figure = plt.scatter(kBTs, ans[h][:,2], 5, c=ones(n_kBT)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-iter_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

#q-gap_in_each_kBT
for j in range(n_kBT):  
    figure = plt.scatter(qs, ans[:,j,0], 5, c=ones(n_q)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#q-iter_in_each_kBT
for j in range(n_kBT):  
    figure = plt.scatter(qs, ans[:,j,2], 5, c=ones(n_q)*kBTs[j],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-iter_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

###################################################################################################################