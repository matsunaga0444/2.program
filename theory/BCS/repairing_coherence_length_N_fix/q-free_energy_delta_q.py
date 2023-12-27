import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N, mu_a, mu_b  = 0.6, 0.3, 0.4
n_kmesh, V, t, mu, gu, n_q, n_kBT, n_scf =10, 1, 1 , 0, 1, 2, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =10000, 1e-10, 1e-6
kBT_a, kBT_b = 0.0001, 1
wide_q = 0.001
qs   = np.linspace(0,wide_q,n_q)         #(np.pi/a)
kBTs = np.linspace(0.0001,1,n_kBT)


###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, q, mu): 
    return 2*t*(cos((k1+q/2)*pi)+cos((k2)*pi)) - mu
    #return 2*t*(np.cos((k1+(q/2)))+np.cos((k2))) - mu + y * 1/2 * gu * B

def e_k_s(k1, k2, q, mu):
    return (e_k_spin(k1, k2, q, mu) + e_k_spin(-1*k1, -1*k2, q, mu))/2

def e_k_a(k1, k2, q, mu):
    return (e_k_spin(k1, k2, q, mu) - e_k_spin(-1*k1, -1*k2, q, mu))/2

def E_k_q(k1, k2, gap, q, mu):
    return sqrt(e_k_s(k1, k2, q, mu)**2 + gap**2)

def E_k_q_s(k1, k2, gap, q, y, mu):
    return E_k_q(k1, k2, gap, q, mu) + y * e_k_a(k1, k2, q, mu)

def Fermi(beta, E):
    #return  1 / (exp(beta*E) + 1 )
    return (1 - tanh(beta*E/2)) /2

def func(k1, k2, gap, q, mu): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1, mu))-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1, mu)))/(2*E_k_q(k1, k2, gap, q, mu))

def rhs(gap, q, mu):
    k1 = -1 + (2 * arange(n_kmesh)) / (n_kmesh)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q, mu)
    return (V / (n_kmesh**2)) * sum(f)

def u_k(e_k_p, e_k_m, delta):
    e_k = (e_k_p + e_k_m)/2
    c2_k = e_k/sqrt(e_k**2 + delta**2)
    c_k = sqrt((1/2)*(1+c2_k))
    return  c_k
    
def v_k(e_k_p, e_k_m, delta):
    e_k = (e_k_p + e_k_m)/2
    c2_k = e_k/sqrt(e_k**2 + delta**2)
    s_k = sqrt((1/2)*(1-c2_k))
    return s_k

def E_k_p(e_k_p, e_k_m, delta):
    e_k = (e_k_p + e_k_m)/2
    return  (e_k_p*e_k + delta**2)/sqrt((e_k**2 + delta**2))

def E_k_m(e_k_p, e_k_m, delta):
    e_k = (e_k_p + e_k_m)/2
    return  (e_k_m*e_k + delta**2)/sqrt((e_k**2 + delta**2))

def n_k(beta, u_k, v_k, E_k_p, E_k_m):
    f_E_k_p = Fermi(beta, E_k_p)
    f_E_k_m = Fermi(beta, E_k_m)
    return (u_k**2) * f_E_k_p + (v_k**2)*(1-f_E_k_m)


###################################################################################################################
##search_N_which_meet_mu_parameter

ans = []
for h in range(n_q):
    ans1 = []
    for j in range(n_kBT): 
        beta, d0 = 1/kBTs[j], 100.0
        mu_0 = mu_a
        mu_1 = mu_b
        for i_search in range(n_search):
            mu = (mu_0 + mu_1) /2
            ##scf_calculation_of_delata
            for i_scf in range(n_scf): # 収束するまで最大1000回ループ
                d1 = rhs(d0, qs[h], mu) 
                if abs(d1-d0) < 1e-10: break # 収束チェック
                d0 = d1
                iter_scf = i_scf
            #U, EからN求める
            delta = d1
            k1 = -1 + 2 * arange(n_kmesh)  / (n_kmesh)
            kx, ky = meshgrid(k1, k1, indexing='ij')
            e_k_p,  e_k_m = e_k_spin(kx, ky, qs[h], mu), e_k_spin(-1*kx, -1*ky, qs[h], mu)
            u_k_f = u_k(e_k_p, e_k_m, delta)
            v_k_f = v_k(e_k_p, e_k_m, delta)
            E_k_p_f=  E_k_p(e_k_p, e_k_m, delta)
            E_k_m_f= E_k_m(e_k_p, e_k_m, delta)
            n_k_f  = n_k(beta, u_k_f, v_k_f, E_k_p_f, E_k_m_f)
            N1 = sum(n_k_f)/n_kmesh**2
            print(mu, N1)
            if abs(N-N1) < 1e-10: break # 収束チェック
            if N1 < N: 
                mu_0 =  mu
            if N1 > N: 
                mu_1 =  mu
            iter_search = i_search
        ans1.append([delta,i_scf, mu, i_search ])
    ans.append(ans1)
ans = array(ans)
print(ans)

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("./output/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
for j in range(n_kBT):   
    for h in range(n_q):
            file.write(str(kBTs[j]) + " " + str(qs[h]) + " " + str(ans[h][j][0]) + " " + str(ans[h][j][2]) + " " +  "\n")
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
ans_q= ans[:,0,0]
###################################################################################################################
#free energy の定義
def F1(h):
    sum = 0
    for n1 in range(n_kmesh):
        k1 = -1 * np.pi + 2 * n1 * np.pi / (n_kmesh)
        for n2 in range(n_kmesh):
            k2 = -1 * np.pi + 2 * n2 * np.pi / (n_kmesh)
            for y in range(-1,1):
                #sum = sum + np.log(1+np.exp(-1*beta*E_k_q_s(k1, k2, ans_q[h], qs[h], y, B)))
                sum = sum + np.log(-2 / np.tanh(-1*beta*E_k_q_s(k1, k2, ans_q[h], qs[h], y, ans[h,0,2])/2)-1)
    return -1*1/beta*sum

def F0(h):
    sum = 0
    for n1 in range(n_kmesh):
        k1 = -1 * np.pi + 2 * n1 * np.pi / (n_kmesh)
        for n2 in range(n_kmesh):
            k2 = -1 * np.pi + 2 * n2 * np.pi / (n_kmesh)
            sum = sum + e_k_spin(-1*k1, -1*k2, qs[h], ans[h,0,2]) - E_k_q_s(k1, k2, ans_q[h], qs[h], -1, ans[h,0,2])
    return sum

def Fc(h):
    return((n_kmesh)**2)*(ans_q[h]**2)/V

def free_energy(h):              #vn0 = (v / n^2) * n0
    return F1(h) + F0(h) + Fc(h) #  

###################################################################################################################
#free energy の計算
kBT = kBTs[0]
beta = 1/kBT
ans_F0 = []
for h in range(n_q):
    ans_F0.append(F0(h))
ans0 = np.array(ans_F0)

ans_F1 = []
for h in range(n_q):
    ans_F1.append(F1(h))
ans1 = np.array(ans_F1)

ans_FC = []
for h in range(n_q):
    ans_FC.append(Fc(h))
ansC = np.array(ans_FC)

ans_F = []
for h in range(n_q):
    ans_F.append(free_energy(h))
ans_F = np.array(ans_F)

###################################################################################################################
#描画
plt.scatter(qs, ans1)
plt.clf()

plt.scatter(qs, ans0)
plt.clf()

plt.scatter(qs, ansC)
plt.clf()

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
kBTs = np.linspace(0, Tc, 1000)
print(ans[0][0][0])
a_0 = calculate_a_1(ans[0,0,0], ans_F[0])
a = calculate_a(ans[0,0,0], ans_F[0], kBTs[0], kBTs, Tc)
b = calculate_b(ans[0,0,0], ans_F[0])
d = calculate_d(ans[0,0,0], ans_F[0], ans[1:,0,0], ans_F[1:], qs[1:])
e = calculate_e(ans[0,0,0], ans_F[0], ans[1:,0,0], ans_F[1:], qs[1:])

###################################################################################################################
## def for calculating coherence length

def c_l_puterbation_extended_GL(a, b, d, e):
    return sqrt(-1*(d[:,None]/a[None,:]) + (e[:,None]/(2*b)) )

###################################################################################################################
## calculate coherence length
print(a, b, d, e)
c_l =  c_l_puterbation_extended_GL(a, b, d, e)
print(c_l)
###################################################################################################################
## plot coherence length
for i_q in range(n_q-1):
    plt.scatter(kBTs, c_l[i_q,:])
plt.savefig("test.png")
plt.clf
