import numpy as np
from numpy import *
import matplotlib.pyplot as plt

###################################################################################################################
##パラメータの調整
N, mu_a, mu_b  = 0.2, -50.0, 1.9
n_kmesh, t = 1000, 1
mu, gu =0, 1
n_U, n_kBT, n_scf =50, 1, 2000  # 7.525 #9.21
n_search, error, check_gap =100, 1e-10, 1e-6
d_delta = 0.0001
Us     = np.linspace(1,10,n_U)
kBTs   = np.linspace(0.001,0.2,n_kBT)

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, k3, q, mu): 
    return 2*t*(cos((k1+q/2)*pi)) - mu
    #return 2*t*(cos((k1+q/2)*pi)+cos((k2)*pi)) - mu
    #return 2*t*(np.cos((k1+(q/2)))+np.cos((k2))) - mu + y * 1/2 * gu * B

def e_k_s(k1, k2, k3, q, mu):
    return (e_k_spin(k1, k2, k3, q, mu) + e_k_spin(-1*k1, -1*k2, -1*k3, q, mu))/2

def e_k_a(k1, k2, k3, q, mu):
    return (e_k_spin(k1, k2, k3, q, mu) - e_k_spin(-1*k1, -1*k2, -1*k3, q, mu))/2

def E_k_q(k1, k2, k3, gap, q, mu):
    return sqrt(e_k_s(k1, k2, k3, q, mu)**2 + gap**2)

def E_k_q_s(k1, k2, k3, gap, q, y, mu):
    return E_k_q(k1, k2, k3, gap, q, mu) + y * e_k_a(k1, k2, k3, q, mu)

def Fermi(beta, E):
    #return  1 / (exp(beta*E) + 1 )
    return (1 - tanh(beta*E/2)) /2

def func(k1, k2, k3, gap, q, mu): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, k3, gap, q, 1, mu))-Fermi(beta, E_k_q_s(k1, k2, k3, gap, q, -1, mu)))/(2*E_k_q(k1, k2, k3, gap, q, mu))

def rhs(gap, q, mu):
    k1 = -1 + (2 * arange(n_kmesh)) / (n_kmesh)
    kx = k1
    ky, kz = 1, 1
    f = func(kx, ky, kz, gap, q, mu)
    return (V / (n_kmesh)) * sum(f)

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
#free energy の定義
def Fn():
    return F1(0,0) + F0(0,0) + Fc(0)

def F1(qs,ans_q):
    y = -1 + 2 * arange(2)
    k1 = -1 * np.pi + 2 * arange(n_kmesh) * np.pi / (n_kmesh)
    kx, y = meshgrid(k1, y, indexing='ij')
    ky, kz = 1, 1
    g = log(1+exp(-1*beta*E_k_q_s(kx, ky, kz, ans_q, qs, y, mu)))
    return -1*(1/beta) * sum(g)

def F0(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(n_kmesh) * np.pi / (n_kmesh)
    kx = k1
    ky, kz = 1, 1
    f = e_k_spin(-1*kx, -1*ky, -1*kz, qs,-1) - E_k_q_s(kx, ky, kz, ans_q, qs, -1, mu)
    return sum(f)

def F0_0(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(n_kmesh) * np.pi / (n_kmesh)
    kx = k1
    ky, kz = 1, 1
    f = e_k_spin(-1*kx, -1*ky, -1*kz, qs, -1, mu)
    return sum(f)

def F0_1(qs,ans_q):
    k1 = -1 * np.pi + 2 * arange(n_kmesh) * np.pi / (n_kmesh)
    kx = k1
    ky, kz = 1, 1
    f = E_k_q_s(kx, ky, kz, ans_q, qs, -1, mu)
    return sum(f)

def Fc(ans_q):
    return(n_kmesh)*(ans_q**2)/V

def free_energy(qs,ans_q):
    return (F1(qs,ans_q) + F0(qs,ans_q) + Fc(ans_q) -Fn()) / n_kmesh#  


###################################################################################################################
##search_N_which_meet_mu_parameter

ans = []
for i_U in range(n_U):
    V = Us[i_U]
    ans_1=[]
    for j in range(n_kBT): 
        beta, d0 = 1/kBTs[j], 1
        mu_0 = mu_a
        mu_1 = mu_b
        for i_search in range(n_search):
            mu = (mu_0 + mu_1) /2
            ##scf_calculation_of_delata
            for i_scf in range(n_scf): # 収束するまで最大1000回ループ
                d1 = rhs(d0, 0, mu) 
                if abs(d1-d0) < 1e-10: break # 収束チェック
                d0 = d1
                iter_scf = i_scf
            #U, EからN求める
            delta = d1
            k1 = -1 + 2 * arange(n_kmesh)  / (n_kmesh)
            kx = k1
            ky, kz = 1, 1
            e_k_p,  e_k_m = e_k_spin(kx, ky, kz, 0, mu), e_k_spin(-1*kx, -1*ky, -1*kz, 0, mu)
            u_k_f = u_k(e_k_p, e_k_m, delta)
            v_k_f = v_k(e_k_p, e_k_m, delta)
            E_k_p_f=  E_k_p(e_k_p, e_k_m, delta)
            E_k_m_f= E_k_m(e_k_p, e_k_m, delta)
            n_k_f  = n_k(beta, u_k_f, v_k_f, E_k_p_f, E_k_m_f)
            N1 = sum(n_k_f)/n_kmesh
            if abs(N-N1) < 1e-10: break # 収束チェック
            if N1 < N: 
                mu_0 =  mu
            if N1 > N: 
                mu_1 =  mu
            iter_search = i_search
        ans_1.append([delta,i_scf, mu, i_search])
    ans.append(ans_1)
ans = array(ans)
print(ans)


###################################################################################################################
##\xiの計算過程

xi=[]
for i_U in range(n_U):
    V = Us[i_U]
    xi_1 =[]
    for i in range(n_kBT):
        ###################################################################################################################
        ##F''(\Delta,0)

        dc = ans[i_U][i][0]
        mu = ans[i_U][i][2]

        d1 = dc + d_delta
        d2 = dc + 2 * d_delta
        qs = np.linspace(0.0,0.001,3) 

        dfj1  = (free_energy(0,d1) - free_energy(0,dc))/(d1-dc)
        dfj2  = (free_energy(0,d2) - free_energy(0,d1))/(d2-d1)
        d2f =   (dfj2-dfj1)/(d2-d1)
        

        ###################################################################################################################
        ##F''(\Delta_c, q)-F(\Delta_c,0)

        ddf_q1 = ((free_energy(qs[1],dc))-(free_energy(qs[0],dc)))/((qs[1]-qs[0])*pi)
        ddf_q2 = ((free_energy(qs[2],dc))-(free_energy(qs[1],dc)))/((qs[2]-qs[1])*pi)
        dddf_q = (ddf_q2 - ddf_q1)/((qs[1]-qs[0])*pi)

        ###################################################################################################################
        ##\xi
        xi_1.append(sqrt(dddf_q/(dc**2 * d2f)))
    xi.append(xi_1)
xi = np.array(xi)

########################################################################################################################
#plot U-xi
for i_kBT in range(n_kBT):
    plt.scatter(Us, xi[:,i_kBT], 5)
plt.savefig("figure/U-xi.png")
plt.clf()

########################################################################################################################
#plot U-mu
for i_kBT in range(n_kBT):
    plt.scatter(Us, ans[:,i,2], 5)
plt.savefig("figure/U-mu.png")
plt.clf()

########################################################################################################################
#plot U-F
for i_kBT in range(n_kBT):
    plt.scatter(Us, ans[:,i,0]/Us, 5)
plt.savefig("figure/U-F.png")
plt.clf()

###################################
##output
file = open("output/kBT-xi-mu", "w")
file.write("##kBT-xi-mu" + "\n")
for i_U in range(n_U):
    for i in range(n_kBT):
        file.write(str(kBTs[i]) + " " + str(xi[i_U][i]) + " " + str(ans[i_U][i][2]) + " "  + "\n")
file.close()

