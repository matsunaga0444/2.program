from numpy import *
import matplotlib.pyplot as plt
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N, mu_a, mu_b  = 0.6, 0.3, 0.4
n_kmesh, V, t = 10, 1, 1
# n_q, n_kBT, n_search, n_scf =10, 10, 1000, 1000 # 7.525 #9.21
#n_q, n_kBT, n_search, n_scf =5, 100, 1000, 1000 # 7.525 #9.21
n_q, n_kBT, n_search, n_scf, check_delta =5, 5, 10000, 1000, 1e-7  # 7.525 #9.21
qs   = linspace(0.0,0.02,n_q)            #(np.pi/a)
kBTs = linspace(1e-5,0.013,n_kBT)   

# plt_parameter
f_x_c, n_delta = 1.1, 1000

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


def calculate_Tc(T_1, T_2, delta_1, delta_2):
    return (T_2*delta_1**2-T_1*delta_2**2)/(delta_1**2-delta_2**2)

def calculate_b(T_1, T_2, delta_1, delta_2):
    return (T_2-T_1)/2*(delta_1**2-delta_2**2)

def calculate_d(q_1, q_2, delta_1, delta_2, T, Tc, b):
    a = (T-Tc)/Tc
    B = q_1**2*delta_1**2*(-2*b*delta_2**2-a)-q_2**2*delta_2**2*(-2*b*delta_1**2-a)
    return B/(-1*q_1**2*q_2**2*(delta_2**2-delta_1**2))

def calculate_e(q_1, q_2, delta_1, delta_2, T, Tc, b):
    a = (T-Tc)/Tc
    A = q_1**2*(-2*b*delta_2**2-a)-q_2**2*(-2*b*delta_1**2-a)
    return A/(2*q_1**2*q_2**2*(delta_2**2-delta_1**2))

def free_energy(T, Tc, b, d, e, q, delta):    #vn0 = (v / n^2) * n0
    a = (T-Tc)/Tc           
    return  a * (delta**2) + b * (delta**4) \
            + d * (q**2)*(delta**2) + e * (q**2)*(delta**4)
#    return (delta**2)*((a + d * (q**2)) + (b+ e * (q**2))*(delta**2))
            
def solution_free_energy_4(a, b, d, e, q):              #vn0 = (v / n^2) * n0
    return  sqrt(-1 * (a + d * (q**2))/(b + e * (q**2)))

def solution_free_energy(a, b, d, q):              #vn0 = (v / n^2) * n0
    return  sqrt(-1 * (a + d * (q**2))/(b))

def minimum_free_energy(a, b, d, e, q):              #vn0 = (v / n^2) * n0
    return  -1 *  (b+ e * (q**2)) * ((a + d * (q**2))/(b + e * (q**2)))**2 /4

def coherence_length_GL_formulation(gap_q, gap_0, q):
    return sqrt((gap_0**2 - gap_q**2) / (gap_0**2*q**2))

def c_l_puterbation_beyond_GL(a, b, d, e):
    return sqrt(-1*(d/a) + (e/2*b) )

def c_l_puterbation_GL(a, d):
    return sqrt(-1*(d/a))

###################################################################################################################
##search_N_which_meet_mu_parameter

ans = []
for h in range(n_q):
    ans1, delta_1, T_1  = [], 0, 0
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
                if h == 0 and d0 > check_delta: 
                    delta_2 = delta_1
                    delta_1 = d0
                    T_2 = T_1
                    T_1 = kBTs[j]
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
            if abs(N-N1) < 1e-10: break # 収束チェック
            if N1 < N: 
                mu_0 =  mu
            if N1 > N: 
                mu_1 =  mu
            iter_search = i_search
        ans1.append([delta,i_scf, mu, i_search ])
    ans.append(ans1)
ans = array(ans)


q_1, q_2, delta_q_1, delta_q_2, T = qs[1], qs[2], ans[1,0,0], ans[2,0,0], kBTs[0]
Tc = calculate_Tc(T_1, T_2, delta_1, delta_2)
a = (T-Tc)/Tc
b  = calculate_b(T_1, T_2, delta_1, delta_2)
d  = calculate_d(q_1, q_2, delta_q_1, delta_q_2, T, Tc, b)
e  = calculate_e(q_1, q_2, delta_q_1, delta_q_2, T, Tc, b)

ans_xsi = []
for i_kBT in range(n_kBT):
    a = (T-Tc)/Tc
    xsi = c_l_puterbation_beyond_GL(a, b, d, e)
    ans_xsi.append([xsi])
ans_xsi = array(ans_xsi)

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("./output/kBT_q_delta_iter-scf_mu_iter-search" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("### kBT q delta iter_scf mu iter_search" +  "\n")
for i_q in range(n_q):   
    for i_kBT in range(n_kBT):
        file.write(str(kBTs[i_kBT]) + " " + str(qs[i_q ]) + " " + str(ans[i_q][i_kBT][0]) + " " + str(ans[i_q][i_kBT][1]) + " " + str(ans[i_q][i_kBT][2]) + " " + str(ans[i_q][i_kBT][3]) +  "\n")
file.close()

# kBT-c_l
file = open("./output/kBT-c_l" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("# kBT-c_l ")
for j in range(n_kBT):   
    file.write(str(kBTs[j]) + " " + str(ans_xsi[j][0]) + " " +  "\n")

# Tc-b-d-e
file = open("./output/Tc-b-d-e" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + "_nscf_" + str(n_scf)  ,"w")
file.write("# Tc-b-d-e")
file.write(str(Tc) + " " + str(b) + " " + str(d) + " " + str(e) + " " +  "\n")

# ###################################################################################################################
# ##figure

#kBT-c_l_with_Tc
figure = plt.scatter(kBTs, ans_xsi[:,0])
plt.savefig("figure/kBT-c_l_with_Tc" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#free-energy
x = ans[0,0,0] * f_x_c * arange(n_delta)/n_delta
for i_q in range(2):
    figure = plt.scatter(x,free_energy(kBTs[0], Tc, b, d, e, qs[i_q], x) , 5, c=ones(n_delta)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[1])
c= plt.colorbar()
plt.savefig("figure/free-energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf() 

#kBT-gap_in_each_q
for i_q in range(n_q):  
    figure = plt.scatter(kBTs, ans[i_q,:,0], 5, c=ones(n_kBT)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

# #kBT-iter_scf_in_each_q
for i_q in range(n_q):  
    figure = plt.scatter(kBTs, ans[i_q,:,1], 5, c=ones(n_kBT)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-iter_scf_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

#kBT-mu_in_each_q
for i_q in range(n_q):  
    figure = plt.scatter(kBTs, ans[i_q,:,2], 5, c=ones(n_kBT)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-mu_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

#kBT-iter_mu_in_each_q
for i_q in range(n_q):  
    figure = plt.scatter(kBTs, ans[i_q,:,3], 5, c=ones(n_kBT)*qs[i_q],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-iter_mu_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

#q-gap_in_each_kBT
for i_kBT in range(n_kBT):    
    figure = plt.scatter(qs, ans[:,i_kBT,0], 5, c=ones(n_q)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-gap_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

# #q-iter_scf_in_each_kBT
for i_kBT in range(n_kBT):    
    figure = plt.scatter(qs, ans[:,i_kBT,1], 5, c=ones(n_q)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-iter_scf_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

# #q-mu_in_each_kBT
for i_kBT in range(n_kBT):    
    figure = plt.scatter(qs, ans[:,i_kBT,2], 5, c=ones(n_q)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-mu_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

# #q-iter_mu_in_each_kBT
for i_kBT in range(n_kBT):    
    figure = plt.scatter(qs, ans[:,i_kBT,3], 5, c=ones(n_q)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/q-iter_mu_in_each_kBT" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n_q) + "_NkBT_" + str(n_kBT) + ".png")
plt.clf()

# ###################################################################################################################
