from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
N, V, t, mu =10, 1, 1, 0
n_q, n_kBT, n_delta =1, 10, 100 # 7.525 #9.21
wide_ans_q = 0.15
qs   = linspace(0.0,0.08,n_q)            #(np.pi/a)
kBTs = linspace(0.001,0.1,n_kBT)      
ans_q= linspace(-1*wide_ans_q,wide_ans_q,n_delta)   

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
    return  1 / (exp(beta*E) + 1 )
    #return (1 - np.tanh(beta*E/2)) /2

def func(k1, k2, gap, q): 
    return gap*(1-Fermi(beta, E_k_q_s(k1, k2, gap, q, 1))-Fermi(beta, E_k_q_s(k1, k2, gap, q, -1)))/(2*E_k_q(k1, k2, gap, q))

def rhs(gap, q):
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = func(kx, ky, gap, q)
    return (V / (N**2)) * sum(f)

###################################################################################################################
#free energy の定義
def F1(i_q, i_delta):
    y = -1 + 2 * arange(2)
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky, y = meshgrid(k1, k1, y, indexing='ij')
    g = log(1+exp(-1*beta*E_k_q_s(kx, ky, ans_q[i_delta], qs[i_q], y)))
    #g =  log(-2 / tanh(-1*beta*E_k_q_s(kx, ky, ans_q[i_delta], qs[i_q], y)/2)-1)
    return -1*(1/beta) * sum(g)

def F0(i_q, i_delta):
    k1 = -1 + 2 * arange(N)  / (N)
    kx, ky = meshgrid(k1, k1, indexing='ij')
    f = e_k_spin(-1*kx, -1*ky, qs[i_q]) - E_k_q_s(kx, ky, ans_q[i_delta], qs[i_q], -1)
    return sum(f)

def Fc(i_delta):
    return(N**2)*(ans_q[i_delta]**2)/V

def free_energy(i_q, i_delta):              #vn0 = (v / n^2) * n0
    return F1(i_q, i_delta) + F0(i_q, i_delta) + Fc(i_delta) #  

###################################################################################################################
#free energy の計算
ans_F_3 = []
for i_kBT in range(n_kBT):
    beta = 1/kBTs[i_kBT]
    ans_F_2 = []
    for i_delta in range(n_delta):
        ans_F_1 = []
        for i_q in range(n_q):
            ans = free_energy(i_q, i_delta)
            ans_F_1.append(ans)
        ans_F_2.append(ans_F_1)
    ans_F_3.append(ans_F_2)
ans = array(ans_F_3)

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("output/delta-free_energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0])   ,"w")
for i_kBT in range(n_kBT):   
    for i_delta in range(n_delta):
        for i_q in range(n_q):
            file.write(str(kBTs[i_kBT]) + " " + str(qs[i_q]) + " " + str(ans_q[i_delta]) + " "+ str(ans[i_kBT,i_delta,i_q]) + " " +  "\n")
file.close()

###################################################################################################################
#描画
# #delta-free_energy_in_each_q
# for h in range(n_q):
#     plt.scatter(ans_q, ans[:,:,h], 5, c=ones(n_delta)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1] )
# c= plt.colorbar()
# plt.savefig("figure/delta-free_energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
# plt.clf()

#delta-free_energy_in_each_kBT
for i_kBT in range(n_kBT):
    plt.scatter(ans_q, ans[i_kBT,:,:]-ans[i_kBT,0,:], 5, c=ones(n_delta)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/delta-free_energy" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
plt.clf() 

#delta-log(free_energy)_in_each_kBT
for i_kBT in range(n_kBT):
    plt.scatter(ans_q, log(-1*ans[i_kBT,:,:]+ans[i_kBT,0,:]), 5, c=ones(n_delta)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/delta-log(free_energy)" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
plt.clf() 


#delta-log(free_energy)_in_each_kBT
for i_kBT in range(n_kBT):
    plt.scatter(ans_q[:n_delta-1], ((log(-1*ans[i_kBT,1:,:]+ans[i_kBT,0,:]) - log(-1*ans[i_kBT,:n_delta-1,:]+ans[i_kBT,0,:])).reshape(99)/(ans_q[1:]-ans_q[:n_delta-1]).reshape(99)) , 5, c=ones(n_delta-1)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
c= plt.colorbar()
plt.savefig("figure/delta-grad(log(free_energy))" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
plt.clf() 

#delta-log(free_energy)_in_each_kBT
for i_kBT in range(n_kBT):
    plt.scatter(ans_q[:n_delta-1], ((log(-1*ans[i_kBT,1:,:]+ans[i_kBT,0,:]) - log(-1*ans[i_kBT,:n_delta-1,:]+ans[i_kBT,0,:])).reshape(99)/(ans_q[1:]-ans_q[:n_delta-1]).reshape(99)) , 5, c=ones(n_delta-1)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
plt.ylim([-20,20])
c= plt.colorbar()
plt.savefig("figure/set_delta-grad(log(free_energy))" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
plt.clf() 

#delta-log(free_energy)_in_each_kBT
for i_kBT in range(n_kBT):
    plt.scatter(ans_q[:n_delta-1], log((log(-1*ans[i_kBT,1:,:]+ans[i_kBT,0,:]) - log(-1*ans[i_kBT,:n_delta-1,:]+ans[i_kBT,0,:])).reshape(99)/(ans_q[1:]-ans_q[:n_delta-1]).reshape(99)) , 5, c=ones(n_delta-1)*kBTs[i_kBT],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
plt.ylim([-20,20])
c= plt.colorbar()
plt.savefig("figure/log(set_delta-grad(log(free_energy)))" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_" + str(qs[0]) + ".png")
plt.clf() 