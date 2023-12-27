import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time




time_scf_start = time()



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
            ans1.append(coherence_length(ans[h][i][j][0], ans[0][i][j][0], qs[h]) )
        ans0.append(ans1)
    ans_c.append(ans0)
ans_c = np.array(ans_c)

time_cal_coherence_finish = time()
time_cal_coherence = time_cal_coherence_finish - time_cal_coherence_start 
print("time_cal_coherence : " + str(time_cal_coherence//3600) + "時間" + str((time_cal_coherence%3600)//60) +"分" + str(time_cal_coherence%60) + "秒")

##output-kBT-q-coherence_length
file = open("./output/coherence" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) ,"w")
for j in range(n2):    
    for i in range(n1):
        for h in range(n0):
                file.write(str(kBTs[j]) + " " + str(qs[h]) + " " + str(ans_c[h][i][j]) + " " +  "\n")
file.close()

##kBT-coherence_length_in_each_q
for j in range(n2):    
    for i in range(n1):
        ans_q = []
        for h in range(n0):
            figure = plt.scatter(kBTs[j], ans_c[h][i][j], 5, c=qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/kBT-coherence_in_each_momentum(q)" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
plt.show()

print(time())

print(ans_c[0][0][0])

from numpy import *
import irbasis3
import matplotlib.pyplot as plt

class parameters:
    def __init__(self):
    ##パラメータの調整
    N, V, t, mu, gu, n0, n1, n2 =10, 1, 1.4, 0, 1, 10, 1, 100  # 7.525 #9.21
    qs   = np.linspace(0,0.02,n0)      #(np.pi/a)
    Bs   = np.linspace(0.0,0.1,n1)     #np.linspace(0,0.08,n1)
    kBTs = np.linspace(0.001,0.02,n2)  

class definition:
    self.def e_k_spin(k1, k2, q, y, B): 
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

class gfunctions:
    def __init__(self, b, p, i):
        self.scf(b, p, i)
        self.set_gkf_p(b, p, i)
        self.set_dkf(b, p)
        self.set_drt(b, p)
        #self.debug(b, p)
        #self.debug_plt(b ,p)
    def scf(self, b, p, i):
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


class fmc_anisotropic_eliashberg_BCS:
    def __init__(self, g, b, p):
        self.set_delta0(g, b, p)
        self.scf(g, b, p)
        #self.debug(g, b, p)
        #self.debug_plot(g, b, p)
        self.output( g, b, p)
    def set_delta0(self, g, b, p):
        self.delta0 = ones(b.niw_F)[:,None] * ones(p.nk)[None,:]
    def scf(self, g, b, p):
        self.delta   = zeros(b.niw_F)[:,None] * zeros(p.nk)[None,:]
        self.delta_temp = self.delta0
        self.iter = 0
        for n in range(100):
            if linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta) <= 1e-6: break
            if abs(self.delta_temp[len(self.delta_temp)//2,0]) <= 1e-3: break
            self.iter = self.iter + 1
            self.delta = self.delta_temp
            self.set_frt(g, b, p)
            self.y_rt = -(p.g0ph**2 * g.drt*self.frt).reshape(b.ntau_B,p.nk1,p.nk2,p.nk3) / p.nk
            self.y_kt = fft.ifftn(self.y_rt,axes=(1,2,3))
            self.y_kt = self.y_kt.reshape(b.ntau_B, p.nk)
            self.y_kl = b.T_F.fit(self.y_kt)
            self.y_kf = (self.y_kl.T @ b.uhat_F).T
            self.delta_temp=  (1-p.mixing)*self.y_kf + p.mixing*self.delta
            print(n,  self.delta_temp[len(self.delta_temp)//2,0], linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta))
    def set_frt(self, g, b, p):
        #print(g.gkf_p*conj(g.gkf_m))
        _ = 1/(g.gkf_p*conj(g.gkf_m)) + self.delta*conj(self.delta)
        self.fkf = self.delta/_
        #print(self.fkf)
        self.fkl = b.M_F.fit(self.fkf)
        self.fkt = dot(b.ul_F_tau_B, self.fkl)
        self.fkt = self.fkt.reshape(b.ntau_B, p.nk1, p.nk2, p.nk3)
        self.frt = fft.fftn(self.fkt, axes=(1,2,3)).reshape(b.ntau_B, p.nk)
        #print(self.frt)
    def debug(self, g, b, p): #self.delta0, self.delta, self.fkf, self.fkl, fkt, self.frt, y_rt, y_kt, y_kl, y_kf
        file = open("output/delta0.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.delta0[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkf.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.fkf[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkl.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.fkf[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.fkt = self.fkt.reshape(b.ntau_B, p.nk)
        file = open("output/fkt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.fkt[k][j])) + "\n")
            file.write("\n")
        file.close()        
        file = open("output/frt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.frt[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.y_rt = self.y_rt.reshape(b.ntau_B,p.nk)
        file = open("output/y_rt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_rt[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_kt[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kl.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.y_kl[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kf.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.y_kf[k][j])) + "\n")
            file.write("\n")
        file.close()  
    def debug_plot(self, g, b, p):
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.delta0[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.delta0).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.delta[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.delta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.fkf[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.fkf).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.fkl[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.fkl).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.fkt[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.fkt).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.frt[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.frt).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_rt[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_rt).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_kt[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_kt).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.y_kl[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.y_kl).png")
        plt.clf()  
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F[:]), abs(self.y_kf[:,j]), s=1)
        plt.savefig("figure/num(b.iw_F)_abs(self.y_kf).png")
        plt.clf()  
    def output(self, g, b, p):
        file = open("output/delta.dat","w")
        for i in range(p.nk):
            for j in range(b.niw_F):
                file.write(str(imag(b.iw_F[j])) + " " + str(abs(self.delta[j][i])) +  "\n" )
            file.write("\n")
        file.close()


if __name__=='__main__':
    p = parameters()
    file = open("output/delta_iter.dat","w")
    file.close()
    deltas = zeros(p.nbeta)[:,None] * zeros(p.nfmc)[None,:]
    iters =  zeros(p.nbeta)[:,None] * zeros(p.nfmc)[None,:]
    betas = 1/ (arange(p.nbeta)/p.nbeta * (1/p.beta_min - 1/p.beta_max) + 1/p.beta_max)
    fmcs  = p.fmc_max*(arange(p.nfmc)/p.nfmc)+p.fmc_min
    for k in range(p.nbeta):
        b = irfunc(p.lamb, betas[k])
        for i in range(p.nfmc):
            g = gfunctions(b, p, i)
            e = fmc_anisotropic_eliashberg_BCS(g, b, p)
            deltas[k,i] = abs(e.delta[len(e.delta)//2,0])
            iters[k,i]  = e.iter
            file = open("output/delta_iter.dat",mode='a')
            file.write(str(1/betas[k]) + " " + str(fmcs[i]) + " " + str(abs(e.delta[len(e.delta)//2,0])) +  "\n" )
            file.write("\n")
            file.close()
    for i in range(p.nfmc): # T-gap in each q
        plt.scatter(1/betas, deltas[:,i], s=1, c= (fmcs[i]) * ones(p.nbeta),  cmap='viridis' ,vmin=fmcs[0], vmax=fmcs[-1])
    c= plt.colorbar()
    plt.title("T-gap_in_each_q", {"fontsize": 20})
    plt.xlabel("tempature(T)")
    plt.ylabel("gap")
    plt.savefig("figure/T-gap_in_each_q.png")
    plt.clf()  
    for i in range(p.nfmc): # T-iter in each q
        plt.scatter(1/betas, iters[:,i], s=1, c= (fmcs[i]) * ones(p.nbeta),  cmap='viridis' ,vmin=fmcs[0], vmax=fmcs[-1])
    c= plt.colorbar()
    plt.title("T-iter_in_each_q", {"fontsize": 20})
    plt.xlabel("tempature(T)")
    plt.ylabel("iteration of scf")
    plt.savefig("figure/T-iter_in_each_q.png")
    plt.clf()      
    for i in range(p.nbeta): # q-gap in each T
        plt.scatter(fmcs, deltas[i,:], s=1, c= (1/betas[i]) * ones(p.nfmc),  cmap='viridis' ,vmin=1/betas[0], vmax=1/betas[-1])
    c= plt.colorbar()
    plt.title("q-gap_in_each_T", {"fontsize": 20})
    plt.xlabel("mometum(q)")
    plt.ylabel("gap")
    plt.savefig("figure/q-gap_in_each_T.png")
    plt.clf()  
    for i in range(p.nbeta): # q-iter in each T
        plt.scatter(fmcs, iters[i,:], s=1, c= (1/betas[i]) * ones(p.nfmc),  cmap='viridis' ,vmin=1/betas[0], vmax=1/betas[-1])
    c= plt.colorbar()
    plt.title("q-iter_in_each_T", {"fontsize": 20})
    plt.xlabel("mometum(q)")
    plt.ylabel("iteration of scf")
    plt.savefig("figure/q-iter_in_each_T.png")
    plt.clf()  

