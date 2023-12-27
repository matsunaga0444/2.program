from numpy import *
import irbasis3
import matplotlib.pyplot as plt

class parameters:
    def __init__(self):
        self.nk1, self.nk2, self.nk3 = 8, 8, 1 
        self.beta_min   = 10
        self.beta_max   = 100000
        self.nbeta  = 1
        self.mu     = 0.1
        self.g0ph   = 2.0
        self.qf_phonon  =  1.0
        self.damp_phonon   = 0.0
        self.damp_electron = 0.0
        self.mixing     = 0.0
        self.lamb = 10000
        self.nk = self.nk1 *self.nk2 * self.nk3
        self.dk1, self.dk2, self.dk3 = 1./self.nk1, 1./self.nk2, 1./self.nk3
        k1, k2, k3 = meshgrid(arange(self.nk1)*self.dk1, arange(self.nk2)*self.dk2, arange(self.nk3)*self.dk3)
        self.k1, self.k2, self.k3 = k1.flatten(), k2.flatten(), k3.flatten()
        self.fmc_max  = 0.05
        self.fmc_min  = 0.0
        self.nfmc  = 1

class irfunc:
    def __init__(self, Lambda, beta, eps=1e-7):
        # initialize
        K              = irbasis3.KernelFFlat(lambda_=Lambda)
        # for Fermion
        self.basis_F   = irbasis3.FiniteTempBasis(K, "F", beta=beta, eps=eps)
        self.M_F       = irbasis3.MatsubaraSampling(self.basis_F)
        self.T_F       = irbasis3.TauSampling(self.basis_F)
        self.tau_F     = self.T_F.sampling_points
        self.iw_F      = 1j*pi*self.M_F.sampling_points/beta
        self.ntau_F    = len(self.tau_F)
        self.niw_F     = len(self.iw_F)
        self.nbasis_F  = self.basis_F.size
        # for Boson
        self.basis_B   = irbasis3.FiniteTempBasis(K, "B", beta=beta, eps=eps)
        self.M_B       = irbasis3.MatsubaraSampling(self.basis_B)
        self.T_B       = irbasis3.TauSampling(self.basis_B)
        self.tau_B     = self.T_B.sampling_points
        self.iw_B      = 1j*pi*self.M_B.sampling_points/beta
        self.ntau_B    = len(self.tau_B)
        self.niw_B     = len(self.iw_B)
        self.nbasis_B  = self.basis_B.size
        # define ul matrix
        self.ul_F_tau_F = array([self.basis_F.u(x) for x in self.tau_F])
        self.ul_F_tau_B = array([self.basis_F.u(x) for x in self.tau_B])
        self.ul_B_tau_F = array([self.basis_B.u(x) for x in self.tau_F])
        self.ul_B_tau_B = array([self.basis_B.u(x) for x in self.tau_B])
        self.uhat_F = self.basis_F.uhat(self.M_F.sampling_points)
        self.uhat_B = self.basis_B.uhat(self.M_B.sampling_points)


class gfunctions:
    def __init__(self, b, p, i):
        self.set_ek_m(b, p, i)
        self.set_ek_p(b, p, i)
        self.set_dkf(b, p)
        self.set_drt(b, p)
        #self.debug(b, p)
        #self.debug_plt(b ,p)
    def set_ek_m(self, b, p, i):
        self.ek_m  = -2*(cos(2*pi*(-1*p.k1+p.fmc_max*(i/p.nfmc)+p.fmc_min))+cos(2*pi*(-1*p.k2))) - p.mu
        #self.gkf_m = 1.0 / (b.iw_F[:,None] - (self.ek_m[None,:]))
    def set_ek_p(self, b, p, i):
        self.ek_p  = -2*(cos(2*pi*(p.k1+p.fmc_max*(i/p.nfmc)+p.fmc_min))+cos(2*pi*p.k2)) - p.mu
        #self.gkf_p = 1.0 / (b.iw_F[:,None] - (self.ek_p[None,:]))
    def set_dkf(self, b, p):
        damping  = -1j * p.damp_phonon * sign(imag(b.iw_B))
        dkf_m    = 1.0 / (b.iw_B - p.qf_phonon - damping)
        dkf_p    = 1.0 / (b.iw_B + p.qf_phonon - damping)
        self.dkf = 0.5 * p.qf_phonon * (dkf_m - dkf_p)
        self.dkf = self.dkf[:,None] * ones(p.nk)[None,:]
    def set_dkf_c(self, b, p):
        self.dkf = p.g0ph**2 * -0.5 * ones(len((b.iw_B)-1)) * 1e-1
        self.dkf = self.dkf[:,None] * ones(p.nk)[None,:]
    def set_drt(self, b, p):
        # k,omega => k,irbasis => k,tau     => r,tau
        self.dkl = b.M_B.fit(self.dkf)
        self.d_kt = dot(b.ul_B_tau_B, self.dkl)
        self.d_kt = self.d_kt.reshape(b.ntau_B, p.nk1, p.nk2, p.nk3)
        self.drt = fft.fftn(self.d_kt, axes=(1,2,3)).reshape(b.ntau_B, p.nk)
    def debug(self, b, p):
        file = open("output/ek_m.dat","w")
        for k in range(len(self.ek_m)):
            file.write(str(k) + " " + str(abs(self.ek_m[k])) + "\n")
        file.write("\n")
        file.close()
        file = open("output/ek_p.dat","w")
        for k in range(len(self.ek_p)):
            file.write(str(k) + " " + str(abs(self.ek_p[k])) + "\n")
        file.write("\n")
        file.close()
        self.d_kt = self.d_kt.reshape(b.ntau_B, p.nk)
        self.d_kl = b.T_B.fit(self.d_kt)
        self.d_kf = (self.d_kl.T @ b.uhat_B).T
        file = open("output/dkf.dat","w")
        for k in range(len(b.iw_B)):
            for j in range(p.nk):
                file.write(str(abs(b.iw_B[k])) + " " + str(abs(self.dkf[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/dkl.dat","w")
        for i in range(p.nk):
            for j in range(len(self.dkl)):
                file.write(str(j) + " " + str(abs(self.dkl[j][i])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/dkt.dat","w")
        for i in range(b.nbasis_B):
            for j in range(p.nk):
                file.write(str(b.tau_B[i]) + " " + str(abs(self.d_kt[i][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/drt.dat","w")
        for i in range(b.ntau_B):
            for j in range(p.nk):
                file.write(str(i) + " " + str(abs(self.drt[i][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/d_kf.dat","w")
        for k in range(b.ntau_B):
            for j in range(p.nk):
                file.write(str(abs(b.iw_B[k])) + " " + str(abs(self.d_kf[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/compare_dkf.dat","w")
        for k in range(b.ntau_B):
            for j in range(p.nk):
                file.write(str(abs(b.iw_B[k])) + " " + str(abs(self.dkf[k][j]))+ " " + str(abs(self.d_kf[k][j])) + "\n")
            file.write("\n")
        file.close()
    def debug_plt(self, b, p):
        plt.scatter(array(range(len(self.ek_p))), abs(self.ek_p[:]), s=1)
        plt.savefig("figure/num(self.ek_p)_abs(self.ek_p).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.gkf_p[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.gkf_p).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_B[:]), abs(self.dkf[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_B)_abs(self.dkf).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(array(range(len(self.dkl))), abs(self.dkl[:,j]), s=1)
        plt.savefig("figure/num(self.dkl)_abs(self.dkl).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(array(b.tau_B[:]), abs(self.d_kt[:,j]), s=1)
        plt.savefig("figure/b.tau_B_abs(self.d_kt).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.tau_B)[:], abs(self.drt[:,j]), s=1)
        plt.savefig("figure/abs(b.tau_B)_abs(self.drt).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_B[:]), abs(self.d_kf[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_B)_abs(d_kf).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_B[:]), abs(self.dkf[:,j]), s=1, c='b')
            plt.scatter(abs(b.iw_B[:]), abs(self.d_kf[:,j]), s=1, c='r')     
        plt.savefig("figure/abs(b.iw_B)_compare(abs(self.dkf),abs(d_kf).png")
        plt.clf()
        

class fmc_anisotropic_eliashberg_BCS:
    def __init__(self, g, b, p):
        self.set_initial(g, b, p)
        self.scf(g, b, p)
        #self.debug(g, b, p)
        #self.debug_plot(g, b, p)
        self.cal_delta(g, b, p)
        self.output( g, b, p)
    def set_initial(self, g, b, p):
        self.zeta0  = ones(b.niw_F)[:,None] * ones(p.nk)[None,:]
        self.chi0   = ones(b.niw_F)[:,None] * ones(p.nk)[None,:]
        self.delta0 = ones(b.niw_F)[:,None] * ones(p.nk)[None,:]
    def scf(self, g, b, p):
        self.zeta   = zeros(b.niw_F)[:,None] * zeros(p.nk)[None,:]
        self.zeta_temp = self.zeta0
        self.chi   = zeros(b.niw_F)[:,None] * zeros(p.nk)[None,:]
        self.chi_temp = self.chi0
        self.delta   = zeros(b.niw_F)[:,None] * zeros(p.nk)[None,:]
        self.delta_temp = self.delta0
        self.iter = 0
        for n in range(100):
            if linalg.norm(self.zeta_temp-self.zeta)/linalg.norm(self.zeta) <= 1e-6 \
               and linalg.norm(self.chi_temp-self.chi)/linalg.norm(self.chi) <= 1e-6 \
               and linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta) <= 1e-6: break
            if abs(self.delta_temp[len(self.delta_temp)//2,0]) <= 1e-10: break
            self.iter = self.iter + 1
            self.zeta = self.zeta_temp
            self.chi = self.chi_temp
            self.delta = self.delta_temp
            self.set_gkfs(g, b, p)
            self.set_frt_zeta(g, b, p)
            self.y_rt_zeta = -(p.g0ph**2 * g.drt * self.frt_zeta).reshape(b.ntau_F,p.nk1,p.nk2,p.nk3) / p.nk
            self.y_kt_zeta = fft.ifftn(self.y_rt_zeta,axes=(1,2,3))
            self.y_kt_zeta = self.y_kt_zeta.reshape(b.ntau_F, p.nk)
            self.y_kl_zeta = b.T_F.fit(self.y_kt_zeta)
            self.y_kf_zeta = 1 - ((self.y_kl_zeta.T @ b.uhat_F).T) / (b.iw_F[:,None]*ones(p.nk)[None,:])
            self.zeta_temp=  (1-p.mixing)*self.y_kf_zeta + p.mixing*self.zeta
            self.set_frt_chi(g, b, p)
            self.y_rt_chi = -(p.g0ph**2 * g.drt*self.frt_chi).reshape(b.ntau_F,p.nk1,p.nk2,p.nk3) / p.nk
            self.y_kt_chi = fft.ifftn(self.y_rt_chi,axes=(1,2,3))
            self.y_kt_chi = self.y_kt_chi.reshape(b.ntau_F, p.nk)
            self.y_kl_chi = b.T_F.fit(self.y_kt_chi)
            self.y_kf_chi = (self.y_kl_chi.T @ b.uhat_F).T
            self.chi_temp=  (1-p.mixing)*self.y_kf_chi + p.mixing*self.chi
            self.set_frt_delta(g, b, p)
            self.y_rt_delta = -(p.g0ph**2 * g.drt*self.frt_delta).reshape(b.ntau_F,p.nk1,p.nk2,p.nk3) / p.nk
            self.y_kt_delta = fft.ifftn(self.y_rt_delta,axes=(1,2,3))
            self.y_kt_delta = self.y_kt_delta.reshape(b.ntau_F, p.nk)
            self.y_kl_delta = b.T_F.fit(self.y_kt_delta)
            self.y_kf_delta = (self.y_kl_delta.T @ b.uhat_F).T 
            self.delta_temp=  (1-p.mixing)*self.y_kf_delta + p.mixing*self.delta
            print(n,  self.zeta_temp[len(self.zeta_temp)//2,0], linalg.norm(self.zeta_temp-self.zeta)/linalg.norm(self.zeta))
            print(n,  self.chi_temp[len(self.chi_temp)//2,0], linalg.norm(self.chi_temp-self.chi)/linalg.norm(self.chi))
            print(n,  self.delta_temp[len(self.delta_temp)//2,0], linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta))
    def set_gkfs(self, g, b, p):
        self.gkf_m = 1.0 / (b.iw_F[:,None] * self.zeta - (g.ek_m[None,:]))
        self.gkf_p = 1.0 / (b.iw_F[:,None] * self.zeta - (g.ek_p[None,:]))
    def set_frt_zeta(self, g, b, p):
        _ = (1/self.gkf_p + self.chi) *(1/conj(self.gkf_m) + self.chi) + self.delta*conj(self.delta)
        self.fkf_zeta = (1j*((b.iw_F[:,None]*ones(b.niw_F)[None,:]).T@self.zeta)+(g.ek_m-g.ek_p)/2)/_
        self.fkl_zeta = b.M_F.fit(self.fkf_zeta)
        self.fkt_zeta = dot(b.ul_F_tau_B, self.fkl_zeta)
        self.fkt_zeta = self.fkt_zeta.reshape(b.ntau_B, p.nk1, p.nk2, p.nk3)
        self.frt_zeta = fft.fftn(self.fkt_zeta, axes=(1,2,3)).reshape(b.ntau_B, p.nk)
    def set_frt_chi(self, g, b, p):
        _ = 1/(self.gkf_p*conj(self.gkf_m)) + self.delta*conj(self.delta)
        self.fkf_chi = (self.chi+(g.ek_m+g.ek_p)/2)/_
        self.fkl_chi = b.M_F.fit(self.fkf_chi)
        self.fkt_chi = dot(b.ul_F_tau_B, self.fkl_chi)
        self.fkt_chi = self.fkt_chi.reshape(b.ntau_B, p.nk1, p.nk2, p.nk3)
        self.frt_chi = fft.fftn(self.fkt_chi, axes=(1,2,3)).reshape(b.ntau_B, p.nk)
    def set_frt_delta(self, g, b, p):
        _ = 1/(self.gkf_p*conj(self.gkf_m)) + self.delta*conj(self.delta)
        self.fkf_delta = self.delta /_
        self.fkl_delta = b.M_F.fit(self.fkf_delta)
        self.fkt_delta = dot(b.ul_F_tau_B, self.fkl_delta)
        self.fkt_delta = self.fkt_delta.reshape(b.ntau_B, p.nk1, p.nk2, p.nk3)
        self.frt_delta = fft.fftn(self.fkt_delta, axes=(1,2,3)).reshape(b.ntau_B, p.nk)
    def cal_delta(self, g, b, p):
        self.delta = self.delta / self.zeta
    def debug(self, g, b, p): #self.delta0, self.delta, self.fkf, self.fkl, fkt, self.frt, y_rt, y_kt, y_kl, y_kf
        file = open("output/gkf_m.dat","w")
        for k in range(len(g.ek_m)):
            for j in range(len(b.iw_F)):
                file.write(str(abs(b.iw_F[j])) + " " + str(abs(self.gkf_m[j][k])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/gkf_p.dat","w")
        for k in range(len(g.ek_p)):
            for j in range(len(b.iw_F)):
                file.write(str(abs(b.iw_F[j])) + " " + str(abs(self.gkf_p[j][k])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/delta0.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.delta0[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/delta.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/zeta.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/chi.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkf.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.fkf_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkl.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.fkf_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.fkt_delta = self.fkt_delta.reshape(b.ntau_B, p.nk)
        file = open("output/fkt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.fkt_delta[k][j])) + "\n")
            file.write("\n")
        file.close()        
        file = open("output/frt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.frt_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.y_rt_delta = self.y_rt_delta.reshape(b.ntau_F,p.nk)
        file = open("output/y_rt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_rt_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_kt_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kl.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.y_kl_delta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kf.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.y_kf_delta[k][j])) + "\n")
            file.write("\n")
        file.close()  
        file = open("output/fkf.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.fkf_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkl.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.fkf_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.fkt_zeta = self.fkt_zeta.reshape(b.ntau_B, p.nk)
        file = open("output/fkt_zeta.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.fkt_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()        
        file = open("output/frt_zeta.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.frt_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.y_rt_zeta = self.y_rt_zeta.reshape(b.ntau_B,p.nk)
        file = open("output/y_rt_zeta.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_rt_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kt_zeta.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_kt_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kl_zeta.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.y_kl_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kf_zeta.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.y_kf_zeta[k][j])) + "\n")
            file.write("\n")
        file.close()  
        file = open("output/fkf_chi.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.fkf_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/fkl_chi.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.fkf_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.fkt_chi = self.fkt_chi.reshape(b.ntau_B, p.nk)
        file = open("output/fkt_chi.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.fkt_chi[k][j])) + "\n")
            file.write("\n")
        file.close()        
        file = open("output/frt_chi.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.frt_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        self.y_rt_chi = self.y_rt_chi.reshape(b.ntau_B,p.nk)
        file = open("output/y_rt_chi.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_rt_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kt_chi.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.y_kt_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kl_chi.dat","w")
        for k in range(b.nbasis_F):
            for j in range(p.nk):
                file.write(str(k) + " " + str(abs(self.y_kl_chi[k][j])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/y_kf_chi.dat","w")
        for k in range(b.niw_F):
            for j in range(p.nk):
                file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.y_kf_chi[k][j])) + "\n")
            file.write("\n")
        file.close()  
    def debug_plot(self, g, b, p):
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.delta0[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.delta0).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.zeta[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.zeta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.chi[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.chi).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.delta[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.delta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.fkf_zeta[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.fkf_zeta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.fkl_zeta[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.fkl_zeta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.fkt_zeta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.fkt_zeta).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.frt_zeta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.frt_zeta).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_rt_zeta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_rt_zeta).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_kt_zeta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_kt_zeta).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.y_kl_zeta[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.y_kl_zeta).png")
        plt.clf()  
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F[:]), abs(self.y_kf_zeta[:,j]), s=1)
        plt.savefig("figure/num(b.iw_F)_abs(self.y_kf_zeta).png")
        plt.clf()  
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.fkf_chi[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.fkf_chi).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.fkl_chi[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.fkl_chi).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.fkt_chi[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.fkt_chi).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.frt_chi[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.frt_chi).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_rt_chi[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_rt_chi).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_kt_chi[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_kt_chi).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.y_kl_chi[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.y_kl_chi).png")
        plt.clf()  
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F[:]), abs(self.y_kf_chi[:,j]), s=1)
        plt.savefig("figure/num(b.iw_F)_abs(self.y_kf_chi).png")
        plt.clf() 
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F), abs(self.fkf_delta[:,j]), s=1)
        plt.savefig("figure/abs(b.iw_F)_abs(self.fkf_delta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.fkl_delta[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.fkl_delta).png")
        plt.clf()
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.fkt_delta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.fkt_delta).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.frt_delta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.frt_delta).png")
        plt.clf()     
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_rt_delta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_rt_delta).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(b.tau_F[:], abs(self.y_kt_delta[:,j]), s=1)
        plt.savefig("figure/(b.tau_F)_abs(self.y_kt_delta).png")
        plt.clf()   
        for j in range(p.nk):
            plt.scatter(arange(b.nbasis_F), abs(self.y_kl_delta[:,j]), s=1)
        plt.savefig("figure/num(b.basis_F)_abs(self.y_kl_delta).png")
        plt.clf()  
        for j in range(p.nk):
            plt.scatter(abs(b.iw_F[:]), abs(self.y_kf_delta[:,j]), s=1)
        plt.savefig("figure/num(b.iw_F)_abs(self.y_kf_delta).png")
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
