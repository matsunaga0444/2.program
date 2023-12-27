from numpy import *
import irbasis3
import matplotlib.pyplot as plt

class parameters:
    def __init__(self):
        self.nk1, self.nk2, self.nk3 = 64, 64, 1 
        self.beta_min   = 20
        self.beta_max   = 200
        self.nbeta  = 5
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
        self.fmc_max  = 0.01
        self.fmc_min  = 0.0
        self.nfmc  = 20
        self.v0 = -1.0

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
        self.ul_F_tau0 = self.basis_F.u(0.0)


class gfunctions:
    def __init__(self, b, p, i):
        self.set_gkf_m(b, p, i)
        self.set_gkf_p(b, p, i)
        #self.debug(b, p)
        #self.debug_plt(b ,p)
    def set_gkf_m(self, b, p, i):
        self.ek_m  = -2*(cos(2*pi*(-1*p.k1+p.fmc_max*(i/p.nfmc)+p.fmc_min))+cos(2*pi*(-1*p.k2))) - p.mu
        self.gkf_m = 1.0 / (b.iw_F[:,None] - (self.ek_m[None,:]))
    def set_gkf_p(self, b, p, i):
        self.ek_p  = -2*(cos(2*pi*(p.k1+p.fmc_max*(i/p.nfmc)+p.fmc_min))+cos(2*pi*p.k2)) - p.mu
        self.gkf_p = 1.0 / (b.iw_F[:,None] - (self.ek_p[None,:]))
    def set_dkf(self, b, p):
        damping  = -1j * p.damp_phonon * sign(imag(b.iw_B))
        dkf_m    = 1.0 / (b.iw_B - p.qf_phonon - damping)
        dkf_p    = 1.0 / (b.iw_B + p.qf_phonon - damping)
        self.dkf = 0.5 * p.qf_phonon * (dkf_m - dkf_p)
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
        file = open("output/gkf_m.dat","w")
        for k in range(len(self.ek_m)):
            for j in range(len(b.iw_F)):
                file.write(str(abs(b.iw_F[j])) + " " + str(abs(self.gkf_m[j][k])) + "\n")
            file.write("\n")
        file.close()
        file = open("output/ek_p.dat","w")
        for k in range(len(self.ek_p)):
            file.write(str(k) + " " + str(abs(self.ek_p[k])) + "\n")
        file.write("\n")
        file.close()
        file = open("output/gkf_p.dat","w")
        for k in range(len(self.ek_p)):
            for j in range(len(b.iw_F)):
                file.write(str(abs(b.iw_F[j])) + " " + str(abs(self.gkf_p[j][k])) + "\n")
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
        self.set_delta0(g, b, p)
        self.scf(g, b, p)
        self.debug(g, b, p)
        #self.debug_plot(g, b, p)
        #self.output( g, b, p)
    def set_delta0(self, g, b, p):
        self.delta0 = ones(b.niw_F)[:,None] * ones(p.nk)[None,:]
    def scf(self, g, b, p):
        self.delta   = zeros(b.niw_F)[:,None] * zeros(p.nk)[None,:]
        self.delta_temp = self.delta0
        self.iter = 0
        for n in range(500):
            if linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta) <= 1e-6: break
            if abs(self.delta_temp[len(self.delta_temp)//2,0]) <= 1e-3: break
            self.iter = self.iter + 1
            self.delta = self.delta_temp
            self.set_fkt(g, b, p)
            self.y_kf = p.v0 * sum(-(self.fkt0)) / p.nk 
            self.delta_temp=  (1-p.mixing)*self.y_kf + p.mixing*self.delta
            print(n,  self.delta_temp[len(self.delta_temp)//2,0], linalg.norm(self.delta_temp-self.delta)/linalg.norm(self.delta))
    def set_fkt(self, g, b, p):
        _ = 1/(g.gkf_p*conj(g.gkf_m)) + self.delta*conj(self.delta)
        self.fkf = self.delta/_
        self.fkl = b.M_F.fit(self.fkf)
        self.fkt = dot(b.ul_F_tau_B, self.fkl)
        self.fkt0 = dot(b.ul_F_tau0, self.fkl)
        #print(b.tau_F)
        #print(b.tau_B)
    def debug(self, g, b, p): #self.delta0, self.delta, self.fkf, self.fkl, fkt, self.frt, y_rt, y_kt, y_kl, y_kf
        # file = open("output/delta0.dat","w")
        # for k in range(b.niw_F):
        #     for j in range(p.nk):
        #         file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.delta0[k][j])) + "\n")
        #     file.write("\n")
        # file.close()
        # file = open("output/fkf.dat","w")
        # for k in range(b.niw_F):
        #     for j in range(p.nk):
        #         file.write(str(abs(b.iw_F[k])) + " " + str(abs(self.fkf[k][j])) + "\n")
        #     file.write("\n")
        # file.close()
        # file = open("output/fkl.dat","w")
        # for k in range(b.nbasis_F):
        #     for j in range(p.nk):
        #         file.write(str(k) + " " + str(abs(self.fkf[k][j])) + "\n")
        #     file.write("\n")
        # file.close()
        self.fkt = self.fkt.reshape(b.ntau_B, p.nk)
        file = open("output/fkt.dat","w")
        for k in range(b.ntau_F):
            for j in range(p.nk):
                file.write(str(b.tau_F[k]) + " " + str(abs(self.fkt[k][j])) + "\n")
            file.write("\n")
        file.close()        
        # file = open("output/y_kf.dat","w")
        # file.write(str(abs(self.y_kf)) + "\n")
        # file.close()  
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
