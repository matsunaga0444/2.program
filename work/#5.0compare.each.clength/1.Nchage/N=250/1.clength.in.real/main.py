from numpy import *

class parameters:
    def __init__(self):
        self.nk   = 250
        self.beta = 1/0.016333333333333335
        self.mu   = 0
        self.u    = 2 * ones(self.nk)
        self.u[self.nk//2:-1] = 0.0

class calculation:
    def __init__(self, p):
        self.H_normal = - (eye(p.nk,k=1) + eye(p.nk,k=-1)) - p.mu * eye(p.nk)
        self.H_normal[0,-1] = self.H_normal[-1,0] = -1
        self.delta    = 0.1*eye(p.nk)
        self.delta0   = 0.1*eye(p.nk)
        self.dim      = self.H_normal.shape[0]

    def set_H_BdG(self, p):
        H11 = self.H_normal
        H12 = self.delta
        H21 = self.delta.T   # delta is set to be real
        H22 = -self.H_normal
        H = array([[H11, H12],[H21, H22]]) / 2.0
        H = transpose(H, (0,2,1,3)).reshape(self.dim*2, self.dim*2)
        self.H_BdG = H
        val, vec = linalg.eigh(self.H_BdG)
        self.E = val[self.dim:]
        self.u = vec[:self.dim,self.dim:]
        self.v = vec[self.dim:,self.dim:]

    def set_delta(self, p):
        f  = 0.5 * (1.0-tanh(0.5*p.beta*self.E))
        _1 = einsum('in,n,nj->ij', self.u, 1-f, conj(self.v.T), optimize=True)
        _2 = einsum('in,n,nj->ij', conj(self.v),   f, self.u.T, optimize=True)
        self.f = _1 - _2
        self.delta = diag(p.u * diag(self.f))
        #self.delta[0,0] = self.delta[0,0]*99/100

    def scf(self, p):
        lam1 = 0.1
        lam0 = 0.0
        for n in range(500):
            self.delta0 = self.delta
            self.set_H_BdG(p)
            self.set_delta(p)
            nrm, err = linalg.norm(self.delta)/sqrt(p.nk), linalg.norm(self.delta-self.delta0)/linalg.norm(self.delta)
            #nrm, err = linalg.norm(self.delta)/sqrt(p.nk), linalg.norm(self.delta[0,0]-self.delta0[0,0])/linalg.norm(self.delta[0,0])
            if (nrm<1e-8) or (err<1e-7): break
            print(n,  nrm, err)

    def output(self, p):
        file = open("delta.dat","w")
        for i in range(p.nk):
            file.write("{0}    {1}    {2}    {3}\n".format(i, real(diag(self.delta))[i], real(diag(self.f))[i], abs(real(diag(self.f-self.f[p.nk//4,p.nk//4]))[i])))
        file.close()

if __name__=='__main__':
    p = parameters()
    c = calculation(p)
    c.scf(p)
    c.output(p)

