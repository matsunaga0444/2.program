import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整(1d)
N, V, t, mu, gu, n0, n1, n2, nscf =10000, 1, 1, 0, 1, 10, 1, 10, 2000  #kBT-gap_in_each_q
#N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 100, 1, 10, 2000  #q-gap_in_each_kBT
qs   = np.linspace(0,0.1,n0)        
Bs   = np.linspace(0.0,0.1,n1)     
kBTs = np.linspace(0.0001,0.0175,n2)
mesh_DOS = 0.01

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, k3, q, y, B): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))) - mu + y * 1/2 * gu * B

###################################################################################################################
## cal DOS
k1 = (-1 + 2 * arange(N)/N) *pi
k2 = 1
k3 = 1

#DOSの要素の列を作る
DOS_mesh = (-(2*t/mesh_DOS) + arange(4*t/mesh_DOS)) 

print(DOS_mesh)
DOS_mesh = DOS_mesh.tolist()
int_DOS =e_k_spin(k1, k2, k3, 0, 0, 0) / mesh_DOS
print(int_DOS)
int_DOS = int_DOS //1
print(int_DOS)
DOS = int_DOS.tolist()
#print(DOS, DOS_mesh)
DOS_count = []
for i in range(len(DOS_mesh)):
    DOS_count.append(DOS.count(DOS_mesh[i]))
print(DOS_count)


###################################################################################################################
## plot DOS
DOS_mesh = array(DOS_mesh)
DOS_count = array(DOS_count)
plt.scatter(mesh_DOS * DOS_mesh, DOS_count/N, 5)
plt.savefig("figure/1D_DOS")
plt.show
