import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整(1d)
N, V, t, mu, gu, n0, n1, n2, nscf =100, 1, 1, 0, 1, 10, 1, 10, 2000  #kBT-gap_in_each_q
mesh_DOS = 0.1

###################################################################################################################
## gap_eq をdef
def e_k_spin(k1, k2, k3, q, y, B): 
    return 2*t*(np.cos((k1+(q/2)*np.pi))+np.cos((k2))) - mu + y * 1/2 * gu * B

###################################################################################################################
## cal DOS

k1 = (-1 + 2 * arange(N)/N) *pi
kx, ky= meshgrid(k1, k1, indexing='ij')
kz = 1

#DOSの要素の列を作る必要あり、
DOS_mesh = (-(4*t/mesh_DOS) + arange(8*t/mesh_DOS)) // 1
print(DOS_mesh)
DOS_mesh = DOS_mesh.tolist()
int_DOS =e_k_spin(kx, ky, kz, 0, 0, 0) / mesh_DOS
print(int_DOS)
int_DOS = int_DOS // 1
int_DOS = int_DOS.reshape(N**2) //1
print(int_DOS)
DOS = int_DOS.tolist()
print(DOS, DOS_mesh)
DOS_count = []
for i in range(len(DOS_mesh)):
    DOS_count.append(DOS.count(DOS_mesh[i]))
#print(DOS_count)


###################################################################################################################
## plot DOS

plt.scatter(DOS_mesh, DOS_count, 5)
plt.savefig("figure/2D_DOS")
plt.show
