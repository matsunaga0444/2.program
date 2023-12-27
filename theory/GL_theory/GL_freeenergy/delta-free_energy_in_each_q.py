from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
n_q, n_delta =2, 100 # 7.525 #9.21
n_d, n_e_d   = 2, 2
qs   =  linspace(0.000,0.01,n_q)            #(np.pi/a)  
b   =   1
ds  =  -1 * linspace(0,8000,n_d)
e_ds = -1 * linspace(0,19/16,n_e_d)

###################################################################################################################
#free energy の定義

def free_energy(b, d, e_d, q, delta):              #vn0 = (v / n^2) * n0
    return  -1 * (delta**2) + b * (delta**4) \
            -1 * ( d * (q**2)*(delta**2) + e_d * d * (q**2)*(delta**4))\

def solution_free_energy(b, d, e_d, q):              #vn0 = (v / n^2) * n0
    return  sqrt((2*b + 2 * e_d*d *(q**2))/(-1*(-1+d*(q**2))))

###################################################################################################################
#free energy の計算
ans = []
for i_d in range(n_d):
    ans1 = []
    for i_e_d in range(n_e_d):
        delta_0 = solution_free_energy(b, ds[i_d], e_ds[i_e_d], qs[0])
        delta_q = solution_free_energy(b, ds[i_d], e_ds[i_e_d], qs[1])
        if delta_0 < delta_q:
            f_0 = free_energy(b, ds[i_d], e_ds[i_e_d], qs[0], delta_0)
            f_q = free_energy(b, ds[i_d], e_ds[i_e_d], qs[1], delta_q)
            if f_0 < f_q:
                ans1.append([b,ds[i_d],e_ds[i_e_d]])
            else:
                ans1.append(["1",b,ds[i_d],e_ds[i_e_d]])
        else:
            ans1.append([0,0,0])
    ans.append(ans1)


###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("output/coefficient_b_" + str(b) + "_ds_[" + str(ds[0]) +","+ str(ds[-1])+","+str(n_d) +"]_e_d_[" + str(e_ds[0])+","+str(e_ds[-1])+","+str(n_e_d) +"]" ,"w") 
file.write("### b , d, e_d "+  "\n")
for i_d in range(n_d):
    for i_e_d in range(n_e_d):
        file.write(str(ans[i_d][i_e_d][0]) + " " + str(ans[i_d][i_e_d][1]) + " "+ str(ans[i_d][i_e_d][2]) + " " +  "\n")
file.close()
