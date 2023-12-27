from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
n_q, n_delta =10, 100 # 7.525 #9.21
wide_ans_q = 0.5
qs   = linspace(0.0,0.08,n_q)            #(np.pi/a)
deltas= linspace(-1*wide_ans_q,wide_ans_q,n_delta)   
a, b, d, = -7e7, -0.1, -0.1, 1, 1, 1


###################################################################################################################
#free energy の定義

def free_energy(i_q, i_delta):              #vn0 = (v / n^2) * n0
    return  a * (deltas[i_delta]**2) + b * (deltas[i_delta]**4) + d * (qs[i_q]**2)*(deltas[i_delta]**2) 

###################################################################################################################
#free energy の計算
ans_F_3 = []
for i_delta in range(n_delta):
    ans_F_1 = []
    for i_q in range(n_q):
        ans = free_energy(i_q, i_delta)
        ans_F_1.append(ans)
    ans_F_3.append(ans_F_1)
ans = array(ans_F_3)

###################################################################################################################
##output   ans[h][i][j][0,1,2]
# kBT-q-gap-iter
file = open("output/delta-free_energy"  "_q_" + str(qs[0])   ,"w") 
for i_delta in range(n_delta):
    for i_q in range(n_q):
        file.write(str(qs[i_q]) + " " + str(deltas[i_delta]) + " "+ str(ans[i_delta,i_q]) + " " +  "\n")
file.close()

###################################################################################################################
#描画
#delta-free_energy_in_each_q
for h in range(n_q):
    plt.scatter(deltas, ans[:,h], 5, c=ones(n_delta)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1] )
c= plt.colorbar()
plt.savefig("figure/delta-free_energy" + "_q_" + str(qs[0]) + ".png")
plt.show()
plt.clf() 

# #delta-free_energy_in_each_kBT
# for h in range(n_kBT):
#     plt.scatter(ans_q, ans[h,:,:], 5, c=ones(n_delta)*kBTs[h],  cmap='viridis' ,vmin=kBTs[0], vmax=kBTs[-1])
# c= plt.colorbar()
# plt.savefig("test_in_each_kBT.png")
# #plt.savefig("figure/Nchange/kBT-gap_in_each_q" + "_N_" + str(N) + "_V_" + str(V) + "_mu_" + str(mu) + "_t_" + str(t) + "_q_[" + str(qs[0]) + "," + str(qs[-1]) + "]_kBT_[" + str(kBTs[0]) + "," + str(kBTs[-1]) + "]_Nq_" + str(n0) + "_NkBT_" + str(n2) + ".png")
# plt.show()