from numpy import *
import matplotlib.pyplot as plt
from time import time
#from scipy.integrate import quad

###################################################################################################################
##パラメータの調整
n_q, n_T =100, 100
Tc = 1
Ts= linspace(0,2*Tc,n_T)
a, b, d, e = -1 * ((Tc-Ts)/Tc), 1, (13/17)*100*2, (-16/17)*100*2
qs = linspace(0,3,n_q)  #sqrt(-1*b/e)-0.001
#f2      = 1/17 * x^2 * (x^2-4) =   1/17 * x^4 -  4/17 * x^2 
#f1      =        x^2 * (x^2-1) =   1    * x^4 -         x^2 =    b * x^4 + a * x^2
#f2 - f1 =        x^2 * (x^2-1) = -16/17 * x^4 + 13/17 * x^2 =    e * q**2 * x^4 + d * (q**2) * x^2

# n_q, n_T =100, 100
# Tc = 1
# Ts= linspace(0,2*Tc,n_T)
# a, b, d, e = -1 * ((Tc-Ts)/Tc), 1, (13/17)*100*2, (-16/17)*100*2
# qs = linspace(0,sqrt(-1*b/e)-0.001,n_q)

###################################################################################################################
#free energy の定義

def free_energy(a, b, d, e, q, delta):              #vn0 = (v / n^2) * n0
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
    return sqrt(-1*(d/a) + (e/(2*b)) )

def c_l_puterbation_GL(a, d):
    return sqrt(-1*(d/a))

###################################################################################################################
#plot q - minimum_free_energy 
plt.scatter(qs, minimum_free_energy(a[0], b, d, e, qs))
plt.savefig("figure/q-minimum_free_energy.png")
plt.show()
plt.clf()

#plot q-solution_free_energy_4
plt.scatter(qs, solution_free_energy_4(a[0], b, d, e, qs))
plt.savefig("figure/q-solution_free_energy_4.png")
plt.show()
plt.clf()

#plot T-coherence_length_GL_formulation_from_delta_q_data_from_beyond_GL_freeenergy
for h in range(n_q):
    plt.scatter(Ts, coherence_length_GL_formulation(solution_free_energy_4(a, b, d, e, qs[h]), solution_free_energy_4(a, b, d, e, qs[0]), qs[h]), 5, c=ones(n_T)*qs[h],  cmap='viridis' ,vmin=qs[0], vmax=qs[-1])
c= plt.colorbar()
plt.savefig("figure/T-coherence_length_GL_formulation_from_delta_q_data_from_beyond_GL_freeenergy.png")
plt.show()
plt.clf()

#plot T-c_l_puterbation_beyond_GL
plt.scatter(Ts, c_l_puterbation_beyond_GL(a, b, d, e))
plt.savefig("figure/T-c_l_puterbation_beyond_GL.png")
plt.clf()

#plot T-c_l_puterbation_GL
plt.scatter(Ts, c_l_puterbation_GL(a, d))
plt.savefig("figure/T-c_l_puterbation_GL.png")
plt.clf()

#plot comopared_T-each_c_l
plt.scatter(Ts, coherence_length_GL_formulation(solution_free_energy_4(a, b, d, e, qs[1]), solution_free_energy_4(a, b, d, e, qs[0]), qs[1]),5 ,marker= "o" ,label ='c_l_GL_form_from_beyond_GL_f')
plt.scatter(Ts, c_l_puterbation_beyond_GL(a, b, d, e), 5,marker= "o" ,label = 'c_l_puterbation_beyond_GL')
plt.scatter(Ts, c_l_puterbation_GL(a, d), 5,marker= "o" ,label = 'c_l_puterbation_GL')
plt.legend()
plt.savefig("figure/comopared_T-each_c_l.png")
plt.show()
plt.clf()


###################################################################################################################
##output  
# T-coherence_length_GL_formulation_from_delta_q_data_from_beyond_GL_freeenergy
file = open("output/T-coherence_length_GL_formulation_from_delta_q_data_from_beyond_GL_freeenergy" ,"w") 
file.write("### T , coherence_length"+  "\n")
for i_T in range(n_T):
    for i_q in range(n_q):
        file.write(str(Ts[i_T]) + " " + str(coherence_length_GL_formulation(solution_free_energy_4(a[i_T], b, d, e, qs[h]), solution_free_energy_4(a[i_T], b, d, e, qs[0]), qs[h])) + " " +  "\n")
file.close()


# T-c_l_puterbation_beyond_GL
file = open("output/T-c_l_puterbation_beyond_GL" ,"w") 
file.write("### T , coherence_length"+  "\n")
for i_T in range(n_T):
    file.write(str(Ts[i_T]) + " " + str(c_l_puterbation_beyond_GL(a[i_T], b, d, e)) + " " +  "\n")
file.close()


# T-c_l_puterbation_GL
file = open("output/T-c_l_puterbation_GL" ,"w") 
file.write("### T , coherence_length"+  "\n")
for i_T in range(n_T):
    file.write(str(Ts[i_T]) + " " + str(c_l_puterbation_GL(a[i_T], d)) + " " +  "\n")
file.close()
