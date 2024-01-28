from numpy import *
import matplotlib.pyplot as plt

delta_c            = 0.12096405360425501
ddF_Delta_c_0      = 0.3258539122621016
dd_dF_Delta_c_0    = 0.31616256992994635

xi = sqrt(dd_dF_Delta_c_0/(delta_c**2 * ddF_Delta_c_0))
xi = sqrt(dd_dF_Delta_c_0/(delta_c**2 * ddF_Delta_c_0))


###################################
##output
file = open("output/xi", "w")
file.write("##xi" + "\n")
for j in range (1):
    file.write(str(xi) + " "   + "\n")
file.close()
