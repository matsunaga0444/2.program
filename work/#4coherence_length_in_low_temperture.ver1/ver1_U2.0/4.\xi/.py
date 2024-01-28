from numpy import *
import matplotlib.pyplot as plt

delta_c            = 0.3404304808723633
ddF_Delta_c_0      = 0.3286393207204735
dd_dF_Delta_c_0    = 0.3161625699028144 

xi = sqrt(dd_dF_Delta_c_0/(delta_c**2 * ddF_Delta_c_0))
xi = sqrt(dd_dF_Delta_c_0/(delta_c**2 * ddF_Delta_c_0))


###################################
##output
file = open("output/xi", "w")
file.write("##xi" + "\n")
for j in range (1):
    file.write(str(xi) + " "   + "\n")
file.close()
