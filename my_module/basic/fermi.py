from numpy import *

def Fermi(beta, E):
    #return  1 / (exp(beta*E) + 1 )
    return (1 - tanh(beta*E/2)) / 2

def derivative_Fermi_function(beta, E):
    #return -1 * beta * exp(beta*E) / (exp(beta*E) + 1 )**2
    return beta*(Fermi(beta, E)**2-Fermi(beta, E))
