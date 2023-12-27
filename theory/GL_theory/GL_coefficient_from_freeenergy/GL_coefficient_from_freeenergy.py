from numpy import *

def f(delta, beta, delta0):
    abs_d = abs(delta)
    f1 = -2*sqrt(2*pi*abs_d/beta**3)*exp(-1*beta*abs_d)
    f2 = -1*abs_d**2*(1/2-log(abs_d/abs(delta0)))
    return f1 + f2

