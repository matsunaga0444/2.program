from numpy import *

def e_k_s_1D(t, k1, q, mu, y, gu, B): 
    return t*((k1+(q/2)*pi)**2) - 2 - mu + y * 1/2 * gu * B

