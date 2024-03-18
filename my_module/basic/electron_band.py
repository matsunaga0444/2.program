from numpy import *

def e_k_s_1D(t, k1, q, mu, y, gu, B): 
    return 2*t*(cos((k1+(q/2)*pi))) - mu + y * 1/2 * gu * B

def e_k_s_2D(t, k1, k2, q, mu, y, gu, B): 
    return 2*t*(cos((k1+(q/2)*pi))+cos((k2))) - mu + y * 1/2 * gu * B

def e_k_s_3D(t, k1, k2, k3, q, mu, y, gu, B): 
    return 2*t*(cos((k1+(q/2)*pi))+cos((k2))+cos((k3))) - mu + y * 1/2 * gu * B

def e_k_s_free_1D(t, k1, q, mu, y, gu, B): 
    return -2*t*((k1+(q/2)*pi)**2) - 2 - mu + y * 1/2 * gu * B
