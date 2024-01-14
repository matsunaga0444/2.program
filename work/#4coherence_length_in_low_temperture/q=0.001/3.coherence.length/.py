from numpy import *
import matplotlib.pyplot as plt


q = 0.001 
invq = 1/q
b, c = 0.257506 , 0.159662
d, e = 8.14854e-07, -6.55084e-06
d, e = -1*invq**2*d, -1*invq**2*e
print(d,e)

delta_c = exp((-2*b-c)/(2*c))
print("delta_c",delta_c)

print(d-e*delta_c**2)
clength = sqrt(2*(d-e*delta_c**2)/c)

###################################
##output
file = open("output/clength", "w")
file.write("##delta_c--clength" + "\n")
file.write(str(delta_c) + " " + str(clength) + "\n")



"""
この方法だと、dが基本的に負となってしまうため、コヒーレンス長が虚数となってしまう。
delta_c近傍でfree energyのdelta依存性がある程度正確に得られているにも関わらず、
コヒーレンス長が虚数となってしまうのはなぜなのか？
コヒーレンス長の定義を変える必要性がある？
"""
