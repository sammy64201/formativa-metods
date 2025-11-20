from pylab import *
import numpy as np

def MC2(N): 
    x = uniform(0,3)
    y = uniform(min(exp(-x**2)), max(exp(-x**2)), N)
    T = y <= exp(-x**2) # test
    exitos = sum(T)
    area = (exitos/N)*(3*(max(exp(-x**2))-min(exp(-x**2))))
    return area

print(MC2(100))


