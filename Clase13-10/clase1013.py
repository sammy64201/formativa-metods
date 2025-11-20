from pylab import *

def IntR(f,a,b,n,pos):
    h = (b-a)/n  # base del rectangulo
    suma = 0
    for i in range(n):
        xi = a+h*(pos+i)
        suma = suma + h*f(xi)
    return suma

# EJEMPLO
def f(x):
    return (cos(x))**2

a,b,n,