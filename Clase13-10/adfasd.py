#una aeroline vende 125 boletos para un aavion que solamente cuenta 120 asientos.
#se sabe qupor datos historicos que el 90% de los pasajeros se presentan a tomar el vuelvo. 
#Cual es la probabilidad, de que todos los pasajerso que se presenten puedan tomar el vuelo?
from pylab import *
import numpy as np

def Vuelos(N):
    suma = 0
    for i in range(N):
        pasajeros = uniform(0,1,125)
        exitos = sum(pasajeros < 0.9)
        if exitos <= 120:
            suma += 1
    return suma/N

print(Vuelos(10000))
