import Henon as Hn
import SympNetR
import SympNetNR
import numpy as np


def SympNet(num_layers, num_sub, reversible):
    
    if reversible=='reversible':
    
        return SympNetR.SympNet(num_layers, num_sub)
    
    if reversible=='non_reversible':
    
        return SympNetNR.SympNet(num_layers, num_sub)

def Henon(num_layers, reversible):
    
     if reversible=='reversible':
    
        return Hn.HenonR(num_layers)
    
     if reversible=='non_reversible':
    
        return Hn.HenonNR(num_layers)


def prova(a,b):
    
    return a+b



def HenonHeiles_read(file_name1, file_name2):
    
    file = open(file_name1, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    
    Ty = []
    Tydot = []
    
    read_y = open(file_name1, "r")
    read_ydot = open(file_name2, "r")
    
    
    for i in range(line_count):
        Ty.append(float(read_y.readline()))
        Tydot.append(float(read_ydot.readline()))
    
    read_y.close()
    read_ydot.close()
    
    print("Number of points: ", len(Ty))
    
    return Ty, Tydot


def PerturbedHamiltonian_read(file_name1, file_name2):
    
    file = open(file_name1, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    
    
    Nlines = int(line_count/2)
    
    u0 = []
    
    prova = open(file_name1, "r")
    for i in range(Nlines):
        x = float(prova.readline())
        y = float(prova.readline())
        u0.append(np.array([x,y]))
    prova.close()
    
    T = []
    
    prova = open(file_name2, "r")
    
    for i in range(Nlines):
        x = float(prova.readline())
        y = float(prova.readline())
        T.append(np.array([x,y]))
    prova.close()
    
    return u0, T
