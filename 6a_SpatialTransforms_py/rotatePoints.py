#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 20:44:19 2019

@author: jasus
"""
import numpy as np
import math
import matplotlib.pyplot as plt
theta = math.pi/6

#tranformation matrix, T
T=[[np.cos(theta),np.sin(theta),0],
[-np.sin(theta),np.cos(theta),0],
[0,0,1]];

#Coords, U,V
x = np.arange(-1,2)
y = np.arange(-1,2)

U,V= np.meshgrid(x,y)
print(U)
print(V)
X=np.zeros([9,3])
for i in range (0,3):
    for j in range (0,3):
        print(U[i,j],V[i,j])
        X[3*i+j,:] = np.array([U[i,j],V[i,j],1])
        Y = np.matmul(X,T)
#print(X)
plt.plot(Y[:,0],Y[:,1],'r*')
plt.imshow()