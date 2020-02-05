#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:42:20 2019

@author: jasus
"""
import numpy as np

def zeroInsertion(vec):
    print(vec)
    ins = np.zeros(1)
    for num in vec:
        #np.append(ins,0)
        np.append(ins,num)
        
        print(num,ins)
    return(ins)
    
a = np.array([1,2,3,4,5])
print(zeroInsertion(a))
