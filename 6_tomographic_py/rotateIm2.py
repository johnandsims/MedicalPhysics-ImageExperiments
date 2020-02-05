#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:49:20 2019

@author: jasus
"""

import numpy as np
from skimage import data
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import math


#rotation:
theta = math.pi/6

#inverse tranformation matrix, T
T=np.array(
[[np.cos(theta),-np.sin(theta),0],
[np.sin(theta),np.cos(theta),0],
[0,0,1]] )

# import image from skimage
im=data.immunohistochemistry()

#convert to grayscale
grayIm = rgb2gray(im)
# Take a portion of original image 
image = grayIm[0:200,0:200]

#image padding
Rows, Cols = np.shape(image);
Diagonal = np.sqrt(Rows**2 + Cols**2)
RowPad = (np.ceil(Diagonal - Rows) + 2).astype('int')
ColPad = (np.ceil(Diagonal - Cols) + 2).astype('int')
imagepad = np.zeros([Rows+RowPad, Cols+ColPad])
imagepad[np.int(np.ceil(RowPad/2)):np.int(np.ceil(RowPad/2)+Rows),
         np.int(np.ceil(ColPad/2)):np.int(np.ceil(ColPad/2)+Cols)] = image

RowImpad, ColImpad = np.shape(imagepad)
# midpoints
midx=np.ceil((RowImpad+1)/2).astype('int')
midy=np.ceil((ColImpad+1)/2).astype('int')

imagerot= np.zeros([RowImpad, ColImpad]) # midx and midy same for both

for i in range (0, RowImpad):
    for j in range (0, ColImpad):

         x= (i-midx)*np.cos(theta)+(j-midy)*np.sin(theta)
         y=-(i-midx)*np.sin(theta)+(j-midy)*np.cos(theta)
         x=np.round(x) + midx
         y=np.round(y) + midy

         if (x>=0 and y>=0 and x<ColImpad and y<RowImpad):
              imagerot[i,j]=imagepad[x.astype('int'),y.astype('int')] #k degrees rotated image         

#Plot results        
fig, ((ax1, ax2,ax3)) = plt.subplots(1,3,figsize=(15,7))
ax1.imshow(image,cmap='gray')
ax1.set_title('Original image')
ax2.imshow(imagepad,cmap='gray')
ax2.set_title('Padded image')
ax3.imshow(imagerot,cmap='gray')
ax3.set_title(f'Rotated image around point ({midx},{midy})')