#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:24:52 2019

@author: jasus
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

im=imageio.imread('../Images/Head.jpeg')

im = rgb2gray(im)
plt.imshow(im,cmap='gray')
plt.show()

row,col=np.shape(im)

Line = im[np.int(row/2),]
plt.plot(Line)
plt.title('Pixel profile along central row of image')



I = np.fft.fft2(im)
magnitude_spectrum_I = 20*np.log(np.abs(I))

Ishift = np.fft.fftshift(I)
magnitude_spectrum_Ishift = 20*np.log(np.abs(Ishift))

#pyplot setting to make space between subplots large, and maintain size of figure.
#plt.subplots_adjust(wspace=0.3)
plt.figure(figsize=(15,4))

centre = [0 ,0]
radius = 50
j1 = fftFilt (centre ,V ,U , f, radius )