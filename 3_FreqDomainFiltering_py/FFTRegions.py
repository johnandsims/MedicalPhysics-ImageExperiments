#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function FFTfilt by John Sims 2019 
 Create mask , apply mask to freq domain , print image , IFFT , print image .
 Function parameter It is the image transformed by FFT . Returns filtered image in
   spatial domain ( j ) .
"""

def fftFilt ( centre, U, V, It, radius):
    #Define mask as points where 
    mask = (U - centre [0] )**2 + (V - centre [1] )**2 < radius
    plt.imshow(mask,cmap='gray')
    plt.show()
    
    J = It * mask ;

    magnitude_spectrum = 20*np.log(np.abs(J))
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title(f'Ideal Low Pass Filter radius={radius} position ({centre[0]}, {centre[1]})'), plt.xticks([]), plt.yticks([])
    plt.show()
 
    j = np.real( np.fft.ifft2( J ) )
    plt.imshow(j, cmap = 'gray')
    plt.title(f"Filtered image"), plt.xticks([]), plt.yticks([])
    plt.show()
    return j

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

im=imageio.imread('../Images/Head.jpeg')

im = rgb2gray(im)
plt.imshow(im,cmap='gray')
plt.show()
M,N=np.shape(im)

u = np.arange(0, M , 1) 
v = np.arange(0, N , 1)

idx = np.where(u > M/2)
u[idx] = u[idx] - M
idy = np.where(v > N/2)
v[idy] = v[idy] - N

V , U  = np.meshgrid ( v , u )
print(V)

f = np.fft.fft2(im)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(im, cmap = 'gray')
plt.title('MRI Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('FFT of MRI image'), plt.xticks([]), plt.yticks([])
plt.show()

centre = [0 ,0]
radius = 5
j1 = fftFilt (centre ,V ,U , f, radius )

centre = [0 ,0]
radius = 100
j1 = fftFilt (centre ,V ,U , f, radius )