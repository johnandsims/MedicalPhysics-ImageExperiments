#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dec  3
@author: j. sims
"""

import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

eye = imageio.imread('./cosfire_validation.png')
plt.imshow(eye)
plt.show()

#Remove red channel (0)
eye = eye[:,:,1:4]
plt.imshow(eye)
plt.show()

eye = cv2.medianBlur(eye,5)
# Need grayscale image for Hough
gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#minRadius adjusted to eliminate many circle which have smaller radius.
# Param has to be kept high to eliminate poor quality circles
#I found that the number 1 detection is the required circle (i.e. it is
# a strong detection), here at x=416.5, y=409.5, radius = 385.5
# Of course, the centre should be close to the centre of the image.
#Unfortunately, the algorithm has no way of setting a range for x and y.

circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=30,param2=60,minRadius=200,maxRadius=700)

#Get first value returned from accumulator
circle = circles[0][0]
circle[2] = circle[2] - 20

r,c = np.shape(gray)
U,V = np.meshgrid(np.arange(0,c),np.arange(0,r))
mask = ( (U-circle[0])**2 + (V-circle[1])**2 < circle[2]**2)

gray2 = gray * mask

fig = plt.figure()
ax = plt.subplot(111)
ax.imshow(gray2)
fig.savefig('./HoughEye.png')
