#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Region Growing algorithm converted from Matlab
# @author: jasus

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                 %
%                                                                 %
% Author: Enio GJERGA                                             %
% Date: 07/05/2015                                                %
% email: enio.gjerga@gmail.com                                    %
% Universiteit Gent, Belgium                                      %
%                                                                 %
% Description: This code segments a region based on the value of  %
% the pixel selected and on which thresholding region it belongs  %
% based on the region growing algorithm.                          %
%                                                                 %
%                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import data
img=data.coins()
I = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#I = im2double(imread('Image_To_Read.tiff'));
plt.imshow(I)
#figure, imshow(I)
#%imtool(I);
row,col  = I.shape

#Isizes = size(I); %size of the image
#threshI = multithresh(I, 3) #thresholding for three regions
#from skimage.filters import threshold_multiotsu
#thresholds = threshold_multiotsu(image)
#threshI = threshold_multiotsu(I, 3) #thresholding for three regions
#[m, n]=ginput(1); 
# pick one pixel of the region to be segmented
m=60
n=130

#c = impixel(I, m, n); %value of the pixel picked
c=I[m,n]
currPix = c #current pixel
surr = np.array([[-1, 0],[ 1, 0],[ 0, -1],[ 0, 1]]) #create a mask which represents the four surrounding pixels
mem = np.zeros([row * col, 3]) #create a register to put the pixel coordinates and pixel value 


mem[0, :] = [m, n, currPix] #insert initial picked pixel to the register
regSize = 1 #initial size
J = np.zeros([row, col]) #create another black image with the same size as the original image
init = 0;
posInList = 1;
k=1 #create the initial condition to run the loop


#The region growing algorithm.
while(k==1):
    
  for l in range (init,posInList): #first pointer on the register
      for j in range(0,4): #second pointer for the neighboring pixels
        m1 = np.int(m + surr[j,0])
        n1 = np.int(n + surr[j,1])
        
        check=(m1>=0) and (n1>=0) and (m1< row) and (n1< col) #simple check if pixel position still inside the image
        
        current = I[m1, n1];
        currPix = current;
        #if(check and currPix<=threshI(2) and (J[m1, n1]==0)): #check if it belongs to the thresholding boundary and if not set yet on the image we want to recreate

        if(check  and (J[m1, n1]==0)): #check if it belongs to the thresholding boundary and if not set yet on the image we want to recreate
            posInList = posInList+1
            mem[posInList, :] = [m1, n1, currPix] #add the new pixel
            J[m1, n1] = 1

  if(posInList == init):  #when there is no more pixels to add
      k = 0 #make k=0 to close the loop
  else:
      init = init+1;
      m = mem[init, 0]
      n = mem[init, 1]
      k = 1 #keep running the loop

plt.imshow(J) #the segmented black and white region
