

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
grayIm = grayIm[0:5,0:3]
row, col = np.shape(grayIm)

rpoint = np.array([2.,1.]) # rotate around point 2,1

#Determine four corners of array
corners = np.array([[0.,0.,col,col],[0.,row,0.,row],[1.,1.,1.,1.]])
corners = np.array([corners[0]+rpoint[0],corners[1]+rpoint[1],corners[2]])
# rotate corners
rcorners = np.matmul(T,corners)

# find max and min values of rotated coords
xmax = np.ceil(rcorners[0].max())
xmax = np.int(xmax)
xmin = np.floor(rcorners[0].min())
ymax = np.ceil(rcorners[1].max())
ymin = np.floor(rcorners[1].min())
xmax = np.int(xmax)
xmin = np.int(xmin)
ymax = np.int(ymax)
ymin = np.int(ymin)

x=np.arange(xmin,xmax+1)
y=np.arange(ymin,ymax+1)
U,V = np.meshgrid(x,y)


newIm = np.zeros([ymax-ymin+1,xmax-xmin+1])
newIm[xmin]

# Origin for transform will be at centre of image.
#  calculate "floor" and "ceiling" of number of cols and rows div by 2     
limcolf = np.floor(col/2)
limrowf = np.floor(row/2)
     
limcolc = np.ceil(col/2)
limrowc = np.ceil(row/2)

#Define Coords, U,V for image
x = np.arange(-limcolf, limcolc)
y = np.arange(-limrowf, limrowc)
V,U = np.meshgrid(x,y)


#Create list of points, X, from U and V
X=np.zeros([row*col,3])
for i in range (0,row):
    for j in range (0,col):
        print(f'{i},{j},{col*i+j}')
        print(np.array([U[i,j],V[i,j],1]))
        X[col*i+j,:] = np.array([V[i,j],U[i,j],1])
 
Y = np.matmul(X,IT)
#print(Y)


# Plot X and Y
fig, ((ax1, ax2)) = plt.subplots(1,2,figsize=(15,7))
ax1.plot(X[:,0],X[:,1],'b*')
ax1.set_title('Original points')
ax1.grid(True)


ax2.plot(Y[:,0],Y[:,1],'r*')
ax2.set_title('inverse transform')
ax2.grid(True)
plt.show()

ind = np.where( (Y[:,0]>=-limcolc) & (Y[:,0]<=limcolc) & (Y[:,1]>=-limrowc) & (Y[:,1]<=limrowc) )


Xselrow = X[ind[0],1] + limrowc
Xselcol = X[ind[0],0] + limcolc


Yselrow = Y[ind[0],1]
Yselcol = Y[ind[0],0]


plt.plot(Yselcol,Yselrow,'r*')
plt.title('inverse transform')
plt.grid(True)
plt.show()

from scipy import interpolate
f = interpolate.interp2d(x, y, grayIm, kind='linear')

zinterp = np.zeros([row,col])

numpix, = np.shape(Xselrow)

for i in range(0,numpix):
    print(f'{Xselrow[i]}, {Xselcol[i]},{Yselrow[i]},{Yselcol[i]}')
    zinterp[np.int(Xselcol[i]), np.int(Xselrow[i])] = f(Yselrow[i],Yselcol[i])


plt.imshow(grayIm)
plt.title('Original image')
plt.grid(True)
plt.colorbar()
plt.show()

plt.imshow(zinterp)
plt.title('rotated image')
plt.grid(True)
plt.colorbar()
plt.show()


