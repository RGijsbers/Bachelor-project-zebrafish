#here comes the code wich hopefully will make a mesh of the tiff files
import numpy as np
import tifffile
import matplotlib.pyplot as plt
# from skimage import exposure, io, util, measure
# from skimage import img_as_ubyte
import sys
import PIL
# import imageio
from PIL import Image
import pandas as pd
import os
from bitmap import BitMap
from mpl_toolkits.mplot3d import Axes3D


# tifffile.askopenfilename()
# tifffile.astype('20190701--20000.tif')
# tifffile.create_output('20190701--20000.tif', shape=(512,512), dtype='unit16')

#zorg dat het werkt voor tifs van enige grote
#begin met zorgen voor tifs van enige diepte

im = tifffile.imread('20190701--20119.tif') # add path to file here
# im.show()
# img = tifffile.imread('20190701--20000.obj')
# img_array= np.array(img)
# print(img_array)
# plt.imshow(img)

imarray = np.array(im)

z_counter = 0
y_counter = 0
x_counter = 0
for element in imarray:
    z_counter += 1
for element in imarray[1]:
    y_counter+=1
    x_counter = len(element)


data = np.zeros((z_counter,x_counter,y_counter), dtype=np.uint8)
counter = 0
# i=2

for i in range(8):
    for j in range(512):
        # print()
        for k in range(512):
            if (imarray[i][j][k]) != 0:
                # print(imarray[i][j][k])
                data[i][j][k] = imarray[i][j][k]
            # print(imarray[i][j][k], end='')

# plt.imshow(data, interpolation='nearest')
# plt.show()


fig = plt.figure(figsize=(10, 10))
 
# Generating a 3D sine wave
ax = plt.axes(projection='3d')
 
# Create axis
axes = [8, 512, 512]
 
# Create Data
# data = np.ones(axes)
 
# Control Tranperency
alpha = 0.5
 
# Control colour
colors = np.empty(axes + [4])
 
colors[0] = [1, 0, 0, alpha]  # red
colors[1] = [0, 1, 0, alpha]  # green
colors[2] = [0, 0, 1, alpha]  # blue
colors[3] = [1, 1, 0, alpha]  # yellow
colors[4] = [1, 1, 1, alpha]  # grey
 
# turn off/on axis
plt.axis('off')
 
# Voxels is used to customizations of
# the sizes, positions and colors.
# ax.voxels(data, facecolors=colors, edgecolors='grey')

ax.voxels(data)

plt.show()
#find a way to save this object
