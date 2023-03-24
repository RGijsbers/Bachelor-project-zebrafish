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
from pygltflib import GLTF2, Scene
from scipy.ndimage import zoom



# tifffile.askopenfilename()
# tifffile.astype('20190701--20000.tif')
# tifffile.create_output('20190701--20000.tif', shape=(512,512), dtype='unit16')

#zorg dat het werkt voor tifs van enige grote
#begin met zorgen voor tifs van enige diepte

im = tifffile.imread('/Users/robertgijsbers/Desktop/20190701--2/20190701--20119.tif')
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

# new_data = zoom(data, (0.5, 1, 1)) #the zoom try

old_set0 = data[0]
new_set0 = data[1]

old_set1 = data[1]
new_set1 = data[2]

old_set2 = data[2]
new_set2 = data[3]

old_set3 = data[3]
new_set3 = data[4]

old_set4 = data[4]
new_set4 = data[5]

old_set5 = data[5]
new_set5 = data[6]

old_set6 = data[6]
new_set6 = data[7]


average_set0 = np.mean( np.array([ old_set0, new_set0 ]), axis=0 ) # the average over 2 layers try
new_average_set0 = average_set0.reshape(1,512,512)

average_set1 = np.mean( np.array([ old_set1, new_set1 ]), axis=0 )
new_average_set1 = average_set1.reshape(1,512,512)

average_set2 = np.mean( np.array([ old_set2, new_set2 ]), axis=0 )
new_average_set2 = average_set2.reshape(1,512,512)

average_set3 = np.mean( np.array([ old_set3, new_set3 ]), axis=0 )
new_average_set3 = average_set3.reshape(1,512,512)

average_set4 = np.mean( np.array([ old_set4, new_set4 ]), axis=0 )
new_average_set4 = average_set4.reshape(1,512,512)

average_set4 = np.mean( np.array([ old_set4, new_set4 ]), axis=0 )
new_average_set4 = average_set4.reshape(1,512,512)

average_set5 = np.mean( np.array([ old_set5, new_set5 ]), axis=0 )
new_average_set5 = average_set5.reshape(1,512,512)

average_set6 = np.mean( np.array([ old_set6, new_set6 ]), axis=0 )
new_average_set6 = average_set6.reshape(1,512,512)


# print(np.shape(new_average_set0))
# print(type(average_set0))



# new_data=np.append(data, new_average_set, axis=0)

new_data = np.insert(data, 1, new_average_set0, axis=0)
new_data = np.insert(new_data, 3, new_average_set1, axis=0)
new_data = np.insert(new_data, 5, new_average_set2, axis=0)
new_data = np.insert(new_data, 7, new_average_set3, axis=0)
new_data = np.insert(new_data, 9, new_average_set4, axis=0)
new_data = np.insert(new_data, 11, new_average_set5, axis=0)
new_data = np.insert(new_data, 13, new_average_set6, axis=0)

print('nieuwe shape:' ,np.shape(new_data))
 
# # Create Data
# # data = np.ones(axes)
 
# # Control Tranperency
# alpha = 0.5
 
# # Control colour
# colors = np.empty(axes + [4])
 
# colors[0] = [1, 0, 0, alpha]  # red
# colors[1] = [0, 1, 0, alpha]  # green
# colors[2] = [0, 0, 1, alpha]  # blue
# colors[3] = [1, 1, 0, alpha]  # yellow
# colors[4] = [1, 1, 1, alpha]  # grey
 
# turn off/on axis
plt.axis('off')
 
# Voxels is used to customizations of
# the sizes, positions and colors.
# ax.voxels(data, facecolors=colors, edgecolors='grey')

ax.voxels(new_data)
# save = ax.voxels(data)

# plt.imsave('3Dtest.obj', save)
# GLTF2.save('3Dtest.gltf', save)
plt.show()

#find a way to save this object
#to try out
#https://plotly.com/python/visualizing-mri-volume-slices/

