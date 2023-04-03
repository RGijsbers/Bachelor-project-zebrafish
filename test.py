import vedo
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from vedo import Volume, show

# vol = vedo.Volume('/Users/robertgijsbers/Desktop/20190701--2/20190701--20119.tif').print()

im = tifffile.imread('/Users/robertgijsbers/Desktop/20190701--2/20190701--20119.tif')
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
for i in range(8):
    for j in range(512):
        # print()
        for k in range(512):
            if (imarray[i][j][k]) != 0:
                # print(imarray[i][j][k])
                data[i][j][k] = imarray[i][j][k]

fig = plt.figure(figsize=(10, 10))
 
# Generating in 3D
ax = plt.axes(projection='3d', autoscale_on = 1)
 
# Create axis
axes = [8, 512, 512]

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

new_data = np.insert(data, 1, new_average_set0, axis=0)
new_data = np.insert(new_data, 3, new_average_set1, axis=0)
new_data = np.insert(new_data, 5, new_average_set2, axis=0)
new_data = np.insert(new_data, 7, new_average_set3, axis=0)
new_data = np.insert(new_data, 9, new_average_set4, axis=0)
new_data = np.insert(new_data, 11, new_average_set5, axis=0)
new_data = np.insert(new_data, 13, new_average_set6, axis=0)

vol = Volume(new_data, c=['white','b','g','r'], mode=1)
vol.add_scalarbar3d()


# optionally mask some parts of the volume (needs mapper='gpu'):
# data_mask = np.zeros_like(data_matrix)
# data_mask[10:65, 10:65, 20:75] = 1
# vol.mask(data_mask)

# vedo.write(objct=vol, fileoutput='/Users/robertgijsbers/Desktop/test.obj')


show(vol, __doc__, axes=1).close()


# iso = vol.isosurface(35)
# iso.write("iso.obj")
# vedo.show(vol, iso, N=2, axes=2)