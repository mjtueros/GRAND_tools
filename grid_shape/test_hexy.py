import hexy as hx
import numpy as np
import matplotlib.pyplot as plt

Nring = 5 # number of hex rings corresponding to 186 antennas
radius = 1000 #stepx
xcube = hx.get_spiral(np.array((0,0,0)), 0,Nring)
xpix = hx.cube_to_pixel(xcube, radius)
xcorn = hx.get_corners(xpix,radius)  

sh = np.array(xcorn).shape
xcorn2=xcorn.transpose(0,2,1) 
hexarray = np.array(xcorn2).reshape((sh[0]*sh[2],sh[1]))   

grid_x = hexarray[:,0]
grid_y = hexarray[:,1]


# remove redundant points
x_pos_flat_fl = grid_x #np.floor(grid_x)
y_pos_flat_fl = grid_y #np.floor(grid_y)
scal = (x_pos_flat_fl-x_pos_flat_fl.min()) + (x_pos_flat_fl.max()-x_pos_flat_fl.min())*(y_pos_flat_fl-y_pos_flat_fl.min())
scal = np.floor(scal)
unique, index = np.unique(scal, return_index=True)
x_pos_new = grid_x[index]
y_pos_new = grid_y[index]

Nant = x_pos_new.shape[0]
print('number of antennas', Nant)

fig, axs = plt.subplots(1,1) 
#axs.plot(hexarray[:,0], hexarray[:,1], 'k.')  
#axs.plot(x_pos_new, y_pos_new, 'k.')       
axs.axis('equal')  
plt.show()


Nrand = 100
randeff = 5
indrand = np.random.randint(0, high=Nant, size=Nrand)
x_pos_new[indrand] += np.random.randn(Nrand) * radius/randeff
y_pos_new[indrand] += np.random.randn(Nrand) * radius/randeff

axs.plot(x_pos_new[indrand], y_pos_new[indrand], 'r.')       
axs.plot(x_pos_new, y_pos_new, 'r.')       
plt.show()
