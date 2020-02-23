import grids 
import matplotlib.pyplot as plt
import numpy as np 


theta = 135  # degrees
theta_rad = theta / 180 * np.pi



pos2, offset2 = grids.create_grid_univ("hexhex", 100, angle = - theta, do_offset=True)
pos, offset = grids.create_grid_univ("hexhex", 100, angle = 0, do_offset=False)
ind = np.arange(0, 216)



plt.figure(3)
plt.clf()
plt.scatter(pos[0], pos[1], c=ind)
plt.plot(offset2[0], offset2[1], 'ro')
plt.plot([offset2[0], offset2[0]+ 300*np.cos(theta_rad)], [offset2[1], offset2[1] + 300*np.sin(theta_rad)], 'm-')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
cbar = plt.colorbar()
cbar.set_label('Antenna id')
plt.title('Real grid on ground')


plt.figure(4)
plt.clf()
plt.scatter(pos2[0], pos2[1], c=ind)
plt.axis('equal')
plt.plot(0, 0, 'ro')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
cbar = plt.colorbar()
cbar.set_label('Antenna id')
plt.plot([0,  300*np.cos(theta_rad*0)], [0,  300*np.sin(theta_rad*0)], 'm-')
plt.title('Rotated and offset grid for simulation')
    



pos2, offset2 = grids.create_grid_univ("rect", 100, angle = - theta, do_offset=True)
pos, offset = grids.create_grid_univ("rect", 100, angle = 0, do_offset=False)
ind = np.arange(0, 15*15)


plt.figure(1)
plt.clf()
plt.scatter(pos[0], pos[1], c=ind)
plt.plot(offset2[0], offset2[1], 'ro')
plt.plot([offset2[0], offset2[0]+ 300*np.cos(theta_rad)], [offset2[1], offset2[1] + 300*np.sin(theta_rad)], 'm-')
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
cbar = plt.colorbar()
cbar.set_label('Antenna id')
plt.title('Real grid on ground')


plt.figure(2)
plt.clf()
plt.scatter(pos2[0], pos2[1], c=ind)
plt.axis('equal')
plt.plot(0, 0, 'ro')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
cbar = plt.colorbar()
cbar.set_label('Antenna id')
plt.plot([0,  300*np.cos(theta_rad*0)], [0,  300*np.sin(theta_rad*0)], 'm-')
plt.title('Rotated and offset grid for simulation')
    




