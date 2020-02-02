from __future__ import absolute_import
import numpy as np
from numpy import *
import astropy.units as u
import logging   
import os,sys,inspect
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 


def create_grid(path,positions,phi,GridShape,xNum,yNum):
    '''
    generate new positions of antennas with specific layout
    write positions in file path/Test/new_antpos.dat
    caution: in this routine, a new layout is generated for each shower

    Parameters:
    path: str
        path of directory where antpos.dat file is located
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation
    phi: float
        shower axis zenith
    GridShape: str
        shape of antenna grid
        'rect' = rectangular
        'hex' = hexagonal
    xNum, yNum: int
        number of antennas for new layout in x, y directions
        only xNum is used for hexagonal grids

    Output:
    new_pos: numpy arrays
        x, y, z coordinates of antenna in new layout
    

    '''

    x_pos = positions[0,:]
    y_pos = positions[1,:]
    z_pos = positions[2,:]

    z_site = 2900. # height of GP300 site in km
    icheck = np.mgrid[160:176:1]

    # geometrical parameters for grid
    minx = min(x_pos)
    maxx = max(x_pos)
    miny = min(y_pos)
    maxy = max(y_pos)

    stepx = (maxx-minx)/(xNum+1)
    stepy = (maxy-miny)/(yNum+1)
    hexradius = stepx # for hexagonal grid

    x_pos_new = x_pos
    y_pos_new = y_pos

    if GridShape == 'check':
       # points at the end of the antpos.dat file 
       logging.debug('create_grid: Interpolation check...')
       x_pos_new = x_pos[icheck]
       y_pos_new = y_pos[icheck]

    if GridShape == 'rect':
        # create rectangular grid 
        logging.debug('create_grid: Generating rectangular grid...')
        grid_x, grid_y = np.mgrid[minx:maxx:(xNum*1j), miny:maxy:(yNum*1j)]
        
        # flatten grids 
        x_pos_new = grid_x.flatten() 
        y_pos_new = grid_y.flatten() 

    
    if GridShape == 'hexrect':
        # create a hexagonal grid fitting into rectangle
        logging.debug('create_grid:Generating hexagonal grid over rectangle...')
        import hexagon as hex
        hexlist = hex.calculate_polygons(minx,miny,maxx,maxy,hexradius)
        sh = array(hexlist).shape
        hexarray = array(hexlist).reshape(sh[0]*sh[1],sh[2]) 
        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]

        # flatten grids 
        x_pos_flat = grid_x.flatten() 
        y_pos_flat = grid_y.flatten() 

        # remove redundant points
        x_pos_flat_fl = np.floor(x_pos_flat)
        y_pos_flat_fl = np.floor(y_pos_flat)
        scal = (x_pos_flat_fl-x_pos_flat_fl.min()) + (x_pos_flat_fl.max()-x_pos_flat_fl.min())*(y_pos_flat_fl-y_pos_flat_fl.min())
        unique, index = np.unique(scal, return_index=True)
        x_pos_new = x_pos_flat[index]
        y_pos_new = y_pos_flat[index]

        # remove points outside of border
        ind = where(x_pos_new > maxx)
        x_pos_new=np.delete(x_pos_new,ind)
        y_pos_new=np.delete(y_pos_new,ind)
        ind = where(x_pos_new < minx)
        x_pos_new=np.delete(x_pos_new,ind)
        y_pos_new=np.delete(y_pos_new,ind)

        ind = where(y_pos_new > maxy)
        x_pos_new=np.delete(x_pos_new,ind)
        y_pos_new=np.delete(y_pos_new,ind)
        ind = where(y_pos_new < miny)
        x_pos_new=np.delete(x_pos_new,ind)
        y_pos_new=np.delete(y_pos_new,ind)


    if GridShape == 'hexhex':
        # create a hexagonal grid with overall hexagonal layout
        logging.debug('create_grid:Generating hexagonal grid in hex layout...')
        import hexy as hx

        Nring = 5 # number of hex rings corresponding to 186 antennas
        radius = 1000 # radius of hex in m
        xcube = hx.get_spiral(np.array((0,0,0)), 0,Nring)
        xpix = hx.cube_to_pixel(xcube, radius)
        xcorn = hx.get_corners(xpix,radius)  
        
        sh = np.array(xcorn).shape
        xcorn2=xcorn.transpose(0,2,1) 
        hexarray = np.array(xcorn2).reshape((sh[0]*sh[2],sh[1]))   

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]


        # remove redundant points
        x_pos_flat_fl = grid_x 
        y_pos_flat_fl = grid_y 
        scal = (x_pos_flat_fl-x_pos_flat_fl.min()) + (x_pos_flat_fl.max()-x_pos_flat_fl.min())*(y_pos_flat_fl-y_pos_flat_fl.min())
        scal = np.floor(scal)
        unique, index = np.unique(scal, return_index=True)
        x_pos_new = grid_x[index]
        y_pos_new = grid_y[index]


    # for now set position of z to site altitude
    z_pos_new = x_pos_new*0 + z_site

    # create new position array
    new_pos = np.stack((x_pos_new,y_pos_new,z_pos_new), axis=0)

    # write new antenna position file
    logging.debug('create_grid: Writing in file '+ path +'/Test/new_antpos.dat...')
    os.makedirs(os.path.join(path,'Test'),exist_ok=True)
    FILE = open(path+ '/Test/new_antpos.dat',"w+" )
    for i in range( 1, len(x_pos_new)+1 ):
        print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
    FILE.close()

    
    return new_pos


def create_grid_univ(directory,GridShape,radius, Nrand=None, randeff=None, DISPLAY=False):
    '''
    generate new positions of antennas with universal layout 
    write positions in file directory/new_antpos.dat
    should be called outside of database reading loop

    Parameters:
    directory: str
        path of root directory of shower library
    GridShape: str
        shape of antenna grid
        'rect' = rectangles tiled over rectangular shape
        'hexhex' = hexagons tiled in overall hexagonal shape
        'hexrand' = hexagons tiled in overal hexagonal shape with Nrand randomly displaced antennas
    radius: float
        radius of hexagon in m
    Nrand: int
        for hexrand option: number of randomly displaced antennas
    randeff: 
        for hexrand option: antennas are displaced following a normal law 
        centered on 0 and of sigma radius/randeff  

    Output:
    new_pos: numpy arrays
        x, y, z coordinates of antenna in new layout
    

    '''

    z_site = 2900. # height of GP300 site in km


    if GridShape == 'rect':
        # create rectangular grid 
        logging.debug('create_grid: Generating rectangular grid...')
        
        xNum = 20
        yNum = 10
      
        grid_x, grid_y = mgrid[0:xNum*radius:radius, 0:yNum*radius:radius]

        grid_x = grid_x - (xNum-1)*radius/2
        grid_y = grid_y - (yNum-1)*radius/2
        
        # flatten grids 
        x_pos_new = grid_x.flatten() 
        y_pos_new = grid_y.flatten() 

    
        # write new antenna position file
        logging.debug('create_grid: Writing in file '+ directory +'/new_antpos_rect_%d.dat...'%radius)
        FILE = open(directory+ '/new_antpos_rect_%d.dat'%radius,"w+" )
        for i in range( 1, len(x_pos_new)+1 ):
            print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
        FILE.close()


    if GridShape == 'hexhex':
        # create a hexagonal grid with overall hexagonal layout
        logging.debug('create_grid:Generating hexagonal grid in hex layout...')
        import hexy as hx

        Nring = 5 # number of hex rings corresponding to 186 antennas
        xcube = hx.get_spiral(np.array((0,0,0)), 0,Nring)
        xpix = hx.cube_to_pixel(xcube, radius)
        xcorn = hx.get_corners(xpix,radius)  
        
        sh = np.array(xcorn).shape
        xcorn2=xcorn.transpose(0,2,1) 
        hexarray = np.array(xcorn2).reshape((sh[0]*sh[2],sh[1]))   

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]


        # remove redundant points
        x_pos_flat_fl = grid_x 
        y_pos_flat_fl = grid_y 
        scal = (x_pos_flat_fl-x_pos_flat_fl.min()) + (x_pos_flat_fl.max()-x_pos_flat_fl.min())*(y_pos_flat_fl-y_pos_flat_fl.min())
        scal = np.floor(scal)
        unique, index = np.unique(scal, return_index=True)
        x_pos_new = grid_x[index]
        y_pos_new = grid_y[index]

        # write new antenna position file
        logging.debug('create_grid: Writing in file '+ directory +'/new_antpos_hex_%d.dat...'%radius)
        FILE = open(directory+ '/new_antpos_hex_%d.dat'%radius,"w+" )
        for i in range( 1, len(x_pos_new)+1 ):
            print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
        FILE.close()




    if GridShape == 'hexrand':
        # create a hexagonal grid with overall hexagonal layout
        logging.debug('create_grid:Generating hexagonal grid in hex layout with random displacements...')
        import hexy as hx

        Nring = 5 # number of hex rings corresponding to 186 antennas
        xcube = hx.get_spiral(np.array((0,0,0)), 0,Nring)
        xpix = hx.cube_to_pixel(xcube, radius)
        xcorn = hx.get_corners(xpix,radius)  
        
        sh = np.array(xcorn).shape
        xcorn2=xcorn.transpose(0,2,1) 
        hexarray = np.array(xcorn2).reshape((sh[0]*sh[2],sh[1]))   

        grid_x = hexarray[:,0]
        grid_y = hexarray[:,1]


        # remove redundant points
        x_pos_flat_fl = grid_x 
        y_pos_flat_fl = grid_y 
        scal = (x_pos_flat_fl-x_pos_flat_fl.min()) + (x_pos_flat_fl.max()-x_pos_flat_fl.min())*(y_pos_flat_fl-y_pos_flat_fl.min())
        scal = np.floor(scal)
        unique, index = np.unique(scal, return_index=True)
        x_pos_new = grid_x[index]
        y_pos_new = grid_y[index]

        # displace Nrand antennas randomly
        Nant = x_pos_new.shape[0]
        indrand = np.random.randint(0, high=Nant, size=Nrand)
        x_pos_new[indrand] += np.random.randn(Nrand) * radius*randeff
        y_pos_new[indrand] += np.random.randn(Nrand) * radius*randeff

        # write new antenna position file
  #      logging.debug('create_grid: Writing in file '+ directory +'/new_antpos_hexrand.dat...')
  #      FILE = open(directory+ '/new_antpos_hexrand.dat',"w+" )
  #      for i in range( 1, len(x_pos_new)+1 ):
  #          print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
   #     FILE.close()



    if DISPLAY:
        fig, axs = plt.subplots(1,1) 
        axs.plot(x_pos_new, y_pos_new, 'k.')       
        axs.axis('equal')  
        plt.show()


    # for now set position of z to site altitude
    z_pos_new = x_pos_new*0 + z_site

    # create new position array
    new_pos = np.stack((x_pos_new,y_pos_new,z_pos_new), axis=0)


    # write new antenna position file
    logging.debug('create_grid: Writing in file '+ directory +'/new_antpos.dat...')
    FILE = open(directory+ '/new_antpos.dat',"w+" )
    for i in range( 1, len(x_pos_new)+1 ):
        print("%i A%i %1.5e %1.5e %1.5e" % (i,i-1,x_pos_new[i-1],y_pos_new[i-1],z_site), end='\n', file=FILE)
    FILE.close()



    return new_pos

