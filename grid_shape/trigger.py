from __future__ import absolute_import
import numpy as np
from numpy import *
import astropy.units as u
import logging   #for...you guessed it...logging
import os,sys,inspect


#from __init__ import phigeo, thetageo
#from .__init__ import phigeo, thetageo

thetageo=147.43 *u.deg # deg, GRAND ->astropy.units
phigeo=0.72*u.deg  # deg, GRAND ->astropy.units


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 



def trig(path, pos, new_pos, new_p2pE, Zenith, Azimuth, EThres, DISPLAY):
    '''
    Interpolates the signal peak-to-peak electric field 
    at new antenna positions read at directory/new_antpos.dat
    
    Parameters:
    path: str
        path of shower event
    pos: numpy array
        x, y, z coordinates of the antennas in the simulation
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout (at 6 check points)
    new_p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm at new antenna positions
    Zenith: float
        shower axis zenith
    Azimuth: float
        shower axis azimuth
    Ethres: float
        threshold energy for interpolation

    DISPLAY: boolean
        if TRUE: 2D map of peak-to-peak electric field 
            at positions of triggered antennas are displayed
 
    Output:
    NT0: int
        number of triggered antennas
    indT0: 2-tuple
        indT0[0]: array of indices of triggered antennas

    '''

    
    indT0 = np.where(new_p2pE >= EThres) # get triggered antennas
    NT0 = np.size(indT0) # number of triggered antennas

    if DISPLAY:
        logging.debug('trig:Plotting...')

        ##### Plot 2d figures of total peak amplitude in positions along North-South and East-West 
        fig1 = plt.figure(figsize=(10,2), dpi=100, facecolor='w', edgecolor='k')
        ax2=fig1.add_subplot(111)
        name = 'total interpolated'
        plt.title(name)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=new_p2pE,  vmin=np.min(new_p2pE), vmax=np.max(new_p2pE),  marker='o', cmap=cm.gnuplot2_r)
        ax2.scatter(new_pos[0,indT0],new_pos[1,indT0], facecolors='none', edgecolors='k')
        plt.xlim((min(pos[0,:]),max(pos[0,:])))
        plt.ylim((min(pos[1,:]),max(pos[1,:])))
        plt.colorbar(col2)
        plt.tight_layout()


        plt.show(block=False)


    return NT0, indT0
