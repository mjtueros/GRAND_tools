from __future__ import absolute_import
import numpy as np
from numpy import *
import astropy.units as u
import logging
import hdf5fileinout as hdf5io
import os,sys,inspect



#thetageo=147.43 *u.deg # deg, GRAND ->astropy.units
#phigeo=0.72*u.deg  # deg, GRAND ->astropy.units


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def get_antenna_pos_zhaires(pathfile):
    '''
    read in path to folder containing the inp-file, trace files and antpos.dat

    Parameters:
    pathfile: str
        complete path of antpos.dat

    Returns:
    number_ant: int
       number of antennas
    positions: numpy array
        x,y,z components of antenna positions in meters
    ID_ant: list
        corresponding antenna ID for identification

    '''

    # antpos.dat-file
    posfile = pathfile #
    mypos = np.genfromtxt(posfile) # positions[:,0]:along North-South, positions[:,1]: along East-West ,positions[:,2]: Up, in m

    ID_ant = mypos.T[0]
    x_pos = mypos.T[2]
    y_pos = mypos.T[3]
    z_pos = mypos.T[4]
    positions = np.stack((x_pos,y_pos,z_pos), axis=0)

    number_ant = len(x_pos) # number of positions in the array
    logging.debug('get_antenna_pos_zhaires: Number of antennas: %i' % number_ant)

    return number_ant, positions, ID_ant


def interpol_check_hdf5(InputFilename, positions, new_pos, p2pE, InterpolMethod, threshold=0,usetrace='efield', DISPLAY=False):
    '''
    Interpolates the signal peak-to-peak electric field at new antenna positions
    Check that the interpolation efficiency at 6 antenna positions available in each shower file

    Parameters:
    InputFilename: str
        HDF5File
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation (not used in trace interpolation method)
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout (at 6 check points)
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm

    threshold: float
        threshold abouve wich the interpolation is computed

    InterpolMethod: str
        interpolation method
        'lin' = linear interpolation from scipy.interpolate
        'rbf' = radial interpolation from scipy.interpolate
        'trace' = interpolation of signal traces:
            generates new interpolated trace files in path/Test/ directory

    DISPLAY: boolean
        if TRUE: 2D maps of peak-to-peak electric field
            at original and interpolated antennas are displayed

    Output:
    interp_err: numpy arrays
        interpolation error at each antenna (interpolated - original)/original
    p2p_total_new: numpy array
        peak-to-peak electric field at new antenna positions

    '''


    # interpolate (check rbf)
    logging.debug('interpol_check:Interpolating...'+str(usetrace))
    #print('Interpolating...'+path)

    number_ant = 160
    icheck = np.mgrid[160:176:1]

    myx_pos = positions[0,0:number_ant-1]
    myy_pos = positions[1,0:number_ant-1]
    myz_pos = positions[2,0:number_ant-1]
    mypositions = np.stack((myx_pos, myy_pos, myz_pos), axis=0)
    myp2p_total = p2pE[3,0:number_ant-1]

    if InterpolMethod == 'lin':
        from scipy.interpolate import griddata
        new_txt = griddata((mypositions[0,:],mypositions[1,:]), myp2p_total, (new_pos[0,:],new_pos[1,:]), method='linear')

        # flatten grids
        p2p_total_new = new_txt.flatten()


    if InterpolMethod == 'rad':
        from scipy.interpolate import Rbf
        rbfi = Rbf(mypositions[0,:],mypositions[1,:],myp2p_total,epsilon=1)
        p2p_total_new = rbfi(new_pos[0,:],new_pos[1,:])


    if InterpolMethod == 'trace':
        from trace_interpol_hdf5 import do_interpolation_hdf5
        OutputFilename = InputFilename + '.Interpolated.'+str(usetrace)+'.hdf5'

        #do_interpolation(AntPath,new_pos,mypositions,Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
        do_interpolation_hdf5(new_pos, InputFilename, OutputFilename, antennamin=0, antennamax=159,threshold=threshold, EventNumber=0,shower_core=np.array([0,0,2900]), DISPLAY=DISPLAY, usetrace=usetrace)

        #NewAntNum = size(new_pos)
        #NewAntNum, NewAntPos, NewAntID = get_antenna_pos_zhaires(NewAntPath)
        #NewP2pE = get_p2p(path+"/Test",NewAntNum)

        NewP2pE = hdf5io.get_p2p_hdf5(OutputFilename,antennamax=15,antennamin=0,usetrace=usetrace)

        p2p_total_new = NewP2pE[3,:]
        p2p_x_new = NewP2pE[0,:]
        p2p_y_new = NewP2pE[1,:]
        p2p_z_new = NewP2pE[2,:]

    # checking the interpolation efficiency
    interp_err = abs(p2p_total_new-p2pE[3,icheck])/p2pE[3,icheck]
    interp_errx = abs(p2p_x_new-p2pE[0,icheck])/p2pE[0,icheck]
    interp_erry = abs(p2p_y_new-p2pE[1,icheck])/p2pE[1,icheck]
    interp_errz = abs(p2p_z_new-p2pE[2,icheck])/p2pE[2,icheck]

    #print(np.shape(p2p_total_new))
    #print(np.shape(p2pE[3,icheck]))
    #print(p2pE[3,icheck])
    #print("interp_err = #{}".format(interp_err))


    if (DISPLAY and InterpolMethod!='trace'):
        logging.debug('interpol_check:Plotting...')

        ##### Plot 2d figures of total peak amplitude in positions along North-South and East-West
        fig1 = plt.figure(10,figsize=(5,7), dpi=100, facecolor='w', edgecolor='k')


        ax1=fig1.add_subplot(211)
        name = 'total'
        plt.title(name)
        ax1.set_xlabel('positions along NS (m)')
        ax1.set_ylabel('positions along EW (m)')
        col1=ax1.scatter(positions[0,:],positions[1,:], c=p2pE[3,:],  vmin=min(myp2p_total), vmax=max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
        plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
        plt.colorbar(col1)
        plt.tight_layout()


        ax2=fig1.add_subplot(212)
        name = 'total interpolated'
        plt.title(name)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=p2p_total_new,  vmin=np.min(myp2p_total), vmax=np.max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
        plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
        plt.colorbar(col2)
        plt.tight_layout()


        plt.show(block=False)


        if (interp_err.min() < 1.e-9):
            fig2 = plt.figure(figsize=(5,7), dpi=100, facecolor='w', edgecolor='k')


            ax1=fig2.add_subplot(211)
            name = 'total'
            plt.title(name)
            ax1.set_xlabel('positions along NS (m)')
            ax1.set_ylabel('positions along EW (m)')
            col1=ax1.scatter(positions[0,:],positions[1,:], c=p2pE[3,:],  vmin=min(myp2p_total), vmax=max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
            plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
            plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
            plt.colorbar(col1)
            plt.tight_layout()


            ax2=fig1.add_subplot(212)
            name = 'total interpolated'
            plt.title(name)
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('y (m)')
            col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=p2p_total_new,  vmin=np.min(myp2p_total), vmax=np.max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
            plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
            plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
            plt.colorbar(col2)
            plt.tight_layout()


            plt.show(block=False)



    return interp_err, p2p_total_new, interp_errx, p2p_x_new, interp_erry, p2p_y_new, interp_errz, p2p_z_new

def interpol_hdf5(InputFilename, OutputFilename,new_pos, p2pE=None,positions=None, InterpolMethod='trace', threshold=0,usetrace='efield', DISPLAY=False):

    '''
    Interpolates the signal peak-to-peak electric field
    at new antenna positions read at directory/new_antpos.dat

    Parameters:
    InputFilename: str

    path: str
        path of shower event
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation (not used in trace method)
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout
    p2pE: numpy array                                         (not used in trace method)
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm


    InterpolMethod: str
        interpolation method
        'lin' = linear interpolation from scipy.interpolate
        'rbf' = radial interpolation from scipy.interpolate
        'trace' = interpolation of signal traces:
            generates new interpolated trace files in path/Test/ directory
    threshold:
         signal above wich the trace interpolation is done (to speed up)
    usetrace:
         efield, voltage, filteredvoltage

    DISPLAY: boolean
        if TRUE: 2D maps of peak-to-peak electric field
            at original and interpolated antennas are displayed

    Output:
    p2p_total_new: numpy array
        peak-to-peak electric field at new antenna positions (in all but the trace method)
        peak to peak of the electric field components, and of the total

    '''

    # interpolate
    logging.debug('interpol:Interpolating...')

    number_ant = 160


    if InterpolMethod == 'trace':

        from trace_interpol_hdf5 import do_interpolation_hdf5
        #OutputFilename = InputFilename + '.Interpolated.'+str(usetrace)+'.hdf5'

        #do_interpolation(AntPath,new_pos,mypositions,Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
        do_interpolation_hdf5(new_pos, InputFilename, OutputFilename, antennamin=0, antennamax=159, EventNumber=0,threshold=threshold,shower_core=np.array([0,0,2900]), DISPLAY=DISPLAY, usetrace=usetrace)

        #NewAntNum = size(new_pos)
        #NewAntNum, NewAntPos, NewAntID = get_antenna_pos_zhaires(NewAntPath)
        #NewP2pE = get_p2p(path+"/Test",NewAntNum)

        NewP2pE = hdf5io.get_p2p_hdf5(OutputFilename,usetrace=usetrace)

        return NewP2pE
        #p2p_total_new = NewP2pE[3,:]

    if InterpolMethod == 'lin':
        myx_pos = positions[0,0:number_ant]
        myy_pos = positions[1,0:number_ant]
        myz_pos = positions[2,0:number_ant]
        mypositions = np.stack((myx_pos, myy_pos, myz_pos), axis=0)
        myp2p_total = p2pE[3,0:number_ant]
        from scipy.interpolate import griddata
        new_txt = griddata((mypositions[0,:],mypositions[1,:]), myp2p_total, (new_pos[0,:],new_pos[1,:]), method='linear')

        # flatten grids
        p2p_total_new = new_txt.flatten()


    if InterpolMethod == 'rad':
        myx_pos = positions[0,0:number_ant]
        myy_pos = positions[1,0:number_ant]
        myz_pos = positions[2,0:number_ant]
        mypositions = np.stack((myx_pos, myy_pos, myz_pos), axis=0)
        myp2p_total = p2pE[3,0:number_ant]
        from scipy.interpolate import Rbf
        rbfi = Rbf(mypositions[0,:],mypositions[1,:],myp2p_total,epsilon = 1)
        p2p_total_new = rbfi(new_pos[0,:],new_pos[1,:])


    if (DISPLAY and InterpolMethod!='trace'):
        logging.debug('interpol:Plotting...')

        ##### Plot 2d figures of total peak amplitude in positions along North-South and East-West
        fig1 = plt.figure(figsize=(5,7), dpi=100, facecolor='w', edgecolor='k')


        ax1=fig1.add_subplot(211)
        name = 'total'
        plt.title(name)
        ax1.set_xlabel('positions along NS (m)')
        ax1.set_ylabel('positions along EW (m)')
        col1=ax1.scatter(positions[0,:],positions[1,:], c=p2pE[3,:],  vmin=min(myp2p_total), vmax=max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
        plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
        plt.colorbar(col1)
        plt.tight_layout()


        ax2=fig1.add_subplot(212)
        name = 'total interpolated'
        plt.title(name)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=p2p_total_new,  vmin=np.min(myp2p_total), vmax=np.max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(new_pos[0,:]*1.1),max(new_pos[0,:]*1.1)))
        plt.ylim(-10000,10000)
        plt.colorbar(col2)
        plt.tight_layout()


        plt.show(block=False)

    return p2p_total_new



