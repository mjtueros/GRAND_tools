from __future__ import absolute_import
import numpy as np
from numpy import *
import astropy.units as u
import logging   
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



def get_p2p(path,number_ant):
    '''
    read in all .trace files located at path and output the peak to peak electric field and amplitude

    Parameters:
    path: str
        path to directory where trace files are located
    number_ant: int
       number of antennas

    Output:
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm  

    '''


    # create an array
    p2p_Ex = np.zeros(number_ant)
    p2p_Ey = np.zeros(number_ant)
    p2p_Ez = np.zeros(number_ant)
    p2p_total = np.zeros(number_ant)

    # read signal at antennas
    for i in range(0, number_ant): # loop over all antennas in folder
        try: 
            txt = np.loadtxt(path+ '/a'+str(i)+'.trace') 
            #txt.T[0]: time in ns, txt.T[1]: North-South, txt.T[2]: East-West, txt.T[3]: Up , all electric field in muV/m

            p2p_Ex[i] = max(txt.T[1])-min(txt.T[1])
            p2p_Ey[i] = max(txt.T[2])-min(txt.T[2])
            p2p_Ez[i] = max(txt.T[3])-min(txt.T[3])
            amplitude = np.sqrt(txt.T[1]**2. + txt.T[2]**2. + txt.T[3]**2.) # combined components
            p2p_total[i] = max(amplitude)-min(amplitude)

        except IOError:
            p2p_Ex[i]=0.
            p2p_Ey[i]=0.
            p2p_Ez[i]=0.
            p2p_total[i] = 0.

    p2pE = np.stack((p2p_Ex, p2p_Ey, p2p_Ez, p2p_total), axis=0)
    return p2pE






def interpol_check(path, positions, new_pos, p2pE, Zenith, Azimuth, InterpolMethod, DISPLAY):
    '''
    Interpolates the signal peak-to-peak electric field at new antenna positions
    Check that the interpolation efficiency at 6 antenna positions available in each shower file
    
    Parameters:
    path: str
        path of shower event
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout (at 6 check points)
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm  
    Zenith: float
        shower axis zenith
    Azimuth: float
        shower axis azimuth

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
    logging.debug('interpol_check:Interpolating...')
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
        from trace_interpol_check import do_interpolation
        NewAntPath = path + '/Test/new_antpos.dat'
        AntPath = path + '/antpos.dat'
        #do_interpolation(AntPath,new_pos,mypositions,Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
        do_interpolation(NewAntPath,AntPath,Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)

        #NewAntNum = size(new_pos)
        NewAntNum, NewAntPos, NewAntID = get_antenna_pos_zhaires(NewAntPath)
        NewP2pE = get_p2p(path+"/Test",NewAntNum)
        p2p_total_new = NewP2pE[3,:]
        

 
    if DISPLAY:
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


    # checking the interpolation efficiency
    interp_err = abs(p2p_total_new-p2pE[3,icheck])/p2pE[3,icheck]

    print("interp_err = #{}".format(interp_err))
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



    return interp_err, p2p_total_new


 
def interpol(directory, path, positions, new_pos, p2pE, Zenith, Azimuth, InterpolMethod, DISPLAY):

    '''
    Interpolates the signal peak-to-peak electric field 
    at new antenna positions read at directory/new_antpos.dat
    
    Parameters:
    directory: str
        path of root directory of shower library where new_antpos.dat is located
    path: str
        path of shower event
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout (at 6 check points)
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm  
    Zenith: float
        shower axis zenith
    Azimuth: float
        shower axis azimuth

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
    p2p_total_new: numpy array
        peak-to-peak electric field at new antenna positions

    '''

    # interpolate 
    logging.debug('interpol:Interpolating...')    

    number_ant = 160
    icheck = np.mgrid[160:176:1]

    myx_pos = positions[0,0:number_ant]
    myy_pos = positions[1,0:number_ant]
    myz_pos = positions[2,0:number_ant]
    mypositions = np.stack((myx_pos, myy_pos, myz_pos), axis=0)
    myp2p_total = p2pE[3,0:number_ant]

    if InterpolMethod == 'lin':
        from scipy.interpolate import griddata
        new_txt = griddata((mypositions[0,:],mypositions[1,:]), myp2p_total, (new_pos[0,:],new_pos[1,:]), method='linear')

        # flatten grids 
        p2p_total_new = new_txt.flatten()


    if InterpolMethod == 'rad':
        from scipy.interpolate import Rbf
        rbfi = Rbf(mypositions[0,:],mypositions[1,:],myp2p_total,epsilon = 1)
        p2p_total_new = rbfi(new_pos[0,:],new_pos[1,:])

 
    if InterpolMethod == 'trace':
        from trace_interpol import do_interpolation

        AntPath = path + '/antpos.dat'
        os.system('rm ' + path + '/Test/*.trace')

        do_interpolation(AntPath,new_pos.transpose(),mypositions.transpose(),Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)

        NewAntNum = size(new_pos)
        NewP2pE = get_p2p(path+"/Test",NewAntNum)
        p2p_total_new = NewP2pE[3,:]
        

    if DISPLAY:
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


 
