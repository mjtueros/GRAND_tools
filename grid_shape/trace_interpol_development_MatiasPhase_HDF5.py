'''Script to perform an interpolation between to electric field traces at a desired position
TODO: use magnetic field values and shower core from config-file
'''
import numpy as np
from scipy import signal
from utils import getn
import operator
import logging
import os

from os.path import split
import sys

from frame import UVWGetter
from io_utils import load_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import hdf5fileinout as hdf5io

import astropy.units as u
#from __init__ import phigeo, thetageo
thetageo=147.43 *u.deg # deg, GRAND ->astropy.units
phigeo=0.72*u.deg  # deg, GRAND ->astropy.units


##no longer in use, i dont unwrap the phase any more
#======================================
def unwrap(phi, ontrue=None):
    """Unwrap the phase so that the absolute difference
      between 2 consecutive phases remains below Pi

    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None

    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace

    Adapted by E. Hivon (2020-02) from A. Zilles' unwrap
    """
    eps = np.finfo(np.pi).resolution
    thr = np.pi - eps
    pi2 = 2. * np.pi
    phi_unwrapped = np.zeros(phi.shape)
    p0  = phi_unwrapped[0] = phi[0]
    l   = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        dp = p1 - p0
        if (np.abs(dp) > thr):
            dl = np.floor_divide(abs(dp), pi2) + 1
            if (dp > 0):
                l -= dl
            else:
                l += dl
        phi_unwrapped[i] = p1 + l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i],           phi[i-1],           abs(phi[i] - phi[i-1]),
                  l, phi_unwrapped[i], phi_unwrapped[i-1], abs(phi_unwrapped[i] - phi_unwrapped[i-1]))

    return phi_unwrapped
#======================================



def interpolate_trace(t1, trace1, x1, t2, trace2, x2, xdes, upsampling=None,  zeroadding=None, ontrue=None, flow=60.e6, fhigh=200.e6):
    """Interpolation of signal traces at the specific position in the frequency domain

    The interpolation of traces needs as input antenna position 1 and 2, their traces (filtered or not)
    in one component, their time, and the desired antenna position and returns the trace ( in x,y,z coordinate system) and the time from the desired antenna position.
    Zeroadding and upsampling of the signal are optional functions.

    IMPORTANT NOTE:
    The interpolation of the phases includes the interpolation of the signal arrival time. A linear interpolation implies a plane radio
    emission wave front, which is a simplification as it is hyperbolic in shape. However, the wave front can be estimated as a plane between two simulated observer positions
    for a sufficiently dense grid of observers, as then parts of the wave front are linear on small scales.

    This script bases on the diploma thesis of Ewa Holt (KIT, 2013) in the context of AERA/AUGER. It is based on the interpolation of the amplitude and the pahse in the frequency domain.
    This can lead to misidentifiying of the correct phase. We are working on the interplementaion on a more robust interpolation of the signal time.
    Feel free to include it if you have some time to work on it. The script is completely modular so that single parts can be substitute easily.


    Parameters:
    ----------
            t1: numpy array, float
                time in ns of antenna 1
            trace1: numpy array, float
                single component of the electric field's amplitude of antenna 1
            x1: numpy array, float
                position of antenna 1
            t2: numpy array, float
                time in ns of antenna 2
            trace2: numpy array, float
                single component of the electric field's amplitude of antenna 2
            x2: numpy array, float
                position of antenna 2
            xdes: numpy arry, float
                antenna position for which trace is desired, in meters
            upsampling: str
                optional, True/False, performs upsampling of the signal, by a factor 8
            zeroadding: str
                optional, True/False, adds zeros at the end of the trace of needed
            ontrue: str
                optional, True/False, just a plotting command
            flow: float
                lower frequency - optional, define the frequency range for plotting, if desired (DISPLAY=True/False)
            fhigh: float
                higher frequency - optional, define the frequency range for plotting, if desired (DISPLAY=True/False)

    Returns:
    ----------
        xnew: numpy array, float
            time for signal at desired antenna position in ns
        tracedes: numpy array, float
            interpolated electric field component at desired antenna position
    """
    DISPLAY = False
    #ontrue=True

    # hand over time traces of one efield component -t1=time, trace1=efield- and the position
    # x1 of the first antenna, the same for the second antenna t2,trace2, x2.
    # xdes is the desired antenna position (m) where you would like to have the efield trace in time
    # if necessary you have to do an upsampling of the trace: upsampling=On
    # onTrue=On would give you printings to the terminal to check for correctness
    # flow= lower freq in Hz, fhigh=higher freq in Hz, not necessarily needed

    factor_upsampling = 1
    if upsampling is not None:
        factor_upsampling = 8
    c = 299792458.e-9  # m/ns

    # calculating weights: should be done with the xyz coordinates
    # since in star shape pattern it is mor a radial function connection the poistion of
    # same signal as linear go for that solution.
    # if lines ar on a line, it will give the same result as before
    tmp1 = np.linalg.norm(x1 - xdes)
    tmp2 = np.linalg.norm(x2 - xdes)

    tmp = 1. / (tmp1 + tmp2)
    weight1 = tmp2 * tmp
    weight2 = tmp1 * tmp

    if np.isinf(weight1):
        print("weight = inf")
        print(x1, x2, xdes)
        weight1 = 1.
        weight2 = 0.
    if np.isnan(weight1):
        print('Attention: projected positions equivalent')
        weight1 = 1.
        weight2 = 0.
    epsilon = np.finfo(float).eps
    if (weight1 > 1. + epsilon) or (weight2 > 1 + epsilon):
        print("weight larger 1: ", weight1, weight2, x1, x2, xdes, np.linalg.norm(
            x2-x1), np.linalg.norm(x2-xdes), np.linalg.norm(xdes-x1))
    if weight1 + weight2 > 1 + epsilon:
        print("PulseShape_Interpolation.py: order in simulated positions. Check whether ring or ray structure formed first")
        print(weight1, weight2, weight1 + weight2)


    #################################################################################
    # Fourier Transforms

    # first antenna
    # upsampling if necessary
    if upsampling is not None:
        trace1 = signal.resample(trace1, len(trace1)*factor_upsampling)
        t1 = np.linspace(t1[0], t1[-1], len(trace1)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        max_element = len(trace1)  # to shorten the array after zeroadding
        print(max_element)
        xnew = np.linspace(t1[0], 1.01*t1[-1],
                              int((1.01*t1[-1]-t1[0])/(t1[2]-t1[1])))
        print(len(xnew))
        xnew = xnew*1.e-9  # ns -> s
        zeros = np.zeros(len(xnew)-max_element)
        f = trace1
        f = np.hstack([f, zeros])

    if zeroadding is None:
        f = trace1
        xnew = t1*1.e-9

    fsample = 1./((xnew[1]-xnew[0]))  # Hz

    freq = np.fft.rfftfreq(len(xnew), 1./fsample)
    FFT_Ey = np.fft.rfft(f)

    Amp = np.abs(FFT_Ey)
    phi = np.angle(FFT_Ey)
    #Eric
    phi_unwrapped = unwrap(phi, ontrue)

    #############################

    # second antenna
    # t in ns, Ex in muV/m, Ey, Ez
    # NOTE: Time binning always 1ns

    # upsampling if needed
    if upsampling is not None:
        trace = signal.resample(trace2, len(trace2)*factor_upsampling)
        trace2 = trace
        t2 = np.linspace(t2[0], t2[-1], len(trace2)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        # get the same length as xnew
        xnew2 = np.linspace(
            t2[0], t2[0] + (xnew[-1]-xnew[0])*1e9, len(xnew))
        xnew2 = xnew2*1.e-9
        f2 = trace2
        f2 = np.hstack([f2, zeros])

    if zeroadding is None:
        f2 = trace2
        xnew2 = t2*1e-9  # ns -> s
    fsample2 = 1./((xnew2[1]-xnew2[0]))  # *1.e-9 to get time in s

    freq2 = np.fft.rfftfreq(len(xnew2), 1./fsample2)
    FFT_Ey = np.fft.rfft(f2)

    Amp2 = np.abs(FFT_Ey)
    phi2 = np.angle(FFT_Ey)
    #Eric
    phi2_unwrapped = unwrap(phi2, ontrue)

    ### Get the pulse sahpe at the desired antenna position

    # get the phase

    #Matias way of doing the phi interpolation.

    #The phases are always between -pi and pi , so there is a discontinuity when you go beyond -pi or pi, you "jump" by 2pi
    #we want to interpolate always in the "closest" direction . This means that if the diference between both phases is bigger than pi, we should add 2*pi to both phases, interpolate, and
    phides=np.zeros(phi.shape)

    #for i in range(0,len(phi)-1):
    #  #print(i)
    #  if(np.abs(phi[i]-phi2[i])>np.pi):
    #    if(phi[i]<phi2[i]):
    #      phides[i]= (weight1 * (phi[i]+2.0*np.pi) + weight2 * phi2[i]) - 2.0*np.pi
    #      #print("1-"+str(weight1)+" phi1:"+str(phi[i]/np.pi)+" |"+str(weight2)+" phi2:"+str(phi2[i]/np.pi)+"phides:"+str(phides[i]/np.pi))
    #    else:
    #      phides[i]= (weight1 * phi[i] + weight2 * (phi2[i]+2*np.pi)) - 2.0*np.pi
    #      #print("2-"+str(weight1)+" phi1:"+str(phi[i]/np.pi)+" |"+str(weight2)+" phi2:"+str(phi2[i]/np.pi)+"phides:"+str(phides[i]/np.pi))
    #  else:
    #    phides[i] = weight1 * phi[i] + weight2 * phi2[i]
    #    #print("3-"+str(weight1)+" phi1:"+str(phi[i]/np.pi)+" |"+str(weight2)+" phi2:"+str(phi2[i]/np.pi)+"phides:"+str(phides[i]/np.pi))
    #  # get to [-pi,+pi} range
    #  if(phides[i]>np.pi):
    #    phides[i]=phides[i]-2*np.pi
    #  elif(phides[i]<-np.pi):
    #    phides[i]=phides[i]+2*np.pi

    # getnp.zeros([len(phi2)]) the angle for the desired position
    #Eric
    phides = weight1 * phi_unwrapped + weight2 * phi2_unwrapped

    if ontrue is not None:
        print(phides)
    if DISPLAY:
        phides2 = phides.copy()

    #Eric re-unwrap: get -pi to +pi range back and check whether phidesis in between (im not wraping any more)
    phides = np.mod(phides + np.pi, 2. * np.pi) - np.pi

    #################################################################################
    ### linearly interpolation of the amplitude

    #Amp, Amp2
    # Since the amplitude shows a continuous unipolar shape, a linear interpolation is sufficient

    Ampdes = weight1 * Amp + weight2 * Amp2
    if DISPLAY:
        Ampdes2 = Ampdes.copy()

    # inverse FFT for the signal at the desired position
    Ampdes = Ampdes.astype(np.complex64)
    phides = phides.astype(np.complex64)
    if DISPLAY:
        phides2 = phides2.astype(np.complex64)
    Ampdes *= np.exp(1j * phides)

    tracedes = (np.fft.irfft(Ampdes))
    tracedes = tracedes.astype(float)

    xdes=(xnew*weight1+xnew2*weight2)
    #print("weights 1:"+str(weight1)+ " 2:"+str(weight2))


    # PLOTTING

    if (DISPLAY):
        import matplotlib.pyplot as plt
        import pylab

        fig1 = plt.figure(1, dpi=120, facecolor='w', edgecolor='k')
        plt.subplot(211)
        plt.plot(freq, phi, 'ro-', label="first")
        plt.plot(freq2, phi2, 'bo-', label="second")
        plt.plot(freq2, phides, 'go--', label="interpolated")
        #plt.plot(freq2, phi_test, 'co--', label= "real")
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"phase (rad)", fontsize=16)
        #plt.xlim(flow, fhigh)

        #pylab.legend(loc='upper left')

        #plt.subplot(312)
        #ax = fig1.add_subplot(3, 1, 2)
        #plt.plot(freq, phi_unwrapped, 'r+')
        #plt.plot(freq2, phi2_unwrapped, 'bx')
        #plt.plot(freq2, phides2, 'g^')
        #plt.plot(freq2, phi_test_unwrapped, 'c^')
        #plt.xlabel(r"Frequency (Hz)", fontsize=16)
        #plt.ylabel(r"phase (rad)", fontsize=16)
        # plt.show()
        # plt.xlim([0,0.1e8])
        # plt.xlim([1e8,2e8])
        # plt.ylim([-10,10])
        # ax.set_xscale('log')
        #plt.xlim(flow, fhigh)

        plt.subplot(212)
        plt.plot(freq, Amp, 'r+')
        plt.plot(freq2, Amp2, 'bx')
        plt.plot(freq2, Ampdes2, 'g^')
        #plt.plot(freq2, Amp_test, 'c^')
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"Amplitude muV/m/Hz ", fontsize=16)
        #print("Min Amplitude: " + str(np.min(Amp)) + " Amplitude 2: " + str(np.min(Amp2)))
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        #plt.ylim([1e1, 10e3])
        #plt.xlim(flow, fhigh)

        plt.show()

################################## CONTROL

    if DISPLAY and False :
        ##### PLOTTING

        import matplotlib.pyplot as plt
        plt.plot(np.real(t1), np.real(trace1), 'g:', label= "antenna 1")
        plt.plot(np.real(t2), np.real(trace2), 'b:', label= "antenna 2")
        plt.plot(np.real(xdes*1e9), np.real(tracedes), 'r-', label= "desired")

        plt.xlabel(r"time (ns)", fontsize=16)
        plt.ylabel(r"Amplitude muV/m ", fontsize=16)
        plt.legend(loc='best')

        plt.show()


#################################



    if zeroadding is True:
        # hand over time of first antenna since interpolation refers to that time
        return xdes[0:max_element]*1.e9, tracedes[0:max_element]

    if upsampling is not None:
        return xdes[0:-1:8]*1.e9, tracedes[0:-1:8]
    else:
        #xnew = np.delete(xnew, -1)
        return xdes*1.e9, tracedes  # back to ns



#im not using this any more, since i chose the interpolation points radially
def _ProjectPointOnLine(a, b, p):
    ''' Helper function
    line defined by vector a and b, project othogonally vector p to it
    '''
    ap = p-a
    ab = b-a
    nrm = np.dot(ab,ab)
    if nrm <= 0.:
        print(a, b)
    point = a + np.dot(ap,ab) / nrm * ab
    return point


#################################


def do_interpolation_hdf5(desired, InputFilename, OutputFilename, antennamin=0, antennamax=159, threshold=0, EventNumber=0,shower_core=np.array([0,0,0]), DISPLAY=False, usetrace='efield'):
    '''
    Reads in arrays, looks for neigbours, calls the interpolation and saves the traces

    Parameters:
    ----------
    desired: str
        numpy array of desired antenna positions (x,y,y info)
    InputFilename: str
        path to HDF5 simulation file
        The script accepts starshape as well as grid arrays
    antennamin,antennamax:int
        the program is designed to tun on 160 antennas. If your simulation has more, you can specify a range to be used
    threshold :float
        minumum value of p2p in at least one channel of the 4 neighbor antennas to proceed with interpolation (8.66 guarantees minumum p2p of 15 in interpolated (15/sqrt3))
    EventNumber: int
        number of event in the file to use. you can process only one at a time
    shower_core: numpy array
        position of shower core for correction (NOT TESTED, CHECK BEFORE USING IT!) In ZHAireS, its the altitude of the simulation, but i dont really get why.
    DISPLAY: True/False
        enables printouts and plots
    usetrace: str (note that for now you can only do one at a time, and on different output files)
        efield
        voltage
        filteredvoltage

    Returns:
    ----------
        --
    Saves traces via index infomation in same folder as desired antenna positions


    NOTE: The selection of the neigbours is sufficiently stable, but does not always pick the "best" neigbour, still looking for an idea
    TODO: Read-in and save only hdf5 files
    '''
    #print(shower_core)
    DEVELOPMENT=False


    antennamax=antennamax+1
    CurrentEventNumber=EventNumber

    CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
    CurrentNumberOfEvents=hdf5io.GetNumberOfEvents(CurrentRunInfo)
    CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,CurrentEventNumber)

    Zenith=hdf5io.GetEventZenith(CurrentRunInfo,0)
    Azimuth=hdf5io.GetEventAzimuth(CurrentRunInfo,0)

    CurrentEventInfo=hdf5io.GetEventInfo(InputFilename,CurrentEventName)
    PhiGeo= hdf5io.GetEventBFieldIncl(CurrentEventInfo) + 90.0 #adjust to GRAND coordinates.
    ThetaGeo= hdf5io.GetEventBFieldDecl(CurrentEventInfo)

    # SIMULATION
    # Read in simulated position list
    CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)
    #one way of putting the antenna information as the numpy array this script was designed to use:
    xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
    ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
    zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
    positions_sims=np.column_stack((xpoints,ypoints,zpoints))

    # DESIRED
    # Hand over a list file including the antenna positions you would like to have. This could be improved by including an ID.
    positions_des = desired #np.loadtxt(desired,usecols=(2,3,4))

    if DISPLAY:
        print('desired positions: '+ str(len(positions_sims)))
        #print(positions_des, len(positions_des))
    if len(positions_des) <=1:
        print("Files of desired positions has to consist of at least two positions, Bug to be fixed")

    if DISPLAY:
        print('simulated positions: ' + str(len(positions_sims)))
        #print(positions_sims, len(positions_sims))
    if len(positions_sims) <=1:
        print("Files of simulated positions has to consist of at least two positions, Bug to be fixed")

    #write the output file headers
    hdf5io.SaveRunInfo(OutputFilename,CurrentRunInfo)
    hdf5io.SaveEventInfo(OutputFilename,CurrentEventInfo,CurrentEventName)
    #not using them, but i could put SignalSim And ShowerSim Info
    #making the table of desired antennas for the file
    DesiredAntennaInfoMeta=hdf5io.CreatAntennaInfoMeta(split(InputFilename)[1],CurrentEventName,AntennaModel="Interpolated")
    DesiredIds=np.arange(0, len(positions_des)) #this could be taken from the input file of desired antennas
    DesiredAntx=positions_des.T[0]
    DesiredAnty=positions_des.T[1]
    DesiredAntz=positions_des.T[2]
    DesiredSlopeA=np.zeros(len(positions_des))
    DesiredSlopeB=np.zeros(len(positions_des))
    DesiredAntennaInfo=hdf5io.CreateAntennaInfo(DesiredIds, DesiredAntx, DesiredAnty, DesiredAntz, DesiredSlopeA, DesiredSlopeB, DesiredAntennaInfoMeta)
    hdf5io.SaveAntennaInfo(OutputFilename,DesiredAntennaInfo,CurrentEventName)
    #here i could save other simulation. For now, i save a copy. I could modify some fields to show this is an interpolation
    CurrentShowerSimInfo=hdf5io.GetShowerSimInfo(InputFilename,CurrentEventName)
    hdf5io.SaveShowerSimInfo(OutputFilename,CurrentShowerSimInfo,CurrentEventName)
    CurrentSignalSimInfo=hdf5io.GetSignalSimInfo(InputFilename,CurrentEventName)
    hdf5io.SaveSignalSimInfo(OutputFilename,CurrentShowerSimInfo,CurrentEventName)


    print("Warning: this routine is hardwired for a starshape pattern of 160 antennas.Check that is your case!")
    # SELECTION: For interpolation only select the desired position which are "in" the plane of simulated antenna positions
    a = positions_sims[0]-positions_sims[10]  #Why this 10, and why that 80?
    a = a/np.linalg.norm(a)
    b = positions_sims[0]-positions_sims[-1]
    b = b/np.linalg.norm(b)
    if(a==b).all():
        a = positions_sims[0]-positions_sims[80]
        a = a/np.linalg.norm(a)
    n=np.cross(a,b)
    n = n/np.linalg.norm(n)

    # test wether desired points are in_plane, needed assumption for interpolation
    ind=[]
    for i in np.arange(0,len(positions_des[:,1])):
        if (np.dot(positions_sims[0]- positions_des[i], n) ==0. ):
            ind.append(i)
        else:
          print("desired positions must be in-plane: if you see this stop and take the time to understand what this code does. not tested.", ind)

    #------------------------
    if DISPLAY:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #ax = fig.gca(projection='3d')

        #ax.scatter(positions_sims[:,0], positions_sims[:,1], positions_sims[:,2], label = "simulated")
        #ax.scatter(positions[:,0], positions[:,1], positions[:,2], label = "desired")
        #ax.scatter(shower_core[0], shower_core[1], shower_core[2], label = "shower core")
        for j in range(0,len(positions_des[:,1])):
                ax.annotate(str(j), ((positions_des[j,0], positions_des[j,1])))
        ax.scatter(positions_sims[:,0], positions_sims[:,1], label = "simulated")
        ax.scatter(positions_des[ind,0], positions_des[ind,1], label = "desired")
        ax.scatter(shower_core[0], shower_core[1],  label = "shower core")

        plt.title("XYZ coordinates")
        plt.legend(loc=2)
        plt.show()
    #------------------------


    ##--##--##--##--##--##--##--##--##-##--##--##-##--##--## START: WRAP UP AS FUNCTION (PROJECTION AND ROTATION)
    #### START: UNDO projection
    #define shower vector
    az_rad=np.deg2rad(180.+Azimuth)#Note ZHAIRES units used
    zen_rad=np.deg2rad(180.-Zenith)

    # shower vector  = direction of line for backprojection, TODO should be substituded bey line of sight Xmax - positions
    v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])
    v = v/np.linalg.norm(v)

    # for back projection position vector line is projected position
    # for back projection normal vector of plane to intercsect == v
    n = v

    for i in np.arange(0,len(positions_des[:,1])):
        b=-np.dot(n,positions_des[i,:])/ np.dot(n, v)
        positions_des[i,:] = positions_des[i,:] + b*v - shower_core # correct by shower core position
    for i in np.arange(0,len(positions_sims[:,1])):
        b=-np.dot(n,positions_sims[i,:])/ np.dot(n, v)
        positions_sims[i,:] = positions_sims[i,:] + b*v - shower_core # correct by shower core position

    ### START: ROTATE INTO SHOWER COORDINATES, and core for offset by core position, alreadz corrected in projection
    #GetUVW = UVWGetter(shower_core[0], shower_core[1], shower_core[2], zenith, azimuth, phigeo, thetageo)
    GetUVW = UVWGetter(0., 0., 0., Zenith, Azimuth, phigeo, thetageo)


    # Rotate only "in"plane desired positions
    pos_des= []
    for i in np.arange(0,len(positions_des[:,1])):
        if i in ind:
            pos_des.append(GetUVW(positions_des[i,:], ))
    pos_des=np.asarray(pos_des)

    # Rotate simulated positions
    pos_sims= np.zeros([len(positions_sims[:,1]),3])
    for i in np.arange(0,len(positions_sims[:,1])):
        pos_sims[i,:] = GetUVW(positions_sims[i,:], )
    ##--##--##--##--##--##--##--##--##-##--##--##-##--##--## END: WRAP UP AS FUNCTION (PROJECTION AND ROTATION)

    # ------------------
    if DISPLAY:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        #ax2 = fig2.gca(projection='3d')

        #ax2.scatter(pos_sims[:,0], pos_sims[:,1], pos_sims[:,2], label = "simulated")
        #ax2.scatter(pos_des[:,0], pos_des[:,1], pos_des[:,2], label = "desired")
        for j in range(0,len(pos_des[:,1])):
                ax2.annotate(str(j), ((pos_des[j,1], pos_des[j,2])))
        ## x component should be 0
        ax2.scatter(pos_sims[:,1], pos_sims[:,2], label = "simulated")
        ax2.scatter(pos_des[:,1], pos_des[:,2], label = "desired")
        ax2.scatter(0, 0, marker ="x",  label = "core")

        plt.title("shower coordinates")
        plt.legend(loc=2)
        plt.show()
    # ------------------

    # calculate radius and angle for simulated positions and store some in list
    points=[]
    for i in np.arange(0,len(pos_sims[:,1])):  # position should be within one plane yz plane, remove x=v component for simplicity
        #points.append([i, pos_sims[i,1], pos_sims[i,2] ])
        theta2 = np.arctan2(pos_sims[i,2], pos_sims[i,1])
        radius2 = np.sqrt( pos_sims[i,1]**2 + pos_sims[i,2]**2 )
        if round(theta2,4) == -3.1416:        #why is this
            theta2*=-1
            print("Check why is this in trace_interpl.py")
        points.append([i, theta2, radius2])


    #i loops only over desired in-plane positions, acting as new reference
    for i in np.arange(0,len(pos_des[:,1])):  # position should be within one plane yz plane, remove x=v component for simplicity
        #print("desired antena:"+str(i))
        theta = np.arctan2(pos_des[i,2], pos_des[i,1])
        radius = np.sqrt( pos_des[i,1]**2 + pos_des[i,2]**2 )
        #print("index of desired antenna ", ind[i], theta, radius, )


        # The 4 quadrants -- in allen 4 Ecken soll Liebe drin stecken
        points_I=[]
        points_II=[]
        points_III=[]
        points_IV=[]

        #m loops over the simulated positions
        for m in np.arange(0,len(points)): # desired points as reference
            delta_phi = points[m][1]-theta
            if delta_phi > np.pi:
                delta_phi = delta_phi -2.*np.pi
            elif delta_phi < -np.pi:
                delta_phi = delta_phi + 2.*np.pi


            delta_r = points[m][2]-radius

            #distance = np.sqrt(delta_r**2 + (delta_r *delta_phi)**2 ) # weighting approach1
            #distance= np.sqrt((pos_sims[m,1]-pos_des[i,1])**2. +(pos_sims[m,2]-pos_des[i,2])**2.) # euclidean distance
            distance= np.sqrt(points[m][2]**2. +radius**2. -2.*points[m][2]*radius* np.cos(points[m][1]-theta) ) #polar coordinates

            if delta_phi >= 0. and  delta_r >= 0:
                points_I.append((m,delta_phi,delta_r, distance))
            if delta_phi >= 0. and  delta_r <= 0:
                points_II.append((m,delta_phi,delta_r, distance))
            if delta_phi <= 0. and  delta_r <= 0:
                points_III.append((m,delta_phi,delta_r, distance))
            if delta_phi <= 0. and  delta_r >= 0:
                points_IV.append((m,delta_phi,delta_r, distance))

        bailoutI=0
        bailoutII=0
        bailoutIII=0
        bailoutIV=0
        if not points_I:
            print("list - Quadrant 1 - empty --> no interpolation for ant", str(ind[i]))
            bailoutI=1
        if not points_II:
            print("list - Quadrant 2 - empty --> no interpolation for ant", str(ind[i]))
            bailoutII=1
            points_II=points_I
        if not points_III:
            print("list - Quadrant 3 - empty --> no interpolation for ant", str(ind[i]))
            bailoutIII=1
            points_III=points_IV
        if not points_IV:
            print("list - Quadrant 4 - empty --> no interpolation for ant", str(ind[i]))
            bailoutIV=1

        if(bailoutII==1 and bailoutIII==1 and bailoutIV==0 and bailoutI==0):
            print("but wait, lets try the inner antennas")

        if(bailoutI==1 or bailoutIV==1 or (bailoutII==1 and bailoutIII==0) or (bailoutII==0 and bailoutII==1)):
          AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,0)
          if(usetrace=='efield'):
              txt0=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
          elif(usetrace=='voltage'):
              txt0=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
          elif(usetrace=='filteredvoltage'):
              txt0=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
          else:
              print("You must specify either efield, voltage or filteredvoltage, bailing out")

          if(usetrace=='efield'):
            efield=np.zeros(np.shape(txt0))
            EfieldTable=hdf5io.CreateEfieldTable(efield, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
            hdf5io.SaveEfieldTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),EfieldTable)
          elif(usetrace=='voltage'):
            voltage=np.zeros(np.shape(txt0))
            VoltageTable=hdf5io.CreateVoltageTable(voltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
            hdf5io.SaveVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)
          elif(usetrace=='filteredvoltage'):
            filteredvoltage=np.zeros(np.shape(txt0))
            VoltageTable=hdf5io.CreateVoltageTable(filteredvoltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
            hdf5io.SaveFilteredVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)



        else:
            # ------------------

            points_I=np.array(points_I, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
            points_II=np.array(points_II, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
            points_III=np.array(points_III, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
            points_IV=np.array(points_IV, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])

            ## Sort points; not optimal (the best) solution for all, but brings stable/acceptable results
            points_I = np.sort(points_I, order=['distance', 'delta_phi', 'delta_r'])
            points_II = np.sort(points_II, order=['distance', 'delta_phi', 'delta_r'])
            points_III = np.sort(points_III, order=[ 'distance','delta_phi', 'delta_r'])
            points_IV = np.sort(points_IV, order=['distance', 'delta_phi', 'delta_r'])
            #indizes of 4 closest neigbours: points_I[0][0], points_II[0][0], points_III[0][0], points_IV[0][0]

            # try to combine the one with roughly the same radius first and then the ones in phi
            #point_online1=_ProjectPointOnLine(pos_sims[points_I[0][0]], pos_sims[points_IV[0][0]], pos_des[i])# Project Point on line 1 - I-IV
            #point_online2=_ProjectPointOnLine(pos_sims[points_II[0][0]], pos_sims[points_III[0][0]], pos_des[i])# Project Point on line 2 - II-III
            #this is not correct,in particular the coordinate [0]  of the point is not correct, so when the wheighting of the two signals is done is incorrect (always ends in nearly 50/50 weight.
            #so, since the weighting is done just for the distance, and that is the only thing why the position is used inside that function,
            #i will input to interpolate_trace the y and z components, and make x=0 (this is becouse in the uxVxB plane one component is 0?


            #points I and IV have a higher radius. lets take the average radius, and the same theta of the desired point as the point1
            #this is stored alreadty in the opoints variable
            meanr=(points[points_I[0][0]][2]+points[points_IV[0][0]][2])/2.0
            #and theta is already available
            point_online1=np.array([0,meanr*np.cos(theta),meanr*np.sin(theta)])


            meanr=(points[points_II[0][0]][2]+points[points_III[0][0]][2])/2.0
            #and theta is already available
            point_online2=np.array([0,meanr*np.cos(theta),meanr*np.sin(theta)])
            #points II and III have a lower radius. lets take the average radius, and the same theta of the desired point as the point2

            #this is to make a list of the indices of antennas in the each quadrant
            if (DISPLAY and False):
              listI=list(list(zip(*points_I))[0])
              listII=list(list(zip(*points_II))[0])
              listIII=list(list(zip(*points_III))[0])
              listIV=list(list(zip(*points_IV))[0])

              mypoints_I=[]
              mypoints_II=[]
              mypoints_III=[]
              mypoints_IV=[]

              for h in listI:
                mypoints_I.append((pos_sims[h,1],pos_sims[h,2]))

              for h in listII:
                mypoints_II.append((pos_sims[h,1],pos_sims[h,2]))

              for h in listIII:
                mypoints_III.append((pos_sims[h,1],pos_sims[h,2]))

              for h in listIV:
                mypoints_IV.append((pos_sims[h,1],pos_sims[h,2]))

              mypoints_I=np.array(mypoints_I)
              mypoints_II=np.array(mypoints_II)
              mypoints_III=np.array(mypoints_III)
              mypoints_IV=np.array(mypoints_IV)

              fig3a = plt.figure()
              ax3a = fig3a.add_subplot(1,1,1)

              for j in range(0,len(pos_sims[:,1])):
                ax3a.annotate(str(j), ((pos_sims[j,1], pos_sims[j,2])))

              ## x component should be 0
              ax3a.scatter(pos_des[i,1], pos_des[i,2], label = "desired")
              ax3a.scatter(mypoints_I[:,0], mypoints_I[:,1], s=90,marker ="D",label = "1")
              ax3a.scatter(mypoints_II[:,0], mypoints_II[:,1], s=90, marker ="D",label = "2")
              ax3a.scatter(mypoints_III[:,0], mypoints_III[:,1], s=90, marker ="D",label = "3")
              ax3a.scatter(mypoints_IV[:,0], mypoints_IV[:,1], s=90,marker ="D",label = "4")

              ax3a.scatter(pos_sims[points_I[0][0],1], pos_sims[points_I[0][0],2], s=20, marker ="D",label = "1")
              ax3a.scatter(pos_sims[points_II[0][0],1], pos_sims[points_II[0][0],2], s=20, marker ="D", label = "2")
              ax3a.scatter(pos_sims[points_III[0][0],1], pos_sims[points_III[0][0],2], s=20, marker ="D", label = "3")
              ax3a.scatter(pos_sims[points_IV[0][0],1], pos_sims[points_IV[0][0],2], s=20, marker ="D",label = "4")



              ax3a.scatter(point_online1[1], point_online1[2], marker ="x")
              ax3a.scatter(point_online2[1], point_online2[2], marker ="x")
              ax3a.scatter(0, 0, marker ="x",  label = "core")
              ax3a.plot([0, pos_des[i,1]], [0, pos_des[i,2]])
              plt.title("my"+str(i))
              plt.legend(loc=2)
              plt.show()
            # ------------------

            ## the interpolation of the pulse shape is performed, in x, y and z component
            # TODO: read-in table instead textfile
            bailout0=0
            bailout1=0
            bailout2=0
            bailout3=0

            AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,points_I[0][0])
            if(usetrace=='efield'):
              txt0=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='voltage'):
              txt0=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='filteredvoltage'):
              txt0=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            else:
              print('You must specify either efield, voltage or filteredvoltage, bailing out')
              return 0

            p2p= np.amax(txt0,axis=0)-np.amin(txt0,axis=0)
            if(max(p2p[1:3])<threshold):
             #print(max(p2p[1:3]))
             bailout0=1

            AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,points_IV[0][0])
            if(usetrace=='efield'):
              txt1=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='voltage'):
              txt1=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='filteredvoltage'):
              txt1=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            else:
              print('You must specify either efield, voltage or filteredvoltage, bailing out')
              return 0

            p2p= np.amax(txt1,axis=0)-np.amin(txt1,axis=0)
            if(max(p2p[1:3])<threshold):
             #print(max(p2p[1:3]))
             bailout1=1

            AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,points_II[0][0])
            if(usetrace=='efield'):
              txt2=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='voltage'):
              txt2=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='filteredvoltage'):
              txt2=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            else:
              print('You must specify either efield, voltage or filteredvoltage, bailing out')
              return 0

            p2p= np.amax(txt2,axis=0)-np.amin(txt2,axis=0)
            if(max(p2p[1:3])<threshold):
             #print(max(p2p[1:3]))
             bailout2=1

            AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,points_III[0][0])
            if(usetrace=='efield'):
              txt3=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='voltage'):
              txt3=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            elif(usetrace=='filteredvoltage'):
              txt3=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
            else:
              print('You must specify either efield, voltage or filteredvoltage, bailing out')
              return 0

            p2p= np.amax(txt3,axis=0)-np.amin(txt3,axis=0)
            if(max(p2p[1:3])<threshold):
             #print(max(p2p[1:3]))
             bailout3=1

            if(bailout0==1 and bailout1==1 and bailout2==1 and bailout3==1):
              print("desired antenna " + str(i) + " had all neighbors below threshold of " + str(threshold)+ ",setting interpolated signal to 0")
              if(usetrace=='efield'):
                efield=np.zeros(np.shape(txt0))
                EfieldTable=hdf5io.CreateEfieldTable(efield, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                hdf5io.SaveEfieldTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),EfieldTable)
              elif(usetrace=='voltage'):
                voltage=np.zeros(np.shape(txt0))
                VoltageTable=hdf5io.CreateVoltageTable(voltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                hdf5io.SaveVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)
              elif(usetrace=='filteredvoltage'):
                filteredvoltage=np.zeros(np.shape(txt0))
                VoltageTable=hdf5io.CreateVoltageTable(filteredvoltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                hdf5io.SaveFilteredVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)

            else:
                #directory=split(array)[0]+"/"
                #print("Read traces from ", directory)
                #directory="/home/mjtueros/GRAND/GP300/GridShape/Stshp_XmaxLibrary_0.1995_85.22_0_Iron_23/"

                #txt0_old = load_trace(directory, points_I[0][0], suffix=".trace")
                #txt1_old = load_trace(directory, points_IV[0][0], suffix=".trace")
                #txt2_old = load_trace(directory, points_II[0][0], suffix=".trace")
                #txt3_old = load_trace(directory, points_III[0][0], suffix=".trace")

                #print(txt0_old-txt0)
                #print(txt1_old-txt1)
                #print(txt2_old-txt2)
                #print(txt3_old-txt3)

                if DEVELOPMENT and DISPLAY:
                  AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,160+i)
                  if(usetrace=='efield'):
                    txtdes=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
                  elif(usetrace=='voltage'):
                    txtdes=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
                  elif(usetrace=='filteredvoltage'):
                    txtdes=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
                  else:
                    print('You must specify either efield, voltage or filteredvoltage, bailing out')
                    return 0
                  #txtdes= load_trace(directory, 160+i, suffix=".trace")

                matias_patch_to_point_I=np.array([0,pos_sims[points_I[0][0]][1],pos_sims[points_I[0][0]][2]])
                matias_patch_to_point_II=np.array([0,pos_sims[points_II[0][0]][1],pos_sims[points_II[0][0]][2]])
                matias_patch_to_point_III=np.array([0,pos_sims[points_III[0][0]][1],pos_sims[points_III[0][0]][2]])
                matias_patch_to_point_IV=np.array([0,pos_sims[points_IV[0][0]][1],pos_sims[points_IV[0][0]][2]])

                xnew1, tracedes1x = interpolate_trace(txt0.T[0], txt0.T[1], matias_patch_to_point_I , txt1.T[0], txt1.T[1], matias_patch_to_point_IV, point_online1 ,upsampling=None, zeroadding=None)
                xnew1, tracedes1y = interpolate_trace(txt0.T[0], txt0.T[2], matias_patch_to_point_I , txt1.T[0], txt1.T[2], matias_patch_to_point_IV, point_online1 ,upsampling=None, zeroadding=None)
                xnew1, tracedes1z = interpolate_trace(txt0.T[0], txt0.T[3], matias_patch_to_point_I , txt1.T[0], txt1.T[3], matias_patch_to_point_IV, point_online1 ,upsampling=None, zeroadding=None)

                #xnew1, tracedes1x = interpolate_trace(txt0.T[0], txt0.T[1], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[1], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None)
                #xnew1, tracedes1y = interpolate_trace(txt0.T[0], txt0.T[2], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[2], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None)
                #xnew1, tracedes1z = interpolate_trace(txt0.T[0], txt0.T[3], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[3], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None)

                xnew2, tracedes2x = interpolate_trace(txt2.T[0], txt2.T[1], matias_patch_to_point_II , txt3.T[0], txt3.T[1], matias_patch_to_point_III, point_online2 ,upsampling=None, zeroadding=None)
                xnew2, tracedes2y = interpolate_trace(txt2.T[0], txt2.T[2], matias_patch_to_point_II , txt3.T[0], txt3.T[2], matias_patch_to_point_III, point_online2 ,upsampling=None, zeroadding=None)
                xnew2, tracedes2z = interpolate_trace(txt2.T[0], txt2.T[3], matias_patch_to_point_II , txt3.T[0], txt3.T[3], matias_patch_to_point_III, point_online2 ,upsampling=None, zeroadding=None)

                #xnew2, tracedes2x = interpolate_trace(txt2.T[0], txt2.T[1], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[1], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None)
                #xnew2, tracedes2y = interpolate_trace(txt2.T[0], txt2.T[2], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[2], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None)
                #xnew2, tracedes2z = interpolate_trace(txt2.T[0], txt2.T[3], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[3], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None)

                ###### Get the pulse shape of the desired position from projection on line1 and 2
                xnew_desiredx, tracedes_desiredx =interpolate_trace(xnew1, tracedes1x, point_online1, xnew2, tracedes2x, point_online2, np.array([0,pos_des[i,1],pos_des[i,2]]), zeroadding=None)
                xnew_desiredy, tracedes_desiredy =interpolate_trace(xnew1, tracedes1y, point_online1, xnew2, tracedes2y, point_online2, np.array([0,pos_des[i,1],pos_des[i,2]]), zeroadding=None)
                xnew_desiredz, tracedes_desiredz =interpolate_trace(xnew1, tracedes1z, point_online1, xnew2, tracedes2z, point_online2, np.array([0,pos_des[i,1],pos_des[i,2]]), zeroadding=None)

                if DISPLAY and False:
                    fig4 = plt.figure()
                    ax4 = fig4.add_subplot(1,2,2)
                    ax4.plot(txt0.T[0], txt0.T[2], label = "I")
                    ax4.plot(txt2.T[0], txt2.T[2], label = "II")
                    ax4.plot(txt3.T[0], txt3.T[2], label = "III")
                    ax4.plot(txt1.T[0], txt1.T[2], label = "IV")
                    ax4.plot(xnew1, tracedes1y, linestyle='--',color='r',label = "I->IV")
                    ax4.plot(xnew2, tracedes2y, linestyle='--', color='b',label = "II->III")
                    ax4.plot(xnew_desiredx, tracedes_desiredy, linestyle='--',color='k', label = "desired")

                    plt.title("Y component Signals:"+str(i))
                    plt.legend(loc=2)

                    ax3 = fig4.add_subplot(1,2,1)

                    ax3.scatter(pos_sims[:,1], pos_sims[:,2], color='c',label = "simulated")
                    ax3.scatter(pos_sims[points_I[0][0],1], pos_sims[points_I[0][0],2], marker ="D",label = "I")
                    ax3.scatter(pos_sims[points_II[0][0],1], pos_sims[points_II[0][0],2],marker ="D", label = "II")
                    ax3.scatter(pos_sims[points_III[0][0],1], pos_sims[points_III[0][0],2], marker ="D", label = "III")
                    ax3.scatter(pos_sims[points_IV[0][0],1], pos_sims[points_IV[0][0],2], marker ="D",label = "IV")

                    #print("desired "+str(pos_des[i]))
                    #print("pointI "+str(points_I[0]))
                    #print("pointII "+str(points_II[0]))
                    #print("pointIII "+str(points_III[0]))
                    #print("pointIV "+str(points_IV[0]))

                    ax3.scatter(point_online1[1], point_online1[2], marker ="x",color='r',label = "I->IV")
                    ax3.scatter(point_online2[1], point_online2[2], marker ="x",color='b',label = "II->III")
                    ax3.scatter(pos_des[i,1], pos_des[i,2], color= 'k',label = "desired")

                    ax3.scatter(0, 0, marker ="x",  label = "core")
                    ax3.plot([0, pos_des[i,1]], [0, pos_des[i,2]])

                    for j in range(0,len(pos_sims[:,1])):
                        ax3.annotate(str(j), ((pos_sims[j,1], pos_sims[j,2])))

                    plt.title("Test Antenna "+str(i))
                    plt.legend(loc=2)
                    #plt.show()

                if DEVELOPMENT and DISPLAY:
                   fig5 = plt.figure()
                   ax5 = fig5.add_subplot(3,2,1)

                   #ax5.plot(xnew_desiredx, tracedes_desiredy, linestyle='--',color='k', label = "desired")
                   #ax5.plot(xnew1, tracedes1y, linestyle='--',color='r',label = "I->IV")
                   #ax5.plot(xnew2, tracedes2y, linestyle='--', color='b',label = "II->III")
                   ax5.plot(txtdes.T[0], tracedes_desiredx, linestyle='--',color='g', label = "shifted desired")
                   ax5.plot(txtdes.T[0], txtdes.T[1], label = "simulation")
                   plt.title("X component Signals:"+str(i))
                   plt.legend(loc=2)

                   ax6 = fig5.add_subplot(3,2,2)
                   ax6.plot(txtdes.T[0], tracedes_desiredx-txtdes.T[1], linestyle='--',color='g', label = "absolute diference")
                   plt.legend(loc=2)

                   ax5b = fig5.add_subplot(3,2,3)
                   ax5b.plot(txtdes.T[0], tracedes_desiredy, linestyle='--',color='g', label = "shifted desired")
                   ax5b.plot(txtdes.T[0], txtdes.T[2], label = "simulation")
                   plt.title("Y component Signals:"+str(i))
                   plt.legend(loc=2)

                   ax6b = fig5.add_subplot(3,2,4)
                   ax6b.plot(txtdes.T[0], tracedes_desiredy-txtdes.T[2], linestyle='--',color='g', label = "absolute diference")
                   plt.legend(loc=2)

                   ax5b = fig5.add_subplot(3,2,5)
                   ax5b.plot(txtdes.T[0], tracedes_desiredz, linestyle='--',color='g', label = "shifted desired")
                   ax5b.plot(txtdes.T[0], txtdes.T[3], label = "simulation")
                   plt.title("Z component Signals:"+str(i))
                   plt.legend(loc=2)

                   ax6b = fig5.add_subplot(3,2,6)
                   ax6b.plot(txtdes.T[0], tracedes_desiredz-txtdes.T[3], linestyle='--',color='g', label = "absolute diference")
                   plt.legend(loc=2)

                if DISPLAY and False:
                  plt.show()

                # ------------------

                # TODO Save as hdf5 file instead of textfile
                #print("Interpolated trace stord as ",split(desired)[0]+ '/a'+str(ind[i])+'.trace')
                #os.system('rm ' + split(desired)[0] + '/*.trace')
                #FILE = open(split(desired)[0]+ '/a'+str(ind[i])+'.trace', "w+" )

                #uncomment this if you want the old style output
                #FILE = open(directory+ '/Test/a'+str(ind[i])+'.trace', "w+" )
                #for j in range( 0, len(xnew_desiredx) ):
                #        print("%3.2f %1.5e %1.5e %1.5e" % (xnew_desiredx[j], tracedes_desiredx[j], tracedes_desiredy[j], tracedes_desiredz[j]), end='\n', file=FILE)
                #FILE.close()

                if(usetrace=='efield'):
                    efield=np.column_stack((xnew_desiredx,tracedes_desiredx,tracedes_desiredy,tracedes_desiredz))
                    EfieldTable=hdf5io.CreateEfieldTable(efield, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    hdf5io.SaveEfieldTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),EfieldTable)
                elif(usetrace=='voltage'):
                    voltage=np.column_stack((xnew_desiredx,tracedes_desiredx,tracedes_desiredy,tracedes_desiredz))
                    VoltageTable=hdf5io.CreateVoltageTable(voltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    hdf5io.SaveVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)
                elif(usetrace=='filteredvoltage'):
                    filteredvoltage=np.column_stack((xnew_desiredx,tracedes_desiredx,tracedes_desiredy,tracedes_desiredz))
                    VoltageTable=hdf5io.CreateVoltageTable(filteredvoltage, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    hdf5io.SaveFilteredVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)


                #delete after iterate
                del points_I, points_II, points_III, points_IV

#-------------------------------------------------------------------



def main():
    if ( len(sys.argv)<1 ):
        print("""
            Example on how to do interpolate a signal
                -- read in list of desired poistion
                -- read in already simulated arrazs
                -- find neigbours and perform interpolation
                -- save interpolated trace

            Usage: python3 interpolate.py <path>
            Example: python3 interpolate.py <path>

            path: Filename and path of the input hdf5file
        """)
        sys.exit(0)

    hdf5file = sys.argv[1]

    # path to list of desied antenna positions, traces will be stored in that corresponding folder
    #desired  = sys.argv[1]
    #desired=np.array([[ 100., 0., 2900.],[ 0., 100., 2900.]])
    desired=np.loadtxt("/home/mjtueros/GRAND/GP300/GridShape/Stshp_XmaxLibrary_0.1995_85.22_0_Iron_23/Test/new_antpos.dat",usecols=(2,3,4))

    OutputFilename=split(hdf5file)[0]+"/InterpolatedAntennas.hdf5"
    # call the interpolation: Angles of magnetic field and shower core information needed, but set to default values
    #do_interpolation(desired,hdf5file, zenith, azimuth, phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
    do_interpolation_hdf5(desired, hdf5file, OutputFilename, antennamin=0, antennamax=159, EventNumber=0,shower_core=np.array([0,0,2900]), DISPLAY=True,usetrace='voltage')

if __name__== "__main__":
  main()




