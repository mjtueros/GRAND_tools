'''
Performs an interpolation of the peak-to-peak electric field at 6 antenna positions for each shower in database.
Check interpolation efficiency.

run ant_trig_db [argv1]

argv1: str
    path+name of shower database
'''
import sys
import os
import logging   #for...you guessed it...logging
import sqlite3   #for the database
import argparse  #for command line parsing
import glob      #for listing files in directories
import importlib #to be able to import modules dynamically (will need it to switch from cluster to local run configurations)
import time      #for the sleep
import datetime  #for the now()
#sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/Scripts") #so that it knows where to find things
import numpy as np
import matplotlib.pyplot as plt
import hdf5fileinout as hdf5io
#from matplotlib.pyplot import cm
import DatabaseFunctions as mydatabase  #my database handling library
import AiresInfoFunctions as AiresInfo
#import interpol_func as intf
import interpol_func_hdf5 as intf
import grids



parser = argparse.ArgumentParser(description='A script to get the CPU time in a library of Simulations')
parser.add_argument('DatabaseFile', #name of the parameter
                    metavar="filename", #name of the parameter value in the help
                    help='The Database of the library .db file') # help message for this parameter
results = parser.parse_args()
dbfile=results.DatabaseFile

dbfile="/home/mjtueros/GRAND/GP300/HDF5StshpLibrary/StshpXmaxLibraryInExa24.01.sql3.db"
#directory where the files from the library are located
Directory = "/home/mjtueros/GRAND/GP300/HDF5StshpLibrary/Outbox"
#what to use in the interpolation (efield, voltage, filteredvoltage)
usetrace='efield'
#threshold abouve wich the interpolation is computed
threshold=0;#26 #8.66 for 15uV , 26 for 45uV
trigger=75
display=False
#logging.debug('This is a debug message')
#logging.info('This is an info message')
#logging.warning('This is a warning message')
#logging.error('This is an error message')
#logging.critical('This is a critical message')
logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.WARNING)
#logging.disable(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


logging.debug("Starting Xmax analysis with %s " % dbfile)

DataBase=mydatabase.ConnectToDataBase(dbfile)
#this is to show the current status of the database
mydatabase.GetDatabaseStatus(DataBase)

#This is how you search on the database, here im selecting everything (To Do: functions to search the database)
#This is to get a cursor on the database. You can think of the cursor as a working environment. You can have many cursors.
CurDataBase = DataBase.cursor()
CurDataBase.execute("SELECT * FROM showers")

#f = open(dbfile+".Xmax", 'w')
#f.write("Id, Energy, Zenith, Azimuth, Primary, SlantXmax, XmaxAltitude, Xxmax, Yxmax,Zxmax\n")

DatabaseRecord = CurDataBase.fetchone()
counterr = 0
countok = 0
InterpErrAll = np.zeros((16,1100))
InterpErrAllx = np.zeros((16,1100))
InterpErrAlly = np.zeros((16,1100))
InterpErrAllz = np.zeros((16,1100))
P2pAll = np.zeros((16,1100))
P2pAllx = np.zeros((16,1100))
P2pAlly = np.zeros((16,1100))
P2pAllz = np.zeros((16,1100))
errortype = np.zeros((16,1100))
errortypex = np.zeros((16,1100))
errortypey = np.zeros((16,1100))
errortypez = np.zeros((16,1100))
while(DatabaseRecord!=None and countok < 1100): #500 events in 30min, withouth tresholding, 700 en 47 min
    #while(DatabaseRecord!=None):
    DatabaseStatus = mydatabase.GetStatusFromRecord(DatabaseRecord) #i do it with a function call becouse if we change the database structure we dont have to change this
    #Directory = mydatabase.GetDirectoryFromRecord(DatabaseRecord)
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)
    JobDirectory = str(Directory)+"/"+str(JobName)
    Tries = mydatabase.GetTriesFromRecord(DatabaseRecord)
    #.
    logging.debug("Reading Job " + JobName + " which was in " + DatabaseStatus + " status ("+str(Tries)+") at " + Directory)
    #.
    if(DatabaseStatus == "RunOK"): #and JobName=="Stshp_XmaxLibrary_0.1995_80.40_180_Gamma_01"):
        try:
            TaskName = mydatabase.GetTasknameFromRecord(DatabaseRecord)
            Id = mydatabase.GetIdFromRecord(DatabaseRecord)
            Energy = mydatabase.GetEnergyFromRecord(DatabaseRecord)
            Zenith = mydatabase.GetZenithFromRecord(DatabaseRecord)
            Azimuth = mydatabase.GetAzimuthFromRecord(DatabaseRecord)
            Primary = mydatabase.GetPrimaryFromRecord(DatabaseRecord)
            Xmax = mydatabase.GetXmaxFromRecord(DatabaseRecord)
            #.
            InputFilename=str(Directory)+"/"+str(JobName)+"/"+str(JobName)+".hdf5"
            #.
            CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
            CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
            CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)
            #.
            #AntNum, AntPos, AntID = intf.get_antenna_pos_zhaires(JobDirectory+'/antpos.dat')
            #one way of putting the antenna information as the numpy array this script was designed to use:
            antennamin=0
            antennamax=176 #NOTE THAT WE ARE GETTING ALL THE ANTENNAS, THE STARSHAPE AND THE RANDOM!
            AntID=CurrentAntennaInfo['ID'].data[antennamin:antennamax]
            AntNum=len(AntID)
            xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
            ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
            zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
            AntPos=np.stack((xpoints,ypoints,zpoints), axis=0)
            #.
            #this gets the p2p values in all chanels, for all simulated antennas.

            p2pE = hdf5io.get_p2p_hdf5(InputFilename,antennamax=175,antennamin=0,usetrace=usetrace)
            peaktime, peakamplitude= hdf5io.get_peak_time_hilbert_hdf5(InputFilename,antennamax=175,antennamin=0, usetrace=usetrace, DISPLAY=False)

            #lets append this to the an tennainfo (once)
            from astropy.table import Table, Column
            from astropy import units as u
            p2pE32=p2pE.astype('f4') #reduce the data type to float 32
            CurrentAntennaInfo.add_column(Column(data=p2pE32[3,:],name='P2P_efield',unit=u.u*u.V/u.m)) #p2p Value of the electric field
            CurrentAntennaInfo.add_column(Column(data=p2pE32[0,:],name='P2Px_efield',unit=u.u*u.V/u.m)) #p2p Value of the electric field
            CurrentAntennaInfo.add_column(Column(data=p2pE32[1,:],name='P2Py_efield',unit=u.u*u.V/u.m)) #p2p Value of the electric field
            CurrentAntennaInfo.add_column(Column(data=p2pE32[2,:],name='P2Pz_efield',unit=u.u*u.V/u.m)) #p2p Value of the electric field
            peakamplitude32=peakamplitude.astype('f4') #reduce the data type to float 32
            CurrentAntennaInfo.add_column(Column(data=peakamplitude32,name='HilbertPeak')) #
            peaktime32=peaktime.astype('f4')
            CurrentAntennaInfo.add_column(Column(data=peaktime32,name='HilbertPeakTime',unit=u.u*u.s)) #
            #hdf5io.SaveAntennaInfo(InputFilename,CurrentAntennaInfo,CurrentEventName,overwrite=True)
            #i get an error when writing an existing table, even if the overwrite is set to true :(
            CurrentAntennaInfo.write(InputFilename, path=CurrentEventName+"/AntennaInfo4", format="hdf5", append=True,  compression=True, serialize_meta=True, overwrite=True)

            #.
            NewPos = grids.create_grid(AntPos,Zenith,'check',20,10) #In Check mode, it will return the last 16 elements of Antpos, so this just Antpos[160:175]
            #.
            InterpErr, p2p_total_new, interp_errx, p2p_x_new, interp_erry, p2p_y_new, interp_errz, p2p_z_new = intf.interpol_check_hdf5(InputFilename, AntPos, NewPos.T, p2pE,'trace',threshold=threshold, usetrace=usetrace,DISPLAY=display)
            #.
            #so these are the relative errors of all interpolated antennas
            InterpErrAll[:,countok] = InterpErr
            InterpErrAllx[:,countok] = interp_errx
            InterpErrAlly[:,countok] = interp_erry
            InterpErrAllz[:,countok] = interp_errz
            #.
            #and this is the p2p value of all interpolated antennas
            P2pAll[:,countok] = p2p_total_new
            P2pAllx[:,countok] = p2p_x_new
            P2pAlly[:,countok] = p2p_y_new
            P2pAllz[:,countok] = p2p_z_new
            #.
            #now, false positives is 1, false negatives is -1, 0  is correct and triggered, -2 is neither
            a=np.arange(0,len(p2p_total_new))
            errortype[:,countok]=[1 if  (p2p_total_new[i] >= trigger and p2pE[3,160+i]<trigger) else -1 if (p2p_total_new[i] < trigger and p2pE[3,160+i]>=trigger) else 0 if (p2p_total_new[i] >= trigger and p2pE[3,160+i]>=trigger) else -2  for i in a]
            errortypex[:,countok]=[1 if  (p2p_x_new[i] >= trigger and p2pE[0,160+i]<trigger) else -1 if (p2p_x_new[i] < trigger and p2pE[0,160+i]>=trigger) else 0 if (p2p_x_new[i] >= trigger and p2pE[0,160+i]>=trigger) else -2 for i in a]
            errortypey[:,countok]=[1 if  (p2p_y_new[i] >= trigger and p2pE[1,160+i]<trigger) else -1 if (p2p_y_new[i] < trigger and p2pE[1,160+i]>=trigger) else 0 if (p2p_y_new[i] >= trigger and p2pE[1,160+i]>=trigger) else -2 for i in a]
            errortypez[:,countok]=[1 if  (p2p_z_new[i] >= trigger and p2pE[2,160+i]<trigger) else -1 if (p2p_z_new[i] < trigger and p2pE[2,160+i]>=trigger) else 0 if (p2p_z_new[i] >= trigger and p2pE[2,160+i]>=trigger) else -2 for i in a]
            #.
            countok += 1
            print("Event #{} done" .format(countok))
            #.
            #.
        except FileNotFoundError:
          logging.error("ant_interpol_chk_db:file not found or invalid:"+TaskName)
          counterr += 1
          #.
    #this is the last order of the while, that will fetch the next record of the database
    DatabaseRecord=CurDataBase.fetchone()
#.
#.
logging.debug('ant_interpol_chk_db: Plotting...')


ind = np.where(InterpErrAll != 0) #here i remove the cases where the error is 0. Shouldnt happen?
myInterpErrAll = InterpErrAll[ind]
myP2pAll= P2pAll[ind]
myerrortype=errortype[ind]

indx = np.where(InterpErrAllx != 0) #here i remove the cases where the error is 0. Shouldnt happen?
myInterpErrAllx = InterpErrAllx[indx]
myP2pAllx= P2pAllx[indx]
myerrortypex=errortypex[indx]

indy = np.where(InterpErrAlly != 0) #here i remove the cases where the error is 0. Shouldnt happen?
myInterpErrAlly = InterpErrAlly[indy]
myP2pAlly= P2pAlly[indy]
myerrortypey=errortypey[indy]

indz = np.where(InterpErrAllz != 0) #here i remove the cases where the error is 0. Shouldnt happen?
myInterpErrAllz = InterpErrAllz[indz]
myP2pAllz= P2pAllz[indz]
myerrortypez=errortypez[indz]

print(np.shape(P2pAll),np.shape(P2pAllx),np.shape(P2pAlly),np.shape(P2pAllz))
print(np.shape(myP2pAll),np.shape(myP2pAllx),np.shape(myP2pAlly),np.shape(myP2pAllz))


##############Plot histogram of relative errors, for all components####################################
fig1 = plt.figure(1,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
mybins = [-1.5,-0.5,0.5,1.5]

ax1=fig1.add_subplot(221)
ax1.set_xlabel('Error type')
ax1.set_ylabel('N')
name = 'clasification errors ' + str(usetrace) + " threshold " + str(threshold) + " trigger " + str(trigger)
plt.title(name)
plt.yscale('log')
plt.hist(myerrortype, bins=mybins,alpha=0.8,label="Total",density=True)


ax2=fig1.add_subplot(222)
ax2.set_xlabel('Error type')
ax2.set_ylabel('N')
name = 'clasification errors x' + str(usetrace) + " threshold " + str(threshold) + " trigger " + str(trigger)
plt.title(name)
plt.yscale('log')
plt.hist(myerrortypex, bins=mybins,alpha=0.8,label="Total",density=True)

ax3=fig1.add_subplot(223)
ax3.set_xlabel('Error type')
ax3.set_ylabel('N')
name = 'clasification errors y' + str(usetrace) + " threshold " + str(threshold) + " trigger " + str(trigger)
plt.title(name)
plt.yscale('log')
plt.hist(myerrortypey, bins=mybins,alpha=0.8,label="Total",density=True)

ax4=fig1.add_subplot(224)
ax4.set_xlabel('Error type')
ax4.set_ylabel('N')
name = 'clasification errors z ' + str(usetrace) + " threshold " + str(threshold) + " trigger " + str(trigger)
plt.title(name)
plt.yscale('log')
plt.hist(myerrortypez, bins=mybins,alpha=0.8,label="Total",density=True)

plt.tight_layout()

##############Plot histogram of relative errors, for all components####################################
fig2 = plt.figure(2,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
mybins = np.linspace(-4,0,79)

ax1=fig2.add_subplot(221)
ax1.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_ylabel('N')
name = 'overall errors ' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
plt.hist(np.log10(myInterpErrAll), bins=mybins,alpha=0.8,label="Total")

ax2=fig2.add_subplot(222)
ax2.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax2.set_ylabel('N')
name = 'overall errors x' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
plt.hist(np.log10(myInterpErrAllx), bins=mybins,alpha=0.8,label="x")

ax3=fig2.add_subplot(223)
ax3.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax3.set_ylabel('N')
name = 'overall errors y' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
plt.hist(np.log10(myInterpErrAlly), bins=mybins,alpha=0.8,label="y")

ax4=fig2.add_subplot(224)
ax4.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax4.set_ylabel('N')
name = 'overall errors z' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
plt.hist(np.log10(myInterpErrAllz), bins=mybins,alpha=0.8,label="z")

plt.tight_layout()


####################Plot 2d histogram, relative errors vs signal, all components##################################3

ind = np.where(myP2pAll != 0) #now i remove the cases where the signal is 0
myInterpErrAll2 = myInterpErrAll[ind]
myP2pAll2= myP2pAll[ind]

indx = np.where(myP2pAllx != 0) #now i remove the cases where the signal is 0
myInterpErrAll2x = myInterpErrAllx[indx]
myP2pAll2x= myP2pAll[indx]

indy = np.where(myP2pAlly != 0) #now i remove the cases where the signal is 0
myInterpErrAll2y = myInterpErrAlly[indy]
myP2pAll2y= myP2pAlly[ind]

indz = np.where(myP2pAllz != 0) #now i remove the cases where the signal is 0
myInterpErrAll2z = myInterpErrAllz[indz]
myP2pAll2z= myP2pAllz[indz]


print(np.shape(P2pAll),np.shape(P2pAllx),np.shape(P2pAlly),np.shape(P2pAllz))
print(np.shape(myP2pAll2),np.shape(myP2pAll2x),np.shape(myP2pAll2y),np.shape(myP2pAll2z))


fig3 = plt.figure(3,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3.add_subplot(221)
name = ' Total ' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')
plt.hist2d(np.log10(myP2pAll2),np.log10(myInterpErrAll2),bins=[100,79],range=[[-2, 3], [-4, 0]])


ax2=fig3.add_subplot(222)
name = ' x ' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
ax2.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax2.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')
plt.hist2d(np.log10(myP2pAll2x),np.log10(myInterpErrAll2x),bins=[100,79],range=[[-2, 3], [-4, 0]])

ax3=fig3.add_subplot(223)
name = ' y ' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
ax3.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax3.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')
plt.hist2d(np.log10(myP2pAll2y),np.log10(myInterpErrAll2y),bins=[100,79],range=[[-2, 3], [-4, 0]])

ax4=fig3.add_subplot(224)
name = ' z ' + str(usetrace) + " threshold " + str(threshold)
plt.title(name)
ax4.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax4.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')
plt.hist2d(np.log10(myP2pAll2z),np.log10(myInterpErrAll2z),bins=[100,79],range=[[-2, 3], [-4, 0]])
plt.tight_layout()



##########################Plot scatter 2d errors vs signal all components###########################3


fig4 = plt.figure(4,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig4.add_subplot(221)
name = 'scatter overall errors'
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')

plt.scatter(np.log10(myP2pAll2),np.log10(myInterpErrAll2), s=1)
plt.xlim(-2,3)
plt.ylim(-4, 0)

ax2=fig4.add_subplot(222)
name = 'scatter overall errors x'
plt.title(name)
ax2.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax2.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')

plt.scatter(np.log10(myP2pAll2x),np.log10(myInterpErrAll2x), s=1)
plt.xlim(-2,3)
plt.ylim(-4, 0)


ax3=fig4.add_subplot(223)
name = 'scatter overall errors y'
plt.title(name)
ax3.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax3.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')

plt.scatter(np.log10(myP2pAll2y),np.log10(myInterpErrAll2y), s=1)
plt.xlim(-2,3)
plt.ylim(-4, 0)

ax4=fig4.add_subplot(224)
name = 'scatter overall errors z'
plt.title(name)
ax4.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax4.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')

plt.scatter(np.log10(myP2pAll2z),np.log10(myInterpErrAll2z), s=1)
plt.xlim(-2,3)
plt.ylim(-4, 0)

plt.tight_layout()




plt.show()
