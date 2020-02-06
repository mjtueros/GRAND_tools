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

#directory where the files from the library are located
Directory = "/home/mjtueros/GRAND/GP300/HDF5StshpLibrary/Outbox"
#what to use in the interpolation (efield, voltage, filteredvoltage)
usetrace='efield'
#threshold abouve wich the interpolation is computed
threshold=26 #8.66 for 15uV , 26 for 45uV
display=False

parser = argparse.ArgumentParser(description='A script to get the CPU time in a library of Simulations')
parser.add_argument('DatabaseFile', #name of the parameter
                    metavar="filename", #name of the parameter value in the help
                    help='The Database of the library .db file') # help message for this parameter


results = parser.parse_args()
dbfile=results.DatabaseFile

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
InterpErrAll = np.zeros((16,1000))
DistanceAnt = np.zeros((16,1000))
P2pAll = np.zeros((16,1000))

#while(DatabaseRecord!=None and countok < 50): #500 events in 30min, withouth tresholding, 700 en 47 min
while(DatabaseRecord!=None):

    DatabaseStatus = mydatabase.GetStatusFromRecord(DatabaseRecord) #i do it with a function call becouse if we change the database structure we dont have to change this
    #Directory = mydatabase.GetDirectoryFromRecord(DatabaseRecord)
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)
    JobDirectory = str(Directory)+"/"+str(JobName)
    Tries = mydatabase.GetTriesFromRecord(DatabaseRecord)

    logging.debug("Reading Job " + JobName + " which was in " + DatabaseStatus + " status ("+str(Tries)+") at " + Directory)

# 0.09665536 0.05555899 0.11996197 0.02175862 0.01625389 0.00539286
# 0.02177963 0.40966389 0.02274206 0.19980328 0.00483166 0.02463318
# 0.13643482 0.00595974 0.04243686 0.01210873

    if(DatabaseStatus == "RunOK"): #and JobName=="Stshp_XmaxLibrary_0.1995_80.40_180_Gamma_01"):
        try:
            TaskName = mydatabase.GetTasknameFromRecord(DatabaseRecord)
            Id = mydatabase.GetIdFromRecord(DatabaseRecord)
            Energy = mydatabase.GetEnergyFromRecord(DatabaseRecord)
            Zenith = mydatabase.GetZenithFromRecord(DatabaseRecord)
            Azimuth = mydatabase.GetAzimuthFromRecord(DatabaseRecord)
            Primary = mydatabase.GetPrimaryFromRecord(DatabaseRecord)
            Xmax = mydatabase.GetXmaxFromRecord(DatabaseRecord)


            #unncecesary?
            #Altitude,Distance,x,y,z = AiresInfo.GetKmXmaxFromSry(str(Directory)+"/"+str(JobName)+"/"+str(TaskName)+".sry","N/A")

            InputFilename=str(Directory)+"/"+str(JobName)+"/"+str(JobName)+".hdf5"


            CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
            CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
            CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)

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

            p2pE = intf.get_p2p_hdf5(InputFilename,antennamax=175,antennamin=0,usetrace=usetrace)
            #print(np.shape(AntPos))


            NewPos = grids.create_grid(AntPos[0:15],Zenith,'check',20,10) #In Check mode, it will return the last 16 elements of Antpos, so this just Antpos[160:175]
            #print(np.shape(NewPos))

            InterpErr,p2p_total_new = intf.interpol_check_hdf5(InputFilename, AntPos, NewPos.T, p2pE,'trace',threshold=threshold, usetrace=usetrace,DISPLAY=display)

            InterpErrAll[:,countok] = InterpErr
            DistanceAnt[:,countok] = np.sqrt(NewPos[0,:]**2 + NewPos[1,:]**2 + NewPos[2,:]**2)
            P2pAll[:,countok] = p2p_total_new


            countok += 1
            print("Event #{} done".format(countok))


        except FileNotFoundError:
          logging.error("ant_interpol_chk_db:file not found or invalid:"+TaskName)
          counterr += 1


    #this is the last order of the while, that will fetch the next record of the database
    DatabaseRecord=CurDataBase.fetchone()


logging.debug('ant_interpol_chk_db: Plotting...')

fig2 = plt.figure(1,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig2.add_subplot(111)
name = 'overall errors ' + str(usetrace) + " threshold " + str(thrshold)
plt.title(name)
ax1.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_ylabel('N')

ind = np.where(InterpErrAll != 0) #here i remove the cases where the error is 0. Shouldnt happen?
myInterpErrAll = InterpErrAll[ind]
myP2pAll= P2pAll[ind]
print(np.shape(InterpErrAll))
print(np.shape(myInterpErrAll))


myhist, mybins = np.histogram(np.log10(myInterpErrAll),bins=30)
plt.hist(np.log10(myInterpErrAll), bins=mybins)
plt.tight_layout()



fig3 = plt.figure(2,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3.add_subplot(111)
name = 'overall errors ' + str(usetrace) + " threshold " + str(thrshold)
plt.title(name)
ax1.set_ylabel('$|E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('Antenna distance from core [m]')

ind = np.mgrid[0:countok:1]
dist = DistanceAnt[:,ind].flatten()
interr = InterpErrAll[:,ind].flatten()

plt.hist2d(dist,interr,bins=30)
plt.tight_layout()



fig3 = plt.figure(3,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3.add_subplot(111)
name = 'overall errors ' + str(usetrace) + " threshold " + str(thrshold)
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('$E_{sim}  [\mu V/m]$')


ind = np.where(myP2pAll != 0) #now i remove the cases where the signal is 0
myInterpErrAll2 = myInterpErrAll[ind]
myP2pAll2= myP2pAll[ind]

print(np.shape(myInterpErrAll2))
print(np.shape(myP2pAll2))

p2p = myP2pAll2


plt.hist2d(np.log10(p2p),np.log10(myInterpErrAll2),bins=100)
plt.tight_layout()



fig3b = plt.figure(3,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3b.add_subplot(111)
name = 'scatter overall errors'
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('$log_{10} E_{sim}  [\mu V/m]$')

plt.scatter(np.log10(p2p),np.log10(myInterpErrAll2), s=1)
plt.tight_layout()




plt.show()
