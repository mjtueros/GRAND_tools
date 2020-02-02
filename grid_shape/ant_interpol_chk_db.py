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
from matplotlib.pyplot import cm 


import DatabaseFunctions as mydatabase  #my database handling library
import AiresInfoFunctions as AiresInfo
import interpol_func as intf
import grids

Directory = "/Users/kotera/BROQUE/Data_GRAND/Matias/StshpLibrary01"

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
logging.disable(logging.DEBUG)

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
InterpErrAll = np.zeros((16,284))
DistanceAnt = np.zeros((16,284))
P2pAll = np.zeros((16,284))

#while(DatabaseRecord!=None and countok < 50):
while(DatabaseRecord!=None):

    DatabaseStatus = mydatabase.GetStatusFromRecord(DatabaseRecord) #i do it with a function call becouse if we change the database structure we dont have to change this
    #Directory = mydatabase.GetDirectoryFromRecord(DatabaseRecord)
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)
    JobDirectory = str(Directory)+"/"+str(JobName)
    Tries = mydatabase.GetTriesFromRecord(DatabaseRecord)

    logging.debug("Reading Job " + JobName + " which was in " + DatabaseStatus + " status ("+str(Tries)+") at " + Directory)

    if(DatabaseStatus == "RunOK"):
        try:
            TaskName = mydatabase.GetTasknameFromRecord(DatabaseRecord)
            Id = mydatabase.GetIdFromRecord(DatabaseRecord)
            Energy = mydatabase.GetEnergyFromRecord(DatabaseRecord)
            Zenith = mydatabase.GetZenithFromRecord(DatabaseRecord)
            Azimuth = mydatabase.GetAzimuthFromRecord(DatabaseRecord)
            Primary = mydatabase.GetPrimaryFromRecord(DatabaseRecord)
            Xmax = mydatabase.GetXmaxFromRecord(DatabaseRecord)
            Altitude,Distance,x,y,z = AiresInfo.GetKmXmaxFromSry(str(Directory)+"/"+str(JobName)+"/"+str(TaskName)+".sry","N/A")

            AntNum, AntPos, AntID = intf.get_antenna_pos_zhaires(JobDirectory+'/antpos.dat')
            p2pE = intf.get_p2p(JobDirectory,AntNum)
            NewPos = grids.create_grid(JobDirectory,AntPos,Zenith,'check',20,10)
            InterpErr,p2p_total_new = intf.interpol_check(JobDirectory,AntPos,NewPos,p2pE,Zenith,Azimuth,'trace',DISPLAY=False)
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
name = 'overall errors'
plt.title(name)
ax1.set_xlabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_ylabel('N')
ind = np.where(InterpErrAll != 0)
myInterpErrAll = InterpErrAll[ind]
myhist, mybins = np.histogram(np.log10(myInterpErrAll),bins=30)
plt.hist(np.log10(myInterpErrAll), bins=mybins)
plt.tight_layout()



fig3 = plt.figure(2,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3.add_subplot(111)
name = 'overall errors'
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('Antenna distance from core [m]')

ind = np.mgrid[0:countok:1]
dist = DistanceAnt[:,ind].flatten()
interr = InterpErrAll[:,ind].flatten()

plt.hist2d(dist,interr,bins=30)
plt.tight_layout()



fig3 = plt.figure(3,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig3.add_subplot(111)
name = 'overall errors'
plt.title(name)
ax1.set_ylabel('$log_{10} |E_{int}-E_{sim}|/E_{sim}$')
ax1.set_xlabel('$E_{sim}  [\mu V/m]$')

p2p = P2pAll[:,ind].flatten()


plt.hist2d(p2p,interr,bins=30)
plt.tight_layout()

plt.show(block=False)
