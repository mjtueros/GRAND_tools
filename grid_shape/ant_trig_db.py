'''
Count triggered antennas for each shower in shower database, given a layout
Performs an interpolation of the peak-to-peak electric field at new layout antenna positions

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
import trigger 
import trace_interpol
import grids


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

logging.debug("Starting trigger analysis with %s " % dbfile)

DataBase=mydatabase.ConnectToDataBase(dbfile)
#this is to show the current status of the database
mydatabase.GetDatabaseStatus(DataBase)

#This is how you search on the database, here im selecting everything (To Do: functions to search the database)
#This is to get a cursor on the database. You can think of the cursor as a working environment. You can have many cursors.
CurDataBase = DataBase.cursor()
CurDataBase.execute("SELECT * FROM showers")

DatabaseRecord = CurDataBase.fetchone()


# directory of library
Directory = "/Users/kotera/BROQUE/Data_GRAND/Matias/StshpLibrary01"

# counters
countok = 0 # number of showers treated in the database
NT0All = np.zeros(284) # array of number of triggered antennas for each shower

# new antenna layout
radius = 1000#1000 # radius of hexagon in m 
NewPos = grids.create_grid_univ(Directory,'rect',radius,DISPLAY=True)


# file for storing triggered antenna information
#TrigFile = open(Directory + '/trig_ant.dat',"w+" )

#while(DatabaseRecord!=None and countok < 30):
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
            p2p_total_new = intf.interpol(Directory,JobDirectory,AntPos,NewPos,p2pE,Zenith,Azimuth,'trace',DISPLAY=False)    

            NT0, indT0 = trigger.trig(JobDirectory,AntPos,NewPos,p2p_total_new,Zenith,Azimuth,EThres=22,DISPLAY=False)
            
            NT0All[countok] = NT0

      #      f.write(str(Id)+", "+str(Energy)+", "+str(Zenith)+", "+str(Azimuth)+", "+str(Primary)+", "+str(Xmax)+", "+str(Altitude)+", "+str(x)+", "+str(y)+", "+str(z)+"\n")

            #s1 = [ "%d "%i for i in indT0[0]] 
            #s2 = ''
            #for s in s1: 
            #    s2+=s 
            #s2 += '\n'
            #TrigFile.write(s2)

            TrigFile = JobDirectory + '/trig_ant.dat'
            np.savetxt(TrigFile, indT0[0], fmt = '%d')
            
            countok += 1
            print("Event #{} done".format(countok))


        except FileNotFoundError:
          logging.error("ant_trig_db:file not found or invalid:"+TaskName)
          counterr += 1


    #this is the last order of the while, that will fetch the next record of the database
    DatabaseRecord=CurDataBase.fetchone()


# close trigger information file
#TrigFile.close()

# calculate triggered events
NThres = 5
indT1 = np.where(NT0All >= NThres)
NT1 = np.size(indT1)  # number of triggered events

# plot histogram of number of triggered antennas among countok showers 
logging.debug('ant_trig_db: Plotting...')

fig2 = plt.figure(figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')
ax1=fig2.add_subplot(111)
name = 'triggered antennas'
plt.title(name)
ax1.set_xlabel('$N_{T0} \,per \,event$')
ax1.set_ylabel('N')
myNT0All = NT0All[0:countok-1]
myhist, mybins = np.histogram(myNT0All,bins=30)
plt.hist(myNT0All, bins=mybins)
plt.tight_layout()

plt.show(block=False)
