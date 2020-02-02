'''
Count triggered antennas for one shower, for a set of layouts
Performs an interpolation of the traces of the peak-to-peak electric field 
at new layout antenna positions

'''
import sqlite3   #for the database
import argparse  #for command line parsing
import sys
import logging  
import DatabaseFunctions as mydatabase  #my database handling library
import interpol_func as intf
import trigger 
import trace_interpol
import grids
import numpy as np




def ant_trig(Directory,DatabaseRecord, rectStep, hexStep, hexrandStep=None):

    # get general information on shower
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)
    JobDirectory = str(Directory)+"/"+str(JobName)
    print(JobDirectory)
    Energy = mydatabase.GetEnergyFromRecord(DatabaseRecord)
    Zenith = mydatabase.GetZenithFromRecord(DatabaseRecord)
    Azimuth = mydatabase.GetAzimuthFromRecord(DatabaseRecord)
    Primary = mydatabase.GetPrimaryFromRecord(DatabaseRecord)
    Xmax = mydatabase.GetXmaxFromRecord(DatabaseRecord)

    # get original antenna position and peak-to-peak amplitudes
    AntNum, AntPos, AntID = intf.get_antenna_pos_zhaires(JobDirectory+'/antpos.dat')
    p2pE = intf.get_p2p(JobDirectory,AntNum)

    # perform trace interpolation on a set of layouts

    # rect
    for r in rectStep:
        pathfile = Directory + '/new_antpos_rect_%d.dat'%r
        print(pathfile)
        NumAnt, NewPos, ID_ant = intf.get_antenna_pos_zhaires(pathfile)
        p2p_total_new = intf.interpol(Directory,JobDirectory,AntPos,NewPos,p2pE,Zenith,Azimuth,'trace',DISPLAY=False)    
        # identify triggered antennas
        NT0, indT0 = trigger.trig(JobDirectory,AntPos,NewPos,p2p_total_new,Zenith,Azimuth,EThres=22,DISPLAY=False)

        # write antenna trigger information file
        TrigFile = JobDirectory + '/trig_ant_rect_%d.dat'%r
        np.savetxt(TrigFile, indT0[0], fmt = '%d')
            
    # hex
    for r in hexStep:
        pathfile = Directory + '/new_antpos_hex_%d.dat'%r
        AntNum, AntPos, AntID = intf.get_antenna_pos_zhaires(pathfile)
        p2p_total_new = intf.interpol(Directory,JobDirectory,AntPos,NewPos,p2pE,Zenith,Azimuth,'trace',DISPLAY=False)    
        # identify triggered antennas
        NT0, indT0 = trigger.trig(JobDirectory,AntPos,NewPos,p2p_total_new,Zenith,Azimuth,EThres=22,DISPLAY=False)

        # write antenna trigger information file
        TrigFile = JobDirectory + '/trig_ant_hex_%d.dat'%r
        np.savetxt(TrigFile, indT0[0], fmt = '%d')

    # hexrand
 #   for i in range(0,Nhexrand):
 #       pathfile = Directory + '/new_antpos_hexrand_%d.dat'%r
 #       NumAnt, NewPos, ID_ant = intf.get_antenna_pos_zhaires(pathfile)
 #       p2p_total_new = intf.interpol(Directory,JobDirectory,AntPos,NewPos,p2pE,Zenith,Azimuth,'trace',DISPLAY=False)    
        # identify triggered antennas
 #       NT0, indT0 = trigger.trig(JobDirectory,AntPos,NewPos,p2p_total_new,Zenith,Azimuth,EThres=22,DISPLAY=False)

        # write antenna trigger information file
 #       TrigFile = JobDirectory + '/trig_ant_hexrand_%d.dat'%r 
 #       np.savetxt(TrigFile, indT0[0], fmt = '%d')

            

def main():

    if ( len(sys.argv)<2 ):
        print("""
           
            Usage: python3 ant_trig.py <dir> <database>
            Example: python3 ant_trig.py <dir> <database>

            dir: path of shower library directory
            jobdir: path of specific shower
        """)
        sys.exit(0)

    Directory = sys.argv[1]
    results = sys.argv[2]


    # create rectangular antenna layouts with various steps
    rectStep = [100, 500, 1000, 2000] # in m
    for r in rectStep:
        NewPos = grids.create_grid_univ(Directory,'rect',r,DISPLAY=False)

    # create hexagonal antenna layouts with various radii
    hexStep = rectStep # radius of hexagon in m 
    for r in rectStep:
        NewPos = grids.create_grid_univ(Directory,'hexhex',r,DISPLAY=False)


    # create hexagonal antenna layouts with fixed radius 
    # and Nrand antennas randomly displaced at various rates
    #radius = 1000
    #NewPos = grids.create_grid_univ(Directory,'hexrand',radius,Nrand=100,randeff=0.2,DISPLAY=True)

    # fetch one event from database
    dbfile=results
    DataBase=mydatabase.ConnectToDataBase(dbfile)
    CurDataBase = DataBase.cursor()
    CurDataBase.execute("SELECT * FROM showers")
    DatabaseRecord = CurDataBase.fetchone()

    # check triggers
    ant_trig(Directory,DatabaseRecord, rectStep, hexStep)





print(__name__)
if __name__ == '__main__':
    main()
