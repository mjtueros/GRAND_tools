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
import interpol_func_hdf5 as intf
import grids
import numpy as np
import hdf5fileinout as hdf5io




def grid_study_on_starshape_hdf5(InputJobName, rectStep, hexStep, hexrandStep=None, method='trace', usetrace='efield', threshold=0):

    InputFileName= InputJobName+str(".hdf5")

    CurrentRunInfo=hdf5io.GetRunInfo(InputFileName)
    CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
    CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFileName,CurrentEventName)
    Zenith=hdf5io.GetEventZenith(CurrentRunInfo,0)
    Azimuth=hdf5io.GetEventAzimuth(CurrentRunInfo,0)
    Primary=hdf5io.GetEventPrimary(CurrentRunInfo,0)
    Energy=hdf5io.GetEventEnergy(CurrentRunInfo,0)
    XmaxDistance=hdf5io.GetEventXmaxDistance(CurrentRunInfo,0)
    SlantXmax=hdf5io.GetEventSlantXmax(CurrentRunInfo,0)
    HadronicModel=hdf5io.GetEventHadronicModel(CurrentRunInfo,0)
    RandomAzimuth=np.random.uniform(0,180)

    print(RandomAzimuth)

    # rect
    for r in rectStep:

        NewPos,RandomCore = grids.create_grid_univ('rect',r,RandomAzimuth,do_offset=True,DISPLAY=True)

        OutputFileName= InputJobName + ".Interpolated.rect_"+str(r)+"_"+str(usetrace)+".hdf5"

        print(OutputFileName)

        p2p_total_new=intf.interpol_hdf5(InputFileName, OutputFileName, NewPos.T, InterpolMethod=method, threshold=threshold,usetrace=usetrace, DISPLAY=True)

        # write antenna trigger information file (this would be cool to have it also inside the interpolated hdf5)
        P2PFile = InputJobName + ".Interpolated.rect_"+str(r)+"_"+str(usetrace)+".P2Pdat"
        np.savetxt(P2PFile, p2p_total_new)

        FILE = open(P2PFile+str(".showerinfo"),"w" )
        print("%s %s %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %s" % (InputJobName,Primary,Energy,Zenith,Azimuth,XmaxDistance,SlantXmax,RandomCore[0],RandomCore[1],RandomAzimuth,HadronicModel), file=FILE)
        FILE.close()


    #hex
    for r in hexStep:

        NewPos,RandomCore = grids.create_grid_univ('hexhex',r,RandomAzimuth,do_offset=True,DISPLAY=True)

        OutputFileName= InputJobName + ".Interpolated.hexhex_"+str(r)+"_"+str(usetrace)+".hdf5"

        print(OutputFileName)

        p2p_total_new=intf.interpol_hdf5(InputFileName, OutputFileName, NewPos.T, InterpolMethod=method, threshold=threshold,usetrace=usetrace, DISPLAY=True)

        # write antenna trigger information file (this would be cool to have inside the interpolated hdf5
        P2PFile = InputJobName + ".Interpolated.hexhex_"+str(r)+"_"+str(usetrace)+".P2Pdat"
        np.savetxt(P2PFile, p2p_total_new)

        FILE = open(P2PFile+str(".showerinfo"),"w" )
        print("%s %s %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %1.5e %s" % (InputJobName,Primary,Energy,Zenith,Azimuth,XmaxDistance,SlantXmax,RandomCore[0],RandomCore[1],RandomAzimuth,HadronicModel), file=FILE)
        FILE.close()


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

            Usage: python3  <database/jobname>
            Example: python3  <database/jobname>

            remeber to changhe the variable Directory Inside
            During development in the laptop It must point to the library base Output
            On production is not used


        """)
        sys.exit(0)

    dbfile = sys.argv[1]
    Directory = "/home/mjtueros/GRAND/GP300/HDF5StshpLibrary/Outbox"



    # create rectangular antenna layouts with various steps
    rectStep = [137.5, 275, 550,825, 1100] # in m
    #for r in rectStep:
    #    NewPos = grids.create_grid_univ(Directory,'rect',r,DISPLAY=False)

    # create hexagonal antenna layouts with various radii
    hexStep = [125, 250, 500, 750, 1000] # radius of hexagon in m
    #for r in rectStep:
    #    NewPos = grids.create_grid_univ(Directory,'hexhex',r,DISPLAY=False)


    # create hexagonal antenna layouts with fixed radius
    # and Nrand antennas randomly displaced at various rates
    #radius = 1000
    #NewPos = grids.create_grid_univ(Directory,'hexrand',radius,Nrand=100,randeff=0.2,DISPLAY=True)

    # during development in my laptop, fetch one event from database and form a JobName including the path. On production, the input is just the job name (job runs on the event directory)
    DataBase=mydatabase.ConnectToDataBase(dbfile)
    CurDataBase = DataBase.cursor()
    CurDataBase.execute("SELECT * FROM showers")
    DatabaseRecord = CurDataBase.fetchone()
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)

    InputJobName=str(Directory)+"/"+str(JobName)+"/"+str(JobName)  #this is for developing in my laptop, in production the InputFilename=JobName
    #InputJobName=sys.argv[1]

    # check triggers
    grid_study_on_starshape_hdf5(InputJobName, rectStep, hexStep, hexrandStep=None, method='trace', usetrace='efield', threshold=0)

    grid_study_on_starshape_hdf5(InputJobName, rectStep, hexStep, hexrandStep=None, method='trace', usetrace='voltage', threshold=0)



print(__name__)
if __name__ == '__main__':
    main()
