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




def ant_trig_hdf5(InputFileName, rectStep, hexStep, hexrandStep=None, method='trace', usetrace='efield', threshold=0):

    # rect
    for r in rectStep:

        NewPos = grids.create_grid_univ(None,'rect',r,DISPLAY=True)

        OutputFileName= InputFileName + ".Interpolated.rect_"+str(r)+"_"+str(usetrace)+".hdf5"

        print(OutputFileName)

        p2p_total_new=intf.interpol_hdf5(InputFileName, OutputFileName, NewPos.T, InterpolMethod=method, threshold=threshold,usetrace=usetrace, DISPLAY=True)

        # write antenna trigger information file (this would be cool to have inside the interpolated hdf5
        P2PFile = InputFileName + ".Interpolated.rect_"+str(r)+"_"+str(usetrace)+".dat"
        np.savetxt(P2PFile, p2p_total_new)

    # hex
    #for r in hexStep:
    #
    #    NewPos = grids.create_grid_univ(None,'hexhex',r,DISPLAY=True)
    #
    #    OutputFileName= InputFileName + ".Interpolated.hexhex_"+str(r)+"_"+str(usetrace)+".hdf5"
    #
    #    print(OutputFileName)
    #
    #    p2p_total_new=intf.interpol_hdf5(InputFileName, OutputFileName, NewPos.T, InterpolMethod=method, threshold=threshold,usetrace=usetrace, DISPLAY=True)
    #
    #    # write antenna trigger information file (this would be cool to have inside the interpolated hdf5
    #    P2PFile = InputFileName + ".Interpolated.hexhex_"+str(r)+"_"+str(usetrace)+".dat"

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

            Usage: python3  <database>
            Example: python3  <database>

            remeber to changhe the variable Directory Inside
            It must point to the library base Output
        """)
        sys.exit(0)

    dbfile = sys.argv[1]
    Directory = "/home/mjtueros/GRAND/GP300/HDF5StshpLibrary/Outbox"



    # create rectangular antenna layouts with various steps
    rectStep = [125, 250, 500, 1000] # in m
    #for r in rectStep:
    #    NewPos = grids.create_grid_univ(Directory,'rect',r,DISPLAY=False)

    # create hexagonal antenna layouts with various radii
    hexStep = rectStep # radius of hexagon in m
    #for r in rectStep:
    #    NewPos = grids.create_grid_univ(Directory,'hexhex',r,DISPLAY=False)


    # create hexagonal antenna layouts with fixed radius
    # and Nrand antennas randomly displaced at various rates
    #radius = 1000
    #NewPos = grids.create_grid_univ(Directory,'hexrand',radius,Nrand=100,randeff=0.2,DISPLAY=True)

    # fetch one event from database
    DataBase=mydatabase.ConnectToDataBase(dbfile)
    CurDataBase = DataBase.cursor()
    CurDataBase.execute("SELECT * FROM showers")
    DatabaseRecord = CurDataBase.fetchone()
    JobName = mydatabase.GetNameFromRecord(DatabaseRecord)

    InputFileName=str(Directory)+"/"+str(JobName)+"/"+str(JobName)+".hdf5"

    # check triggers
    ant_trig_hdf5(InputFileName, rectStep, hexStep, hexrandStep=None, method='trace', usetrace='efield', threshold=0)





print(__name__)
if __name__ == '__main__':
    main()
