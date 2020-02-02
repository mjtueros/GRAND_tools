import sqlite3
import logging
import datetime

def CreateAndConnectToDataBase(databasename):
  DataBase = sqlite3.connect(databasename)
  cursor = DataBase.cursor()
  cursor.execute("CREATE TABLE IF NOT EXISTS showers(id integer PRIMARY KEY AUTOINCREMENT, JobName text UNIQUE, status text, directory text, taskname text, tries int, energy float, zenith float, azimuth float, primry text, slantxmax float, d2xmax float)")
  logging.debug('Opening database' + databasename)
  cursor.close()
  return DataBase

def ConnectToDataBase(databasename):
  DataBase = sqlite3.connect(databasename)
  cursor = DataBase.cursor()
  logging.debug('Opening database' + databasename)
  cursor.close()
  return DataBase


#########################################################################################################
#while getting a record with DatabaseRecord=cursor.fetchone(), the returned element is a tuple.
def GetIdFromRecord(DatabaseRecord):
    return DatabaseRecord[0]

def GetNameFromRecord(DatabaseRecord):
    return DatabaseRecord[1]

def GetStatusFromRecord(DatabaseRecord):
    return DatabaseRecord[2]

def GetDirectoryFromRecord(DatabaseRecord):
    return DatabaseRecord[3]

def GetTasknameFromRecord(DatabaseRecord):
    return DatabaseRecord[4]

def GetTriesFromRecord(DatabaseRecord):
    return int(DatabaseRecord[5])

def GetEnergyFromRecord(DatabaseRecord):
    return float(DatabaseRecord[6])

def GetZenithFromRecord(DatabaseRecord):
    return float(DatabaseRecord[7])

def GetAzimuthFromRecord(DatabaseRecord):
    return float(DatabaseRecord[8])

def GetPrimaryFromRecord(DatabaseRecord):
    return str(DatabaseRecord[9])

def GetXmaxFromRecord(DatabaseRecord):
    return float(DatabaseRecord[10])

def GetD2XmaxFromRecord(DatabaseRecord):
    return float(DatabaseRecord[11])

def CreateNewDatabaseRecord(DataBase,config,JobName):
    #new entries on the database are only created when a new job is added to the queue. So, the starting status is always "Queued"
    #the only available information is the current directory of the job, that will be the workbench, becouse that is where all jobs start, and that this is the firsttry
    #the rest of the fields are initialized as -1 for numeric fields, and N/A for text fields
    #the database functionality automatically checks for duplicate entries (job name has to be unique), and assigns a new id.

    #JobName text, status text, directory text, taskname text, tries int, energy float, zenith float, azimuth float, prim text, slantxmax float, d2xmax float)")
    record=(JobName ,'Queued', config.WorkbenchDir+"/"+JobName,"N/A",1,-1.0 , -1.0, -1.0, "N/A",-1.0,-1.0)
    cursor = DataBase.cursor()
    cursor.execute('INSERT INTO showers(JobName, status, directory, taskname, tries, energy, zenith, azimuth, primry, slantxmax, d2xmax) VALUES( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', record)
    lastrowid=cursor.lastrowid
    cursor.close()
    return lastrowid

def CheckIfJobNameExists(DataBase,JobName):
    cursor = DataBase.cursor()
    cursor.execute("SELECT * FROM showers WHERE JobName = '%s'" % JobName)
    DatabaseSelection=cursor.fetchall()
    itexists=len(DatabaseSelection)
    cursor.close()
    return(itexists)

def FetchJobRecord(DataBase,JobName):
    cursor = DataBase.cursor()
    cursor.execute("SELECT * FROM showers WHERE JobName = '%s'" % JobName)
    DatabaseSelection=cursor.fetchall()
    cursor.close()
    #To Do: Handle this in a more profesional and elegant way.
    if(len(DatabaseSelection)==1):
      return(DatabaseSelection[0])
    else:
      return(DatabaseSelection)

#######################################################################################################
def GetCurrentJobStatus(StatusFile):
# this function gets the last Status from the status file
# it returns the time of the last status, and the status.
# To do: Handle the error when the file is not found
  try:
    with open(StatusFile, 'r') as f:
      lines = f.read().splitlines()
      last_line = lines[-1]
      splited_line=last_line.rsplit(' ', 1)
      #date_time_obj = datetime.datetime.strptime(splited_line[0], '%Y-%m-%d %H:%M:%S.%f')
      #print('Date:', date_time_obj.date())
      #print('Time:', date_time_obj.time())
  except IOError:
      logging.critical("could not open StatusFile or not found:"+StatusFile)
      splited_line=("0","FileNotFound")
  finally:
    return splited_line[0],splited_line[1]


def UpdateJobStatus(StatusFileName,Status):
    now=datetime.datetime.now()
    with open(StatusFileName,'a') as f:
      f.write(str(now) + " " + Status +"\n")

#########################################################################################################
def UpdateRecordStatus(DataBase,Id,NewStatus):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set status = \""+NewStatus+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set status = \""+NewStatus+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordDirectory(DataBase,Id,NewDirectory):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set directory = \""+NewDirectory+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set directory = \""+NewDirectory+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)


def UpdateRecordAzimuth(DataBase,Id,Azimuth):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set azimuth = \""+str(Azimuth)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set azimuth = \""+str(Azimuth)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordZenith(DataBase,Id,Zenith):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set zenith = \""+str(Zenith)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set zenith = \""+str(Zenith)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordEnergy(DataBase,Id,Energy):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set energy = \""+str(Energy)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set energy = \""+str(Energy)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordPrimary(DataBase,Id,Primary):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set primry = \""+str(Primary)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set primry = \""+Primary+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)


def UpdateRecordSlantXmax(DataBase,Id,SlantXmax):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set slantxmax = \""+str(SlantXmax)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set slantxmax = \""+str(SlantXmax)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordD2Xmax(DataBase,Id,D2Xmax):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set d2xmax = \""+str(D2Xmax)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set d2xmax = \""+str(D2Xmax)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)


def UpdateRecordTaskName(DataBase,Id,TaskName):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set taskname = \""+str(TaskName)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set taskname = \""+str(TaskName)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)

def UpdateRecordTries(DataBase,Id,Tries):
    try:
        cursor = DataBase.cursor()
        sql_update_query = "Update showers set tries = \""+str(Tries)+"\" where id = " + str(Id)
        #print(sql_update_query)
        cursor.execute(sql_update_query)
        DataBase.commit()
        cursor.close()

    except sqlite3.Error as error:
        sql_update_query = "Update showers set tries = \""+str(Tries)+"\" where id = " + str(Id)
        print("Failed to update sqlite table", error)
        print("with querry: "+sql_update_query)



##########################################################################################################
def GetNQueued(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'Queued'")
  NQueued = cursor.fetchall()
  cursor.close()
  return int(NQueued[0][0])

def GetNRunning(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'Running'")
  NRunning = cursor.fetchall()
  cursor.close()
  return int(NRunning[0][0])

def GetNRunComplete(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'RunComplete'")
  NRunComplete = cursor.fetchall()
  cursor.close()
  return NRunComplete[0][0]

def GetNRunOK(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'RunOK'")
  NRunOK = cursor.fetchall()
  cursor.close()
  return NRunOK[0][0]

def GetNRunFailed(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'RunFailed'")
  NRunFailed = cursor.fetchall()
  cursor.close()
  return NRunFailed[0][0]

def GetNRunCanceled(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers WHERE status = 'RunCanceled'")
  NRunCanceled = cursor.fetchall()
  cursor.close()
  return NRunCanceled[0][0]

def GetNTotal(DataBase):
  cursor = DataBase.cursor()
  cursor.execute("SELECT count(*) FROM showers")
  NTotal = cursor.fetchall()
  cursor.close()
  return NTotal[0][0]


def GetDatabaseStatus(DataBase):

  NQueued = GetNQueued(DataBase)

  NRunning = GetNRunning(DataBase)

  NComplete = GetNRunComplete(DataBase)

  NOK = GetNRunOK(DataBase)

  NFailed = GetNRunFailed(DataBase)

  NCanceled = GetNRunCanceled(DataBase)

  logging.info("#########################################################################")
  logging.info("# Database Status                                                       #")
  logging.info("#########################################################################")
  logging.info("# Queued Jobs : " + str(NQueued))
  logging.info("# Running Jobs: " + str(NRunning))
  logging.info("# RunComplete : " + str(NComplete))
  logging.info("# RunOK       : " + str(NOK))
  logging.info("# RunCanceled : " + str(NCanceled))
  logging.info("# RunFailed   : " + str(NFailed))
  logging.info("#########################################################################")
