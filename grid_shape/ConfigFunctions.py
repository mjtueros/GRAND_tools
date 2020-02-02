import os
import logging

try:
    import configparser  # Python 3
except ImportError:
    import ConfigParser as configparser  # Python 2

def load(config_file):

    if config_file == '':
        sys.exit("configuration filename is expected")
    elif not os.path.isfile(config_file):
        logging.critical("Conf (%s) not found. Using defaults." % config_file)
        sys.exit("a valid configuration filename is expected")

    config = configparser.ConfigParser()
    config.read(config_file)

    # General settings
    sect = 'General'
    globals()['User'] = config.get(sect, 'User')
    globals()['MaxConcurrentJobs'] = config.get(sect, 'MaxConcurrentJobs')
    globals()['MaxQueuedJobs'] = config.get(sect, 'MaxQueuedJobs')
    globals()['MaxRetries'] = config.get(sect, 'MaxRetries')
    globals()['RunnerSleepsFor'] = config.get(sect, 'RunnerSleepsFor')

    # Library settings
    sect = 'Library'
    globals()['LibraryName'] = config.get(sect, 'Name')
    globals()['InboxDir'] = config.get(sect, 'InboxDir')
    globals()['OutboxDir'] = config.get(sect, 'OutboxDir')
    globals()['WorkbenchDir'] = config.get(sect, 'WorkbenchDir')
    globals()['ScratchDir'] = config.get(sect, 'ScratchDir')

    # Sofware settings
    sect = 'Software'
    globals()['UserBin'] = config.get(sect, 'UserBin')
    globals()['ProcessName'] = config.get(sect, 'ProcessName')
    globals()['UserLibrary'] = config.get(sect, 'UserLibrary')
    globals()['SystemLibrary'] = config.get(sect, 'SystemLibrary')

    logging.debug("Finished loading configuration from %s " % config_file)



def show():
    logging.info("This is Aires Library Runner v0.1")
    logging.info("============================================================================")
    logging.info('User ' + User + ' runing library ' + LibraryName)
    logging.info('Max Concurrent Jobs ' + MaxConcurrentJobs + ' MaxQueuedJobs ' + MaxQueuedJobs)
    logging.info('Max Retries ' + MaxRetries)
    logging.info("============================================================================")
    logging.info("Directories")
    logging.info("============================================================================")
    logging.info('Inbox:' + InboxDir)
    logging.info('Workbench:' + WorkbenchDir)
    logging.info('Outbox:' + OutboxDir)
    logging.info('Scratch:' + ScratchDir)
    logging.info("============================================================================")
    logging.info("Software")
    logging.info("============================================================================")
    logging.info('UserBin:' + UserBin)
    logging.info('ProcessName:' + ProcessName)
    logging.info('UserLibrary:' + UserLibrary)
    logging.info('SystemLibrary:' + SystemLibrary)
    logging.info("============================================================================")





