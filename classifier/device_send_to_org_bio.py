from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
ORG_BIO_API_DIR = path.abspath(path.join(THIS_DIR, 'org_bio_api'))

sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)
sys.path.append(ORG_BIO_API_DIR)

from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from modbus_api.Modbus_Registers_Exceptions import ModbusRequestFailedException
from device_api.Device_Database import find_device
from device_api.Logger import Logger
from device_api.Argument_Parser import Default_Arg_Parser
from org_bio_api.Org_Bio_Ids_Handler import pk_data_to_org_bio_data
import threading
from script_stopper import ScriptStopper
import time
import uuid
import requests
import json
import difflib

# printing the value of unique MAC
# address using uuid and getnode() function 

arg_parser = Default_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

mutex = threading.Lock() 
devDataRead = False
logger = None
mac=hex(uuid.getnode())[2:]
mac = '%s:%s:%s:%s:%s:%s' % (mac[0:2],mac[2:4],mac[4:6],mac[6:8],mac[8:10],mac[10:12])
uid = None
deviceTypeId = None

ss = ScriptStopper()

def modbusReader():
    while(1):
        try:
            global logger
            global devDataRead
            global options
            global uid
            global deviceTypeId
            if(options.ip_address):
                modbus = Modbus_Handler_TCPIP(slave_ids=[int(options.slave_address)],ip_address=options.ip_address,port=80)
            else:
                modbus = Modbus_Handler_Serial_RTU(slave_ids=[int(options.slave_address)],port=options.serial_port)
            dev = find_device(modbus,int(options.slave_address))
            dev.print_info()
            uid = dev.metadata.uid
            deviceTypeId = dev.metadata.device_type_id
            logger = Logger(dev.get_data_type())
            while(True):
                dev.read_data(True,options.pc_synch)
                if(devDataRead):
                    devDataRead = False
                mutex.acquire()
                logger.write_data(dev.get_data())
                mutex.release()
                if(ss.script_stopped()):
                    ss.stop()
                    exit()
        except KeyboardInterrupt:
            exit()
        except ModbusRequestFailedException:
            print('Connection Error, reconnecting...')


def bioOrgWriter():
    global logger
    global devDataRead
    global deviceTypeId
    while(1):
        try:
            if(logger):
                data=logger.get_data()
                convertedDat = json.dumps(pk_data_to_org_bio_data(data.__dict__,mac,uid,deviceTypeId))
                hdrs={
                    'Content-Type': 'application/json',
                    'accept':'*/*'
                }
                r = requests.post("https://org-bio.azurewebsites.net/api/Device", data = convertedDat,headers=hdrs)
            if(ss.script_stopped()):
                ss.stop()
                exit()
            time.sleep(1)
        except KeyboardInterrupt:
            exit()

if __name__ == '__main__':
    dataReadThead = threading.Thread(target=modbusReader)
    dataReadThead.start()
    time.sleep(1)
    bioOrgWriterThread = threading.Thread(target=bioOrgWriter)
    bioOrgWriterThread.start()
