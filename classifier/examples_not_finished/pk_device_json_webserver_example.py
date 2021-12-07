from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from modbus_api.Modbus_Registers_Exceptions import ModbusRequestFailedException
from device_api.Device_Database import find_device
from device_api.Logger import Logger
from device_api.Argument_Parser import Web_Server_Access_Arg_Parser
from flask import Flask
import threading

arg_parser = Web_Server_Access_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

mutex = threading.Lock() 
api = Flask(__name__)
devDataRead = False
dl = None

def modbusReader():
    while(1):
        try:
            global dl
            global devDataRead
            global options
            if(options.ip_address):
                modbus = Modbus_Handler_TCPIP(slave_ids=[int(options.slave_address)],ip_address=options.ip_address,port=80)
            else:
                modbus = Modbus_Handler_Serial_RTU(slave_ids=[int(options.slave_address)],port=options.serial_port)
            dev = find_device(modbus,int(options.slave_address))
            dev.print_info()
            logger = Logger(dev.get_data_type())
            while(True):
                data = dev.read_data()
                if(devDataRead):
                    devDataRead = False
                mutex.acquire()
                logger.write_data(data)
                mutex.release()
        except KeyboardInterrupt:
            exit()
        except ModbusRequestFailedException:
            print('Connection Error, reconnecting...')

@api.route('/data', methods=['GET'])
def get_dev():
    global logger
    global devDataRead
    if(dl!=None):
        mutex.acquire()
        rslt = logger.get_data()
        devDataRead=True
        mutex.release()
        return rslt
    return {}

if __name__ == '__main__':
    dataReadThead = threading.Thread(target=modbusReader)
    dataReadThead.start()
    api.run(host=options.host,port=options.outputPort) 

