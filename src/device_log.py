from os import path, getcwd
import sys

THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from device_api.Argument_Parser import Device_Log_Arg_Parser
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from modbus_api.Modbus_Registers_Exceptions import ModbusRequestFailedException
from device_api.Device_Database import find_device,look_for_device_in_db
from device_api.Logger import Logger
from script_stopper import ScriptStopper

arg_parser = Device_Log_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

ss = ScriptStopper()
while(True):
    try:
        if(options.ip_address):
            modbus = Modbus_Handler_TCPIP(slave_ids=[options.slave_address],ip_address=options.ip_address,port=80)
        else:
            modbus = Modbus_Handler_Serial_RTU(slave_ids=[options.slave_address],port=options.serial_port)
        dev = find_device(modbus,options.slave_address)
        look_for_device_in_db(dev.metadata.uid)
        dev.print_info()
        logger = Logger(dev.get_data_type())
        while(True):
            dev.read_data(True,options.pc_synch)
            logger.write_data(dev.get_data())
            if(ss.script_stopped()):
                logger.write_file(options.output_file_name)
                ss.stop()
                exit()
    except KeyboardInterrupt:
        logger.write_file(options.output_file_name)
        exit()
    except ModbusRequestFailedException:
        print('Connection Error, reconnecting...')