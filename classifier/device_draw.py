from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from modbus_api.Modbus_Registers_Exceptions import ModbusRequestFailedException
from device_api.Argument_Parser import Default_Arg_Parser
from device_api.Device_Database import find_device,look_for_device_in_db
from drawnow import drawnow
from script_stopper import ScriptStopper

arg_parser = Default_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

num_retries = 5

ss = ScriptStopper()
for i in range(num_retries):
    try:
        if(options.ip_address):
            modbus = Modbus_Handler_TCPIP(slave_ids=[int(options.slave_address)],ip_address=options.ip_address,port=80)
        else:
            modbus = Modbus_Handler_Serial_RTU(slave_ids=[int(options.slave_address)],port=options.serial_port)
        dev = find_device(modbus,int(options.slave_address))
        look_for_device_in_db(dev.metadata.uid)
        dev.print_info()
        while(1):
            dev.read_data(False,options.pc_synch)
            drawnow(dev.draw_with_plt,False, False,False)
            if(ss.script_stopped()):
                ss.stop()
                exit()
    except KeyboardInterrupt:
        exit()
    except ModbusRequestFailedException:
        print('Connection Error, reconnecting...Attempt %d/%d' % ((i + 1), num_retries))
