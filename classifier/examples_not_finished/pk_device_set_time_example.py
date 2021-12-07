from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from device_api.Argument_Parser import Default_Arg_Parser
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from device_api.Device_Database import find_device
from time import time, sleep

arg_parser = Default_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

if(options.ip_address):
    modbus = Modbus_Handler_TCPIP(slave_ids=[int(options.slave_address)],ip_address=options.ip_address,port=80)
else:
    modbus = Modbus_Handler_Serial_RTU(slave_ids=[int(options.slave_address)],port=options.serial_port)

dev = find_device(modbus,int(options.slave_address))
dev.print_info()

if('rtc' in dev.__annotations__.keys()):
    print('RTC available')
    timeNow = int(time())
    dev.set_unix_epoch(timeNow)
    sleep(0.5)
    print('Time set:',dev.get_unix_epoch())
else:
    print('RTC is not available, cant set time')