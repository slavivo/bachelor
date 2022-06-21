# Mandatory import section
from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU
from device_api.EEGV1 import EEGV1
from drawnow import drawnow

modbus = Modbus_Handler_Serial_RTU(slave_ids=[4],port="/dev/ttyACM0")
eeg = EEGV1(modbusHandler=modbus,slaveId=4)
eeg.print_info()

while(1):
    data=eeg.read_data(only_new=True)
    drawnow(eeg.draw,False, False,False)
