from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from modbus_api.Modbus_Handler import Modbus_Handler_TCPIP
from device_api.ECGV3 import ECGV3
from drawnow import drawnow
from time import time


modbus = Modbus_Handler_TCPIP(slave_ids=[4],ip_address="192.168.1.13",port=80)
ecg = ECGV3(modbus_handler=modbus,slave_id=4)
ecg.print_info()
ecg.set_unix_epoch(int(time()))

while(1):
    data=ecg.read_data(only_new=False)
    drawnow(ecg.draw_with_plt,False, False,False)
