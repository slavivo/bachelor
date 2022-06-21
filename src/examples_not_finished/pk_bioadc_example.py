from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from modbus_api.Modbus_Handler import Modbus_Handler_TCPIP
from device_api.BioAdc import BioAdc
from time import time

modbus = Modbus_Handler_TCPIP(slave_ids=[4],ip_address="192.168.1.11",port=80,timeout=1)
bioAdc = BioAdc(modbus_handler=modbus,slave_id=4)
print(bioAdc.print_info())
times = []
bioAdc.set_unix_epoch(int(time()))
print(bioAdc.get_time_stamp_ms())
while(1):
    data = time()
    bioAdc.read_data()
    times.append(time()-data)
    if(len(times)>1000):
        times.pop(0)
