from os import path, getcwd
import sys
THIS_DIR = getcwd()
MODBUS_API_DIR = path.abspath(path.join(THIS_DIR, 'modbus_api'))
DEVICE_API_DIR = path.abspath(path.join(THIS_DIR, 'device_api'))
sys.path.append(MODBUS_API_DIR)
sys.path.append(DEVICE_API_DIR)

from device_api.Argument_Parser import Device_DB_Access_Arg_Parser
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU,Modbus_Handler_TCPIP
from device_api.Device_Database import find_device
import json
import datetime

arg_parser = Device_DB_Access_Arg_Parser()
arg_parser.parse(sys.argv[1:])
options = arg_parser.get_options()

if(options.ip_address):
    modbus = Modbus_Handler_TCPIP(slave_ids=[int(options.slave_address)],ip_address=options.ip_address,port=80)
else:
    modbus = Modbus_Handler_Serial_RTU(slave_ids=[int(options.slave_address)],port=options.serial_port)

dev = find_device(modbus,int(options.slave_address))
firmware_info = dev.get_firmware_info()
metadata = dev.get_metadata()

f = open("device_db.json") 
structure = json.load(f)
f.close()
if(not(metadata.uid in structure)):
    print('Unlabeled device found, labeling')
    newId=len(structure.keys())

    structure[metadata.uid]={
        'pkId':newId,
        'deviceName':str(options.device_name),
        'deviceTypeName':metadata.device_type_name,
        'projectName':str(options.project_name),
        'uploadTime':firmware_info.upload_time.strftime("%Y/%m/%d, %H:%M:%S"),
        'labelTime':datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"),
        'deviceTypeId':metadata.device_type_id,
        'gitCommitId':firmware_info.git_commit
    }

    if(options.comment):
        structure[metadata.uid]['comment']=str(options.comment)

    f = open("device_db.json","w+") 
    json.dump(structure, f,indent=4, sort_keys=True)
    f.close()
    print('Labeled')
else:
    print('Device already labeled')