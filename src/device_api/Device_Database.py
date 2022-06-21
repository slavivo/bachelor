import typing
import json

from base.Device_Exceptions import Not_Existing_Class_Exception
from base.Device import Device, Device_Type_Register, Modbus_Handler

from EEGV1 import EEGV1
from ECGV3 import ECGV3
from BioAdc import BioAdc
from FastImu import FastImu
from SmartAHRSV1 import SmartAHRSV1


class Device_Database:
    """
    Class uses only for creating Device object basing on device id value.
    For every new Device created it must be added to dictionary
    """

    map: typing.Dict[int, Device] = {
        0: EEGV1,
        1: ECGV3,
        3: BioAdc,
        4: SmartAHRSV1,
        6: ECGV3,
        7: BioAdc,
        8: FastImu,
        9: SmartAHRSV1
    }

    def find_device_class(self, type_id: int) -> Device:
        """Returns Device object"""
        class_reference = self.map.get(type_id)
        if (not class_reference):
            raise Not_Existing_Class_Exception(type_id)
        else:
            return class_reference


def find_device(modbus_handler: Modbus_Handler, slave_id:int) -> Device:
    """Returns Device object"""
    type_id_reg = Device_Type_Register(slave_id=slave_id, modbus_handler=modbus_handler)
    type_id = type_id_reg.get_info()
    database = Device_Database()
    device_class = database.find_device_class(type_id)
    return device_class(modbus_handler, slave_id, type_id)

def look_for_device_in_db(uid:str):
    f = open("device_db.json") 
    structure = json.load(f)
    f.close()
    if(not(uid in structure)):
        print('Unlabeled device')
    else:
        print('DB info:\n\tProject Name: %s' % (str(structure[uid]['projectName'])))
        if('comment' in  structure[uid]):
            print('\tComment: %s ' % (str(structure[uid]['comment'])))
        if('deviceName' in  structure[uid]):
            print('\tDevice Name: %s ' % (str(structure[uid]['deviceName'])))
