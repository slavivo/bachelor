import typing
from abc import ABCMeta, abstractmethod

from base.Data import Data, Firmware_Information, Metadata
from base.Device_Info_Registers import *
from base.Common import Time_Stamp_Register
from base.Device_Exceptions import Unsupported_Modbus_Handler_Exception, Incorrect_Device_Type_Id

from modbus_api.Modbus_Handler import Modbus_Handler


class Device(metaclass=ABCMeta):
    '''
    Device class is an abstract class for any Prokyber device.
    '''

    firmware_info: Firmware_Information
    metadata: Metadata
    supported_device_type_ids: typing.List[int] = NotImplemented
    supported_Modbus_Handlers: typing.List[Modbus_Handler] = NotImplemented
    data: Data = NotImplemented

    def __init__(self, modbus_handler: Modbus_Handler, slave_id: int, device_type_id: int = None):
        if (type(modbus_handler) not in self.supported_Modbus_Handlers):
            raise Unsupported_Modbus_Handler_Exception(modbus_handler=modbus_handler)
        if (device_type_id == None): 
            type_id_reg = Device_Type_Register(slave_id=slave_id, modbus_handler=modbus_handler)
            device_type_id = type_id_reg.get_info()
        upload_time_reg = Firmware_Upload_Date_Register(slave_id=slave_id, modbus_handler=modbus_handler)
        uid_reg = Unique_ID_Register(slave_id=slave_id, modbus_handler=modbus_handler)
        git_commit_reg = Git_Commid_ID_Register(slave_id=slave_id, modbus_handler=modbus_handler)
        self.firmware_info = Firmware_Information.create(upload_time_reg.get_info(), git_commit_reg.get_info())
        self.metadata = Metadata.create(device_type_id, type(self).__name__, uid_reg.get_info())
        self.time_Stamp_Register = Time_Stamp_Register(slave_id=slave_id, modbus_handler=modbus_handler)
        self._check_device()
        self._create_registers(modbus_handler, slave_id)
        self._post_init(modbus_handler, slave_id)

    @abstractmethod
    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int) -> None:
        """
        Creates attributes associated with Modbus registers
        """
        pass

    @abstractmethod
    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int) -> None:
        """Any other actions with object that could be processed after __init__ method"""
        pass

    @abstractmethod
    def read_data(self, only_new: bool = True, pc_time = False) -> None:
        """Reads data from device and saves to _data"""
        pass

    @abstractmethod
    def draw_with_plt(self) -> None:
        """Draws curently saved data in _data with matplotlib"""
        pass
    
    def _check_device(self) -> None:
        """
        Checks if connected device type id corresponds the current Device class.
        """
        if(self.metadata.device_type_id not in self.supported_device_type_ids):
            raise Incorrect_Device_Type_Id(self.metadata.id, self.supported_device_type_ids)

    def get_data(self) -> Data:
        """Returns _data contents"""
        return self.data

    def get_firmware_info(self) -> Firmware_Information:
        '''Returns Device_Information dataclass'''
        return self.firmware_info

    def get_metadata(self) -> Metadata:
        '''Returns Device_Metadata class part of Device_Information dataclass'''
        return self.metadata

    def get_time_stamp_ms(self):
        '''Returns time stamp value read from register'''
        return self.time_Stamp_Register.read()[0]

    def get_data_type(self) -> type:
        '''Returns type of _data attribute. Uses only for Logger configuration'''
        return type(self.data)

    def print_info(self) -> None:
        print("%s\n%s" % (self.firmware_info, self.metadata.__str__()))
