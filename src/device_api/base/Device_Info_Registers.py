from datetime import datetime
from abc import abstractmethod

from modbus_api.Modbus_Handler import Modbus_Handler
from modbus_api.modbus_registers import Holding_16Bit_Unsigned
from modbus_api.convert import uint16_to_uint32, uint32_to_uint64, uint16_to_hex_string


class Device_Info_Register(Holding_16Bit_Unsigned):
    """Base class for Modbus registers that storing Prokyber device information"""
    reg_number: int = NotImplemented
    reg_count: int = NotImplemented

    def __init__(self, slave_id: int, modbus_handler: Modbus_Handler):
        super(Holding_16Bit_Unsigned, self).__init__(reg_number=self.reg_number, slave_id=slave_id, reg_count=self.reg_count, modbus_handler=modbus_handler)

    @abstractmethod
    def get_info(self):
        pass


class Firmware_Upload_Date_Register(Device_Info_Register):
    """Modbus register storing data and time of firmware uploading to the Prokyber device"""
    reg_number: int = 40001
    reg_count: int = 4

    def get_info(self) -> datetime:
        """Returns Python datetime variable with wifrmware upload date"""
        result = self.read()
        output_data_msb = uint16_to_uint32(result[0], result[1])
        output_data_lsb = uint16_to_uint32(result[2], result[3])
        output_data = uint32_to_uint64(output_data_msb, output_data_lsb)
        return datetime.fromtimestamp(output_data)


class Unique_ID_Register(Device_Info_Register):
    """Modbus register storing unique identification number of Prokyber device"""
    reg_number: int = 40005
    reg_count: int = 6

    def get_info(self) -> str:
        """Returns unique identification number"""
        return uint16_to_hex_string(self.read())


class Device_Type_Register(Device_Info_Register):
    """Modbus register storing device identification number of Prokyber device"""
    reg_number: int = 40011
    reg_count: int = 1

    def get_info(self) -> int:
        """Returns device identification number"""
        return self.read()[0]


class Git_Commid_ID_Register(Device_Info_Register):
    """Modbus register storing git commit number for firmware of Prokyber device"""
    reg_number: int = 40013
    reg_count: int = 10

    def get_info(self) -> str:
        """Returns git commit number"""
        return uint16_to_hex_string(self.read())
