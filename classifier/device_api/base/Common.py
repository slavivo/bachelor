from modbus_api.Modbus_Handler import Modbus_Handler
from modbus_api.modbus_registers import Holding_16Bit_Unsigned, Holding_32Bit_Unsigned, Holding_64Bit_Unsigned


class Time_Stamp_Register(Holding_64Bit_Unsigned):
    """Base class for registers that repeats in every Device"""    
    _reg_number: int = 40023
    _reg_count:int = 1

    def __init__(self, slave_id: int, modbus_handler: Modbus_Handler):
        super().__init__(reg_number=self._reg_number, slave_id=slave_id, reg_count=self._reg_count, modbus_handler=modbus_handler)


class RTC:
    def __init__(self, reg_number:int, slave_id: int, modbus_handler: Modbus_Handler):
        self.time_set_ctrl_register = Holding_16Bit_Unsigned(reg_number=reg_number+2, slave_id=slave_id, modbus_handler=modbus_handler)
        self.time_set_data_register = Holding_32Bit_Unsigned(reg_number=reg_number+4, slave_id=slave_id, modbus_handler=modbus_handler)
        self.time_get_data_register = Holding_32Bit_Unsigned(reg_number=reg_number, slave_id=slave_id, modbus_handler=modbus_handler)

    def set_unix_epoch(self, epoch):
        self.time_set_data_register.write([epoch])
        self.time_set_ctrl_register.write([1])

    def get_unix_epoch(self):
        return self.time_get_data_register.read()[0]
