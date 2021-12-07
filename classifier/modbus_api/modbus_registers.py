from abc import ABCMeta, abstractmethod
import time
from modbus_api.ModbusHandlerExceptions import ModbusHandlerExceptions
from modbus_api.Modbus_Connection_Handler import Modbus_Connection_Handler
from modbus_api.Modbus_Handler import Modbus_Handler
from modbus_api.Modbus_Registers_Exceptions import *

COIL_OFFSET = 0
DISCRETE_INPUT_OFFSET = 10000
INPUT_OFFSET = 30000
HOLDING_OFFSET = 40000
REG_NUMBER_MAX = 10000


class ModbusRegister(metaclass=ABCMeta):

    _offset: int = NotImplemented
    _modbus_connection: Modbus_Connection_Handler
    _reg_number: int
    _slave_id: int
    _reg_count: int

    def __init__(self, reg_number: int, slave_id: int, modbus_handler: Modbus_Handler, reg_count: int = 1):
        if ((reg_number > self._offset) and (reg_number + (reg_count-1) < self._offset + REG_NUMBER_MAX)):
            self._reg_number = reg_number
        elif ((reg_number > 0) and (reg_number + (reg_count-1) < REG_NUMBER_MAX)):
            self._reg_number = reg_number + self._offset
        else:
            raise RegisterNumerException(self)
        self._reg_count = reg_count
        self._modbus_connection = modbus_handler.get_modbus_connection_handler()
        self._slave_id = slave_id

    @property
    @abstractmethod
    def _write_fcn(self, val: list):
        pass

    def write(self, val: list):
        if (type(val) is not list):
            raise NotAListWriteValueException(val)
        elif (len(val) > self._reg_count):
            raise RegistersCountExceedException(self, self._reg_count)
        else:
            try:
                self._write_fcn(val=val)
            except ModbusHandlerExceptions.ModbusIOError:
                raise ModbusRequestFailedException()

    @property
    @abstractmethod
    def _read_fcn(self, reg_count: int):
        pass

    def read(self, reg_count: int = None):
        if (reg_count == None):
            reg_count = self._reg_count
        if ((reg_count > self._reg_count) or (reg_count <= 0)):
            raise RegistersCountExceedException(self, reg_count)
        else:
            try:
                retval = self._read_fcn(reg_count)
            except ModbusHandlerExceptions.ModbusIOError:
                raise ModbusRequestFailedException()
            else:
                return retval

    


class Coil(ModbusRegister):

    _offset: int = COIL_OFFSET

    def _write_fcn(self, val: int):
        self._modbus_connection.write_coils(reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_coils(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Discrete_Input(ModbusRegister):

    _offset: int = DISCRETE_INPUT_OFFSET

    def _write_fcn(self, val:any):
        raise ReadOnlyAccessException(self)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_discrete_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input(ModbusRegister):

    _offset: int = INPUT_OFFSET

    def _write_fcn(self, val: any):
        raise ReadOnlyAccessException(self)


class Input_16Bit_Unsigned(Input):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint16_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_16Bit_Signed(Input):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int16_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Unsigned(Input):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint32_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Signed(Input):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int32_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Float(Input):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_float32_inputs(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_FIFO(Input):

    def __init__(self, reg_number: int, slave_id: int, modbus_handler:Modbus_Handler, reg_count: int = 1):
        Input.__init__(self, reg_number=reg_number, slave_id=slave_id,modbus_handler=modbus_handler, reg_count=reg_count)
        self._last_time_stamps_ms = [0 for i in range(reg_count)]

    def read(self, reg_count: int = None, only_new:bool = False,pc_time:bool = False):
        if (reg_count == None):
            reg_count = self._reg_count
        if ((reg_count > self._reg_count) or (reg_count <= 0)):
            raise RegistersCountExceedException(self, reg_count)
        else:
            try:
                result = self._read_fcn(reg_count=reg_count)
                retval = [[[], []] for i in range(reg_count)]
                if(pc_time):
                    pc_time_now = int(time.time()*1000)
                for i in range(reg_count):
                    retval[i][0] = result[i][0]
                    fifo_size = len(retval[i][0])
                    fifo_end_ms = result[i][1][0]
                    fifo_time_step_ms = 1000/result[i][1][1]
                    fifo_begin_ms = fifo_end_ms - fifo_time_step_ms*(fifo_size - 1)
                    retval[i][1] = [(fifo_begin_ms + fifo_time_step_ms*j) for j in range(fifo_size)]
                    if(pc_time):
                        pc_fifo_end_ms = pc_time_now
                        pc_fifo_begin_ms=pc_fifo_end_ms - fifo_time_step_ms*(fifo_size - 1)
                        pc_time_ret = [(pc_fifo_begin_ms + fifo_time_step_ms*j) for j in range(fifo_size)]
                    if (only_new == True):
                        if(self._last_time_stamps_ms[i] == fifo_end_ms):
                            retval[i][0] = []
                            retval[i][1] = []                            
                        if ((self._last_time_stamps_ms[i] > fifo_begin_ms) and (self._last_time_stamps_ms[i] < fifo_end_ms)):
                            evaluated_begin_idx = int((self._last_time_stamps_ms[i]-fifo_begin_ms)/(fifo_end_ms - fifo_begin_ms) * fifo_size)-2
                            for j in range(evaluated_begin_idx, fifo_size):
                                if (self._last_time_stamps_ms[i] >= retval[i][1][j-1]) and (self._last_time_stamps_ms[i] <= retval[i][1][j]):
                                    retval[i][0] = retval[i][0][j:]
                                    retval[i][1] = pc_time_ret[j:] if pc_time else retval[i][1][j:]
                                    break
                    self._last_time_stamps_ms[i] = fifo_end_ms
            except ModbusHandlerExceptions.ModbusIOError:
                raise ModbusRequestFailedException()
            else:
                return retval


class Input_16Bit_Unsigned_FIFO(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint16_inputs_fifo(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_16Bit_Signed_FIFO(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int16_inputs_fifo(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)

class Input_16Bit_Signed_FIFONew(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int16_inputs_fifo_new(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Unsigned_FIFO(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint32_inputs_fifo(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Signed_FIFO(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int32_inputs_fifo(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Input_32Bit_Float_FIFO(Input_FIFO):

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_float32_inputs_fifo(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding(ModbusRegister):

    _offset = HOLDING_OFFSET


class Holding_16Bit_Unsigned(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_uint16_holdings(reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint16_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding_16Bit_Signed(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_int16_holdings(
            reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int16_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding_32Bit_Unsigned(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_uint32_holdings(
            reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint32_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding_32Bit_Signed(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_int32_holdings(
            reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_int32_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding_32Bit_Float(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_float32_holdings(
            reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_float32_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)


class Holding_64Bit_Unsigned(Holding):

    def _write_fcn(self, val: list):
        self._modbus_connection.write_uint64_holdings(
            reg_address=self._reg_number, val=val, slave_id=self._slave_id)

    def _read_fcn(self, reg_count: int):
        return self._modbus_connection.read_uint64_holdings(reg_address=self._reg_number, reg_count=reg_count, slave_id=self._slave_id)
