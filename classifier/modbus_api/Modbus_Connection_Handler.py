from abc import ABCMeta
import pymodbus
from pymodbus.client.sync import BaseModbusClient, ModbusSerialClient, ModbusTcpClient 
from pymodbus.exceptions import ConnectionException, ModbusIOException
from pymodbus.pdu import ExceptionResponse
import pdu_FIFO 
from ModbusHandlerExceptions import ModbusHandlerExceptions
import convert
        
class Modbus_Connection_Handler(metaclass=ABCMeta):

    modbus_handler: BaseModbusClient

    def __init__(self,number_of_retries):
        self.exceptionHandler = ModbusHandlerExceptions()
        self.set_number_of_retries(number_of_retries)

    def set_number_of_retries(self,numer_of_retries):
        self.number_of_retries = numer_of_retries

    def connect(self):
        return self.modbus_handler.connect()

    def _connection_Interrupt_Handler(self,result):
        if (type(result)==ModbusIOException):
            self.exceptionHandler.generateException("ModbusIOError")
        if(type(result)==ConnectionException):
            self.exceptionHandler.generateException("ModbusConnectionError")
        if(result==None):
            raise Exception("Unknown problem")

    def _try_to_encode(self,result):
        try:
            encoded = result.encode()
        except AttributeError:
            self.exceptionHandler.generateException("EncodeError")
        if type(encoded) is not list:
            if(type(result)==ExceptionResponse):
                self.exceptionHandler.generateException(encoded)
            else:
                self.exceptionHandler.generateException("NotAListReturned")
        return encoded

    def register_fifo_request_and_response(self):
        self.modbus_handler.register(pdu_FIFO.ReadFifoRequest.ReadFifoResponse)
        self.modbus_handler.register(pdu_FIFO.Read2FifosRequest.Read2FifosResponse)
        self.modbus_handler.register(pdu_FIFO.ReadFifoAndTimeRequest.ReadFifoAndTimeResponse)

    def write_coils(self,reg_address,val,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.write_coils(address = reg_address-1,values = val,unit = slave_id)
                if(isinstance(result,Exception)):
                    raise result
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result

    def read_coils(self,reg_address,reg_count,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.read_coils(address = reg_address-1,count = reg_count,unit = slave_id)
                if(isinstance(result,Exception)):
                    raise result
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result.bits

    def read_discrete_inputs(self,reg_address,reg_count,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.read_discrete_inputs(address = reg_address-1,count = reg_count,unit = slave_id)
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result.bits 

    def _read_inputs(self,reg_address,reg_count,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.read_input_registers(address = reg_address-1,count = reg_count,unit = slave_id)
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result.registers

    def read_uint16_inputs(self,reg_address,reg_count,slave_id):
        result = self._read_inputs(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        return result

    def read_int16_inputs(self,reg_address,reg_count,slave_id):
        result = self._read_inputs(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_int16,result))
        return output_data

    def read_uint32_inputs(self,reg_address,reg_count,slave_id):
        result = self._read_inputs(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_uint32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def read_int32_inputs(self,reg_address,reg_count,slave_id):
        result = self._read_inputs(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_int32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def read_float32_inputs(self,reg_address,reg_count,slave_id):
        result = self._read_inputs(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_float32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def _read_inputs_fifo(self,reg_address,reg_count,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.execute(pdu_FIFO.ReadFifoAndTimeRequest(address=reg_address-1,count = reg_count, unit = slave_id))
                if(isinstance(result,Exception)):
                    raise result
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break
            
        self._connection_Interrupt_Handler(result)
        result = self._try_to_encode(result)
        fifos = [[result[1][i],[result[0],result[2][i]]] for i in range(reg_count)]
        return fifos

    def read_uint16_inputs_fifo(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        return result

    def read_int16_inputs_fifo(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        for i in range(reg_count):
            result[i][0] = list(map(convert.uint16_to_int16,result[i][0]))
        return result

    def read_int16_inputs_fifo_new(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        for i in range(reg_count):
            result[i][0] = list(map(convert.uint16_to_int16,result[i][0]))
        return result

    def read_uint32_inputs_fifo(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = [[[],result[i*2][1]] for i in range(reg_count)]
        for i in range (reg_count):
            output_data[i][0] = list(map(convert.uint16_to_uint32,result[i*2][0],result[i*2+1][0]))
        return output_data

    def read_int32_inputs_fifo(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = [[[],result[i*2][1]] for i in range(reg_count)]
        for i in range(reg_count):
            output_data[i][0] = list(map(convert.uint16_to_int32,result[i*2][0],result[i*2+1][0]))
        return output_data

    def read_float32_inputs_fifo(self,reg_address,reg_count,slave_id):
        result = self._read_inputs_fifo(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = [[[],result[i*2][1]] for i in range(reg_count)]
        for i in range(reg_count):
            output_data[i][0] = list(map(convert.uint16_to_float32,result[i*2][0],result[i*2+1][0]))
        return output_data

    def _write_holdings(self,reg_address,val,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.write_registers(address = reg_address-1,values = val,unit = slave_id)
                if(isinstance(result,Exception)):
                    raise result
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result

    def write_uint16_holdings(self,reg_address,val,slave_id):
        result = self._write_holdings(reg_address=reg_address,val=val,slave_id=slave_id)
        return result

    def write_int16_holdings(self,reg_address,val,slave_id):
        uint16_val = list(map(convert.int16_to_uint16,val))
        result = self._write_holdings(reg_address=reg_address,val=uint16_val,slave_id=slave_id)
        return result

    def write_uint32_holdings(self,reg_address,val,slave_id):
        val = list(map(convert.uint32_to_uint16,val))
        uint16_val = [0 for i in range(len(val)*2)]
        for i in range(len(val)):
            uint16_val[i*2],uint16_val[i*2+1] = val[i][0],val[i][1]
        result = self._write_holdings(reg_address=reg_address,val=uint16_val,slave_id=slave_id)
        return result

    def write_uint64_holdings(self,reg_address,val,slave_id):
        val = list(map(convert.uint64_to_uint16,val))
        uint16_val = [0 for i in range(len(val)*4)]
        for i in range(len(val)):
            uint16_val[i*4],uint16_val[i*4+1],uint16_val[i*4+2],uint16_val[i*4+3] = val[i][0],val[i][1],val[i][2],val[i][3]
        result = self._write_holdings(reg_address=reg_address,val=uint16_val,slave_id=slave_id)
        return result

    def write_int32_holdings(self,reg_address,val,slave_id):
        val = list(map(convert.int32_to_uint16,val))
        uint16_val = [0 for i in range(len(val)*2)]
        for i in range(len(val)):
            uint16_val[i*2],uint16_val[i*2+1] = val[i][0],val[i][1]
        result = self._write_holdings(reg_address=reg_address,val=uint16_val,slave_id=slave_id)
        return result

    def write_float32_holdings(self,reg_address,val,slave_id):
        val = list(map(convert.float32_to_uint16,val))
        uint16_val = [0 for i in range(len(val)*2)]
        for i in range(len(val)):
            uint16_val[i*2],uint16_val[i*2+1] = val[i][0],val[i][1]
        result = self._write_holdings(reg_address=reg_address,val=uint16_val,slave_id=slave_id)
        return result

    def _read_holdings(self,reg_address,reg_count,slave_id):
        for x in range(self.number_of_retries):
            try:
                result = self.modbus_handler.read_holding_registers(address = reg_address-1,count = reg_count,unit = slave_id)
                if(isinstance(result,Exception)):
                    raise result
            except ModbusIOException:
                result = ModbusIOException()
            except ConnectionException:
                result = ConnectionException()
            else:
                break 
        self._connection_Interrupt_Handler(result)
        return result.registers

    def read_uint16_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        return result

    def read_int16_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_int16,result))
        return output_data

    def read_uint32_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_uint32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def read_uint64_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count*4,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_uint64,result[0:reg_count*4:4],result[1:reg_count*4:4],result[2:reg_count*4:4],result[3:reg_count*4:4]))
        return output_data

    def read_int32_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_int32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def read_float32_holdings(self,reg_address,reg_count,slave_id):
        result = self._read_holdings(reg_address=reg_address,reg_count=reg_count*2,slave_id=slave_id)
        output_data = list(map(convert.uint16_to_float32,result[0:reg_count*2:2],result[1:reg_count*2:2]))
        return output_data

    def ping_device_register(self,slave_id):
        self.write_uint16_holdings(reg_address=40001,val=[1],slave_id=slave_id)
        result = self.read_uint16_holdings(reg_address=40001,reg_count=1,slave_id=slave_id)[0]
        return result==1

class Modbus_Connection_Handler_Serial(Modbus_Connection_Handler):
    def __init__(self,number_of_retries):
        Modbus_Connection_Handler.__init__(self,number_of_retries=number_of_retries)
        
class Modbus_Connection_Handler_Serial_RTU(Modbus_Connection_Handler_Serial):
    def __init__(self,port,stopbits,bytesize,parity,baudrate,timeout,number_of_retries):
        Modbus_Connection_Handler_Serial.__init__(self,number_of_retries=number_of_retries)
        self.modbus_handler = ModbusSerialClient(method="rtu", port=port, stopbits=stopbits , bytesize=bytesize, parity=parity, baudrate=baudrate, timeout=timeout)
        self.register_fifo_request_and_response()

class Modbus_Connection_Handler_TCPIP(Modbus_Connection_Handler):
    def __init__(self,host,port,timeout,number_of_retries):
        Modbus_Connection_Handler.__init__(self,number_of_retries=number_of_retries)
        self.modbus_handler = ModbusTcpClient(host=host, port=port ,timeout=timeout)
        self.register_fifo_request_and_response()
        