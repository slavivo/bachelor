import struct
from pymodbus.pdu import ModbusRequest,ModbusResponse,ModbusExceptions



class DataOut:
    registers = []
    
class DataOutTwoRegs:
    registers0 = []
    registers1 = []

class Read2FifosRequest(ModbusRequest):

    function_code = 66
    _rtu_frame_size = 8

    class Read2FifosResponse(ModbusResponse):

        function_code = 65
        _rtu_byte_count_pos = 3

        def __init__(self, values=None, **kwargs):
            ModbusResponse.__init__(self, **kwargs)
            self.values = values or []

        def encode(self):
            result = bytes([len(self.values) * 2])
            for register in self.values:
                result += struct.pack('>H', register)
            return result

        def decode(self, data):
            byte_count = int(data[1])
            self.values = []
            for i in range(2, byte_count + 1, 2):
                self.values.append(struct.unpack('>H', data[i:i + 2])[0])

    def __init__(self, address=None, **kwargs):
        ModbusRequest.__init__(self, **kwargs)
        self.address = address
        self.count = 2

    def encode(self):
        return struct.pack('>HH', self.address, self.count)

    def decode(self, data):
        self.address, self.count = struct.unpack('>HH', data)

    def execute(self, context):
        if not (1 <= self.count <= 0x7d0):
            return self.doException(ModbusExceptions.IllegalValue)
        if not context.validate(self.function_code, self.address, self.count):
            return self.doException(ModbusExceptions.IllegalAddress)
        values = context.getValues(self.function_code, self.address,
                                   self.count)
        return self.Read2FifosResponse(values)

class ReadFifoRequest(ModbusRequest):

    function_code = 66
    _rtu_frame_size = 8

    class ReadFifoResponse(ModbusResponse):

        function_code = 24
        _rtu_byte_count_pos = 3

        def __init__(self, values=None, **kwargs):
            ModbusResponse.__init__(self, **kwargs)
            self.values = values or []

        def encode(self):
            """ Encodes response pdu
            :returns: The encoded packet message
            """
            # print(len(self.values))
            result = bytes([len(self.values) * 2])
            for register in self.values:
                result += struct.pack('>H', register)
            return result

        def decode(self, data):
            """ Decodes response pdu
            :param data: The packet data to decode
            """
            byte_count = int(data[1])
            self.values = []
            for i in range(2, byte_count + 1, 2):
                self.values.append(struct.unpack('>H', data[i:i + 2])[0])

class ReadFifoAndTimeRequest(ModbusRequest):

    function_code = 67
    # _rtu_frame_size = 8
    # fifoFreq = 100

    class ReadFifoAndTimeResponse(ModbusResponse):

        function_code = 67
        _rtu_double_byte_count_pos = 10

        def __init__(self, values=None, **kwargs):
            ModbusResponse.__init__(self, **kwargs)
            self.values = values or []

        def encode(self):
            """ Encodes response pdu
            :returns: The encoded packet message
            """
            return self.values

        def decode(self, data):
            """ Decodes response pdu
            :param data: The packet data to decode
            """
            timeFromStart = struct.unpack('>Q', data[0:8])[0]
            byte_count = struct.unpack('>H',data[8:10])[0]
            byteCounter = 10
            fifoAllRegs = []
            sampleFreqs = []
            while(byteCounter-6 < byte_count):
                sampleFreqs.append(struct.unpack('>H', data[byteCounter:byteCounter + 2])[0])
                byteCounter = byteCounter + 2
                nextRegCount = struct.unpack('>H', data[byteCounter:byteCounter + 2])[0]
                byteCounter = byteCounter + 2
                fifoRegs = []
                for i in range(nextRegCount):
                    fifoRegs.append(struct.unpack('>H', data[byteCounter + i*2:byteCounter + i*2 + 2])[0])
                byteCounter = byteCounter + nextRegCount*2
                fifoAllRegs.append(fifoRegs)
            self.values = [timeFromStart,fifoAllRegs,sampleFreqs]

    def __init__(self,address=None,count = 1, **kwargs):
        ModbusRequest.__init__(self, **kwargs)
        self.address = address
        self.count = count

    def encode(self):
        return struct.pack('>HH', self.address, self.count)

    def decode(self, data):
        self.address, self.count = struct.unpack('>HH', data)

    def execute(self, context):
        if not (1 <= self.count <= 0x7d0):
            return self.doException(ModbusExceptions.IllegalValue)
        if not context.validate(self.function_code, self.address, self.count):
            return self.doException(ModbusExceptions.IllegalAddress)
        values = context.getValues(self.function_code, self.address,
                                   self.count)
        return self.ReadFifoResponse(values)
