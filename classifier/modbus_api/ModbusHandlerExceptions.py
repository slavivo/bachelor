
class ModbusHandlerExceptions:
    class illegalFunction(Exception):
        pass

    class illegalDataAddress(Exception):
        pass

    class illegalDataValue(Exception):
        pass

    class slaveDeviceFailure(Exception):
        pass

    class acknowlege(Exception):
        pass

    class slaveDeviceBusy(Exception):
        pass

    class negativeAcknowlege(Exception):
        pass

    class memoryParityError(Exception):
        pass

    class gatewayPathUnavailable(Exception):
        pass

    class gatewayTargetDeviceFailedToRespond(Exception):
        pass 

    class EncodeError(Exception):
        pass

    class NotAListReturned(Exception):
        pass

    class NotBytesReturned(Exception):
        pass

    class RegsFIFOsHaveDifferentSizes(Exception):
        pass

    class ModbusIOError(Exception):
        pass

    class ModbusConnectionError(Exception):
        pass

    class unknownException(Exception):
        pass
    
    def illegalFunctionExc(self):
        raise self.illegalFunction("Device returned modbus exception code 0x01, see modbus manual for detailed error description")
    
    def illegalDataAddressExc(self):
        raise self.illegalDataAddress("Device returned modbus exception code 0x02, see modbus manual for detailed error description")
    
    def illegalDataValueExc(self):
        raise self.illegalDataValue("Device returned modbus exception code 0x03, see modbus manual for detailed error description")

    def slaveDeviceFailureExc(self):
        raise self.slaveDeviceFailure("Device returned modbus exception code 0x04, see modbus manual for detailed error description")
    
    def acknowlegeExc(self):
        raise self.acknowlege("Device returned modbus exception code 0x05, see modbus manual for detailed error description")

    def slaveDeviceBusyExc(self):
        raise self.slaveDeviceBusy("Device returned modbus exception code 0x06, see modbus manual for detailed error description")

    def negativeAcknowlegeExc(self):
        raise self.negativeAcknowlege("Device returned modbus exception code 0x07, see modbus manual for detailed error description")

    def memoryParityErrorExc(self):
        raise self.memoryParityError("Device returned modbus exception code 0x08, see modbus manual for detailed error description")

    def gatewayPathUnavailableExc(self):
        raise self.gatewayPathUnavailable("Device returned modbus exception code 0x0A, see modbus manual for detailed error description")

    def gatewayTargetDeviceFailedToRespondExc(self):
        raise self.gatewayTargetDeviceFailedToRespond("Device returned modbus exception code 0x0B, see modbus manual for detailed error description")

    def EncodeErrorExc(self):
        raise self.EncodeError("General error, could not decode slaves response")

    def NotAListReturnedExc(self):
        raise self.NotAListReturned("General error, encode function didnt return list")

    def NotBytesReturnedExc(self):
        raise self.NotBytesReturned("General error, encode function didnt return bytes")

    def ModbusIOErrorExc(self):
        raise self.ModbusIOError("Error resulting from data i/o, slave didnt respond?")

    def ModbusConnectionExc(self):
        raise self.ModbusIOError("Connection error happened")        

    exceptionSwitcher = {
            b'\x01': illegalFunctionExc,
            b'\x02': illegalDataAddressExc,
            b'\x03': illegalDataValueExc,
            b'\x04': slaveDeviceFailureExc,
            b'\x05': acknowlegeExc,
            b'\x06': slaveDeviceBusyExc,
            b'\x07': negativeAcknowlegeExc,
            b'\x08': memoryParityErrorExc,
            b'\x0A': gatewayPathUnavailableExc,
            b'\x0B': gatewayTargetDeviceFailedToRespondExc,
            "EncodeError": EncodeErrorExc,
            "NotAListReturned":NotAListReturnedExc,
            "NotBytesReturned": NotBytesReturned,
            "NotAListReturned":NotAListReturnedExc,
            "ModbusIOError":ModbusIOErrorExc,
            "ModbusConnectionError":ModbusConnectionExc
        }
    
    def generateException(self,key):
        func = self.exceptionSwitcher.get(key)
        if(func is not None):
            func(self)
        else:
            raise self.unknownException
        