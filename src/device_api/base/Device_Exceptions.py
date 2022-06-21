from modbus_api.Modbus_Handler import Modbus_Handler

class Incorrect_Device_Type_Id(Exception):
    
    def __init__(self, this_device_type_name, supported_device_type_names):
        self.__message = "Device type retrieved from HW that is not compatible with class you are using... Hw device type: " + str(this_device_type_name) + ", device types supported by this class:" + str(supported_device_type_names)
        Exception.__init__(self, self.__message)

class Not_Existing_Class_Exception(Exception):

    def __init__(self, type_id: int) -> None:
        self.__message = "There is no implemented class for type_id \'" + str(type_id) + "\'"
        Exception.__init__(self, self.__message)

class Unsupported_Modbus_Handler_Exception(Exception):
    def __init__(self, modbus_handler: Modbus_Handler):
        self.__message = "Modbus hadnler with type: " + str(type(modbus_handler).__name__) + " not supported for this device"
        Exception.__init__(self, self.__message)