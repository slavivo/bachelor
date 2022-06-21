import serial.tools.list_ports
from abc import ABCMeta,abstractmethod
from Modbus_Connection_Handler import Modbus_Connection_Handler_Serial_RTU,Modbus_Connection_Handler_TCPIP
from IP_Lookup import Ip_Lookup

class Modbus_Handler(metaclass=ABCMeta):
    
    def __init__(self,**kwargs):
        self._modbus_connection_handler = None
        self._connection_points = []
        self._active_slave_ids = []

    @property
    @abstractmethod
    def connection_points_fcn(self):
        pass 

    @property
    @abstractmethod
    def init_modbus_connection_handler(self,connection_point):
        pass

    @property
    @abstractmethod
    def set_connection_point(self,connection_point):
        pass

    def find_connection_points(self):
        self._connection_points = self.connection_points_fcn()

    def find_active_devices_and_specify_connection_point(self):
        if (self._connection_points!=[]):
            for connection_point in self._connection_points:
                self.init_modbus_connection_handler(connection_point=connection_point)
                try:   
                    self.try_to_connect()
                except:
                    pass
                else:
                    self.set_connection_point(connection_point=connection_point)
                    break
        else:
            raise self.NoActiveConnectionPointsException()

    def try_to_connect(self):
        if (self._modbus_connection_handler.connect()==True):
            for slave_id in self._slaves_ids:
                try:
                    self._modbus_connection_handler.read_uint16_holdings(reg_address=40001,reg_count=1,slave_id=slave_id)
                except:
                    pass
                else:
                    self._active_slaves_ids.append(slave_id)
            if (self._active_slaves_ids==[]):
                raise self.NoActiveDevicesException()
        else:
            raise self.ModbusSocketFailedException()

    def get_modbus_connection_handler(self):
        return self._modbus_connection_handler

    def get_slave_ids(self):
        return self._active_slave_ids

    class NoActiveConnectionPointsException(Exception):
        def __init__(self):
            self.__message = "No active port/wi-fi connection"
            Exception.__init__(self,self.__message)

    class NoActiveDevicesException(Exception):
        def __init__(self):
            self.__message = "No active device"
            Exception.__init__(self,self.__message)

    class ModbusSocketFailedException(Exception):
        def __init__(self):
            self.__message = "pymodbus failed to create socket"
            Exception.__init__(self,self.__message)

class Modbus_Handler_Serial_RTU(Modbus_Handler):
    def __init__(self,**kwargs):
        Modbus_Handler.__init__(self)
        self._port = kwargs.get('port',None)
        self._stopbits = kwargs.get('stopbits',1)
        self._bytesize = kwargs.get('bytesize',8)
        self._parity = kwargs.get('parity','E')
        self._baudrate = kwargs.get('baudrate',460800)
        self._timeout = kwargs.get('timeout',1)
        self._number_of_retries = kwargs.get('number_of_retries',20)
        self._slaves_ids = kwargs.get('slave_ids',[i for i in range(1,256)])
        if (self._port == None):
            self.find_connection_points()
        else:
            self._connection_points.append(self._port)
        self.find_active_devices_and_specify_connection_point()

    def connection_points_fcn(self):
        ports = []
        try:
            ports_info = serial.tools.list_ports.comports() 
        except:
            return ports
        ports = [port_info.device for port_info in ports_info]
        return ports

    def init_modbus_connection_handler(self,connection_point):
        self._modbus_connection_handler = Modbus_Connection_Handler_Serial_RTU(port=connection_point,stopbits=self._stopbits,bytesize=self._bytesize,parity=self._parity,baudrate=self._baudrate,timeout=self._timeout,number_of_retries=self._number_of_retries)

    def set_connection_point(self,connection_point):
        self._port = connection_point

class Modbus_Handler_TCPIP(Modbus_Handler):
    def __init__(self,**kwargs):
        Modbus_Handler.__init__(self)
        self._host = kwargs.get('ip_address',None)
        self._host_port = kwargs.get('port','80')
        self._timeout = kwargs.get('timeout',1)
        self._number_of_retries = kwargs.get('number_of_retries',20)
        self._slaves_ids = kwargs.get('slave_ids',[i for i in range(1,256)])
        if (self._host == None):
            self.find_connection_points()
        else:
            self._connection_points.append(self._host)
        self.find_active_devices_and_specify_connection_point()

    def connection_points_fcn(self):
        ip_lookup = Ip_Lookup()
        return ip_lookup.find_hosts_ip_addresses()

    def init_modbus_connection_handler(self,connection_point):
        self._modbus_connection_handler = Modbus_Connection_Handler_TCPIP(host=connection_point,port=self._host_port,timeout=self._timeout,number_of_retries=self._number_of_retries)

    def set_connection_point(self,connection_point):
        self._host = connection_point
