from colorama import Fore

class ModbusRequestFailedException(Exception):
        def __init__(self):
            self.__message = Fore.RED + "Modbus request has been failed." + Fore.RESET
            Exception.__init__(self, self.__message)

class ReadOnlyAccessException(Exception):
    def __init__(self, source):
        self.__message = Fore.RED + "Read-Only access for " + Fore.YELLOW + \
            str(type(source).__name__) + Fore.RED + \
            " registers." + Fore.RESET
        Exception.__init__(self, self.__message)

class RegisterNumerException(Exception):
    def __init__(self, source):
        self.__message = Fore.RED + "Parameter " + Fore.YELLOW + "reg_number" + Fore.RED + " and " + Fore.YELLOW + "reg_count" + Fore.RED + \
            " for " + Fore.YELLOW + str(type(source).__name__) + Fore.RED + \
            " Class should be value according to the modbus protocol." + Fore.RESET
        Exception.__init__(self, self.__message)

class RegistersCountExceedException(Exception):
    def __init__(self, source, reg_count: int):
        self.__message = Fore.RED + "Value of " + Fore.YELLOW + "reg_count" + Fore.RED + " parameter should be in range of " + Fore.YELLOW + \
            "[" + str(1) + "," + str(source._reg_count) + "]" + Fore.RED + ", but " + \
            Fore.YELLOW + str(reg_count) + Fore.RED + \
            " were passed." + Fore.RESET
        Exception.__init__(self, self.__message)
        
class NotAListWriteValueException(Exception):
    def __init__(self, val):
        self.__message = Fore.RED + "Parameter " + Fore.YELLOW + "val" + Fore.RED + " should be type of " + Fore.YELLOW + \
            "list" + Fore.RED + ", but " + Fore.YELLOW + \
            str(type(val)) + Fore.RED + " was passed." + Fore.RESET
        Exception.__init__(self, self.__message)