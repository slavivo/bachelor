from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace

class Empty_Parser(metaclass=ABCMeta):
    _epilog: str = NotImplemented
    _Argument_Parser: ArgumentParser
    _options: Namespace

    def __init__(self):
        self._Argument_Parser = ArgumentParser(epilog=self._epilog)
        self._add_Specific_Arguments()

    @abstractmethod
    def _add_Specific_Arguments(self):
        pass

    @abstractmethod
    def _process_Specific_Arguments(self):
        pass

    def parse(self, args):
        self._options = self._Argument_Parser.parse_args(args)
        self._process_Specific_Arguments()

    def get_options(self):
        return self._options

class Arg_Parser(metaclass=ABCMeta):

    _epilog: str = NotImplemented
    _Argument_Parser: ArgumentParser
    _options: Namespace

    def __init__(self):
        self._Argument_Parser = ArgumentParser(epilog=self._epilog)
        self._add_Default_Arguments()
        self._add_Specific_Arguments()

    def _add_Default_Arguments(self):
        self._Argument_Parser.add_argument('-i', '--ip', metavar='ip_address', dest='ip_address', type=str, help="Ip address of the device")
        self._Argument_Parser.add_argument('-s', '--serial', metavar='serial_port', dest='serial_port', default="/dev//ttyACM0", type=str, help="Serial port name of the device")
        self._Argument_Parser.add_argument('-a', '--slave_address', metavar='slave_address', dest='slave_address', default=4, type=int, help="Slave id address")
        self._Argument_Parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help="Verbose mode")
        self._Argument_Parser.add_argument('-c', '--pc_synch', dest='pc_synch', action='store_true', help="Use pc time as input registers time")

    @abstractmethod
    def _add_Specific_Arguments(self):
        pass

    def _process_Default_Arguments(self):
        if((not self._options.ip_address) and (not self._options.serial_port)):
            print("No IP or serial port specified, use -i or -s arguments to specify")
            exit()

        if((not self._options.slave_address)):
            print("No slave id specified, use -a argument to specify")
            exit()

    @abstractmethod
    def _process_Specific_Arguments(self):
        pass

    def parse(self, args):
        self._options = self._Argument_Parser.parse_args(args)
        self._process_Default_Arguments()
        self._process_Specific_Arguments()

    def get_options(self):
        return self._options


class Default_Arg_Parser(Arg_Parser):
    _epilog = "Arguments parser for default communication with Prokyber device"

    def _add_Specific_Arguments(self):
        pass

    def _process_Specific_Arguments(self):
        pass


class Web_Server_Access_Arg_Parser(Arg_Parser):
    _epilog = "Arguments parser for default communication with Prokyber device and access to web server"

    def _add_Specific_Arguments(self):
        self._Argument_Parser.add_argument("-ho", "--host", metavar='host', dest='host', type=str, help="Host.")
        self._Argument_Parser.add_argument("-p", "--output_port", metavar='output_port', dest='output_port', type=str, help="Output port.")

    def _processSpecificArguments(self):
        pass


class Device_DB_Access_Arg_Parser(Arg_Parser):
    _epilog = "Arguments parser for default communication with Prokyber device and access to data base"

    def _add_Specific_Arguments(self):
        self._Argument_Parser.add_argument("-com", "--comment", metavar='comment', dest='comment', type=str, help="Comment to add")
        self._Argument_Parser.add_argument("-prj", "--project_name", metavar='project_name', dest='project_name', type=str)
        self._Argument_Parser.add_argument("-nm", "--device_name", metavar='device_name', dest='device_name', type=str)


    def _process_Specific_Arguments(self):
        if((not self._options.project_name)):
            print("No project name specified, use -prj argument to specify")
            exit()
        if((not self._options.device_name)):
            print("No device name specified, use -nm argument to specify")
            exit()

class File_Draw_Arg_Parser(Empty_Parser):
    _epilog = "Arguments parser for drawing data from csv files"

    def _add_Specific_Arguments(self):
        self._Argument_Parser.add_argument("-i", "--input", help="Input file.")
        self._Argument_Parser.add_argument("-d", "--delimiter", help="Delimiter.")
        self._Argument_Parser.add_argument("-x", "--x_series", help="x series name.")
        self._Argument_Parser.add_argument("-y", "--y_series", help="y series name.")
        self._Argument_Parser.add_argument("-y1", "--y1_series", help="y1 series name.")
        self._Argument_Parser.add_argument("-y2", "--y2_series", help="y2 series name.")
        self._Argument_Parser.add_argument("-xn", "--x_series_number", help="x series number.")
        self._Argument_Parser.add_argument("-yn", "--y_series_number", help="y series number.")
        self._Argument_Parser.add_argument("-y1n", "--y1_series_number", help="y1 series number.")
        self._Argument_Parser.add_argument("-y2n", "--y2_series_number", help="y2 series number.")
        self._Argument_Parser.add_argument("-s", "--subplot",action='store_true', help="Draws every graph in subplot.")
        self._Argument_Parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")

    def _process_Specific_Arguments(self):
        pass

class Device_Log_Arg_Parser(Arg_Parser):
    _epilog = "Arguments parser for logging data from Prokyber device"

    def _add_Specific_Arguments(self):
        self._Argument_Parser.add_argument("-o", "--output_file_name", help="Output file name.")

    def _process_Specific_Arguments(self):
        pass
