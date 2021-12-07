# Mandatory fixed import section
from base.Device import Device, Modbus_Handler

# Mandatory floating import section depending on class implementation
from base.Data import BioADC_Data
from base.Common import RTC
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU
from modbus_api.modbus_registers import Input_32Bit_Float_FIFO

# Optional import section depending on class implementation
from matplotlib import pyplot as plt


class BioAdc(Device):

    supported_device_type_ids = [3, 7]
    supported_Modbus_Handlers = [Modbus_Handler_Serial_RTU]
    data: BioADC_Data = BioADC_Data()
    biop_reg: Input_32Bit_Float_FIFO
    rtc: RTC

    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int):
        self._biop_reg = Input_32Bit_Float_FIFO(30001, slave_id, modbus_handler, 2)
        self._rtc = RTC(40071, slave_id, modbus_handler)


    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int):
        pass

    def read_data(self, only_new: bool = True,pc_time = False):
        data = self._biop_reg.read(only_new=only_new,pc_time=pc_time)
        self.data.write(data[0][1], data[0][0], data[1][0])

    def draw_with_plt(self):
        plt.subplot(211, ylabel="Voltage[mV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.channel0)
        plt.subplot(212, ylabel="Voltage[mV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.channel1)

    def set_unix_epoch(self, epoch):
        self._rtc.set_unix_epoch(epoch)

    def get_unix_epoch(self):
        return self._rtc.get_unix_epoch()
        