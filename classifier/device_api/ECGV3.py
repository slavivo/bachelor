# Mandatory fixed import section
from base.Device import Device
from modbus_api.Modbus_Handler import Modbus_Handler

# Mandatory floating import section depending on class implementation
from base.Data import ECGV3_Data
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU, Modbus_Handler_TCPIP
from modbus_api.modbus_registers import Input_32Bit_Float_FIFO
from base.Common import RTC

# Optional import section depending on class implementation
from matplotlib import pyplot as plt


class ECGV3(Device):

    supported_device_type_ids = [1, 6]
    supported_Modbus_Handlers = [Modbus_Handler_TCPIP, Modbus_Handler_Serial_RTU]
    data: ECGV3_Data = ECGV3_Data()
    biop_reg: Input_32Bit_Float_FIFO
    rtc: RTC

    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int):
        self.biop_reg = Input_32Bit_Float_FIFO(30001, slave_id, modbus_handler, 4)
        self.rtc = RTC(40071, slave_id, modbus_handler)

    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int):
        pass

    def read_data(self, only_new: bool = True,pc_time = False):
        data = self.biop_reg.read(only_new=only_new,pc_time=pc_time)
        self.data.write(data[0][1], data[0][0], data[1][0], data[2][0], data[3][0])

    def draw_with_plt(self):
        plt.subplot(411, ylabel="Bioimp.[kOhm]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.body_impedance_kOhm)
        plt.subplot(412, ylabel="Voltage[mV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.ecg_mV)
        plt.subplot(413, ylabel="Bioimp.[kOhm]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.hand_impedance_kOhm)
        plt.subplot(414, ylabel="Voltage[mV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.tenzo_mV)

    def set_unix_epoch(self, epoch):
        self._rtc.set_unix_epoch(epoch)

    def get_unix_epoch(self):
        return self._rtc.get_unix_epoch()
