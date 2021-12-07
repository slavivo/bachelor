# Mandatory fixed import section
from base.Device import Device
from modbus_api.Modbus_Handler import Modbus_Handler

# Mandatory floating import section depending on class implementation
from base.Data import FastImu_Data
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU
from modbus_api.modbus_registers import Input_16Bit_Signed_FIFO

# Optional import section depending on class implementation
from matplotlib import pyplot as plt


class FastImu(Device):

    supported_device_type_ids = [8]
    supported_Modbus_Handlers = [Modbus_Handler_Serial_RTU]
    data: FastImu_Data = FastImu_Data()
    imu_reg: Input_16Bit_Signed_FIFO

    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int):
        self.imu_reg = Input_16Bit_Signed_FIFO(30001, slave_id, modbus_handler, 6)

    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int):
        pass

    def read_data(self, only_new: bool = True,pc_time = False):
        data = self.imu_reg.read(only_new=only_new,pc_time=pc_time)
        self.data.write(data[0][1], data[0][0], data[1][0], data[2][0], data[3][0], data[4][0], data[5][0])

    def draw_with_plt(self):
        plt.subplot(611, ylabel="gyroscopeX[mdps]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.gyroscopeX_mdps)
        plt.subplot(612, ylabel="gyroscopeY[mdps]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.gyroscopeY_mdps)
        plt.subplot(613, ylabel="gyroscopeZ[mdps]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.gyroscopeZ_mdps)
        plt.subplot(614, ylabel="accelerationX[g]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.accelerationX_g)
        plt.subplot(615, ylabel="accelerationY[g]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.accelerationY_g)
        plt.subplot(616, ylabel="accelerationZ[g]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.accelerationZ_g)
