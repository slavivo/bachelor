# Mandatory fixed import section
from base.Device import Device
from modbus_api.Modbus_Handler import Modbus_Handler

# Mandatory floating import section depending on class implementation
from base.Data import EEGV1_Data, EEGV1_Sound_Data
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU
from modbus_api.modbus_registers import Holding_16Bit_Unsigned, Holding_64Bit_Unsigned, Input_16Bit_Signed_FIFO

# Optional import section depending on class implementation
from matplotlib import pyplot as plt

SOUND_CTRL = [2]
SOUND_ADJUST_DOWN = [4]
SOUND_ADJUST_UP = [8]

class EEGV1(Device):

    supported_device_type_ids = [0]
    supported_Modbus_Handlers = [Modbus_Handler_Serial_RTU]
    data: EEGV1_Data = EEGV1_Data()
    data_audio: EEGV1_Sound_Data = EEGV1_Sound_Data()
    biop_reg: Input_16Bit_Signed_FIFO
    sound_reg: Holding_64Bit_Unsigned
    sound_ctrl_reg: Holding_16Bit_Unsigned
    playing_start_time = 0

    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int):
        self.biop_reg = Input_16Bit_Signed_FIFO(30001, slave_id, modbus_handler, 2)
        self.sound_reg = Holding_64Bit_Unsigned(40071, slave_id, modbus_handler, 3)
        self.sound_ctrl_reg = Holding_16Bit_Unsigned(40083, slave_id, modbus_handler)

    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int):
        pass

    def read_data(self, only_new: bool = True,pc_time = False):
        data_biopotencial = self.biop_reg.read(only_new=only_new,pc_time=pc_time)
        time_stamp = self.get_time_stamp_ms()
        audio_data = self.sound_reg.read()
        self.data.write(data_biopotencial[0][1], data_biopotencial[0][0], data_biopotencial[1][0])
        self.data_audio.write(time_stamp, audio_data[0], audio_data[1], audio_data[2])

    def draw_with_plt(self):
        plt.subplot(211, ylabel="Fp1[uV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.voltage_fp1_uV)
        plt.subplot(212, ylabel="Cz[uV]", xlabel="Time[ms]")
        plt.plot(self.data.time_ms, self.data.voltage_cz_uV)

    def get_data(self):
        return self.data

    def get_data_audio(self):
        return self.data_audio

    def get_audio_data_type(self) -> type:
        return type(self.data_audio)

    def start_playing(self):
        self.read_data()
        if (not self.is_playing()):
            self.sound_ctrl_reg.write(SOUND_CTRL)
    
    def is_playing(self) -> bool:
        return (self.data_audio.sound_start_ms > self.data_audio.sound_stop_ms)

    def increase_volume(self):
        self.sound_ctrl_reg.write(SOUND_ADJUST_UP)

    def decrease_volume(self):
        self.sound_ctrl_reg.write(SOUND_ADJUST_DOWN)
