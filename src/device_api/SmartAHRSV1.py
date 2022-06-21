# Mandatory fixed import section
from base.Device import Device
from modbus_api.Modbus_Handler import Modbus_Handler

# Mandatory floating import section depending on class implementation
from base.Data import SmartAHRS_Data
from modbus_api.Modbus_Handler import Modbus_Handler_Serial_RTU
from modbus_api.modbus_registers import Input_32Bit_Float_FIFO
from base.Common import RTC

# Optional import section depending on class implementation
from matplotlib.collections import PolyCollection
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


class SmartAHRSV1(Device):

    supported_device_type_ids = [4, 9]
    supported_Modbus_Handlers = [Modbus_Handler_Serial_RTU]
    data: SmartAHRS_Data = SmartAHRS_Data()
    smart_reg: Input_32Bit_Float_FIFO
    rtc: RTC

    def _create_registers(self, modbus_handler: Modbus_Handler, slave_id: int):
        self.smart_reg = Input_32Bit_Float_FIFO(30001, slave_id, modbus_handler, 22)
        self.rtc = RTC(40071, slave_id, modbus_handler)

    def _post_init(self, modbus_handler: Modbus_Handler, slave_id: int):
        self.load_file()

    def read_data(self, only_new: bool = True,pc_time = False):
        data = self.smart_reg.read(reg_count=22, only_new=only_new,pc_time=pc_time)
        self.data.time_ms = data[0][1]
        self.data.accelaration_aX_g = data[0][0]
        self.data.accelaration_aY_g = data[1][0]
        self.data.accelaration_aZ_g = data[2][0]
        self.data.gyroscope_aX_mdps = data[3][0]
        self.data.gyroscope_aY_mdps = data[4][0]
        self.data.gyroscope_aZ_mdps = data[5][0]
        self.data.magnetometer_aX_mT = data[6][0]
        self.data.magnetometer_aY_mT = data[7][0]
        self.data.magnetometer_aZ_mT = data[8][0]
        self.data.euler_Yaw = data[9][0]
        self.data.euler_Roll = data[10][0]
        self.data.euler_Pitch = data[11][0]
        self.data.quaternion_aW = data[12][0]
        self.data.quaternion_aX = data[13][0]
        self.data.quaternion_aY = data[14][0]
        self.data.quaternion_aZ = data[15][0]
        self.data.linearise_acceleration_aX_g = data[16][0]
        self.data.linearise_acceleration_aY_g = data[17][0]
        self.data.linearise_acceleration_aZ_g = data[18][0]
        self.data.gravity_aX_g = data[19][0]
        self.data.gravity_aY_g = data[20][0]
        self.data.gravity_aZ_g = data[21][0]

    def load_file(self):
        self.V, self.F = [], []
        with open("bunny.obj") as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                values = line.split()
                if not values:
                    continue
                if values[0] == 'v':
                    self.V.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    self.F.append([int(x) for x in values[1:4]])

    def draw_with_plt(self):
        w = self.data.quaternion_aW[-1]
        x = self.data.quaternion_aX[-1]
        y = self.data.quaternion_aY[-1]
        z = self.data.quaternion_aZ[-1]
        
        q1 = Quaternion(w, y, z, x)
        q2 = Quaternion(axis=[1, 0, 0], angle=3.14159265)

        q = q1*q2
        els = q.elements
        r = R.from_quat([els[0], els[1], els[2], els[3]])

        V, F = np.array(self.V), np.array(self.F)-1

        V = r.apply(V)

        V = (V-(V.max(0)+V.min(0))/2)/max(V.max(0)-V.min(0))
        V = np.c_[V, np.ones(len(V))]
        V /= V[:, 3].reshape(-1, 1)
        V = V[F]
        T = V[:, :, :2]
        Z = -V[:, :, 2].mean(axis=1)
        I = np.argsort(Z)
        T = T[I, :]

        fig = plt.gcf()
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1],
                          ylim=[-1, +1], frameon=False)
        collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor="None", edgecolor="black")
        ax.add_collection(collection)
