import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pandas import DataFrame


@dataclass(frozen=True)
class Metadata(object):
    """
    Class for storing device metadata. 
    This class should not (and can not) be created directly
    and all attributes are only read only
    """

    device_type_id: int
    device_type_name: str
    uid: str

    def __str__(self):
        return "Metadata:\n\tDevice Type ID: %s\n\tDevice Type Name: %s\n\tUID: %s" % (self.device_type_id, self.device_type_name, self.uid)

    @classmethod
    def create(cls, device_type_id: int, device_type_name: str, uid: str):
        """Returns instance of class Device_Metadata class"""
        return cls(device_type_id, device_type_name, uid)

@dataclass(frozen=True)
class Firmware_Information(object):
    """
    Class for storing device information. 
    This class should not (and can not) be created directly
    and all attributes are only read only
    """

    upload_time: datetime
    git_commit: str

    def __str__(self):
        return "Additional information:\n\tFirmaware upload time: %s\n\tFirmware GIT commit id: %s" % (self.upload_time, self.git_commit)

    @classmethod
    def create(cls, upload_time: datetime, commit:str):
        """Returns instance of class Device_Information class"""
        return cls(upload_time, commit)


@dataclass(init=True)
class Data(object):
    """
    Abstract class for any Data class implementation.
    For every Prokyber device there should be appripriate 
    Data class with its owm unique data fields.
    Exception is time_ms attribute which is common
    in all implementations
    """

    time_ms: typing.List[float] = field(default_factory=list)

    def get(self, key) -> typing.List[float]:
        """Returns datafield. Made only to resolve type cast"""
        return self.__dict__.get(key)

    def extend(self, data) -> None:
        """Extends all internal fields"""
        for key in self.__dict__.keys():
            self.get(key).extend(data.__dict__.get(key))

    def flush(self):
        """Flushes all internal fields"""
        for key in self.__dict__.keys():
            self.get(key).__init__()

    def from_dataframe(self, dataframe: DataFrame):
        """Fills fields of Data class from pandas dataframe"""
        for key in self.__dict__.keys():
            self.get(key).extend(dataframe.get(key))

    def from_dataframe_drop_na(self, dataframe: DataFrame):
        """Fills fields of Data class from pandas dataframe without NA"""
        for key in self.__dict__.keys():
            self.get(key).extend(dataframe.get(key).dropna())

    def from_dataframe_drop_na_and_duplicates(self, dataframe: DataFrame):
        """Fills fields of Data class from pandas dataframe without NA and duplicates"""
        for key in self.__dict__.keys():
            self.get(key).extend(dataframe.get(key).dropna().drop_duplicates())

    def cast_to(self, cast_to: type, attr: str):
        """Converts all elements in attribute 'attr' to specified 'cast_to' data type"""
        for i in range(len(self.__getattribute__(attr))):
            self.__getattribute__(attr)[i] = cast_to(self.__getattribute__(attr)[i])

    @abstractmethod
    def write(self) -> None:
        """Writes data to fields"""
        pass
    

@dataclass
class FastImu_Data(Data):
    """Data class for Fast Imu Prokyber device"""
    gyroscopeX_mdps: typing.List[float] = field(default_factory=list)
    gyroscopeY_mdps: typing.List[float] = field(default_factory=list)
    gyroscopeZ_mdps: typing.List[float] = field(default_factory=list)
    accelerationX_g: typing.List[float] = field(default_factory=list)
    accelerationY_g: typing.List[float] = field(default_factory=list)
    accelerationZ_g: typing.List[float] = field(default_factory=list)

    def write(self, time:list, gX:list, gY:list, gZ:list, aX:list, aY:list, aZ:list) -> None:
        self.time_ms = time
        self.gyroscopeX_mdps = (list(map(lambda x: x*70, gX)))
        self.gyroscopeY_mdps = (list(map(lambda x: x*70, gY)))
        self.gyroscopeZ_mdps = (list(map(lambda x: x*70, gZ)))
        self.accelerationX_g = (list(map(lambda x: x*0.000488, aX)))
        self.accelerationY_g = (list(map(lambda x: x*0.000488, aY)))
        self.accelerationZ_g = (list(map(lambda x: x*0.000488, aZ)))

@dataclass
class SmartAHRS_Data(Data):
    """Data class for Smart AHRS Prokyber device"""
    accelaration_aX_g: typing.List[float] = field(default_factory=list)
    accelaration_aY_g: typing.List[float] = field(default_factory=list)
    accelaration_aZ_g: typing.List[float] = field(default_factory=list)
    gyroscope_aX_mdps: typing.List[float] = field(default_factory=list)
    gyroscope_aY_mdps: typing.List[float] = field(default_factory=list)
    gyroscope_aZ_mdps: typing.List[float] = field(default_factory=list)
    magnetometer_aX_mT: typing.List[float] = field(default_factory=list)
    magnetometer_aY_mT: typing.List[float] = field(default_factory=list)
    magnetometer_aZ_mT: typing.List[float] = field(default_factory=list)
    euler_Yaw: typing.List[float] = field(default_factory=list)
    euler_Roll: typing.List[float] = field(default_factory=list)
    euler_Pitch: typing.List[float] = field(default_factory=list)
    quaternion_aW: typing.List[float] = field(default_factory=list)
    quaternion_aX: typing.List[float] = field(default_factory=list)
    quaternion_aY: typing.List[float] = field(default_factory=list)
    quaternion_aZ: typing.List[float] = field(default_factory=list)
    linearise_acceleration_aX_g: typing.List[float] = field(default_factory=list)
    linearise_acceleration_aY_g: typing.List[float] = field(default_factory=list)
    linearise_acceleration_aZ_g: typing.List[float] = field(default_factory=list)
    gravity_aX_g: typing.List[float] = field(default_factory=list)
    gravity_aY_g: typing.List[float] = field(default_factory=list)
    gravity_aZ_g: typing.List[float] = field(default_factory=list)

    def write(self) -> None:
        pass


@dataclass
class ECGV3_Data(Data):
    """Data class for ECGV3 Prokyber device"""
    body_impedance_kOhm: typing.List[float] = field(default_factory=list)
    ecg_mV: typing.List[float] = field(default_factory=list)
    hand_impedance_kOhm: typing.List[float] = field(default_factory=list)
    tenzo_mV: typing.List[float] = field(default_factory=list)

    def write(self, time:list, body_impedance:list, ecg:list, hand_impedance:list, tenzo:list) -> None:
        self.time_ms = time
        self.body_impedance_kOhm = body_impedance
        self.ecg_mV = ecg
        self.hand_impedance_kOhm = hand_impedance
        self.tenzo_mV = tenzo


@dataclass
class BioADC_Data(Data):
    """Data class for BioADC Prokyber device"""
    channel0: typing.List[float] = field(default_factory=list)
    channel1: typing.List[float] = field(default_factory=list)

    def write(self, time:list, channel0:list, channel1:list) -> None:
        self.time_ms = time
        self.channel0 = channel0
        self.channel1 = channel1


@dataclass
class EEGV1_Data(Data):
    """Data class for EEGV1 Prokyber device"""
    voltage_fp1_uV: typing.List[float] = field(default_factory=list)
    voltage_cz_uV: typing.List[float] = field(default_factory=list)

    def write(self, time:list, fp1:list, cz:list) -> None:
        self.time_ms = time
        self.voltage_fp1_uV = list(map(lambda x: float(x) * 0.219727, fp1))
        self.voltage_cz_uV = list(map(lambda x: float(x) * 0.219727, cz))


@dataclass
class EEGV1_Sound_Data(Data):
    """Data class for EEGV1 Prokyber device"""
    reaction_time_ms: typing.List[float] = field(default_factory=list)
    sound_start_ms: typing.List[float] = field(default_factory=list)
    sound_stop_ms: typing.List[float] = field(default_factory=list)

    def write(self, time: float, reaction: float, sound_start: float, sound_stop: float) -> None:
        self.time_ms = [time]
        self.reaction_time_ms = [reaction]
        self.sound_start_ms = [sound_start]
        self.sound_stop_ms = [sound_stop]
