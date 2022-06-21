from base.Data import Data
import pandas as pd
from datetime import datetime
from queue import Queue

class Logger:
    _buffer: Data
    _buffer_2: Data

    def __init__(self, Dtype: type):
        self._buffer = Dtype()
        self._buffer_2 = Dtype()

    def write_data(self, data: Data) -> None:
        self._buffer_2.extend(data)
        self._buffer.extend(data)

    def get_data(self):
        return self._buffer

    def flush(self):
        self._buffer.flush()

    def __to_data_frame(self, first=True) -> None:
        if first:
            self._dataframe =  pd.DataFrame(self._buffer.__dict__, dtype=float)
        else:
            self._dataframe =  pd.DataFrame(self._buffer_2.__dict__, dtype=float)

    def write_file(self, name: str = None, sep: str = ";") -> None:
        self.__to_data_frame(first=False)
        if (not name):
            name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.csv")
        self._dataframe.to_csv(name, sep, index=False)

    def write_queue(self, queue) -> None:
        self.__to_data_frame()
        queue.put(self._dataframe)
        self.flush()