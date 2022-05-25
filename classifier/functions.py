import numpy as np
import pandas as pd
from scipy.spatial import distance


def resample(df):
    # Default is 10hz
    duration = 4 # duration of the sample in seconds
    time = df['time_ms']
    start = time[0]
    end = start + 1000 * duration
    index = None
    for i in range(10):  # Find closest time value
        index = time.where(time == end).first_valid_index()
        if index:
            break
        end += 1
    if not index:
        return pd.DataFrame(), 0
    df_tmp = df.head(index)
    df_tmp.index = df_tmp['time_ms'].astype('datetime64[ms]')
    df_tmp = df_tmp.resample('1ms').last()
    df_tmp = df_tmp.interpolate() # linear interpolation
    df_tmp = df_tmp.resample('100ms').mean()
    if df_tmp.shape[0] == duration * 10 + 1:
        df_tmp = df_tmp.iloc[1:]
    df_tmp.index = pd.RangeIndex(start=0, stop=duration*10, step=1)
    df_tmp.drop('time_ms', inplace=True, axis=1)
    return df_tmp, index


def scale(x, scaler):
    for i in range(x.shape[0]):
        x[i, :, :] = scaler.transform(x[i, :, :])
    return x


def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost
    return cumdist[an, bn]