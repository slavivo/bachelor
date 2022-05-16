import numpy as np
import pandas as pd
from scipy.spatial import distance


def resample(df):
    # Default is 10hz
    time = df['time_ms']
    start = time[0]
    end = start + 10 * 300  # 300 * 10 ms -> 3s
    index = None
    for i in range(10):  # Find closest time value
        index = time.where(time == end).first_valid_index()
        if index:
            break
        end += 1
    if not index:
        return pd.DataFrame(), 0
    df_tmp = df.head(index)
    df_tmp['time_delta'] = pd.to_timedelta(df_tmp['time_ms'], 'ms')
    df_tmp.index = df_tmp['time_delta']
    df_tmp = df_tmp.resample('1000ms').mean()
    df_tmp.index = pd.RangeIndex(start=0, stop=3, step=1)
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