# utils.py
import numpy as np

def normalize_to_unit(x, clip=(0.0,1.0)):
    lo, hi = clip
    x = np.asarray(x, dtype=float)
    x = np.clip(x, lo, hi)
    # if already in [0,1] this is identity; otherwise you can scale externally
    return (x - lo) / (hi - lo) if hi != lo else x

def moving_average(arr, window=3):
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad start with initial values
    pad = np.full(window-1, out[0]) if out.size>0 else np.zeros(window-1)
    return np.concatenate([pad, out])
