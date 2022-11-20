import numpy as np
import pandas as pd
def moving_average(x, w,mode="same"):
    """
    calculate moving average of an array x
    w is the number of samples you want to average
    """
    return np.convolve(x, np.ones(w), mode) / w

def rmse_norm(y1, y2):
    """
    rmse value of two vectors
    """
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))