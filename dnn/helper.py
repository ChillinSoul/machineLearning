import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def data_standardizer(data_frame:pd.DataFrame):
    standardScaler = StandardScaler()
    data_frame = standardScaler.fit_transform(data_frame)
    data = pd.DataFrame(data_frame)
    return data

def revenue_log(y):
    res = np.log(y)
    if res == float('-inf'):
        return 0.1
    else:
        return res
    
def devide_by_1_000_000(y):
    return y/1000000
def revenue_exp(y):
    return np.exp(y)

