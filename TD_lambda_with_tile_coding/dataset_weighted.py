import numpy as np
import itertools
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT
import pandas as pd
import neurokit2 as nk
import mne
from sklearn.preprocessing import normalize
import torch




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# important variables
PRED_LENGTH = 1000 # how many samples in the future are we predicting
WINDOW_LENGTH = 4000 # how many samples we look to make predictions
TOTAL_LENGTH = PRED_LENGTH + WINDOW_LENGTH

SLIDE_SIZE = 100
INPUT_DIM = 5
OUTPUT_DIM = 5

# load data
file = "data/r04.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
print(type(raw_data))


# after normalization 
raw_data = normalize(raw_data,axis=1,norm="max")
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names
print(raw_data.shape)
# cleanup the data
for i in range(len(raw_data)):
    raw_data[i] = nk.ecg_clean(raw_data[i],method="neurokit")

weighted_data = []

for i in range(len(raw_data)):
    alpha = 0.1
    beta = alpha
    s = 0  
    data = [] 
    for t in range(len(raw_data[i])):
        #print(len(raw_data[i]), t)
        data.append((1 - beta) * s + beta * raw_data[i][t])
        s = data[t]
        beta = alpha/s

    weighted_data.append(data) 

new_data = np.array(weighted_data)
print(new_data.shape)

print(type(new_data))  

df = pd.DataFrame(new_data.T).iloc[0:5000]
df.plot(kind="line",figsize=(15,3))
plt.show()


# feature range
feature_ranges = []
for i in range(len(new_data)):
    feature_ranges.append([min(new_data[i]), max(new_data[i])])

#print(feature_ranges)

