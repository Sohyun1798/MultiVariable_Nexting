import numpy as np
import itertools
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT
import pandas as pd
import neurokit2 as nk
import mne
from sklearn.preprocessing import normalize
import torch


def get_tiles(self, feature_ranges):
        """
        Takes a feature range from heartbeat dataset  
        and returns a numpy array of active tiles.
        
        returns:
        tiles - np.array, active tiles
        """
        
        position_scaled = 0
        velocity_scaled = 0
        
        # ----------------
        # your code here
        position_min = -1.2
        position_max = 0.5
        velocity_min = -0.07
        velocity_max = 0.07
        
        
        position_scaled = self.num_tiles* ((position - position_min)/(position_max - position_min))
        velocity_scaled = self.num_tiles* ((velocity - velocity_min)/(velocity_max - velocity_min))
        
      
        
        # ----------------
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tc.tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])
       
        
        return np.array(tiles)



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


# after normalization 
raw_data = normalize(raw_data,axis=1,norm="max")
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names

# cleanup the data
for i in range(len(raw_data)):
    raw_data[i] = nk.ecg_clean(raw_data[i],method="neurokit")

# feature range
feature_ranges = []
for i in range(len(raw_data)):
    feature_ranges.append([min(raw_data[i]), max(raw_data[i])])

print("Feature Ranges: ", feature_ranges)