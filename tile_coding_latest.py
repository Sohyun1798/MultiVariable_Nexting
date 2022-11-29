import numpy as np
import itertools
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT
import pandas as pd
import neurokit2 as nk
import mne
from sklearn.preprocessing import normalize
import torch


class HeartBeatTileCoder:
    def __init__(self, iht_size, num_tilings, num_tiles):
        """
        Initializes the HeartBeat Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        
    
    def get_tiles(self, position):
        """
        Takes in a position from the heartbeat environment
        and returns a numpy array of active tiles.
        
        returns:
        tiles - np.array, active tiles
        """
    
        
        position_scaled = 0
        
        position_min = -0.33
        position_max = 0.4
        
        
        position_scaled = self.num_tiles* ((position - position_min)/(position_max - position_min))
       #print(position_scaled)
        
      
        
        tiles_ = tiles(self.iht, self.num_tilings, [position_scaled])
       
        
        return np.array(tiles_)



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
#for i in range(len(raw_data)):
    #feature_ranges.append([min(raw_data[i]), max(raw_data[i])])

feature_ranges.append([min(raw_data[0]), max(raw_data[0])])


hbtc = HeartBeatTileCoder(iht_size=4096, num_tilings=8, num_tiles=4)
#print("hbtc", hbtc)

#print("Feature Ranges: ", feature_ranges)

pos_tests = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=10)
#vel_tests = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=8)
#tests = list(itertools.product(pos_tests, vel_tests))
#print("position", position_bound)

t = []

for test in pos_tests:
    position = test
    #print(test)
    heart = hbtc.get_tiles(position=position)
    t.append(heart)

print(t)
