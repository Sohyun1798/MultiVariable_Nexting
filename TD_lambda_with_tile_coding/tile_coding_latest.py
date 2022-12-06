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
        
    
    def get_tiles(self, sensor0, sensor1, row1, row2):
        """
        Takes in a position from the heartbeat environment
        and returns a numpy array of active tiles.
        
        returns:
        tiles - np.array, active tiles
        """
    
        
        sensor0_scaled = 0
        sensor1_scaled = 0
        
        sensor0_min = feature_ranges[row1][0]
        #print(feature_ranges[0][0])
        sensor0_max = feature_ranges[row1][1]
        sensor1_min = feature_ranges[row2][0]
        sensor1_max = feature_ranges[row2][1]
        
        
        sensor0_scaled = self.num_tiles* ((sensor0 - sensor0_min)/(sensor0_max - sensor0_min))
        sensor1_scaled = self.num_tiles* ((sensor1 - sensor1_min)/(sensor1_max - sensor1_min))
       #print(position_scaled)
        
      
        
        tiles_ = tiles(self.iht, self.num_tilings, [sensor0_scaled,sensor1_scaled])
       
        
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

#print(type(new_data))  

#df = pd.DataFrame(new_data.T).iloc[0:5000]
#df.plot(kind="line",figsize=(15,3))
#plt.show()


# feature range
feature_ranges = []
for i in range(len(new_data)):
    feature_ranges.append([min(new_data[i]), max(new_data[i])])

#print(feature_ranges)

# feature range
#feature_ranges = []
#for i in range(len(raw_data)):
    #feature_ranges.append([min(raw_data[i]), max(raw_data[i])])

#feature_ranges.append([min(raw_data[0]), max(raw_data[0])])


hbtc = HeartBeatTileCoder(iht_size=256, num_tilings=16, num_tiles=4)
#print("hbtc", hbtc)

#print("Feature Ranges: ", feature_ranges)
#tiles for sensor 0 and 1

heart_sensor0 = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[1][0], feature_ranges[1][1], num=10)
dot_0_1 = list(itertools.product(heart_sensor0, heart_sensor1))
#print(len(dot_0_1))

t = []
row1 = 0
row2 = 1
for feature in dot_0_1:
    sensor0, sensor1 = feature
    print(sensor0, sensor1)
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor1, row1, row2)

    t.append(heart)

tiles_0_1 = np.array(t)
print(tiles_0_1)


#tiles for sensor 0 and 2

heart_sensor0 = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[2][0], feature_ranges[2][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 0
row2 = 2

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 1024)
tiles_0_2 = np.array(t)
#tiles_0_2.append(tiles_0_2 + 1024)
#print(tiles_0_2[2])

#tiles for sensor 0 and 3

heart_sensor0 = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[3][0], feature_ranges[3][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 0
row2 = 3

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 2048)
tiles_0_3 = np.array(t)

#tiles for sensor 0 and 4
heart_sensor0 = np.linspace(feature_ranges[0][0], feature_ranges[0][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[4][0], feature_ranges[4][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 0
row2 = 4

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 3072)
tiles_0_4 = np.array(t)

#tiles for sensor 1 and 2
heart_sensor0 = np.linspace(feature_ranges[1][0], feature_ranges[1][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[2][0], feature_ranges[2][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 1
row2 = 2

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 4096)
tiles_1_2 = np.array(t)

#tiles for sensor 1 and 3

heart_sensor0 = np.linspace(feature_ranges[1][0], feature_ranges[1][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[3][0], feature_ranges[3][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 1
row2 = 3

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 5120)
tiles_1_3 = np.array(t)

#tiles for sensor 1 and 4
heart_sensor0 = np.linspace(feature_ranges[1][0], feature_ranges[1][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[4][0], feature_ranges[4][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 1
row2 = 4

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 6144)
tiles_1_4 = np.array(t)

#tiles for sensor 2 and 3

heart_sensor0 = np.linspace(feature_ranges[2][0], feature_ranges[2][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[3][0], feature_ranges[3][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 2
row2 = 3

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 7168)
tiles_2_3 = np.array(t)

#tiles for sensor 2 and 4

heart_sensor0 = np.linspace(feature_ranges[2][0], feature_ranges[2][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[4][0], feature_ranges[4][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 2
row2 = 4

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 8192)
tiles_2_4 = np.array(t)

#tiles for sensor 3 and 4

heart_sensor0 = np.linspace(feature_ranges[3][0], feature_ranges[3][1], num=10)
heart_sensor1 = np.linspace(feature_ranges[4][0], feature_ranges[4][1], num=10)
dot_0_2 = list(itertools.product(heart_sensor0, heart_sensor1))
#print("position", position_bound)

t = []
row1 = 3
row2 = 4

for feature in dot_0_2:
    sensor0, sensor2 = feature
    #print(test)
    heart = hbtc.get_tiles(sensor0, sensor2, row1, row2)

    t.append(heart)

t = list(np.asarray(t) + 9216)
tiles_3_4 = np.array(t)

active_tiles_0_1 = np.zeros(10240)

for i in range(len(tiles_0_1)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_0_1[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_0_2)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_0_2[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_0_3)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_0_3[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_0_4)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_0_4[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_1_2)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_1_2[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_1_3)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_1_3[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_1_4)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_1_4[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_2_3)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_2_3[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_2_4)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_2_4[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
#print(active_tiles_0_1)

for i in range(len(tiles_3_4)):
    #print("active_tiles",tiles_0_1[i])
    active_tiles_0_1[tiles_3_4[i]] = 1
    #m = tiles_0_1[i]
    #print("where 1", m, active_tiles_0_1[m])
print(active_tiles_0_1.shape)

c = 0
for i in range(len(active_tiles_0_1)):
    if active_tiles_0_1[i]==1:
        c = c + 1

print(c)


"""
#convert array to matrix for multiplication
    x_t = np.asmatrix(get_feature(t))
    x_next = np.asmatrix(get_feature(t + 1))
    w = np.asmatrix(w)
    z = np.asmatrix(z)
"""
