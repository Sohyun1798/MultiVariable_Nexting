import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import mne
from sklearn.preprocessing import normalize
import torch

def createTilings(feature_ranges, number_tilings, bins, offsets):
    """
    feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings; example: 3 tilings
    bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """

    tilings = []

    for i in range(number_tilings):
        tiling_bin = bins[i]
        tiling_offset = offsets[i]

        tiling = [] #for each feature dimension

        for j in range(len(feature_ranges)):
            feature_range = feature_ranges[j]
            feat_tiling = np.linspace(feature_range[0], feature_range[1], tiling_bin[j] +1 )[1:-1] + tiling_offset[j]
            tiling.append(feat_tiling)
        tilings.append(tiling)

    return np.array(tilings)

def tileCoding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """

    num_d = len(feature)
    feature_codings = []

    for tiling in tilings:
        feature_coding = []

        for i in range(num_d):
            i_feature = feature[i]
            i_tiling = tiling[i] #tiling on that dimension
            i_coding = np.digitize(i_feature, i_tiling)
            feature_coding.append(i_coding)
        
        feature_codings.append(feature_coding)

    return np.array(feature_codings)

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

NUMBER_TILINGS = 8
bins = []
offsets = []

for i in range(NUMBER_TILINGS):
    bins.append([len(item) + 1 for item in raw_data])
    offsets.append([ 0.2 for _ in raw_data]) #

print(bins)
print(offsets)

tilings = createTilings(feature_ranges, NUMBER_TILINGS, bins, offsets)
print(tilings.shape)

class qvalueFunction:

    def __init__(self, tilings, actions, lr):
        self.tilings = tilings
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in self.tilings]
        self.q_tables = [np.zeros(shape=(state_size+(len(self.actions),))) for state_size in self.state_sizes]

    def value(self, state, action):
        state_codings = tileCoding(state, self.tilings)
        action_idx = self.actions.index(action)

        value = 0
        
        for coding, q_table in zip(state_codings, self.q_tables):
            value += q_table[tuple(coding) + (action_idx,)]

        return value / self.num_tilings

    def update(self, state, action, target):
        state_codings = tileCoding(state, self.tilings)
        action_idx = self.actions.index(action)

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = target - q_table[tuple(coding) + (action_idx,)]
            q_table[tuple(coding) + (action_idx,)] += self.lr * delta

    # get the # of steps to reach the goal under current state value function
    def stepcost(self, state):
        costs = []
        for action in self.actions:
            costs.append(self.value(state,action))
            
        return -np.max(costs)





