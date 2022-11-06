import numpy as np

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
            feat_tiling = np.linspace(feature_range[0], feature_range[1], tiling_bin[j] + 1)[1:-1] + tiling_offset[j]
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


