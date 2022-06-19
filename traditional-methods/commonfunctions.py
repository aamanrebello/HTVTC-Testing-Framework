import numpy as np

#Returns a list based on the provided range. Start and end inclusive.
def generate_range(start, end, interval):
    return np.linspace(start, end, int(round((end-start)/interval, 0))+1)


def truncate_features(data_array, no_features):
    NO_SAMPLES = len(data_array)
    array_copy = np.zeros(shape=(NO_SAMPLES, no_features))
    for i in range(NO_SAMPLES):
        all_features_i = data_array[i]
        truncated_features_i = all_features_i[:no_features]
        array_copy[i] = truncated_features_i
    return array_copy        
