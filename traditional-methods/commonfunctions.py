import numpy as np

#Returns a list based on the provided range. Start and end inclusive.
def generate_range(start, end, interval):
    return np.linspace(start, end, int(round((end-start)/interval, 0))+1)
