#Common functions that are used by all the workspaces
import itertools
import random
import numpy as np
from tensorsearch import sortHyperparameterValues, findBestValues, hyperparametersFromIndices

#Returns the result of tensorsearch.findbestValues in sorted order
def sortedBestValues(tensor, number_of_values, smallest=True):
    result_dict = findBestValues(tensor, smallest=smallest, number_of_values=number_of_values)
    sorted_dict = sortHyperparameterValues(result_dict)
    return sorted_dict

#Generates an incomplete tensor from a fully known one, randomly sampling the
#indices and providing them along with the incomplete tensor
def generate_incomplete_tensor(tensor, known_fraction=0.25):
    shape = tensor.shape
    #Generate all possible combinations of indices
    total_elements = 1
    value_lists = []
    for dim_size in shape:
        value_lists.append([el for el in range(dim_size)])
        total_elements *= dim_size
    all_indices = list(itertools.product(*value_lists))
    #Randomly sample elements
    no_elements = int(known_fraction*total_elements)
    # Randomly sample from all_indices
    sampled_indices = random.sample(all_indices, no_elements)
    # Generate tensor with all unknown indices set to zero
    incomplete_tensor = np.zeros(shape=shape)
    for index in sampled_indices:
        incomplete_tensor[index] = tensor[index]
    return incomplete_tensor, sampled_indices

#Calculates Hamming distance and augmented Hamming distance between 2 arrays of
#tuples of the same length
def Hamming_distance(arr1, arr2, augmented=False):
    if len(arr1) != len(arr2):
        raise ValueError('For Hamming distance, arrays must be of same length.')
    distance = 0
    for index in range(len(arr1)):
        tup1 = arr1[index]
        tup2 = arr2[index]
        if tup1 != tup2:
            if augmented:
                pointwise_distance = np.linalg.norm(np.array(tup1) - np.array(tup2))
                distance += pointwise_distance
            else:
                distance += 1
    return distance

#Calculates number of common elements between 2 arrays
def common_count(arr1, arr2):
    set1, set2 = set(arr1), set(arr2)
    return len(set1.intersection(set2))

#Calculates norm difference between 2 tensors (numpy arrays)
def norm_difference(ten1, ten2, order=2):
    return np.linalg.norm(np.ndarray.flatten(ten1 - ten2), ord=order)
