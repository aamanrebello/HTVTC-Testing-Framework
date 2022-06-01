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
def randomly_sample_tensor(raw_tensor, known_fraction=0.25):
    #Remove all dimensions of size 1
    tensor = np.squeeze(raw_tensor)
    #Obtain shape
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


#Generates an incomplete tensor from a fully known one, sampling the
#indices uniformly in each fibre along the largest dimension.
def uniformly_sample_tensor(raw_tensor, known_fraction=0.25):
    #Remove all dimensions of size 1
    tensor = np.squeeze(raw_tensor)
    #Find number of elements in tensor
    total_elements = tensor.size
    #Obtain shape
    original_shape = tensor.shape
    #Find size of largest dimension
    largest_dim_size = max(original_shape)
    #Find number of fibres
    no_fibres = int(total_elements/largest_dim_size)
    #Find index of largest dimension
    largest_dim_index = np.argmax(original_shape)
    #Number of samples per fibre
    samples_per_fibre = int(largest_dim_size*known_fraction)
    #Find remainder due to int truncation
    remainder = int(known_fraction*total_elements) - samples_per_fibre*no_fibres 
    #Move largest dimension to the last
    tensor = np.moveaxis(tensor, largest_dim_index, -1)
    adjusted_shape = tensor.shape
    #Create array of zeros in original shape
    incomplete_tensor = np.zeros(original_shape)
    #Generate all possible combinations of indices in smaller dimensions
    value_lists = []
    for dim_size in adjusted_shape[:-1]:
        value_lists.append([el for el in range(dim_size)])
    smaller_indices = list(itertools.product(*value_lists))
    #Generate indices of elements evenly spaced out in a fibre in the non-remainder case
    step = largest_dim_size//samples_per_fibre
    non_remainder_indices = [i for i in range(0, largest_dim_size, step)]
    #Generate indices of elements evenly spaced out in a fibre in the remainder case
    step = largest_dim_size//(samples_per_fibre + 1)
    remainder_indices = [i for i in range(0, largest_dim_size, step)]
    #Iterate through smaller indices
    sampled_indices = []
    iterations = 0
    for index in smaller_indices:
        fibre = tensor[index]
        index_list = non_remainder_indices
        if iterations < remainder:
            index_list = remainder_indices
        for fibre_index in index_list:
            adjusted_fibre_index = (fibre_index + iterations)%largest_dim_size
            tensor_index = tuple(list(index[:largest_dim_index]) + [adjusted_fibre_index] + list(index[largest_dim_index:]))
            incomplete_tensor[tensor_index] = fibre[adjusted_fibre_index]
            sampled_indices.append(tensor_index)
        iterations += 1
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
