import tensorly as tl
import numpy as np

#Converts a 1D index to a higher dimensional index-----------------------------------------------------
def higherDimensionalIndex(index, dimensions):
    dimension_list = [d for d in dimensions] # Convert dimensions tuple into list
    no_dimensions = len(dimension_list) # Find number of dimensions
    higher_dimensional_index_list = tl.zeros(no_dimensions) # Create list to store calculated indices
    # Calculate indices
    for i in range(-1,-(no_dimensions+1),-1):
        higher_dimensional_index_list[i] = index % dimension_list[i]
        index = index // dimension_list[i]
    # Convert to tuple
    return tuple(higher_dimensional_index_list)
#-----------------------------------------------------------------------------------------------------


#Function to find smallest or largest elements in tensor along with indices---------------------------
def findBestValues(tensor, smallest=True, number_of_values=1):
    tensor_dimensions = tl.shape(tensor) # Obtain tensor dimensions
    vector = tensor.flatten(order='C') # flatten tensor along increasing order of dimensions
    # Obtain indices in flattened array
    indices = None
    if smallest:
        indices = np.argpartition(vector, number_of_values)[:number_of_values]
    else:
        indices = np.argpartition(vector, -number_of_values)[-number_of_values:]
    values = vector[indices] # Obtain the largest/smallest values
    higher_dimensional_indices = [higherDimensionalIndex(index, tensor_dimensions) for index in indices]
    return {
        'values': values,
        'indices': higher_dimensional_indices
        }
#-----------------------------------------------------------------------------------------------------


#Given a list of indices, return the corresponding hyperparameters in the tensor representation-------
def hyperparametersFromIndices(index_list, hyperparameter_ranges_dict):
    # Throw exception if number of indices is unequal to the number of hyperparameters
    if len(index_list[0]) != len(hyperparameter_ranges_dict.keys()):
        raise ValueError(f'The indices ({len(index_list[0])}) and hyperparameter configuration ({len(hyperparameter_ranges_dict.keys())}) have unequal dimensions.')
    hyperparameter_values = []
    for index in index_list:
        dimension_index = 0
        current_hyperparameter_values = {}
        # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values
        for key in hyperparameter_ranges_dict.keys():
            info = hyperparameter_ranges_dict[key]
            value_list_index = int(index[dimension_index])
            #If the values are already provided as a list
            if 'values' in info.keys():
                value_list = info['values']
                current_hyperparameter_values[key] = value_list[value_list_index]
            #If the values are provided as a range, no need to create a value list
            else:
                start = float(info['start'])
                interval = float(info['interval'])
                current_hyperparameter_values[key] = start + value_list_index*interval
            dimension_index += 1
        # Update outer values
        hyperparameter_values.append(current_hyperparameter_values)
    return hyperparameter_values
#-----------------------------------------------------------------------------------------------------
