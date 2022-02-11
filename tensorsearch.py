import tensorly as tl
import numpy as np

# Converts a 1D index to a higher dimensional index
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

# Function
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
