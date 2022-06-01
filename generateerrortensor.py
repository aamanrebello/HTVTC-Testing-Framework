import numpy as np
import random
import itertools

#Based on provided numerical ranges for the different hyperparameters, generates an incomplete tensor with random elements set
#to results of the evaluation function, evaluated on hyperparameters corresponding to the nonzero position
def generateIncompleteErrorTensor(eval_func, ranges_dict, known_fraction, metric, eval_trials=5, **kwargs):

    evaluation_mode = 'prediction'
    if 'evaluation_mode' in kwargs.keys():
        evaluation_mode = kwargs['evaluation_mode']

    hyperparameter_values = {}
    tensor_dimensions_list = []
    tensor_elements = 1

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values
    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        # Update outer values
        hyperparameter_values[key] = value_list
        tensor_dimensions_list.append(len(value_list))
        tensor_elements *= len(value_list)

    tensor_dimensions_tuple = tuple(tensor_dimensions_list)
    error_tensor = np.zeros(tensor_dimensions_tuple)
    known_elements = int(known_fraction*tensor_elements)

    # Generate list of all possible index combinations
    value_lists = []
    for i in range(len(tensor_dimensions_tuple)):
        value_lists.append([el for el in range(tensor_dimensions_tuple[i])])
    all_indices = list(itertools.product(*value_lists))

    # Randomly sample from all_indices
    sampled_indices = random.sample(all_indices, known_elements)

    # Populate the empty tensor
    for tensor_index in sampled_indices:
        current_hyperparameter_values = {}
        # Obtain the tensor index and the hyperparameter values to pass to eval_func
        dimension_index = 0
        for key in hyperparameter_values.keys():
            value_list = hyperparameter_values[key]
            value_index = tensor_index[dimension_index]
            current_hyperparameter_values[key] = value_list[value_index]
            dimension_index += 1
        # Assign to tensor using result of eval_func averaged over multiple trials
        eval_result_avg = 0
        for trial in range(eval_trials):
            eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
        error_tensor[tuple(tensor_index)] = eval_result_avg

    #return populated tensor and indices
    return error_tensor, sampled_indices
