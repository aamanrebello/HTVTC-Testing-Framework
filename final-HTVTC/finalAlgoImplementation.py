#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from crosstechnique import generateCrossComponents, noisyReconstruction
from tensorsearch import findBestValues, hyperparametersFromIndices
from generateerrortensor import generateIncompleteErrorTensor


#Helper function to narrow down the search space through the range dictionary.
def update_ranges_dict(ranges_dict, selected_combination, min_real_interval):
    #Iterate over each hyperparameter
    for hyperparameter in ranges_dict.keys():
        hyp_info = ranges_dict[hyperparameter]
        selected_value = selected_combination[hyperparameter]
        #There is no general way to compare categorical hyperparameters, so these are not interfered with.
        if hyp_info['type'] == 'CATEGORICAL':
            continue
        if hyp_info['type'] in ('INTEGER', 'REAL'):
            #In case a list of values is provided, the list is sorted and a sub-list of half the size, with the selected
            #value at the centre is taken.
            if 'values' in hyp_info.keys():
                no_values = len(hyp_info['values'])
                if no_values == 1:
                    continue
                sorted_values = sorted(hyp_info['values'])
                selected_index = sorted_values.index(selected_value)
                quarter_values = int(round(no_values/4))
                lower_index = max(selected_index - quarter_values, 0)
                #If the upper index is allowed to be equal to the lower index, this would result in an empty list
                upper_index = max( min(selected_index + quarter_values, no_values), lower_index + 1)
                ranges_dict[hyperparameter]['values'] = sorted_values[lower_index:upper_index]
            else:
                #Adjust the resolution interval by halving it without letting it fall below the minimum interval size. 
                new_interval = None
                # For an integer hyperparameter, the resolution must be an integer.
                if hyp_info['type'] == 'INTEGER':
                    new_interval = max(hyp_info['interval']//2, 1)
                # For a general real-valued hyperparamater, there is no restriction on the resolution.
                elif hyp_info['type'] == 'REAL':
                    new_interval = max(hyp_info['interval']/2, min_real_interval)
                ranges_dict[hyperparameter]['interval'] = new_interval
                #Adjust endpoints of search space.
                search_space_size = hyp_info['end'] - hyp_info['start']
                # Add and subtract quarter size from the current selected value, so the new limits fit the resolution interval.
                # The new limits should also respect the original limits of the search space.
                quarter_size = search_space_size/4
                new_end = min( round( (selected_value + quarter_size)/new_interval ) * new_interval, hyp_info['end'] )
                new_start = max( round( (selected_value - quarter_size)/new_interval ) * new_interval, hyp_info['start'] )
                ranges_dict[hyperparameter]['end'] = new_end
                ranges_dict[hyperparameter]['start'] = new_start            
    #Return updated ranges dict.
    return ranges_dict

                
#The overall implementation.
#List of kwargs:
#   - min_interval: The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
#   - max_completion_cycles: The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
#   - max_size_gridsearch: The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
#   - evaluation_mode: may be needed to configure the evaluation function that populates the tensor.
#   - eval_trials: The number of evaluations of the evaluation function needed to generate one tensor element. Default 1.
#   - tucker_rank_list: The assumed Tucker rank of the completed tensor
def final_HTVTC(ranges_dict, eval_func, metric, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 0
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    for cycle_num in range(max_completion_cycles):
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #Add to history
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value
            bestValue = findBestValues(full_tensor, smallest=True, number_of_values=1)
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history


import time

#Repeat of the above function that records the timestamps at the end of each cycle
def final_HTVTC_profiling(ranges_dict, eval_func, metric, **kwargs):

    # Deal with kwargs that are not passed into tensor generation------------------------------------------------------
    kwargskeys = kwargs.keys()
    #The minimum resolution interval required for real-valued hyperparameter. For integers, the minimum is 1.
    min_interval = 1
    if 'min_interval' in kwargskeys:
        min_interval = kwargs['min_interval']
    #The maximum number of tensor completions that are needed. The algorithm may terminate before completing this many completions.
    max_completion_cycles = 1
    if 'max_completion_cycles' in kwargskeys:
        max_completion_cycles = kwargs['max_completion_cycles']
    #The maximum number of elements before a grid search can be performed. If 0, this means there will be no grid search.
    max_size_gridsearch = 0
    if 'max_size_gridsearch' in kwargskeys:
        max_size_gridsearch = kwargs['max_size_gridsearch']
    # The number of evaluations of the evaluation function needed to generate one tensor element.
    eval_trials = 1
    if 'eval_trials' in kwargskeys:
        eval_trials = kwargs['eval_trials']

    #Begin time measurement
    start_time = time.perf_counter_ns()

    #Perform the repeated tensor completions----------------------------------------------------------------------------
    history = []
    selected_combination = None
    #Used for profiling
    timestamps = []

    for cycle_num in range(max_completion_cycles):
        #Perform the tensor completion
        body, joints, arms = generateCrossComponents(eval_func=eval_func, ranges_dict=ranges_dict, metric=metric, eval_trials=eval_trials, **kwargs)
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Find best value
        bestValue = findBestValues(completed_tensor, smallest=True, number_of_values=1)
        index_list, value_list = bestValue['indices'], bestValue['values']
        #Obtain hyperparameter from it
        combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
        selected_combination = combinations[0]
        #Add to history
        history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'tensor completion'})
        #Record timestamp
        timestamp = time.perf_counter_ns()
        time_since_start = timestamp - start_time
        timestamps.append(time_since_start)
        
        #If below limit, perform grid search and break.
        if completed_tensor.size < max_size_gridsearch:
            #Generate complete tensor
            full_tensor, _ = generateIncompleteErrorTensor(eval_func=eval_func, ranges_dict=ranges_dict, known_fraction=1, metric=metric, eval_trials=eval_trials, **kwargs)
            #Find best value
            bestValue = findBestValues(full_tensor, smallest=True, number_of_values=1)
            index_list, value_list = bestValue['indices'], bestValue['values']
            #Obtain hyperparameter from it
            combinations = hyperparametersFromIndices(index_list, ranges_dict, ignore_length_1=True)
            selected_combination = combinations[0]
            #Add to history
            history.append({'combination': selected_combination, 'predicted_loss': value_list[0], 'method': 'grid search'})
            #Record timestamp
            timestamp = time.perf_counter_ns()
            time_since_start = timestamp - start_time
            timestamps.append(time_since_start)
            break
        
        #Only need to update the ranges dict if we are using it in the next loop iteration.
        if cycle_num == max_completion_cycles - 1:
            break
        ranges_dict = update_ranges_dict(ranges_dict, selected_combination, min_interval)
        
    #return the optimal hyperparameter combination as decided by the algorithm-------------------------------------------
    return selected_combination, history, timestamps

    
