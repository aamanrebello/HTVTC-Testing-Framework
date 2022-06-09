#IMPORTS==============================================
import unittest
import json
import numpy as np
import time

#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from generateerrortensor import generateIncompleteErrorTensor
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from tensorcompletion import tensorcomplete_TMac_TT
from tensorcompletion import ket_augmentation, inverse_ket_augmentation
from tensorsearch import findBestValues, hyperparametersFromIndices
import regressionmetrics
import classificationmetrics

quantity = 'EXEC-TIME'

known_fraction = 0.25

#Load dataset
task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data)
func = evaluationFunctionGenerator(data_split, algorithm='random-forest', task=task)

#Start timer/memory profiler/CPU timer
start_time = None
if quantity == 'EXEC-TIME':
    import time
    start_time = time.perf_counter_ns()
elif quantity == 'CPU-TIME':
    import time
    start_time = time.process_time_ns()
elif quantity == 'MAX-MEMORY':
    import tracemalloc
    tracemalloc.start()

#Generate range dictionary
ranges_dict = {
        'no_trees': {
            'values':[1,10,20,30,40]
        },
        'max_tree_depth': {
            'values':[1, 5, 10, 15, 20]
        },
        'bootstrap': {
            'values': [True, False]
        },
        'min_samples_split': {
            'start': 2.0,
            'end': 10.0,
            'interval': 1.0,
        },
        'no_features': {
            'start': 1.0,
            'end': 10.0,
            'interval': 1.0,
        },
    }

#Generate incomplete tensor
incomplete_tensor, known_indices = generateIncompleteErrorTensor(func, ranges_dict, known_fraction, metric=classificationmetrics.KullbackLeiblerDivergence, evaluation_mode='probability')
print('TENSOR GENERATED')

expected_rank = [3,3,3,1]
#Apply tensor completion
predicted_tensor, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, expected_rank, convergence_tolerance=1e-15, iteration_limit=100000)

#Find best hyperparameter value
result = findBestValues(predicted_tensor, smallest=True, number_of_values=1)
values, indices = result['values'], result['indices']
hyperparameter_values = hyperparametersFromIndices(indices, ranges_dict)

#End timer/memory profiler/CPU timer
result = None
if quantity == 'EXEC-TIME':
    end_time = time.perf_counter_ns()
    result = end_time - start_time
elif quantity == 'CPU-TIME':
    end_time = time.process_time_ns()
    result = end_time - start_time
elif quantity == 'MAX-MEMORY':
    _, result = tracemalloc.get_traced_memory()
    tracemalloc.stop()

#Find the true loss for the selcted combination
true_value = func(metric=classificationmetrics.KullbackLeiblerDivergence, **hyperparameter_values[0])

print(f'hyperparameters: {hyperparameter_values}')
print(f'function values: {values}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')
