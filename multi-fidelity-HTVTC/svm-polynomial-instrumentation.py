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
from trainmodels import evaluationFunctionGenerator, crossValidationFunctionGenerator
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
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data, method = 'cross_validation')

budget_type = 'features'
budget_fraction = 0.5
func = crossValidationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task, budget_type=budget_type, budget_fraction=budget_fraction)

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
        'C': {
            'start': 0.1,
            'end': 3.0,
            'interval': 0.1,
        },
        'gamma': {
            'start': 0.1,
            'end': 3.0,
            'interval': 0.1,
        },
        'constant_term': {
            'start': 0.0,
            'end': 3.0,
            'interval': 0.1,
        },
        'degree': {
            'start': 0.0,
            'end': 3.0,
            'interval': 1.0,
        }
    }

#Generate incomplete tensor
incomplete_tensor, known_indices = generateIncompleteErrorTensor(func, ranges_dict, known_fraction, metric=classificationmetrics.hingeLoss, evaluation_mode='raw-score', eval_trials=1)
print('TENSOR GENERATED')

expected_rank = [2,2,2]
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

#Recreate cross-validation generator
data_split = trainTestSplit(adjusted_data, method = 'cross_validation')
#Find the true loss for the selcted combination
truefunc = crossValidationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)
true_value = truefunc(metric=classificationmetrics.hingeLoss, **hyperparameter_values[0])

print(f'hyperparameters: {hyperparameter_values}')
print(f'function values: {values}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')
