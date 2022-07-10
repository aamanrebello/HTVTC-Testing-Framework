#IMPORTS==============================================
import unittest
import json
import numpy as np

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

quantity = 'MAX-MEMORY'

known_fraction = 0.25

#Load dataset
task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)

budget_type = 'samples'
budget_fraction = 1.0
func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task, budget_type=budget_type, budget_fraction=budget_fraction)

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
        'start': 0.05,
        'end': 5.00,
        'interval': 0.05,
        },
    'gamma': {
        'start': 0.05,
        'end': 5.00,
        'interval': 0.05,
        }
    }

#Generate incomplete tensor
incomplete_tensor, known_indices = generateIncompleteErrorTensor(func, ranges_dict, known_fraction, metric=classificationmetrics.indicatorFunction)
print('TENSOR GENERATED')

expected_rank = [1]
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
truefunc = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)
true_value = truefunc(metric=classificationmetrics.indicatorFunction, **hyperparameter_values[0])

print(f'hyperparameters: {hyperparameter_values}')
print(f'function values: {values}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')


