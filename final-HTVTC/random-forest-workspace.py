#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from finalAlgoImplementation import final_HTVTC
import regressionmetrics
import classificationmetrics

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='random-forest', task=task)
metric = classificationmetrics.KullbackLeiblerDivergence

#Start timer/memory profiler/CPU timer
a = None
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

ranges_dict = {
        'no_trees': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 40.0,
            'interval': 5.0,
        },
        'max_tree_depth': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 20.0,
            'interval': 5.0,
        },
        'bootstrap': {
            'type': 'CATEGORICAL',
            'values': [True, False]
        },
        'min_samples_split': {
            'type': 'INTEGER',
            'start': 2.0,
            'end': 11.0,
            'interval': 2.0,
        },
        'no_features': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 11.0,
            'interval': 2.0,
        },
    }

recommended_combination, history = final_HTVTC(eval_func=func, ranges_dict=ranges_dict, metric=metric, max_completion_cycles=4, max_size_gridsearch=51)

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
data_split = trainTestSplit(binary_data, method = 'cross_validation')
#Find the true loss for the selected combination
truefunc = crossValidationFunctionGenerator(data_split, algorithm='random-forest', task=task)    
true_value = truefunc(metric=metric, **recommended_combination)

print(f'hyperparameters: {recommended_combination}')
print(f'history: {history}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')
