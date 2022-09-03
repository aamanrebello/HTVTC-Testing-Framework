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
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)
metric = classificationmetrics.hingeLoss

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
        'C': {
            'type': 'REAL',
            'start': 0.1,
            'end': 3.0,
            'interval': 0.1,
        },
        'gamma': {
            'type': 'REAL',
            'start': 0.1,
            'end': 3.0,
            'interval': 0.1,
        },
        'constant_term': {
            'type': 'REAL',
            'start': 0.0,
            'end': 3.0,
            'interval': 0.1,
        },
        'degree': {
            'type': 'INTEGER',
            'start': 0.0,
            'end': 3.0,
            'interval': 1.0,
        }
    }

recommended_combination, history = final_HTVTC(eval_func=func, ranges_dict=ranges_dict, metric=metric)

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
true_value = truefunc(metric=metric, **recommended_combination)

print(f'hyperparameters: {recommended_combination}')
print(f'history: {history}')
print(f'True value: {true_value}')
print(f'{quantity}: {result}')
