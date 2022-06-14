#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bayes_opt import BayesianOptimization
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'
trials = 50
pval = 1

task = 'classification'
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data)
func = evaluationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)


def objective(C, gamma, constant_term, degree):
    #subtract from 1 because the library only supports maximise
    return pval - func(C=C, gamma=gamma, constant_term=constant_term, degree=degree, metric=classificationmetrics.hingeLoss)


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

pbounds = {'C': (0.1, 3.0), 'gamma': (0.1, 3.0), 'constant_term': (0.0, 3.0), 'degree': (0.0, 3.0)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
    verbose=0
)

optimizer.maximize(
    init_points=10,
    n_iter=trials,
)

#resource_usage = getrusage(RUSAGE_SELF)
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
    
print('\n\n\n')
best = optimizer.max
best_params = best['params']
best_score = pval - best['target']
print(f'Number of trials: {trials}')
print(f'Best params: {best_params}')
print(f'Best score: {best_score}')
print(f'{quantity}: {result}')
#print(f'Resource usage: {resource_usage}')
