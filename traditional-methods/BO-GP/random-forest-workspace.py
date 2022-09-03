#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bayes_opt import BayesianOptimization
from trainmodels import evaluationFunctionGenerator, crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'
trials = 30
pval = 1

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='random-forest', task=task)

def objective(no_trees, max_tree_depth, bootstrap_ind, min_samples_split, no_features):
    no_trees  = int(no_trees)
    max_tree_depth = int(max_tree_depth)
    min_samples_split = int(min_samples_split)
    no_features = int(no_features)
    bootstrap = True
    if bootstrap_ind > 0:
        bootstrap = False
    #subtract from 1 because the library only supports maximise
    return pval - func(no_trees=no_trees, max_tree_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, no_features=no_features, metric=classificationmetrics.KullbackLeiblerDivergence)

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

#Begin optimisation
pbounds = {'no_trees': (1, 40), 'max_tree_depth': (1, 20), 'bootstrap_ind': (-1,1), 'min_samples_split': (2,10), 'no_features': (1,10)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
    verbose = 0
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
