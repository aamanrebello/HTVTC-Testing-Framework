#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data)
func = evaluationFunctionGenerator(data_split, algorithm='random-forest', task=task)


def objective(trial):
    no_trees = trial.suggest_categorical("no_trees", [1,10,20,30,40])
    max_tree_depth = trial.suggest_categorical("max_tree_depth", [1, 5, 10, 15, 20])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 11, step=1)
    no_features = trial.suggest_int("no_features", 1, 11, step=1)
    
    return func(no_trees=no_trees, max_tree_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, no_features=no_features, metric=classificationmetrics.KullbackLeiblerDivergence)

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

min_samples_split_search_range = generate_range(2.0, 10.0, 1.0)
no_features_search_range = generate_range(1.0, 10.0, 1.0)
search_space = {"no_trees": [1,10,20,30,40], "max_tree_depth": [1, 5, 10, 15, 20], "bootstrap": [True, False], "min_samples_split": min_samples_split_search_range, "no_features": no_features_search_range}
optuna.logging.set_verbosity(optuna.logging.FATAL)
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, timeout=600)

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
print(f'Number of trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial}')
print(f'{quantity}: {result}')
#print(f'Resource usage: {resource_usage}')
