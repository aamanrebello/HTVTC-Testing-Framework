#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from sklearn.neighbors import KNeighborsRegressor
from commonfunctions import generate_range, truncate_features
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from trainmodels import crossValidationFunctionGenerator
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'

task = 'regression'
data = loadData(source='sklearn', identifier='diabetes', task=task)
metric = regressionmetrics.logcosh
MAX_FEATURES = 10
MIN_FEATURES = 4

resolution = 0.2

def obtain_hyperparameters(trial):
    N = trial.suggest_int("N", 1, 101, step=1)
    p = trial.suggest_int("p", 1, 101, step=1)
    weightingFunction = trial.suggest_categorical("weightingFunction", ['uniform'])
    distanceFunction = trial.suggest_categorical("distanceFunction", ['minkowski'])
    return N, p, weightingFunction, distanceFunction


def objective(trial):
    N, p, weightingFunction, distanceFunction = obtain_hyperparameters(trial)
    metric_value = None

    for no_features in generate_range(MIN_FEATURES,MAX_FEATURES,1):
        data_split = trainTestSplit(data, method='cross_validation')
        fraction = no_features/MAX_FEATURES + 1e-3
        func = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task, budget_type='features', budget_fraction=fraction)
        metric_value = func(N=N, weightingFunction=weightingFunction, distanceFunction=distanceFunction, p=p, metric=metric)
        #Check for pruning
        trial.report(metric_value, fraction)
        if trial.should_prune():
            #print('=======================================================================================================')
            raise optuna.TrialPruned()

    #Would return the metric for fully trained model (on full dataset)
    return metric_value
    

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

optuna.logging.set_verbosity(optuna.logging.FATAL)
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=MIN_FEATURES, max_resource=MAX_FEATURES, reduction_factor=2
    ),
)
study.optimize(objective, n_trials=40)

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
