#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from sklearn import svm
from commonfunctions import generate_range
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from trainmodels import crossValidationFunctionGenerator
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)


metric=classificationmetrics.indicatorFunction
resolution = 0.2


def obtain_hyperparameters(trial):
    C = trial.suggest_float("C", 0.05, 5.05, step=0.05)
    gamma = trial.suggest_float("gamma", 0.05, 5.05, step=0.05)
    return C, gamma


def objective(trial):
    C, gamma = obtain_hyperparameters(trial)
    metric_value = None

    for fraction in generate_range(resolution,1,resolution):
        data_split = trainTestSplit(data, method='cross_validation')
        func = crossValidationFunctionGenerator(data_split, algorithm='svm-rbf', task=task, budget_type='samples', budget_fraction=fraction)
        metric_value = func(C, gamma, metric=metric)
        #Check for pruning
        trial.report(metric_value, fraction)
        if trial.should_prune():
            print('=======================================================================================================')
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
        min_resource=resolution, max_resource=1, reduction_factor=2
    ),
)
study.optimize(objective, n_trials=50)

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
