#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from sklearn.ensemble import RandomForestClassifier
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
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)

metric = classificationmetrics.KullbackLeiblerDivergence
resolution = 0.2


def obtain_hyperparameters(trial):
    no_trees = trial.suggest_categorical("no_trees", [1,10,20,30,40])
    max_tree_depth = trial.suggest_categorical("max_tree_depth", [1, 5, 10, 15, 20])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 11, step=1)
    no_features = trial.suggest_int("no_features", 1, 11, step=1)
    return no_trees, max_tree_depth, bootstrap, min_samples_split, no_features


def objective(trial):
    no_trees, max_tree_depth, bootstrap, min_samples_split, no_features = obtain_hyperparameters(trial)
    metric_value = None

    for fraction in generate_range(resolution,1,resolution):
        data_split = trainTestSplit(binary_data, method='cross_validation')
        func = crossValidationFunctionGenerator(data_split, algorithm='random-forest', task=task, budget_type='samples', budget_fraction=fraction)
        metric_value = func(no_trees=no_trees, max_tree_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, no_features=no_features, metric=metric)
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
        min_resource=resolution, max_resource=1, reduction_factor=2
    ),
)
study.optimize(objective, n_trials=10)

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
