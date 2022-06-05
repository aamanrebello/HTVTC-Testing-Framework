import optuna
from optuna.samplers import RandomSampler
#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from optuna.samplers import RandomSampler
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

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

start_time = time.perf_counter()

study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=10)
#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'Best trial: {study.best_trial}')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')
