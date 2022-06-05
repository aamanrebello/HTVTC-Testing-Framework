#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from optuna.samplers import TPESampler
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)
func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)


def objective(trial):
    C = trial.suggest_float("C", 0.05, 5.05, step=0.05)
    gamma = trial.suggest_float("gamma", 0.05, 5.05, step=0.05)
    
    return func(C, gamma, metric=classificationmetrics.indicatorFunction)

start_time = time.perf_counter()

study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, n_trials=10)
#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')
