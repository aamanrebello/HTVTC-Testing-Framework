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

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data)
func = evaluationFunctionGenerator(data_split, algorithm='knn-classification', task=task)


def objective(trial):
    N = trial.suggest_int("N", 1, 101, step=1)
    p = trial.suggest_int("p", 1, 101, step=1)
    weightingFunction = trial.suggest_categorical("weightingFunction", ['uniform', 'distance'])
    distanceFunction = trial.suggest_categorical("distanceFunction", ['minkowski'])
    
    return func(N=N, weightingFunction=weightingFunction, distanceFunction=distanceFunction, p=p, metric=classificationmetrics.indicatorFunction)

start_time = time.perf_counter()

N_search_range = generate_range(1, 100, 1)
p_search_range = generate_range(1, 100, 1)
search_space = {"N": N_search_range, "weightingFunction": ['uniform', 'distance'], "distanceFunction": ['minkowski'], "p": p_search_range}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, timeout=600)
#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')

