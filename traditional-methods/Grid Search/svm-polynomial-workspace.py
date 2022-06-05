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
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data)
func = evaluationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)


def objective(trial):
    C = trial.suggest_float("C", 0.1, 3.1, step=0.1)
    gamma = trial.suggest_float("gamma", 0.1, 3.1, step=0.1)
    constant_term = trial.suggest_float("constant_term", 0.0, 3.1, step=0.1)
    degree = trial.suggest_float("degree", 0.0, 3.1, step=0.1)
    
    return func(C=C, gamma=gamma, constant_term=constant_term, degree=degree, metric=classificationmetrics.hingeLoss)

start_time = time.perf_counter()

C_search_range = generate_range(0.1, 3.0, 0.1)
gamma_search_range = generate_range(0.1, 3.0, 0.1)
constant_term_search_range = generate_range(0.0, 3.0, 0.1)
degree_search_range = generate_range(0.0, 3.0, 0.1)

search_space = {"C": C_search_range, "gamma": gamma_search_range, "constant_term": constant_term_search_range, "degree": degree_search_range}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
study.optimize(objective, timeout=600)
#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')

