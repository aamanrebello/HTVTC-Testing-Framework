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
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)
func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)


def objective(trial):
    C = trial.suggest_float("C", 0.05, 5.05, step=0.05)
    gamma = trial.suggest_float("gamma", 0.05, 5.05, step=0.05)
    
    return func(C, gamma, metric=classificationmetrics.indicatorFunction)


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

C_search_range = generate_range(0.05, 5.0, 0.05)
gamma_search_range = generate_range(0.05, 5.0, 0.05)
search_space = {"C": C_search_range, "gamma": gamma_search_range}
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

