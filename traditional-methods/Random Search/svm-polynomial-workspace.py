#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from optuna.samplers import RandomSampler
import json
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator, crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)

timestamps = []
validation_losses = []
best_loss = None

def objective(trial):
    C = trial.suggest_float("C", 0.1, 3.1)
    gamma = trial.suggest_float("gamma", 0.1, 3.1)
    constant_term = trial.suggest_float("constant_term", 0.0, 3.1)
    degree = trial.suggest_float("degree", 0.0, 3.1)

    loss = func(C=C, gamma=gamma, constant_term=constant_term, degree=degree, metric=classificationmetrics.hingeLoss)
    timestamp = time.perf_counter_ns()
    #Update best loss based on received loss value
    global best_loss
    if best_loss is None:
        best_loss = loss
    else:
        best_loss = min(best_loss, loss)
    #Store timestamp and loss
    timestamps.append(timestamp)
    validation_losses.append(best_loss)
    return loss

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
study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=100000)

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


#Process time stamps
for i in range(len(timestamps)):
    timestamps[i] -= start_time

graph_stats = {
    'time': timestamps,
    'loss': validation_losses
}
PATH = 'graphs/SVM-polynomial.json'
with open(PATH, 'w') as fp:
        json.dump(graph_stats , fp)