#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn import svm
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'classification'
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data)
train_X = data_split['training_features']
train_y = data_split['training_labels']
validation_X = data_split['validation_features']
validation_y = data_split['validation_labels']

metric = classificationmetrics.hingeLoss
MAXVAL = 10

def objective(fraction, C, gamma, constant_term, degree):
    training_size = len(train_X)
    #Generate fraction of training data
    trial_size = int(fraction*training_size)
    trial_train_X = train_X[:trial_size]
    trial_train_y = train_y[:trial_size]
    #Train SVM with hyperparameters on data
    clf = svm.SVC(C = C, kernel = 'poly', gamma = gamma, coef0 = constant_term, degree = degree)
    #Make prediction
    clf.fit(trial_train_X, trial_train_y)
    trial_validation_predictions = clf.decision_function(validation_X)
    return metric(validation_y, trial_validation_predictions)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    C = cs.UniformHyperparameter('C', 0.1, 3.0)
    gamma = cs.UniformHyperparameter('gamma', 0.1, 3.0)
    constant_term = cs.UniformHyperparameter('constant_term', 0.0, 3.0)
    degree = cs.UniformHyperparameter('degree', 0.0, 3.0)
    configspace = cs.ConfigurationSpace([C, gamma, constant_term, degree])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=1, eta=2)

    quantity = 'MAX-MEMORY'

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

    logs = opt.optimize()

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

    print(logs)
    print(f'{quantity}: {result}')
