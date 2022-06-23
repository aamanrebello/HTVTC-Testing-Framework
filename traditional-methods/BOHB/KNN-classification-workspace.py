#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn.neighbors import KNeighborsClassifier
from commonfunctions import generate_range, truncate_features
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data)
train_X = data_split['training_features']
train_y = data_split['training_labels']
validation_X = data_split['validation_features']
validation_y = data_split['validation_labels']

metric = classificationmetrics.indicatorFunction
MAXVAL = 13
MINVAL = 2

def objective(no_features, N, p, weightingFunction, distanceFunction):
    #Generate training and validation data with truncated samples
    trial_train_X = truncate_features(train_X, int(no_features))
    trial_validation_X = truncate_features(validation_X, int(no_features))
    #Train model with hyperparameters on data
    clf = KNeighborsClassifier(n_neighbors=int(N), weights=weightingFunction, p=p, metric=distanceFunction)
    #Make prediction
    clf.fit(trial_train_X, train_y)
    trial_validation_predictions = clf.predict(trial_validation_X)
    return metric(validation_y, trial_validation_predictions)
    
def evaluate(params, n_iterations):
    no_features = n_iterations
    return objective(**params, no_features=no_features)


if __name__ == '__main__':
    N = cs.IntegerUniformHyperparameter('N', 1, 100)
    p = cs.IntegerUniformHyperparameter('p', 1, 100)
    weightingFunction = cs.CategoricalHyperparameter('weightingFunction', ['uniform', 'distance'])
    distanceFunction = cs.CategoricalHyperparameter('distanceFunction', ['minkowski'])
    configspace = cs.ConfigurationSpace([N, p, weightingFunction, distanceFunction])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=MINVAL, eta=2)

    quantity = 'CPU-TIME'

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
