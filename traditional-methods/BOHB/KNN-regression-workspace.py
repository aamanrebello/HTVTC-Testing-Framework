#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn.neighbors import KNeighborsRegressor
from commonfunctions import generate_range, truncate_features
from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'regression'
data = loadData(source='sklearn', identifier='california_housing', task=task)

metric = regressionmetrics.mae
MAXVAL = 10
MINVAL = 4

def objective(no_features, N, p, weightingFunction, distanceFunction):
    data_split = trainTestSplit(data, method='cross_validation')
    fraction = no_features/MAXVAL + 1e-3
    func = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task, budget_type='features', budget_fraction=fraction)
    return func(N=N, weightingFunction=weightingFunction, distanceFunction=distanceFunction, p=p, metric=metric)
    
    
def evaluate(params, n_iterations):
    no_features = n_iterations
    return objective(**params, no_features=no_features)


if __name__ == '__main__':
    N = cs.IntegerUniformHyperparameter('N', 1, 100)
    p = cs.IntegerUniformHyperparameter('p', 1, 100)
    weightingFunction = cs.CategoricalHyperparameter('weightingFunction', ['uniform'])
    distanceFunction = cs.CategoricalHyperparameter('distanceFunction', ['minkowski'])
    configspace = cs.ConfigurationSpace([N, p, weightingFunction, distanceFunction])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=2.5, eta=2)

    quantity = 'EXEC-TIME'

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
