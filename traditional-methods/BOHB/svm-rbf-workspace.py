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
from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)

metric = classificationmetrics.indicatorFunction
MAXVAL = 10

def objective(fraction, C, gamma):
    data_split = trainTestSplit(data, method='cross_validation')
    func = crossValidationFunctionGenerator(data_split, algorithm='svm-rbf', task=task, budget_type='samples', budget_fraction=fraction)
    return func(C, gamma, metric=metric)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    C = cs.UniformHyperparameter('C', 0.05, 5.0)
    gamma = cs.UniformHyperparameter('gamma', 0.05, 5.0)
    configspace = cs.ConfigurationSpace([C, gamma])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=1, eta=2)

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
