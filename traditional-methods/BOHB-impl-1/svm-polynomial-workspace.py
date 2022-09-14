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
data = loadData(source='sklearn', identifier='iris', task=task)
binary_data = extractZeroOneClasses(data)
adjusted_data = convertZeroOne(binary_data, -1, 1)
data_split = trainTestSplit(adjusted_data)

metric = classificationmetrics.hingeLoss
MAXVAL = 10

def objective(fraction, C, gamma, constant_term, degree):
    data_split = trainTestSplit(adjusted_data, method='cross_validation')
    func = crossValidationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task, budget_type='samples', budget_fraction=fraction)
    return func(C, gamma, constant_term, degree, metric=metric)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    C = cs.UniformHyperparameter('C', 0.1, 3.0)
    gamma = cs.UniformHyperparameter('gamma', 0.1, 3.0)
    constant_term = cs.UniformHyperparameter('constant_term', 0.0, 3.0)
    degree = cs.UniformHyperparameter('degree', 0.0, 3.0)
    configspace = cs.ConfigurationSpace([C, gamma, constant_term, degree])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=6, eta=2)

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
