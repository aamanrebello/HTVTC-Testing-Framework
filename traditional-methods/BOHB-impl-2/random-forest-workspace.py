#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn.ensemble import RandomForestClassifier
from commonfunctions import generate_range
from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)

metric = classificationmetrics.KullbackLeiblerDivergence
MAXVAL = 10

def objective(fraction, no_trees, max_tree_depth, bootstrap, min_samples_split, no_features):
    data_split = trainTestSplit(binary_data, method='cross_validation')
    func = crossValidationFunctionGenerator(data_split, algorithm='random-forest', task=task, budget_type='samples', budget_fraction=fraction)
    return func(no_trees=no_trees, max_tree_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, no_features=no_features, metric=metric)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    no_trees = cs.CategoricalHyperparameter("no_trees", [1,10,20,30,40])
    max_tree_depth = cs.CategoricalHyperparameter("max_tree_depth", [1, 5, 10, 15, 20])
    bootstrap = cs.CategoricalHyperparameter("bootstrap", [True, False])
    min_samples_split = cs.IntegerUniformHyperparameter("min_samples_split", 2, 10)
    no_features = cs.IntegerUniformHyperparameter("no_features", 1, 10)
    
    configspace = cs.ConfigurationSpace([no_trees, max_tree_depth, bootstrap, min_samples_split, no_features])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=2, eta=2)

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
