#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bayes_opt import BayesianOptimization
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
func = evaluationFunctionGenerator(data_split, algorithm='random-forest', task=task)

def objective(no_trees, max_tree_depth, bootstrap_ind, min_samples_split, no_features):
    no_trees  = int(no_trees)
    max_tree_depth = int(max_tree_depth)
    min_samples_split = int(min_samples_split)
    no_features = int(no_features)
    bootstrap = True
    if bootstrap_ind > 0:
        bootstrap = False
    #subtract from 1 because the library only supports maximise
    return 1 - func(no_trees=no_trees, max_tree_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, no_features=no_features, metric=classificationmetrics.KullbackLeiblerDivergence)

#Begin measuring time
start_time = time.perf_counter()

#Begin optimisation
pbounds = {'no_trees': (1, 40), 'max_tree_depth': (1, 20), 'bootstrap_ind': (-1,1), 'min_samples_split': (2,10), 'no_features': (1,10)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'best combination: {optimizer.max}')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')
