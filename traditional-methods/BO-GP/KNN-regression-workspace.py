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

task = 'regression'
data = loadData(source='sklearn', identifier='diabetes', task=task)
data_split = trainTestSplit(data)
func = evaluationFunctionGenerator(data_split, algorithm='knn-regression', task=task)


def objective(N, p, wf):
    weightingFunction = 'uniform'
    if wf > 0:
        weightingFunction = 'distance'
    distanceFunction = 'minkowski'
    #subtract from 1 because the library only supports maximise
    return 50 - func(N=N, weightingFunction=weightingFunction, distanceFunction=distanceFunction, p=p, metric=regressionmetrics.logcosh)

start_time = time.perf_counter()

pbounds = {'N': (1, 100), 'p': (1, 100), 'wf': (-1,1)}

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
