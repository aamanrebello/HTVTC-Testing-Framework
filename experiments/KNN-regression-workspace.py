#IMPORTS==============================================
import unittest
import numpy as np

#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from generateerrortensor import generateIncompleteErrorTensor
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics


#OVERALL CONFIGURATION================================
BASE_PATH = 'saved-arrays/KNN-regression/'
FILE_NAME = 'diabetes-logcosh-100-2.npy'
PATH = BASE_PATH + FILE_NAME
load_tensor = False

#OBTAIN TENSOR========================================
tensor = None

if load_tensor:
    tensor = np.load(PATH)
else:
    task = 'regression'
    data = loadData(source='sklearn', identifier='diabetes', task=task)
    data_split = trainTestSplit(data)
    func = evaluationFunctionGenerator(data_split, algorithm='knn-regression', task=task)

    ranges_dict = {
        'N': {
            'start': 1.0,
            'end': 100.0,
            'interval': 1.0,
        },
        'weightingFunction': {
            'values': ['uniform', 'distance'],
        },
        'distanceFunction': {
            'values': ['minkowski']
        },
        'p': {
            'start': 1.0,
            'end': 100.0,
            'interval': 1.0,
        }
    }
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=regressionmetrics.logcosh, eval_trials=5)
    np.save(file=PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')
