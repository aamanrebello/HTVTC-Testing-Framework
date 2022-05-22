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
BASE_PATH = 'saved-arrays/random-forest/'
FILE_NAME = 'wine-probability-KLD-5-2.npy'
PATH = BASE_PATH + FILE_NAME
load_tensor = True

#OBTAIN TENSOR========================================
tensor = None

if load_tensor:
    tensor = np.load(PATH)
else:
    task = 'classification'
    data = loadData(source='sklearn', identifier='wine', task=task)
    binary_data = extractZeroOneClasses(data)
    data_split = trainTestSplit(binary_data)
    func = evaluationFunctionGenerator(data_split, algorithm='random-forest', task=task)

    ranges_dict = {
        'no_trees': {
            'values':[1,10,20,30,40]
        },
        'max_tree_depth': {
            'values':[1, 5, 10, 15, 20]
        },
        'bootstrap': {
            'values': [True, False]
        },
        'min_samples_split': {
            'start': 2.0,
            'end': 10.0,
            'interval': 1.0,
        },
        'no_features': {
            'start': 1.0,
            'end': 10.0,
            'interval': 1.0,
        },
    }
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.KullbackLeiblerDivergence, evaluation_mode='probability')
    np.save(file=PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')
print(tensor)
