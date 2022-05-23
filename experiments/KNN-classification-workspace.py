#IMPORTS==============================================
import unittest
import json
import numpy as np

#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from generateerrortensor import generateIncompleteErrorTensor
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from tensorsearch import higherDimensionalIndex, findBestValues, hyperparametersFromIndices
import regressionmetrics
import classificationmetrics


#OVERALL CONFIGURATION================================
BASE_PATH = 'saved-arrays/KNN-classification/'
FILE_NAME = 'wine-indicator-100-1'
ARR_EXTN = '.npy'
ARR_PATH = BASE_PATH + FILE_NAME + ARR_EXTN
RANGE_DICT_EXTN = '.json'
RANGE_DICT_PATH = BASE_PATH + FILE_NAME + '-ranges' + RANGE_DICT_EXTN
load_tensor = True

#OBTAIN TENSOR========================================
tensor = None
ranges_dict = None

if load_tensor:
    tensor = np.load(ARR_PATH)
    with open(RANGE_DICT_PATH, 'r') as fp:
        ranges_dict = json.load(fp)
else:
    task = 'classification'
    data = loadData(source='sklearn', identifier='wine', task=task)
    binary_data = extractZeroOneClasses(data)
    data_split = trainTestSplit(binary_data)
    func = evaluationFunctionGenerator(data_split, algorithm='knn-classification', task=task)

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
    with open(RANGE_DICT_PATH, 'w') as fp:
        json.dump(ranges_dict , fp)
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.indicatorFunction, eval_trials=5)
    np.save(file=ARR_PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')

#OBTAIN BEST HYPERPARAMETER COMBINATIONS=============
BEST_FRACTION = 0.001
number_elements = int(BEST_FRACTION*(tensor.size))
result_dict = findBestValues(tensor, smallest=True, number_of_values=number_elements)
best_combinations = hyperparametersFromIndices(result_dict['indices'], ranges_dict)
print(f'STAGE 2 - TRUE BEST COMBINATIONS IDENTIFIED - {best_combinations}')
print(result_dict['values'])
