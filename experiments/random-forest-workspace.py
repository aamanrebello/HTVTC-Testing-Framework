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
BASE_PATH = 'saved-arrays/random-forest/'
FILE_NAME = 'wine-probability-KLD-5-1'
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
    with open(RANGE_DICT_PATH, 'w') as fp:
        json.dump(ranges_dict , fp)
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.KullbackLeiblerDivergence, evaluation_mode='probability')
    np.save(file=ARR_PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')

#OBTAIN BEST HYPERPARAMETER COMBINATIONS=============
BEST_FRACTION = 0.01
number_elements = int(BEST_FRACTION*(tensor.size))
result_dict = findBestValues(tensor, smallest=True, number_of_values=number_elements)
best_combinations = hyperparametersFromIndices(result_dict['indices'], ranges_dict)
print(f'STAGE 2 - TRUE BEST COMBINATIONS IDENTIFIED - {best_combinations}')
print(result_dict['values'])
