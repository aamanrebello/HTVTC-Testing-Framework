#IMPORTS==============================================
import unittest
import json
import numpy as np

#Enable importing code from parent directory
import os, sys
p1 = os.path.abspath('..')
sys.path.insert(1, p1)
p2 = os.path.abspath('../experiments')
sys.path.insert(1, p2)

from generateerrortensor import generateIncompleteErrorTensor
from trainmodels import evaluationFunctionGenerator, crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from tensorsearch import findBestValues
import regressionmetrics
import classificationmetrics


#OVERALL CONFIGURATION================================
BASE_PATH = 'saved-cross-validation-arrays/KNN-regression/'
FILE_NAME = 'calh-mae-100-1'
ARR_EXTN = '.npy'
ARR_PATH = BASE_PATH + FILE_NAME + ARR_EXTN
RANGE_DICT_EXTN = '.json'
RANGE_DICT_PATH = BASE_PATH + FILE_NAME + '-ranges' + RANGE_DICT_EXTN
load_tensor = False

#GENERATE COMPLETE TENSOR=======================
tensor = None
ranges_dict = None

task = 'regression'
data = loadData(source='sklearn', identifier='california_housing', task=task)
data_split = trainTestSplit(data, method = 'cross_validation')
metric = regressionmetrics.mae

if load_tensor:
    tensor = np.load(ARR_PATH)
    with open(RANGE_DICT_PATH, 'r') as fp:
        ranges_dict = json.load(fp)
else:
    nobudgetfunc = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task)

    ranges_dict = {
        'N': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 100.0,
            'interval': 1.0,
        },
        'weightingFunction': {
            'type': 'CATEGORICAL',
            'values': ['uniform'],
        },
        'distanceFunction': {
            'type': 'CATEGORICAL',
            'values': ['minkowski']
        },
        'p': {
            'type': 'INTEGER',
            'start': 1.0,
            'end': 100.0,
            'interval': 1.0,
        }
    }

    with open(RANGE_DICT_PATH, 'w') as fp:
        json.dump(ranges_dict , fp)

    tensor, _ = generateIncompleteErrorTensor(nobudgetfunc, ranges_dict, 1.0, metric=metric, eval_trials=1)
    tensor = np.squeeze(tensor)
    np.save(file=ARR_PATH, arr=tensor)

bestValue = findBestValues(tensor, smallest=True, number_of_values=1)
print(bestValue)
