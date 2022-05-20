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
BASE_PATH = 'saved-arrays/svm-polynomial/'
FILE_NAME = 'iris-hinge-30-1.npy'
PATH = BASE_PATH + FILE_NAME
load_tensor = False

#OBTAIN TENSOR========================================
tensor = None

if load_tensor:
    tensor = np.load(PATH)
else:
    task = 'classification'
        data = loadData(source='sklearn', identifier='iris', task=task)
        binary_data = extractZeroOneClasses(data)
        adjusted_data = convertZeroOne(binary_data, -1, 1)
        data_split = trainTestSplit(adjusted_data)
        func = evaluationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)

        ranges_dict = {
            'C': {
                'start': 0.1,
                'end': 3.0,
                'interval': 0.1,
            },
            'gamma': {
                'start': 0.1,
                'end': 3.0,
                'interval': 0.1,
            },
            'constant_term': {
                'start': 0.0,
                'end': 3.0,
                'interval': 0.1,
            },
            'degree': {
                'start': 0.0,
                'end': 3.0,
                'interval': 1.0,
            }
        }
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.hingeLoss, evaluation_mode='raw-score')
        np.save(file=PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')
