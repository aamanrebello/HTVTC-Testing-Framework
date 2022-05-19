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
FILE_NAME = 'breast-cancer-indicator-100-1.npy'
PATH = BASE_PATH + FILE_NAME
load_tensor = False

#OBTAIN TENSOR========================================
tensor = None

if load_tensor:
    tensor = np.load(PATH)
else:
    task = 'classification'
    data = loadData(source='sklearn', identifier='breast_cancer', task=task)
    data_split = trainTestSplit(data)
    func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)

    ranges_dict = {
        'C': {
            'start': 0.05,
            'end': 5.00,
            'interval': 0.05,
            },
        'gamma': {
            'start': 0.05,
            'end': 5.00,
            'interval': 0.05,
            }
        }
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.indicatorFunction)
    np.save(file=PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')
