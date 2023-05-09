#IMPORTS==============================================
import unittest
import json
import numpy as np

#Enable importing code from parent directory
import os, sys
p = os.path.abspath('..')
sys.path.insert(1, p)

from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses
from commonfunctions import Hamming_distance, norm_difference, sortedBestValues, common_count
from cross_sampling import cross_sample_tensor
from tensorsearch import sortHyperparameterValues, findBestValues, hyperparametersFromIndices
from crosstechnique import noisyReconstruction, noiselessReconstruction
import classificationmetrics
import time


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

    tensor, _ = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.indicatorFunction, eval_trials=5)
    np.save(file=ARR_PATH, arr=tensor)

tensor = np.squeeze(tensor)
print(f'STAGE 1 - TENSOR GENERATED - shape: {tensor.shape}')

#OBTAIN BEST ELEMENTS IN THE TENSORS=============
smallest = True
#Obtain the best 10% in sorted order
no_elements_10pc = int(0.1*(tensor.size))
sorted_dict_10pc = sortedBestValues(tensor, smallest=smallest, number_of_values=no_elements_10pc)
#Obtain the best 5% in sorted order
no_elements_5pc = int(0.05*(tensor.size))
sorted_dict_5pc = sortedBestValues(tensor, smallest=smallest, number_of_values=no_elements_5pc)
#The best 1%
no_elements_1pc = int(0.01*(tensor.size))
sorted_dict_1pc = sortedBestValues(tensor, smallest=smallest, number_of_values=no_elements_1pc)
#The top 20
sorted_dict_top20 = sortedBestValues(tensor, smallest=smallest, number_of_values=20)


#TEST TENSOR COMPLETION================================
tensor_norm = np.linalg.norm(tensor)

class TestTensorCompletion_Cross(unittest.TestCase):

    def test_default_rank(self):
        #Begin time measurement
        start_time = time.process_time_ns()
        #Cross sample tensor
        body, joints, arms, no_elements = cross_sample_tensor(tensor, tucker_rank_list=[2,2,1])
        print(f'sampled elements: {no_elements}')
        print(f'sampled elements ratio: {no_elements/tensor.size}')
        #Apply tensor completion
        completed_tensor = noisyReconstruction(body, joints, arms)
        #Finish time measurement
        end_time = time.process_time_ns()
        proc_time = end_time - start_time
        print(f'Processing time: {proc_time}')
        
        #Check norm difference from true tensor
        diff = norm_difference(completed_tensor, tensor)
        #Find ratio to tensor norm
        norm_difference_ratio = diff/tensor_norm
        print(f'norm difference ratio: {norm_difference_ratio}')
        
        #Compare top 10% according in predicted and true tensor
        sorted_predicted_dict_10pc = sortedBestValues(completed_tensor, smallest=smallest, number_of_values=no_elements_10pc)
        true_indices = sorted_dict_10pc['indices']
        predicted_indices = sorted_predicted_dict_10pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_10pc['values'])
        predicted_values = np.array(sorted_predicted_dict_10pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        value_disagreement = 0
        for i in range(no_elements_10pc):
            if true_values[i] != predicted_values[i]:
                value_disagreement += 1
        print(f'Error in top 10% {norm_error}, value disagreement: {value_disagreement/no_elements_10pc}')
    
        completed = True
        self.assertTrue(completed)

if __name__ == '__main__':
    unittest.main()
