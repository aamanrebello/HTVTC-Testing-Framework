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
from commonfunctions import Hamming_distance, norm_difference, sortedBestValues, common_count
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_TKD_Geng_Miles, tensorcomplete_TMac_TT
from tensorcompletion import ket_augmentation, inverse_ket_augmentation
import regressionmetrics
import classificationmetrics


#OVERALL CONFIGURATION================================
BASE_PATH = 'saved-cross-validation-arrays/KNN-classification/'
FILE_NAME = 'wine-indicator-100-1'
ARR_EXTN = '.npy'
ARR_PATH = BASE_PATH + FILE_NAME + ARR_EXTN
RANGE_DICT_EXTN = '.json'
RANGE_DICT_PATH = BASE_PATH + FILE_NAME + '-ranges' + RANGE_DICT_EXTN
load_tensor = True

#GENERATE COMPLETE TENSOR=======================
tensor = None
ranges_dict = None

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')

if load_tensor:
    tensor = np.load(ARR_PATH)
    with open(RANGE_DICT_PATH, 'r') as fp:
        ranges_dict = json.load(fp)
else:
    nobudgetfunc = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task)
    
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

    tensor, _ = generateIncompleteErrorTensor(nobudgetfunc, ranges_dict, 1.0, metric=classificationmetrics.indicatorFunction, eval_trials=1)
    np.save(file=ARR_PATH, arr=tensor)

tensor = np.squeeze(tensor)
print(f'STAGE 1 - COMPLETE TENSOR GENERATED')

#GENERATE INCOMPLETE TENSOR===========================
budget_type = 'samples'
budget_fraction = 0.95
budgetfunc = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task, budget_type=budget_type, budget_fraction=budget_fraction)

known_fraction = 0.25
incomplete_tensor, known_indices = generateIncompleteErrorTensor(budgetfunc, ranges_dict, known_fraction, metric=classificationmetrics.indicatorFunction, eval_trials=1)
incomplete_tensor = np.squeeze(incomplete_tensor)
removethird = lambda a: (a[0],a[1],a[3])
known_indices = list(map(removethird, known_indices))

print(f'STAGE 2 - INCOMPLETE TENSOR GENERATED')

#OBTAIN BEST HYPERPARAMETER COMBINATIONS=============
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
print(f'STAGE 3 - TRUE BEST COMBINATIONS IDENTIFIED')

#TEST TENSOR COMPLETION================================
tensor_norm = np.linalg.norm(tensor)
ratio_threshold = 5

TT_rank = [3,1]
Tucker_rank = [2,2,1]

class TestTensorCompletion_TMAC_TT(unittest.TestCase):

    def test_TMac_TT_top10pc(self):
        #Apply tensor completion
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, TT_rank, convergence_tolerance=1e-15, iteration_limit=100000)
        #Check norm difference from true tensor
        diff = norm_difference(TMAC_TT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'TMAC-TT (10%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 10% according to predicted tensor
        sorted_predicted_dict_10pc = sortedBestValues(TMAC_TT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_10pc)
        true_indices = sorted_dict_10pc['indices']
        predicted_indices = sorted_predicted_dict_10pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'TMAC-TT (10%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_10pc['values'])
        predicted_values = np.array(sorted_predicted_dict_10pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_TMac_TT_top5pc(self):
        #Apply tensor completion
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, TT_rank, convergence_tolerance=1e-15, iteration_limit=100000)
        #Check norm difference from true tensor
        diff = norm_difference(TMAC_TT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'TMAC-TT (5%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_5pc = sortedBestValues(TMAC_TT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_5pc)
        true_indices = sorted_dict_5pc['indices']
        predicted_indices = sorted_predicted_dict_5pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'TMAC-TT (5%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_5pc['values'])
        predicted_values = np.array(sorted_predicted_dict_5pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_TMac_TT_top1pc(self):
        #Apply tensor completion
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, TT_rank, convergence_tolerance=1e-15, iteration_limit=100000)
        #Check norm difference from true tensor
        diff = norm_difference(TMAC_TT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'TMAC-TT (1%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_1pc = sortedBestValues(TMAC_TT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_1pc)
        true_indices = sorted_dict_1pc['indices']
        predicted_indices = sorted_predicted_dict_1pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'TMAC-TT (1%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_1pc['values'])
        predicted_values = np.array(sorted_predicted_dict_1pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_TMac_TT_top20(self):
        #Apply tensor completion
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, TT_rank, convergence_tolerance=1e-15, iteration_limit=100000)
        #Check norm difference from true tensor
        diff = norm_difference(TMAC_TT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'TMAC-TT (top 20) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 20 according to predicted tensor
        sorted_predicted_dict_top20 = sortedBestValues(TMAC_TT_PREDICTED_TENSOR, smallest=smallest, number_of_values=20)
        true_indices = sorted_dict_top20['indices']
        predicted_indices = sorted_predicted_dict_top20['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'TMAC-TT (top 20) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_top20['values'])
        predicted_values = np.array(sorted_predicted_dict_top20['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    @classmethod
    def tearDownClass(TestTensorCompletion):
        print()
        print('-------------------------------')
        print()


@unittest.skip('not needed')
class TestTensorCompletion_Geng_Miles(unittest.TestCase):

    def test_Geng_Miles_top10pc(self):
        #Apply tensor completion
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, Tucker_rank, hooi_tolerance=1e-3, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(GENG_MILES_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'Geng-Miles (10%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 10% according to predicted tensor
        sorted_predicted_dict_10pc = sortedBestValues(GENG_MILES_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_10pc)
        true_indices = sorted_dict_10pc['indices']
        predicted_indices = sorted_predicted_dict_10pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'Geng-Miles (10%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_10pc['values'])
        predicted_values = np.array(sorted_predicted_dict_10pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_Geng_Miles_top5pc(self):
        #Apply tensor completion
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, Tucker_rank, hooi_tolerance=1e-3, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(GENG_MILES_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'Geng-Miles (5%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_5pc = sortedBestValues(GENG_MILES_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_5pc)
        true_indices = sorted_dict_5pc['indices']
        predicted_indices = sorted_predicted_dict_5pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'Geng-Miles (5%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_5pc['values'])
        predicted_values = np.array(sorted_predicted_dict_5pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_Geng_Miles_top1pc(self):
        #Apply tensor completion
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, Tucker_rank, hooi_tolerance=1e-3, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(GENG_MILES_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'Geng-Miles (1%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_1pc = sortedBestValues(GENG_MILES_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_1pc)
        true_indices = sorted_dict_1pc['indices']
        predicted_indices = sorted_predicted_dict_1pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'Geng-Miles (1%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_1pc['values'])
        predicted_values = np.array(sorted_predicted_dict_1pc['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    def test_Geng_Miles_top20(self):
        #Apply tensor completion
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, Tucker_rank, hooi_tolerance=1e-3, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(GENG_MILES_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'Geng-Miles (top 20) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 20 according to predicted tensor
        sorted_predicted_dict_top20 = sortedBestValues(GENG_MILES_PREDICTED_TENSOR, smallest=smallest, number_of_values=20)
        true_indices = sorted_dict_top20['indices']
        predicted_indices = sorted_predicted_dict_top20['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'Geng-Miles (top 20) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
        true_values = np.array(sorted_dict_top20['values'])
        predicted_values = np.array(sorted_predicted_dict_top20['values'])
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in hyperparameter values: {norm_error}')
        print(ratio)
        print(hamming_distance)
        print(aug_hamming_distance)
        print(common)
        print(norm_error)
        completed = True
        self.assertTrue(completed)

    @classmethod
    def tearDownClass(TestTensorCompletion):
        print()
        print('-------------------------------')
        print()

if __name__ == '__main__':
    unittest.main()
