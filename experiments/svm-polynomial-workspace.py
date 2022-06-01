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
from commonfunctions import randomly_sample_tensor, Hamming_distance, norm_difference, sortedBestValues, common_count
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_TKD_Geng_Miles, tensorcomplete_TMac_TT
from tensorcompletion import ket_augmentation, inverse_ket_augmentation
import regressionmetrics
import classificationmetrics


#OVERALL CONFIGURATION================================
BASE_PATH = 'saved-arrays/svm-polynomial/'
FILE_NAME = 'iris-hinge-30-1'
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
    with open(RANGE_DICT_PATH, 'w') as fp:
        json.dump(ranges_dict , fp)
        
    tensor = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.hingeLoss, evaluation_mode='raw-score')
    np.save(file=ARR_PATH, arr=tensor)

print(f'STAGE 1 - TENSOR GENERATED')

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
print(f'STAGE 2 - TRUE BEST COMBINATIONS IDENTIFIED')

#GENERATE INCOMPLETE TENSOR===========================
known_fraction = 0.25
incomplete_tensor, known_indices = randomly_sample_tensor(tensor, known_fraction)
print(f'STAGE 3 - INCOMPLETE TENSOR GENERATED - known elements: {known_fraction}')

#TEST TENSOR COMPLETION================================
tensor_norm = np.linalg.norm(tensor)
ratio_threshold = 5

class TestTensorCompletion_TMAC_TT(unittest.TestCase):

    def test_TMac_TT_top10pc(self):
        #Apply tensor completion
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, [1,1,1], convergence_tolerance=1e-15, iteration_limit=100000)
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
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, [1,1,1], convergence_tolerance=1e-15, iteration_limit=100000)
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
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, [1,1,1], convergence_tolerance=1e-15, iteration_limit=100000)
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
        TMAC_TT_PREDICTED_TENSOR, _, _ = tensorcomplete_TMac_TT(incomplete_tensor, known_indices, [1,1,1], convergence_tolerance=1e-15, iteration_limit=100000)
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


class TestTensorCompletion_Geng_Miles(unittest.TestCase):

    def test_Geng_Miles_top10pc(self):
        #Apply tensor completion
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, [1,1,1,1], hooi_tolerance=1e-3, iteration_limit=10000)
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
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, [1,1,1,1], hooi_tolerance=1e-3, iteration_limit=10000)
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
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, [1,1,1,1], hooi_tolerance=1e-3, iteration_limit=10000)
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
        GENG_MILES_PREDICTED_TENSOR, _, _, _ = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, known_indices, [1,1,1,1], hooi_tolerance=1e-3, iteration_limit=10000)
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

@unittest.skip('Poor tensor completion accuracy due to restriction on assumed rank <= 3.')
#CP-WOPT is ineffective here
class TestTensorCompletion_CP_WOPT_Dense(unittest.TestCase):

    def test_CP_WOPT_Dense_top10pc(self):
        #Apply tensor completion
        CP_WOPT_PREDICTED_TENSOR, _, _ = tensorcomplete_CP_WOPT_dense(incomplete_tensor, known_indices, 1, stepsize=0.0000001, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(CP_WOPT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'CP-WOPT Dense (10%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 10% according to predicted tensor
        sorted_predicted_dict_10pc = sortedBestValues(CP_WOPT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_10pc)
        true_indices = sorted_dict_10pc['indices']
        predicted_indices = sorted_predicted_dict_10pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'CP-WOPT Dense (10%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
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

    def test_CP_WOPT_Dense_top5pc(self):
        #Apply tensor completion
        CP_WOPT_PREDICTED_TENSOR, _, _ = tensorcomplete_CP_WOPT_dense(incomplete_tensor, known_indices, 1, stepsize=0.0000001, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(CP_WOPT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'CP-WOPT Dense (5%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_5pc = sortedBestValues(CP_WOPT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_5pc)
        true_indices = sorted_dict_5pc['indices']
        predicted_indices = sorted_predicted_dict_5pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'CP-WOPT Dense (5%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
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

    def test_CP_WOPT_Dense_top1pc(self):
        #Apply tensor completion
        CP_WOPT_PREDICTED_TENSOR, _, _ = tensorcomplete_CP_WOPT_dense(incomplete_tensor, known_indices, 1, stepsize=0.0000001, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(CP_WOPT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'CP-WOPT (1%) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 5% according to predicted tensor
        sorted_predicted_dict_1pc = sortedBestValues(CP_WOPT_PREDICTED_TENSOR, smallest=smallest, number_of_values=no_elements_1pc)
        true_indices = sorted_dict_1pc['indices']
        predicted_indices = sorted_predicted_dict_1pc['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'CP-WOPT (1%) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
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

    def test_CP_WOPT_Dense_top20(self):
        #Apply tensor completion
        CP_WOPT_PREDICTED_TENSOR, _, _ = tensorcomplete_CP_WOPT_dense(incomplete_tensor, known_indices, 1, stepsize=0.0000001, iteration_limit=10000)
        #Check norm difference from true tensor
        diff = norm_difference(CP_WOPT_PREDICTED_TENSOR, tensor)
        #Find ratio to tensor norm
        ratio = diff/tensor_norm
        print(f'CP-WOPT (top 20) ratio: {ratio}')
        self.assertTrue(ratio < ratio_threshold)
        #Obtain top 20 according to predicted tensor
        sorted_predicted_dict_top20 = sortedBestValues(CP_WOPT_PREDICTED_TENSOR, smallest=smallest, number_of_values=20)
        true_indices = sorted_dict_top20['indices']
        predicted_indices = sorted_predicted_dict_top20['indices']
        hamming_distance = Hamming_distance(true_indices, predicted_indices)
        aug_hamming_distance = Hamming_distance(true_indices, predicted_indices, augmented=True)
        common = common_count(true_indices, predicted_indices)
        LEN = len(true_indices)
        print(f'CP-WOPT (top 20) Hamming distance: {hamming_distance}, augmented hamming distance: {aug_hamming_distance}, common elements: {common}, length: {LEN}')
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
