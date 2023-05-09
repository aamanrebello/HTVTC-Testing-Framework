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

    tensor, _ = generateIncompleteErrorTensor(func, ranges_dict, 1.0, metric=classificationmetrics.hingeLoss, evaluation_mode='raw-score')
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

#TEST TENSOR COMPLETION================================
tensor_norm = np.linalg.norm(tensor)

class TestTensorCompletion_Cross(unittest.TestCase):

    def test_default_rank(self):
        print('=================================Low Rank Tensor Completion==============================================================')
        #Begin time measurement
        start_time = time.process_time_ns()
        #Cross sample tensor
        body, joints, arms, no_elements = cross_sample_tensor(tensor, tucker_rank_list=[2,2,2,2])
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


    def test_GP_regression(self):
        print('=================================Compare with GP Regression==============================================================')
        shape = list(tensor.shape)

        #Used to transform hyperparameter combination dict to a list that can be used for ML
        def transform_combination(combo_dict):
            return list(combo_dict.values())

        # Generate training data set
        TR_hyperparameters = []
        TR_validation_losses = []
        for shape_id in range(len(shape)):
            dim_size = shape[shape_id]
            base_index = [0]*len(shape)
            for id in range(dim_size):
                base_index[shape_id] = id
                hyperparameter_combination = hyperparametersFromIndices([base_index], ranges_dict, True)[0]
                validation_loss = tensor[tuple(base_index)]
                TR_hyperparameters.append(transform_combination(hyperparameter_combination))
                TR_validation_losses.append(validation_loss)

        # Generate validation data set (the entire tensor)
        import itertools

        value_lists = []
        for i in range(len(shape)):
            value_lists.append([el for el in range(shape[i])])
        tensor_indices = list(itertools.product(*value_lists))

        VAL_hyperparameters = list(map(transform_combination, hyperparametersFromIndices(tensor_indices, ranges_dict, True)))
        VAL_validation_losses = [0]*len(tensor_indices)
        for id in range(len(tensor_indices)):
            t_id = tuple(tensor_indices[id])
            VAL_validation_losses[id] = tensor[t_id] 

        # Run GP regression
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(TR_hyperparameters, TR_validation_losses)
        predictions, stds = gpr.predict(VAL_hyperparameters, return_std=True)

        #Calculate metrics
        tensor_norm = np.linalg.norm(tensor)
        norm_difference_ratio = norm_difference(predictions, VAL_validation_losses)/tensor_norm
        print(f'Norm  Difference Ratio: {norm_difference_ratio}')
        sorted_true_loss_indices = np.argsort(VAL_validation_losses)[0:no_elements_10pc]
        sorted_pred_loss_indices = np.argsort(predictions)[0:no_elements_10pc]
        common = common_count(sorted_true_loss_indices, sorted_pred_loss_indices)
        print(f'Common elements in top 10%: {common}')
        true_values = np.sort(VAL_validation_losses)[0:no_elements_10pc]
        predicted_values = np.sort(predictions)[0:no_elements_10pc]
        norm_error = np.linalg.norm(true_values - predicted_values)/(np.linalg.norm(true_values) + 1e-10)
        print(f'Error in top 10%: {norm_error}')
    
        completed = True
        self.assertTrue(completed)

if __name__ == '__main__':
    unittest.main()
