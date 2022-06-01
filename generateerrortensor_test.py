import unittest
from generateerrortensor import generateIncompleteErrorTensor
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import numpy as np

class TestGenerateErrorTensor(unittest.TestCase):

    # Using mock evaluation function
    def test_mock_example(self):
        eval_func = lambda metric, alpha=0, beta=0, **kwargs : alpha + beta

        ranges_dict = {
            'alpha': {
                'start': 1,
                'end': 5,
                'interval': 0.1,
            },
            'beta': {
                'start': 0,
                'end': 10,
                'interval': 1,
            }
        }

        tensor, indices = generateIncompleteErrorTensor(eval_func, ranges_dict, 0.4, metric=regressionmetrics.mse)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (41, 11))
        self.assertTrue(np.count_nonzero(tensor) == 180)
        #Test indices
        self.assertTrue(len(indices) == 180)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 40)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)


    # Specify known fraction as 0.0
    def test_allzero(self):
        eval_func = lambda metric, alpha=0, beta=0, gamma=0, **kwargs : (alpha + beta)*gamma

        ranges_dict = {
            'alpha': {
                'start': 1,
                'end': 5,
                'interval': 0.1,
            },
            'beta': {
                'start': 0,
                'end': 10,
                'interval': 1,
            },
            'gamma': {
                'start': 1,
                'end': 2.2,
                'interval': 0.1,
            },
        }

        tensor, indices = generateIncompleteErrorTensor(eval_func, ranges_dict, 0.0, metric=classificationmetrics.hingeLoss)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (41, 11, 13))
        self.assertTrue(np.allclose(tensor, np.zeros((41,11,13))))
        #Test indices
        self.assertTrue(len(indices) == 0)


    # Using real data------------------------------------------------------------------------------

    # ridge regression
    def test_ridge_regression(self):
        task = 'regression'
        data = loadData(source='sklearn', identifier='california_housing', task=task)
        data_split = trainTestSplit(data)
        func = evaluationFunctionGenerator(data_split, algorithm='ridge-regression', task=task)

        ranges_dict = {
            'alpha': {
                'start': 0,
                'end': 3,
                'interval': 0.05,
            },
        }

        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.3, metric=regressionmetrics.mape)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (61,))
        self.assertTrue(np.count_nonzero(tensor) == 18)
        #Test indices
        self.assertTrue(len(indices) == 18)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 60)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

    # svm-rbf
    def test_svm_rbf(self):
        task = 'classification'
        data = loadData(source='sklearn', identifier='breast_cancer', task=task)
        data_split = trainTestSplit(data)
        func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)

        ranges_dict = {
            'C': {
                'start': 0.1,
                'end': 5.0,
                'interval': 0.1,
            },
            'gamma': {
                'start': 0.1,
                'end': 1.0,
                'interval': 0.1,
            }
        }
        
        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.2, metric=classificationmetrics.indicatorFunction)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (50,10))
        self.assertTrue(np.count_nonzero(tensor) == 100)
        #Test indices
        self.assertTrue(len(indices) == 100)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 49)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

    # svm-polynomial
    def test_svm_polynomial(self):
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
        
        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.1, metric=classificationmetrics.hingeLoss, eval_trials=2, evaluation_mode='raw-score')
        #Test tensor
        self.assertTrue(np.shape(tensor) == (30,30,31,4))
        self.assertTrue(np.count_nonzero(tensor) <= 31*12*30)
        #Test indices
        self.assertTrue(len(indices) == 31*12*30)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 30)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

    # KNN-regression
    def test_KNN_regression(self):
        task = 'regression'
        data = loadData(source='sklearn', identifier='diabetes', task=task)
        data_split = trainTestSplit(data)
        func = evaluationFunctionGenerator(data_split, algorithm='knn-regression', task=task)

        ranges_dict = {
            'N': {
                'start': 1.0,
                'end': 20.0,
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
                'end': 10.0,
                'interval': 1.0,
            }
        }
        
        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.5, metric=regressionmetrics.logcosh, eval_trials=10)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (20,2,1,10))
        self.assertTrue(np.count_nonzero(tensor) == 200)
        #Test indices
        self.assertTrue(len(indices) == 200)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 19)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

    # KNN-classification
    def test_KNN_classification(self):
        task = 'classification'
        data = loadData(source='sklearn', identifier='wine', task=task)
        binary_data = extractZeroOneClasses(data)
        data_split = trainTestSplit(binary_data)
        func = evaluationFunctionGenerator(data_split, algorithm='knn-classification', task=task)

        ranges_dict = {
            'N': {
                'start': 1.0,
                'end': 20.0,
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
                'end': 10.0,
                'interval': 0.1,
            }
        }
        
        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.3, metric=classificationmetrics.indicatorFunction, eval_trials=5)
        #Test tensor
        self.assertTrue(np.shape(tensor) == (20,2,1,91))
        self.assertTrue(np.count_nonzero(tensor) <= 1092)
        #Test indices
        self.assertTrue(len(indices) == 1092)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 90)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

    # Random forest
    def test_random_forest(self):
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
        
        tensor, indices = generateIncompleteErrorTensor(func, ranges_dict, 0.4, metric=classificationmetrics.KullbackLeiblerDivergence, evaluation_mode='probability')
        #Test tensor
        self.assertTrue(np.shape(tensor) == (5,5,2,9,10))
        self.assertTrue(np.count_nonzero(tensor) <= 5*5*2*9*10*0.4)
        #Test indices
        self.assertTrue(len(indices) == 5*5*2*9*10*0.4)
        maxvalues = list(map(max, indices))
        self.assertTrue(max(maxvalues) <= 9)
        minvalues = list(map(min, indices))
        self.assertTrue(min(minvalues) >= 0)

if __name__ == '__main__':
    unittest.main()
