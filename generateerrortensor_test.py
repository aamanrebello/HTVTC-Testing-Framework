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

        result = generateIncompleteErrorTensor(eval_func, ranges_dict, 0.4, metric=regressionmetrics.mse)
        self.assertTrue(np.shape(result) == (41, 11))
        self.assertTrue(np.count_nonzero(result) == 180)


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

        result = generateIncompleteErrorTensor(eval_func, ranges_dict, 0.0, metric=classificationmetrics.hingeLoss)
        self.assertTrue(np.shape(result) == (41, 11, 13))
        self.assertTrue(np.allclose(result, np.zeros((41,11,13))))

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

        result = generateIncompleteErrorTensor(func, ranges_dict, 0.3, metric=regressionmetrics.mape)
        self.assertTrue(np.shape(result) == (61,))
        self.assertTrue(np.count_nonzero(result) == 18)

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
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 0.2, metric=classificationmetrics.indicatorFunction)
        self.assertTrue(np.shape(result) == (50,10))
        self.assertTrue(np.count_nonzero(result) == 100)

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
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 0.1, metric=classificationmetrics.hingeLoss, eval_trials=2, evaluation_mode='raw-score')
        self.assertTrue(np.shape(result) == (30,30,31,4))
        self.assertTrue(np.count_nonzero(result) <= 31*12*30)

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
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 0.5, metric=regressionmetrics.logcosh, eval_trials=10)
        self.assertTrue(np.shape(result) == (20,2,1,10))
        self.assertTrue(np.count_nonzero(result) == 200)

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
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 0.3, metric=classificationmetrics.indicatorFunction, eval_trials=5)
        self.assertTrue(np.shape(result) == (20,2,1,91))
        self.assertTrue(np.count_nonzero(result) <= 1092)

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
        
        result = generateIncompleteErrorTensor(func, ranges_dict, 0.4, metric=classificationmetrics.KullbackLeiblerDivergence, evaluation_mode='probability')
        self.assertTrue(np.shape(result) == (5,5,2,9,10))
        self.assertTrue(np.count_nonzero(result) <= 5*5*2*9*10*0.4)

if __name__ == '__main__':
    unittest.main()
