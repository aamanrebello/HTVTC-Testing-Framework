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
        data = loadData(source='sklearn', identifier='diabetes', task=task)
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

if __name__ == '__main__':
    unittest.main()
