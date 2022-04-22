from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses
import regressionmetrics
import classificationmetrics
import unittest

#---------CURRENTLY UNUSED----------------------------------------
mockRegressionData = {
    'training_features': [[1,2], [2,2], [0,1], [3,4], [2,3]],
    'validation_features': [[1,1], [3,3]],
    'training_labels': [0.1, 0.4, 0.0, -0.4, -0.2],
    'validation_labels': [0.2, 0.2]
    }

mockClassificationData = {
    'training_features': [[1,2], [2,2], [0,1], [3,4], [2,3]],
    'validation_features': [[1,1], [3,3]],
    'training_labels': [0, 1, 0, 0, 1],
    'validation_labels': [0, 1]
    }
#------------------------------------------------------------------


class TestEvaluationFunctionGenerator(unittest.TestCase):

    def test_incorrect_task(self):
        #Perform test for ridge regression
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockRegressionData, algorithm = 'ridgeregression', task = 'classification')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

        #Perform test for svm-rbf
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockClassificationData, task = 'regression')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

        #Perform test for svm-polynomial
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockClassificationData, algorithm = 'svm-polynomial', task = '***')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

        #Perform test for KNN-regression
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockRegressionData, algorithm = 'knn-regression', task = 'rgression')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

        #Perform test for KNN-classification
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockClassificationData, algorithm = 'knn-classification', task = 'clasification')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

        #Perform test for random forest
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockClassificationData, algorithm = 'random-forest', task = 'regression')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)


    def test_unknown_algorithm(self):
        #Perform test for unrecognised algorithm
        test = False
        try:
            result = evaluationFunctionGenerator(data = mockClassificationData, algorithm = 'unknown', task = 'classification')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The algorithm specified is not recognised.')
            test = True
        self.assertTrue(test)

    def test_ridge_regression(self):
        #Check whether algorithm returns credible results
        task = 'regression'
        data = loadData(source='sklearn', identifier='diabetes', task=task)
        data_split = trainTestSplit(data)
        func = evaluationFunctionGenerator(data_split, algorithm='ridge-regression', task=task)
        self.assertIsNotNone(func)
        
        error = func(alpha=0.1, metric=regressionmetrics.mse)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    def test_svm_rbf(self):
        #Check whether algorithm returns credible results
        task = 'classification'
        data = loadData(source='sklearn', identifier='iris', task=task)
        binary_data = extractZeroOneClasses(data)
        data_split = trainTestSplit(binary_data)
        func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)
        self.assertIsNotNone(func)
        
        error = func(C=1.0, gamma=0.1, metric=classificationmetrics.indicatorFunction)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    def test_svm_polynomial(self):
        #Check whether algorithm returns credible results
        task = 'classification'
        data = loadData(source='sklearn', identifier='wine', task=task)
        binary_data = extractZeroOneClasses(data)
        data_split = trainTestSplit(binary_data)
        func = evaluationFunctionGenerator(data_split, algorithm='svm-polynomial', task=task)
        self.assertIsNotNone(func)
        
        error = func(C=1.0, gamma=0.1, constant_term=0.1, degree=1, metric=classificationmetrics.indicatorFunction)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    def test_KNN_regression(self):
        #Check whether algorithm returns credible results
        task = 'regression'
        data = loadData(source='sklearn', identifier='california_housing', task=task)
        data_split = trainTestSplit(data, validation_proportion=0.3)
        func = evaluationFunctionGenerator(data_split, algorithm='knn-regression', task=task)
        self.assertIsNotNone(func)
        
        error = func(N=5, weightingFunction='uniform', distanceFunction='minkowski', p=2, metric=regressionmetrics.mae)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    def test_KNN_classification(self):
        #Check whether algorithm returns credible results
        task = 'classification'
        data = loadData(source='sklearn', identifier='breast_cancer', task=task)
        data_split = trainTestSplit(data)
        func = evaluationFunctionGenerator(data_split, algorithm='knn-classification', task=task)
        self.assertIsNotNone(func)
        
        error = func(N=10, weightingFunction='distance', distanceFunction='minkowski', p=1, metric=classificationmetrics.indicatorFunction)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    def test_random_forest(self):
        #Check whether algorithm returns credible results
        task = 'classification'
        data = loadData(source='sklearn', identifier='breast_cancer', task=task)
        data_split = trainTestSplit(data, validation_proportion=0.3)
        func = evaluationFunctionGenerator(data_split, algorithm = 'random-forest', task=task)
        self.assertIsNotNone(func)
        
        error = func(15, 10, True, 2, 5, metric=classificationmetrics.indicatorFunction)
        self.assertIsNotNone(error)
        self.assertTrue(isinstance(error, float))
        self.assertTrue(error >= 0)

    
if __name__ == '__main__':
    # Run tests
    unittest.main()
