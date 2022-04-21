from trainmodels import evaluationFunctionGenerator
import unittest

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
    
if __name__ == '__main__':
    # Run tests
    unittest.main()
