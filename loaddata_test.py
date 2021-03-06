from loaddata import generateReturnDict, manipulateLocalData, loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
from loaddata import ONLINE_DOWNLOAD_PATH
import pandas as pd
import numpy as np
import unittest
import warnings
import os

TEST_PATH = 'datasets/test_case.csv'

#-------------------------------------------------------------------------------
class TestGenerateReturnDict(unittest.TestCase):

    def test_function(self):
        features = np.array([[1,2], [3,4], [5,6], [7,8]])
        labels = np.array([9, 10, 11, 12])
        result = generateReturnDict(features, labels)
        self.assertTrue( np.array_equal(result['features'], features) )
        self.assertTrue( np.array_equal(result['labels'], labels) )
#--------------------------------------------------------------------------------


#--------------------------------------------------------------------------------
class TestManipulateLocalData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up test file
        data = {
            'Name': ['Tom', 'Jane', 'Krisha', 'John'],
            'Age': [20, 21, 19, 18],
            'Height': [180, 179, 182, 177]
        }
        df = pd.DataFrame(data)
        df.to_csv(TEST_PATH)

    @classmethod
    def tearDownClass(cls):
        os.remove(TEST_PATH)

    def test_function(self):
        result = manipulateLocalData(file_path=TEST_PATH, feature_attributes=['Age', 'Height'], label_attributes=['Name'])
        features = np.array([[20, 180], [21, 179], [19, 182], [18, 177]])
        labels = np.array([['Tom'], ['Jane'], ['Krisha'], ['John']])
        self.assertTrue( np.array_equal(result['features'], features) )
        self.assertTrue( np.array_equal(result['labels'], labels) )
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
class TestExtractZeroOneClasses(unittest.TestCase):

    def test_default_params(self):
        data = {
            'features': [[1,2], [2,2], [1,1], [0,3], [2,4]],
            'labels': [0, 1, 2, 3, 0]
        }
        result = extractZeroOneClasses(data)
        self.assertTrue( np.array_equal(result['features'], [[1,2], [2,2], [2,4]]) )
        self.assertTrue( np.array_equal(result['labels'], [0, 1, 0]) )

    def test_non_default_params(self):
        data = {
            'features': [[1,2], [2,2], [1,1], [0,3], [2,4]],
            'labels': [2, 3, 2, 3, 0]
        }
        result = extractZeroOneClasses(data, 2, 3)
        self.assertTrue( np.array_equal(result['features'], [[1,2], [2,2], [1,1], [0,3]]) )
        self.assertTrue( np.array_equal(result['labels'], [0, 1, 0, 1]) )
#-----------------------------------------------------------------------------------
        

#-----------------------------------------------------------------------------------
class TestConvertZeroOne(unittest.TestCase):

    def test_default_params(self):
        data = {
            'features': [[1,2], [2,2], [1,1], [0,3], [2,4]],
            'labels': [1, 0, 2, 0, 3]
        }
        result = convertZeroOne(data, 3, 1)
        self.assertTrue( np.array_equal(result['features'], data['features']) )
        self.assertTrue( np.array_equal(result['labels'], [1, 3, 2, 3, 3]) )

    def test_non_default_params(self):
        data = {
            'features': [[1,2], [2,2], [1,1], [0,3], [2,4]],
            'labels': [2, 3, 2, 3, 3]
        }
        result = convertZeroOne(data, -1, 1, 2, 3)
        self.assertTrue( np.array_equal(result['features'], data['features']) )
        self.assertTrue( np.array_equal(result['labels'], [-1, 1, -1, 1, 1]) )

    def test_no_change(self):
        data = {
            'features': [[1,2], [2,2], [1,1], [0,3], [2,4]],
            'labels': [2, 3, 2, 3, 3]
        }
        result = convertZeroOne(data, -1, 1)
        self.assertTrue( np.array_equal(result['features'], data['features']) )
        self.assertTrue( np.array_equal(result['labels'], [2, 3, 2, 3, 3]) )
#------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------
class TestLoadData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up test file
        data = {
            'Name': ['Tom', 'Jane', 'Krisha', 'John'],
            'Age': [20, 21, 19, 18],
            'Height': [180, 179, 182, 177]
        }
        df = pd.DataFrame(data)
        df.to_csv(TEST_PATH)

    @classmethod
    def tearDownClass(cls):
        # Remove test files
        os.remove(TEST_PATH)
        os.remove(ONLINE_DOWNLOAD_PATH)

    def test_iris_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'iris')
        self.assertEqual(np.shape(result['features']), (150,4))
        self.assertEqual(np.shape(result['labels']), (150,))

    def test_diabetes_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'diabetes', 'regression')
        self.assertEqual(np.shape(result['features']), (442,10))
        self.assertEqual(np.shape(result['labels']), (442,))

    def test_digits_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'digits', 'imageclassification')
        self.assertEqual(np.shape(result['features']), (1797,64))
        self.assertEqual(np.shape(result['labels']), (1797,))

    def test_wine_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'wine', 'classification')
        self.assertEqual(np.shape(result['features']), (178,13))
        self.assertEqual(np.shape(result['labels']), (178,))

    def test_breast_cancer_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'breast_cancer')
        self.assertEqual(np.shape(result['features']), (569,30))
        self.assertEqual(np.shape(result['labels']), (569,))

    def test_california_housing_data(self):
        warnings.filterwarnings('ignore')
        result = loadData('sklearn', 'california_housing', 'regression')
        self.assertEqual(np.shape(result['features']), (20640,8))
        self.assertEqual(np.shape(result['labels']), (20640,))

    def test_unknown_sklearn(self):
        test = False
        try:
            result = loadData('sklearn', 'unknown', 'regression')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The specified sklearn dataset is not recognised.')
            test = True
        self.assertTrue(test)

    def test_wrong_sklearn_task_for_data(self):
        test = False
        try:
            result = loadData('sklearn', 'iris', 'regression')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The specified sklearn dataset does not fit the specified machine learning task.')
            test = True
        self.assertTrue(test)

    def test_online_data(self):
        url = 'http://www.commsp.ee.ic.ac.uk/~mandic/FSPML_Course/snp_500_2015_2019.csv'
        result = loadData('online', url, 'regression', feature_attributes=['Open', 'Close'], label_attributes=['High', 'Low'])
        self.assertEqual(np.shape(result['features']), (1006,2))
        self.assertEqual(np.shape(result['labels']), (1006,2))

    def test_local_data(self):
        result = loadData('local', TEST_PATH, feature_attributes=['Age'], label_attributes=['Height'])
        self.assertEqual(np.shape(result['features']), (4,1))
        self.assertEqual(np.shape(result['labels']), (4,1))

    def test_unknown_data_source(self):
        test = False
        try:
            result = loadData('unknown', '', '')
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The specified source of data is not recognised.')
            test = True
        self.assertTrue(test)

    def test_no_feature_attributes(self):
        test = False
        try:
            result = loadData('local', TEST_PATH, label_attributes=['attr'])
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The feature and label attributes of the data both need to be specified.')
            test = True
        self.assertTrue(test)

    def test_no_label_attributes(self):
        test = False
        try:
            result = loadData('local', TEST_PATH, feature_attributes=['attr'])
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The feature and label attributes of the data both need to be specified.')
            test = True
        self.assertTrue(test)
#------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------
class TestTrainTestSplit(unittest.TestCase):

    def test_separate_nonzero_validation_zero_test(self):
        data = loadData('sklearn', 'iris')
        result = trainTestSplit(data)
        self.assertEqual(len(result['training_features']), 120)
        self.assertEqual(len(result['training_labels']), 120)
        self.assertEqual(len(result['validation_features']), 30)
        self.assertEqual(len(result['validation_labels']), 30)
        self.assertEqual(len(result['test_features']), 0)
        self.assertEqual(len(result['test_labels']), 0)

    def test_separate_zero_validation_nonzero_test(self):
        data = loadData('sklearn', 'iris')
        result = trainTestSplit(data, test_proportion=0.3, validation_proportion=0.0)
        self.assertEqual(len(result['training_features']), 105)
        self.assertEqual(len(result['training_labels']), 105)
        self.assertEqual(len(result['validation_features']), 0)
        self.assertEqual(len(result['validation_labels']), 0)
        self.assertEqual(len(result['test_features']), 45)
        self.assertEqual(len(result['test_labels']), 45)

    def test_cross_validation_nonzero_validation_zero_test(self):
        data = loadData('sklearn', 'diabetes', 'regression')
        result = trainTestSplit(data, method='cross_validation', validation_proportion=0.2)
        self.assertEqual(result['no_splits'], 5)
        training_indices, validation_indices = next(result['index_generator'])
        self.assertEqual(len(training_indices), 353)
        self.assertEqual(len(validation_indices), 89)
        self.assertEqual(len(result['test_features']), 0)
        self.assertEqual(len(result['test_labels']), 0)
        self.assertEqual(len(result['trainval_features']), 442)
        self.assertEqual(len(result['trainval_labels']), 442)
                         

    def test_cross_validation_zero_validation(self):
        data = loadData('sklearn', 'diabetes', 'regression')
        test = False
        try:
            result = trainTestSplit(data, method='cross_validation', validation_proportion=0.0)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'Validation data size cannot be zero for cross validation.')
            test = True
        self.assertTrue(test)

    def test_unknown_method_of_split(self):
        data = loadData('sklearn', 'diabetes', 'regression')
        test = False
        try:
            result = trainTestSplit(data, method='unknown', validation_proportion=0.0)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The specified method to split the data is not recognised.')
            test = True
        self.assertTrue(test)
#------------------------------------------------------------------------------


if __name__ == '__main__':
    # Run tests
    unittest.main()
