from tensorsearch import higherDimensionalIndex, findBestValues, hyperparametersFromIndices
import tensorly as tl
import numpy as np
import unittest

#------------------------------------------------------------------------------------------
class TestHigherDimensionalIndex(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(higherDimensionalIndex(0, (2,4,7)), (0,0,0))

    def test_prod_minus_one(self):
        self.assertEqual(higherDimensionalIndex(3*4*5-1, (3,4,5)), (2,3,4))

    def test_prod(self):
        self.assertEqual(higherDimensionalIndex(3*8*3, (3,8,3)), (0,0,0))

    def test_random_index_1(self):
        self.assertEqual(higherDimensionalIndex(9, (2,4,7)), (0,1,2))

    def test_random_index_2(self):
        self.assertEqual(higherDimensionalIndex(11, (3,4,5)), (0,2,1))

    def test_random_index_3(self):
        self.assertEqual(higherDimensionalIndex(61, (3,8,3)), (2,4,1))

RANDOM_TENSOR = tl.tensor([[[[38,  8],
                             [81, 81],
                             [31, 30]],

                            [[86, 21],
                             [64, 50],
                             [26, 74]]],


                           [[[99, 14],
                             [39,  4],
                             [78, 98]],

                            [[72, 54],
                             [20, 24],
                             [42,  4]]],


                           [[[84,  6],
                             [ 1, 52],
                             [23, 23]],

                            [[22, 89],
                             [62, 51],
                             [18, 29]]]])
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
class TestFindBestValues(unittest.TestCase):

    def test_arange_min_1(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor)
        self.assertEqual(result['values'], [0])
        self.assertEqual(result['indices'], [(0,0,0)])

    def test_arange_max_1(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor, smallest=False)
        self.assertEqual(result['values'], [23])
        self.assertEqual(result['indices'], [(3,1,2)])

    def test_arange_min_5(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor, number_of_values=5)
        self.assertTrue( np.allclose(result['values'], [0, 1, 2, 3, 4]) )
        self.assertTrue( np.allclose(result['indices'], [(0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1)]) )

    def test_arange_max_4(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor, smallest=False, number_of_values=4)
        self.assertTrue( np.allclose(result['values'], [20, 21, 22, 23]) )
        self.assertTrue( np.allclose(result['indices'], [(3,0,2), (3,1,0), (3,1,1), (3,1,2)]) )

    def test_ones_min_6(self):
        tensor = tl.tensor(np.ones(81).reshape((3,3,3,3)))
        result = findBestValues(tensor, number_of_values=6)
        self.assertTrue( np.allclose(result['values'], [1, 1, 1, 1, 1, 1]) )

    def test_zeros_max_2(self):
        tensor = tl.tensor(np.zeros(32).reshape((2,2,2,2,2)))
        result = findBestValues(tensor, smallest=False, number_of_values=2)
        self.assertTrue( np.allclose(result['values'], [0,0]) )

    def test_random_min_1(self):
        result = findBestValues(RANDOM_TENSOR)
        self.assertEqual(result['values'], [1])
        self.assertEqual(result['indices'], [(2,0,1,0)])

    def test_random_max_1(self):
        result = findBestValues(RANDOM_TENSOR, smallest=False)
        self.assertEqual(result['values'], [99])
        self.assertEqual(result['indices'], [(1,0,0,0)])

    def test_random_min_3(self):
        result = findBestValues(RANDOM_TENSOR, number_of_values=3)
        self.assertTrue( np.allclose(result['values'], [4,1,4]) )
        self.assertTrue( np.allclose(result['indices'], [(1,1,2,1), (2,0,1,0), (1,0,1,1)]) )

    def test_random_max_4(self):
        result = findBestValues(RANDOM_TENSOR, smallest=False, number_of_values=4)
        self.assertTrue( np.allclose(result['values'], [86, 89, 99, 98]) )
        self.assertTrue( np.allclose(result['indices'], [(0,1,0,0), (2,1,0,1), (1,0,0,0), (1,0,2,1)]) )
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
class TestHyperparametersFromIndices(unittest.TestCase):

    def test_unequal_dimensions(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor, smallest=False, number_of_values=4)
        # Construct a range dictionary commensurate with tensor dimensions
        range_dict = {
            'alpha': {
                'start': 1,
                'end': 4,
                'interval': 1,
            },
        }
        test = False
        try:
            #Obtain hyperparameters corresponding to the best values
            hyperparameters = hyperparametersFromIndices(result['indices'], range_dict)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The indices (3) and hyperparameter configuration (1) have unequal dimensions.')
            test = True
        self.assertTrue(test)
        

    def test_arange_max_4(self):
        tensor = tl.tensor(np.arange(24).reshape((4,2,3)))
        result = findBestValues(tensor, smallest=False, number_of_values=4)
        # Construct a range dictionary commensurate with tensor dimensions
        range_dict = {
            'alpha': {
                'start': 1,
                'end': 4,
                'interval': 1,
            },
            'beta': {
                'values': [True, False],
            },
            'gamma': {
                'start': 0.1,
                'end': 0.9,
                'interval': 0.4,
            },
        }
        #Obtain hyperparameters corresponding to the best values
        hyperparameters = hyperparametersFromIndices(result['indices'], range_dict)
        #Check the hyperparameters
        expected_result = [{'alpha': 4, 'beta': True, 'gamma': 0.9},
                           {'alpha': 4, 'beta': False, 'gamma': 0.1},
                           {'alpha': 4, 'beta': False, 'gamma': 0.5},
                           {'alpha': 4, 'beta': False, 'gamma': 0.9}]
        for i in range(len(expected_result)):
            self.assertEqual( expected_result[i], hyperparameters[i] )

    def test_random_min_3(self):
        result = findBestValues(RANDOM_TENSOR, number_of_values=3)
        # Construct a range dictionary commensurate with tensor dimensions
        range_dict = {
            'a': {
                'values': ['a', 'b', 'c']
            },
            'b': {
                'values': [1,2],
            },
            'c': {
                'start': 0,
                'end': 2,
                'interval': 1,
            },
            'd': {
                'values': [True, False],
            },
        }
        #Obtain hyperparameters corresponding to the best values
        hyperparameters = hyperparametersFromIndices(result['indices'], range_dict)
        #Check the hyperparameters
        expected_result = [{'a': 'b', 'b': 2, 'c': 2, 'd': False},
                           {'a': 'c', 'b': 1, 'c': 1, 'd': True},
                           {'a': 'b', 'b': 1, 'c': 1, 'd': False}]
        for i in range(len(expected_result)):
            self.assertEqual( expected_result[i], hyperparameters[i] )
#------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
