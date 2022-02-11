from tensorsearch import higherDimensionalIndex, findBestValues
import tensorly as tl
import numpy as np
import unittest


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

if __name__ == '__main__':
    unittest.main()
