import unittest
import numpy as np
import tensorly as tl
import itertools
import random
import time
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_CP_WOPT_sparse


class TestTensorcomplete_CP_WOPT_dense(unittest.TestCase):

    def test_matrix_completion(self):
        arr = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        known_indices = [(1,4), (1,7), (4,5), (5,4), (11,11), (7,10), (0,0)]
        t, f, i = tensorcomplete_CP_WOPT_dense(arr, known_indices, 2, stepsize=0.01)
        for index in known_indices:
            self.assertTrue(np.isclose(t[index], arr[index]))
        self.assertTrue(np.linalg.norm(t-arr) < 12)

    def test_3D_tensor(self):
        arr = np.array([[[ 10.,  0.],
                 [ 0.,  18.],
                 [ 0.,  0.],
                 [ 15.,  0.]],
                [[ 0.,  24.],
                 [ 0.,  36.],
                 [ 0.,  24.],
                 [ 30.,  0.]],
                [[ 0.,  0.],
                 [ 15.,  18.],
                 [ 0.,  0.],
                 [ 0.,  18.]]])
        #10/24 of elements known
        known_indices = [(0,0,0), (0,1,1), (0,3,0), (1,0,1), (1,1,1), (1,2,1), (1,3,0), (2,1,0), (2,1,1), (2,3,1)]
        ans = np.array([[[ 10.,  12.],
                         [ 15.,  18.],
                         [ 10.,  12.],
                         [ 15.,  18.]],
                        [[ 20.,  24.],
                         [ 30.,  36.],
                         [ 20.,  24.],
                         [ 30.,  36.]],
                        [[ 10.,  12.],
                         [ 15.,  18.],
                         [ 10.,  12.],
                         [ 15.,  18.]]])
        t, f, i = tensorcomplete_CP_WOPT_dense(arr, known_indices, 1, stepsize=0.001)
        self.assertTrue(np.isclose(ans, t, atol=1e-3).all())

    def test_4D_tensor(self):
        arr = np.array([[[[  0,   0],
         [  2,   0]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [  0,   9]]],


       [[[ 0,   0],
         [ 0,  0]],

        [[-12,  16],
         [0,  0]],

        [[-18,  24],
         [0,  0]]],


       [[[  0,   0],
         [  6,   9]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [ 18,  27]]],


       [[[ 0,   0],
         [ 0,  0]],

        [[-12,  16],
         [-18,  24]],

        [[0, 0],
         [0, 0]]]])
        #One third of elements known
        known_indices = [(0,0,0,0), (0,0,1,0), (0,1,1,0), (0,2,1,1), (1,1,0,0), (1,1,0,1), (1,2,0,0), (1,2,0,1), (2,0,0,1), (2,0,1,1), (2,2,1,0), (2,2,1,1), (3,1,0,0), (3,1,0,1), (3,1,1,0), (3,1,1,1)]

        ans = np.array([[[[  0,   0],
         [  2,   3]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [  6,   9]]],


       [[[ -6,   8],
         [ -5,  18]],

        [[-12,  16],
         [-18,  24]],

        [[-18,  24],
         [-15,  54]]],


       [[[  0,   0],
         [  6,   9]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [ 18,  27]]],


       [[[ -6,   8],
         [ -1,  24]],

        [[-12,  16],
         [-18,  24]],

        [[-18,  24],
         [ -3,  72]]]])
        t, f, i = tensorcomplete_CP_WOPT_dense(arr, known_indices, 2, stepsize=0.001)
        comparison = np.isclose(t, arr, atol=1e-3)
        #First check known elements are equal
        for index in known_indices:
            self.assertTrue(comparison[index])
            comparison[index] = False
        #Check if any of the other elements are predicted correctly
        self.assertTrue(comparison.any())

    
    def test_504030_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 5
        a1 = scaling*np.random.normal(size=(50,))
        b1 = scaling*np.random.normal(size=(40,))
        c1 = scaling*np.random.normal(size=(30,))
        a2 = scaling*np.random.normal(size=(50,))
        b2 = scaling*np.random.normal(size=(40,))
        c2 = scaling*np.random.normal(size=(30,))
        a3 = scaling*np.random.normal(size=(50,))
        b3 = scaling*np.random.normal(size=(40,))
        c3 = scaling*np.random.normal(size=(30,))
        t1 = tl.tenalg.outer([tl.tenalg.outer([a1, b1]), c1])
        t2 = tl.tenalg.outer([tl.tenalg.outer([a2, b2]), c2])
        t3 = tl.tenalg.outer([tl.tenalg.outer([a3, b3]), c3])
        overall_tensor = t1 + t2 + t3
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (50,40,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*50*40*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_CP_WOPT_dense(incomplete_tensor, sampled_indices, 3, stepsize=0.0000001)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 DENSE AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')


class TestTensorcomplete_CP_WOPT_sparse(unittest.TestCase):
        
    def test_matrix_completion(self):
        arr = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
        known_indices = [(1,4), (1,7), (4,5), (5,4), (11,11), (7,10), (0,0)]
        t, f, i = tensorcomplete_CP_WOPT_sparse(arr, known_indices, 2, stepsize=0.01)
        for index in known_indices:
            self.assertTrue(np.isclose(t[index], arr[index]))
        self.assertTrue(np.linalg.norm(t-arr) < 12)

    def test_3D_tensor(self):
        arr = np.array([[[ 10.,  0.],
                 [ 0.,  18.],
                 [ 0.,  0.],
                 [ 15.,  0.]],
                [[ 0.,  24.],
                 [ 0.,  36.],
                 [ 0.,  24.],
                 [ 30.,  0.]],
                [[ 0.,  0.],
                 [ 15.,  18.],
                 [ 0.,  0.],
                 [ 0.,  18.]]])
        #10/24 of elements known
        known_indices = [(0,0,0), (0,1,1), (0,3,0), (1,0,1), (1,1,1), (1,2,1), (1,3,0), (2,1,0), (2,1,1), (2,3,1)]
        ans = np.array([[[ 10.,  12.],
                         [ 15.,  18.],
                         [ 10.,  12.],
                         [ 15.,  18.]],
                        [[ 20.,  24.],
                         [ 30.,  36.],
                         [ 20.,  24.],
                         [ 30.,  36.]],
                        [[ 10.,  12.],
                         [ 15.,  18.],
                         [ 10.,  12.],
                         [ 15.,  18.]]])
        t, f, i = tensorcomplete_CP_WOPT_sparse(arr, known_indices, 1, stepsize=0.001)
        self.assertTrue(np.isclose(ans, t, atol=1e-3).all())

    def test_4D_tensor(self):
        arr = np.array([[[[  0,   0],
         [  2,   0]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [  0,   9]]],


       [[[ 0,   0],
         [ 0,  0]],

        [[-12,  16],
         [0,  0]],

        [[-18,  24],
         [0,  0]]],


       [[[  0,   0],
         [  6,   9]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [ 18,  27]]],


       [[[ 0,   0],
         [ 0,  0]],

        [[-12,  16],
         [-18,  24]],

        [[0, 0],
         [0, 0]]]])
        #One third of elements known
        known_indices = [(0,0,0,0), (0,0,1,0), (0,1,1,0), (0,2,1,1), (1,1,0,0), (1,1,0,1), (1,2,0,0), (1,2,0,1), (2,0,0,1), (2,0,1,1), (2,2,1,0), (2,2,1,1), (3,1,0,0), (3,1,0,1), (3,1,1,0), (3,1,1,1)]

        ans = np.array([[[[  0,   0],
         [  2,   3]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [  6,   9]]],


       [[[ -6,   8],
         [ -5,  18]],

        [[-12,  16],
         [-18,  24]],

        [[-18,  24],
         [-15,  54]]],


       [[[  0,   0],
         [  6,   9]],

        [[  0,   0],
         [  0,   0]],

        [[  0,   0],
         [ 18,  27]]],


       [[[ -6,   8],
         [ -1,  24]],

        [[-12,  16],
         [-18,  24]],

        [[-18,  24],
         [ -3,  72]]]])
        t, f, i = tensorcomplete_CP_WOPT_sparse(arr, known_indices, 2, stepsize=0.001)
        comparison = np.isclose(t, arr, atol=1e-3)
        #First check known elements are equal
        for index in known_indices:
            self.assertTrue(comparison[index])
            comparison[index] = False
        #Check if any of the other elements are predicted correctly
        self.assertTrue(comparison.any())

    def test_504030_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 5
        a1 = scaling*np.random.normal(size=(50,))
        b1 = scaling*np.random.normal(size=(40,))
        c1 = scaling*np.random.normal(size=(30,))
        a2 = scaling*np.random.normal(size=(50,))
        b2 = scaling*np.random.normal(size=(40,))
        c2 = scaling*np.random.normal(size=(30,))
        a3 = scaling*np.random.normal(size=(50,))
        b3 = scaling*np.random.normal(size=(40,))
        c3 = scaling*np.random.normal(size=(30,))
        t1 = tl.tenalg.outer([tl.tenalg.outer([a1, b1]), c1])
        t2 = tl.tenalg.outer([tl.tenalg.outer([a2, b2]), c2])
        t3 = tl.tenalg.outer([tl.tenalg.outer([a3, b3]), c3])
        overall_tensor = t1 + t2 + t3
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (50,40,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*50*40*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_CP_WOPT_sparse(incomplete_tensor, sampled_indices, 3, stepsize=0.0000001)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 SPARSE AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

if __name__ == '__main__':
    unittest.main()
