import unittest
import numpy as np
import tensorly as tl
import itertools
import random
import time
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_CP_WOPT_sparse


class TestTensorcomplete_CP_WOPT_dense(unittest.TestCase):

    
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
