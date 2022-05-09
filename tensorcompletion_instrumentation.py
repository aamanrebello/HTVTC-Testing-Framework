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
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 DENSE CPD CONSTRUCT AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

    def test_504030_random_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 2.5
        overall_tensor = scaling*np.random.normal(size=(50,40,30))
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
        t, f, i = tensorcomplete_CP_WOPT_dense(incomplete_tensor, sampled_indices, 5, stepsize=0.0000001, iteration_limit=100000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 DENSE RANDOM TENSOR AFTER CONVERGENCE-----')
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
        
        
class TestTensorcomplete_TKD_Geng_Miles(unittest.TestCase):
    
    def test_504030_tensor(self):
        #Generate random factor matrices and core tensor
        start_time = time.perf_counter()
        factor_scaling = 5
        core_scaling = 1.5
        #Use different random distributions as tensors are initialised with normal distributed values in the algorithm
        A = factor_scaling*np.random.poisson(size=(50,3))
        B = factor_scaling*np.random.poisson(size=(40,4))
        C = factor_scaling*np.random.poisson(size=(30,2))
        core = core_scaling*np.random.chisquare(df=2, size=(3,4,2))
        overall_tensor = tl.tucker_tensor.tucker_to_tensor((core, [A,B,C]))
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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, sampled_indices, [3,4,2], hooi_tolerance=1e-3)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 TKD Geng Miles AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')
        
    def test_504030_random_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 2.5
        overall_tensor = scaling*np.random.normal(size=(50,40,30))
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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, sampled_indices, [5,5,5], hooi_tolerance=1e-3, iteration_limit=10000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 TKD Geng Miles RANDOM TENSOR AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')
        
        
if __name__ == '__main__':
    unittest.main()
