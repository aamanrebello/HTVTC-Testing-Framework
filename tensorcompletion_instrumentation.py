import unittest
import numpy as np
import tensorly as tl
import itertools
import random
import time
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_CP_WOPT_sparse, tensorcomplete_TKD_Geng_Miles, tensorcomplete_TKD_Gradient, tensorcomplete_TMac_TT
from tensorcompletion import ket_augmentation, inverse_ket_augmentation

@unittest.skip('')
class TestTensorcomplete_CP_WOPT_dense(unittest.TestCase):
    
    def test_504030_CPD_tensor(self):
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
    
    def test_30303030_CPD_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 5
        a1 = scaling*np.random.normal(size=(30,))
        b1 = scaling*np.random.normal(size=(30,))
        c1 = scaling*np.random.normal(size=(30,))
        d1 = scaling*np.random.normal(size=(30,))
        a2 = scaling*np.random.normal(size=(30,))
        b2 = scaling*np.random.normal(size=(30,))
        c2 = scaling*np.random.normal(size=(30,))
        d2 = scaling*np.random.normal(size=(30,))
        a3 = scaling*np.random.normal(size=(30,))
        b3 = scaling*np.random.normal(size=(30,))
        c3 = scaling*np.random.normal(size=(30,))
        d3 = scaling*np.random.normal(size=(30,))
        t1 = tl.tenalg.outer([a1, b1, c1, d1])
        t2 = tl.tenalg.outer([a2, b2, c2, d2])
        t3 = tl.tenalg.outer([a3, b3, c3, d3])
        overall_tensor = t1 + t2 + t3
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (30,30,30,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*30*30*30*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_CP_WOPT_dense(incomplete_tensor, sampled_indices, 3, stepsize=0.000000001, convergence_tolerance=1e-13, iteration_limit=15000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 30x30x30x30 DENSE CPD CONSTRUCT AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')
    
    def test_504030_TKD_tensor(self):
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
        t, f, i = tensorcomplete_CP_WOPT_dense(incomplete_tensor, sampled_indices, 15, stepsize=0.000000001, convergence_tolerance=1e-13, iteration_limit=200000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 DENSE TKD CONSTRUCT AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')
        
    def test_504030_random_tensor(self):
        #Generate random tensor
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

@unittest.skip('')            
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
        t, f, i = tensorcomplete_CP_WOPT_sparse(incomplete_tensor, sampled_indices, 3, stepsize=0.0000001, iteration_limit=250)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 SPARSE AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

    def test_30303030_tensor(self):
        #Generate random vectors whose outer producs generate the rank-1 components
        start_time = time.perf_counter()
        scaling = 5
        a1 = scaling*np.random.normal(size=(30,))
        b1 = scaling*np.random.normal(size=(30,))
        c1 = scaling*np.random.normal(size=(30,))
        d1 = scaling*np.random.normal(size=(30,))
        a2 = scaling*np.random.normal(size=(30,))
        b2 = scaling*np.random.normal(size=(30,))
        c2 = scaling*np.random.normal(size=(30,))
        d2 = scaling*np.random.normal(size=(30,))
        a3 = scaling*np.random.normal(size=(30,))
        b3 = scaling*np.random.normal(size=(30,))
        c3 = scaling*np.random.normal(size=(30,))
        d3 = scaling*np.random.normal(size=(30,))
        t1 = tl.tenalg.outer([a1, b1, c1, d1])
        t2 = tl.tenalg.outer([a2, b2, c2, d2])
        t3 = tl.tenalg.outer([a3, b3, c3, d3])
        overall_tensor = t1 + t2 + t3
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (30,30,30,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*30*30*30*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_CP_WOPT_sparse(incomplete_tensor, sampled_indices, 3, stepsize=0.00000005, convergence_tolerance=1e-13, iteration_limit=15)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 30x30x30x30 SPARSE AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')
        
@unittest.skip('')      
class TestTensorcomplete_TKD_Geng_Miles(unittest.TestCase):
    
    def test_504030_TKD_tensor(self):
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
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x20 TKD Geng Miles (TKD Tensor) AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')
     
    def test_30303030_TKD_tensor(self):
        #Generate random factor matrices and core tensor
        start_time = time.perf_counter()
        factor_scaling = 2.5
        core_scaling = 3
        #Use different random distributions as tensors are initialised with normal distributed values in the algorithm
        A = factor_scaling*np.random.poisson(size=(30,3))
        B = factor_scaling*np.random.poisson(size=(30,4))
        C = factor_scaling*np.random.poisson(size=(30,2))
        D = factor_scaling*np.random.poisson(size=(30,5))
        core = core_scaling*np.random.chisquare(df=2, size=(3,4,2,5))
        overall_tensor = tl.tucker_tensor.tucker_to_tensor((core, [A,B,C,D]))
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (30,30,30,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*30*30*30*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, sampled_indices, [3,4,2,5], hooi_tolerance=1e-3, iteration_limit=10000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 30x30x30x30 TKD Geng Miles (TKD Tensor) AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')
     
    def test_504030_CPD_tensor(self):
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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(incomplete_tensor, sampled_indices, [3,3,3], hooi_tolerance=1e-3, iteration_limit=10000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 TKD Geng Miles (CPD Tensor) AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')
      
    def test_504030_random_tensor(self):
        #Generate random tensor
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
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 TKD Geng Miles RANDOM TENSOR AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Converged?: {c}')
        print(f'Execution time: {end_time - start_time}')

@unittest.skip('')
class TestTensorcomplete_TKD_Gradient(unittest.TestCase):
    
    def test_504030_TKD_tensor(self):
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
        t, f, i = tensorcomplete_TKD_Gradient(incomplete_tensor, sampled_indices, [3,4,2], stepsize=0.00000000000005, iteration_limit=20000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 TKD Gradient (TKD Tensor) AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')
    
    #Takes up too much memory on my PC due to gradient calculation of core tensor taking too much space (6D tensor)
    def test_30303030_TKD_tensor(self):
        #Generate random factor matrices and core tensor
        start_time = time.perf_counter()
        factor_scaling = 2.5
        core_scaling = 3
        #Use different random distributions as tensors are initialised with normal distributed values in the algorithm
        A = factor_scaling*np.random.poisson(size=(30,3))
        B = factor_scaling*np.random.poisson(size=(30,4))
        C = factor_scaling*np.random.poisson(size=(30,2))
        D = factor_scaling*np.random.poisson(size=(30,5))
        core = core_scaling*np.random.chisquare(df=2, size=(3,4,2,5))
        overall_tensor = tl.tucker_tensor.tucker_to_tensor((core, [A,B,C,D]))
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (30,30,30,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*30*30*30*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_TKD_Gradient(incomplete_tensor, sampled_indices, [3,4,2,5], stepsize=0.000000000001, iteration_limit=20000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 30x30x30x30 TKD Gradient (TKD Tensor) AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

class TestTensorcomplete_TMac_TT(unittest.TestCase):
    @unittest.skip('')
    def test_504030_tensor(self):
        #Generate tensor-train order 3 tensors
        start_time = time.perf_counter()
        scaling = 3
        A = scaling*np.random.normal(size=(1, 50, 4))
        B = scaling*np.random.normal(size=(4, 40, 3))
        C = scaling*np.random.normal(size=(3, 30, 1))
        overall_tensor = tl.tt_tensor.tt_to_tensor([A, B, C])
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
        t, f, i = tensorcomplete_TMac_TT(incomplete_tensor, sampled_indices, [4, 3], convergence_tolerance=1e-15, iteration_limit=100000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 TMAC-TT AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')
    @unittest.skip('')
    def test_30303030_tensor(self):
        #Generate tensor-train order 3 tensors
        start_time = time.perf_counter()
        scaling = 3
        A = scaling*np.random.normal(size=(1, 30, 4))
        B = scaling*np.random.normal(size=(4, 30, 3))
        C = scaling*np.random.normal(size=(3, 30, 5))
        D = scaling*np.random.normal(size=(5, 30, 1))
        overall_tensor = tl.tt_tensor.tt_to_tensor([A, B, C, D])
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (30,30,30,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.05*30*30*30*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        t, f, i = tensorcomplete_TMac_TT(incomplete_tensor, sampled_indices, [4, 3, 5], convergence_tolerance=1e-15, iteration_limit=100000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 30x30x30x30 TMAC-TT AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

    def test_504030_random_tensor(self):
        #Generate random tensor
        start_time = time.perf_counter()
        scaling = 2.5
        overall_tensor = scaling*np.random.normal(size=(50,40,30))
        #Generate all possible combinations of indices
        value_lists = []
        for dim_size in (50,40,30):
            value_lists.append([el for el in range(dim_size)])
        all_indices = list(itertools.product(*value_lists))
        #Randomly sample 5% of elements
        no_elements = int(0.1*50*40*30)
        # Randomly sample from all_indices
        sampled_indices = random.sample(all_indices, no_elements)
        # Generate tensor with all unknown indices set to zero
        incomplete_tensor = np.zeros(shape=np.shape(overall_tensor))
        for index in sampled_indices:
           incomplete_tensor[index] = overall_tensor[index]
        #Apply tensor completion to incomplete tensor
        t, f, i = tensorcomplete_TMac_TT(incomplete_tensor, sampled_indices, [5, 5], convergence_tolerance=1e-15, iteration_limit=150000)
        difference = np.linalg.norm(np.ndarray.flatten(t - overall_tensor))
        end_time = time.perf_counter()
        print('\n----SUBJECTIVE TEST RESULTS for 50x40x30 DENSE RANDOM TENSOR AFTER CONVERGENCE-----')
        print(f'Norm Difference Between predicted and true: {difference}')
        print(f'Objective function value: {f}')
        print(f'Number of iterations: {i}')
        print(f'Execution time: {end_time - start_time}')

        
if __name__ == '__main__':
    unittest.main()
