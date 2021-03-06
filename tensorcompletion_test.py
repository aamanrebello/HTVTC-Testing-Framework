import unittest
import numpy as np
import tensorly as tl
from tensorcompletion import tensorcomplete_CP_WOPT_dense, tensorcomplete_CP_WOPT_sparse, tensorcomplete_TKD_Geng_Miles, tensorcomplete_TKD_Gradient, tensorcomplete_TMac_TT


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
        
        
class TestTensorcomplete_TKD_Geng_Miles(unittest.TestCase):

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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(arr, known_indices, [2,2], hooi_tolerance=1e-3)
        self.assertTrue(c)
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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(arr, known_indices, [1,1,1], 1e-3)
        self.assertTrue(c)
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
        t, f, i, c = tensorcomplete_TKD_Geng_Miles(arr, known_indices, [2,2,2,2], 1e-3, 1e-6)
        self.assertTrue(c)
        comparison = np.isclose(t, ans, atol=0.5)
        #First check known elements are equal
        for index in known_indices:
            self.assertTrue(comparison[index])
            comparison[index] = False
        #Check if any of the other elements are predicted correctly
        self.assertTrue(comparison.any())


class TestTensorcomplete_TKD_Gradient(unittest.TestCase):
    
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
        t, f, i = tensorcomplete_TKD_Gradient(arr, known_indices, [2,2])
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
        t, f, i = tensorcomplete_TKD_Gradient(arr, known_indices, [1,1,1], stepsize=0.00001)
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
        t, f, i = tensorcomplete_TKD_Gradient(arr, known_indices, [2,2,2,2], stepsize=0.000001, convergence_tolerance=0.5e-4)
        comparison = np.isclose(t, ans, atol=0.5)
        #First check known elements are equal
        for index in known_indices:
            self.assertTrue(comparison[index])
            comparison[index] = False
        #Check if any of the other elements are predicted correctly
        self.assertTrue(comparison.any())


class TestTensorcomplete_TMac_TT(unittest.TestCase):
    
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
        t, f, i = tensorcomplete_TMac_TT(arr, known_indices, [2,2])
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
        #Interchange axes for ket augmentation and update known indices
        t, f, i = tensorcomplete_TMac_TT(arr, known_indices, [1,1,1])
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
        t, f, i = tensorcomplete_TMac_TT(arr, known_indices, [2,2,2,2])
        comparison = np.isclose(t, ans, atol=0.5)
        #First check known elements are equal
        for index in known_indices:
            self.assertTrue(comparison[index])
            comparison[index] = False
        #Check if any of the other elements are predicted correctly
        self.assertTrue(comparison.any())

if __name__ == '__main__':
    unittest.main()
