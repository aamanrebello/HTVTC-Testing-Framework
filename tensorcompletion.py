import tensorly as tl
import numpy as np
from tensorly.decomposition import parafac

def tensorcomplete_CP_WOPT_dense(np_array, known_indices, rank, stepsize=0.01):
    #INITIALISATION-----------------------
    #Generate tensor from provided numpy array
    tensor = tl.tensor(np_array)
    #Obtain weighting tensor
    weighting_tensor = np.zeros(shape=np.shape(np_array))
    for index in known_indices:
        weighting_tensor[index] = 1
    #Obtain tensor Y from original paper (constant across iterations)
    tensor_Y = np.multiply(weighting_tensor, np_array)
    #Obtain squared norm of tensor Y
    Y_sq_norm = tl.tenalg.inner(tensor_Y, tensor_Y)
    #Initialise factor matrices as left singular vectors of n-mode flattening
    CPD_factors = []
    Ndims = len(np.shape(np_array))
    for mode in range(Ndims):
        unfolded_tensor = tl.unfold(tensor, mode=mode)
        u,_,_ = np.linalg.svd(unfolded_tensor, full_matrices=False)
        factor_matrix_estimate = u[:,0:rank]
        CPD_factors.append(factor_matrix_estimate)
    #In this form of the CPD, the weights of all rank-1 components are 1
    CPD_weights = np.ones(shape=(rank,))
    CPD_estimate = (CPD_weights, CPD_factors)
        
    #ITERATIONS----------------------------
    def stop_condition(prev_F, curr_F, tol=1e-8):
        return abs(prev_F - curr_F)/(prev_F+tol) < tol
    predicted_tensor = None
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_fval = 1
    current_fval = 0
    while not stop_condition(previous_fval, current_fval):
        #Obtain tensor Z from original paper (changes across iterations)
        predicted_tensor = tl.cp_to_tensor(CPD_estimate)
        tensor_Z = tl.tensor(np.multiply(weighting_tensor, predicted_tensor))
        #Obtain squared norm of tensor Z
        Z_sq_norm = tl.tenalg.inner(tensor_Z, tensor_Z)
        #Obtain function value
        previous_fval = current_fval
        current_fval = 0.5*Y_sq_norm + 0.5*Z_sq_norm - tl.tenalg.inner(tensor_Y, tensor_Z)
        #Difference between tensors Y and Z
        tensor_T = tensor_Y - tensor_Z
        #Gradient update of each A(n) wrt objective function.
        for mode in range(Ndims):
            leave_one_out_factors = CPD_factors[0:mode] + CPD_factors[mode+1:]
            continued_product = tl.tenalg.khatri_rao(leave_one_out_factors)
            gradient = -np.matmul(tl.unfold(tensor_T, mode=mode), continued_product)
            CPD_factors[mode] = CPD_factors[mode] - stepsize*gradient
        iterations+=1
    return predicted_tensor, current_fval, iterations


def tensorcomplete_TKD(np_array):
    tensor = tensorly.tensor(np_array)
    pass
