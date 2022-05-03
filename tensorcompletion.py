import tensorly as tl
import numpy as np
from tensorly.decomposition import parafac

#=================================================================================
#CP-WOPT when the tensor is treated as dense
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
#=================================================================================================


#=================================================================================================
#CP-WOPT when the tensor is treated as sparse
def tensorcomplete_CP_WOPT_sparse(np_array, known_indices, rank, stepsize=0.01):
    #INITIALISATION-----------------------
    #Generate tensor from provided numpy array
    tensor = tl.tensor(np_array)
    #Sort the known indices
    known_indices.sort()
    no_known_indices = len(known_indices)
    #Find elements corresponding to indices and take norm of the vector
    y = [np_array[index] for index in known_indices]
    y_sq_norm = np.inner(y,y)
    #Initialise factor matrices as left singular vectors of n-mode flattening
    CPD_factors = []
    Ndims = len(np.shape(np_array))
    for mode in range(Ndims):
        unfolded_tensor = tl.unfold(tensor, mode=mode)
        u,_,_ = np.linalg.svd(unfolded_tensor, full_matrices=False)
        factor_matrix_estimate = u[:,0:rank]
        CPD_factors.append(factor_matrix_estimate)
    
    #ITERATIONS----------------------------
    def stop_condition(prev_F, curr_F, tol=1e-8):
        return abs(prev_F - curr_F)/(prev_F+tol) < tol
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_fval = 1
    current_fval = 0
    while not stop_condition(previous_fval, current_fval):
        #Obtain vector z from original paper (changes across iterations)
        z = np.zeros(shape=(no_known_indices,))
        for q in range(no_known_indices):
            for r in range(rank):
                #Take product across factor matrices
                element_product = 1
                for n in range(Ndims):
                    factor_matrix = CPD_factors[n]
                    index = known_indices[q]
                    element = factor_matrix[index[n], r]
                    element_product *= element
                #Add products for each column of factor matrices
                z[q] += element_product
        #Obtain squared norm of vector z
        z_sq_norm = np.inner(z,z)
        #Obtain function value
        previous_fval = current_fval
        current_fval = 0.5*y_sq_norm + 0.5*z_sq_norm - np.inner(y, z)
        #Difference between vectors y and z
        t = y - z
        #Gradient update of each A(mode) wrt objective function.
        tensor_shape = np.shape(np_array)
        for mode in range(Ndims):
            mode_size = tensor_shape[mode]
            gradient_matrix = np.zeros(shape=(mode_size, rank))
            #For each of the r columns
            for r in range(rank):
                #u is as described in the paper - it holds the continued products
                u = t
                # For each of the factor matrices
                for n in range(Ndims):
                    if n == mode:
                        continue
                    #Find vector v as described in the paper 
                    factor_matrix = CPD_factors[n]
                    v = np.zeros(shape=(no_known_indices,))
                    for q in range(no_known_indices):
                        index = known_indices[q]
                        v[q] = factor_matrix[index[n], r]
                    u = np.multiply(u, v)
                #Find r^th column of mode^th gradient matrix
                for j in range(mode_size):
                    for q in range(no_known_indices):
                        index = known_indices[q]
                        if j == index[mode]:
                            gradient_matrix[j, r] += u[q]        
            #Use gradient matrix to update factor matrix
            CPD_factors[mode] = CPD_factors[mode] + stepsize*gradient_matrix       
        iterations+=1
    #In this form of the CPD, the weights of all rank-1 components are 1
    CPD_weights = np.ones(shape=(rank,))
    CPD_estimate = (CPD_weights, CPD_factors)
    predicted_tensor = tl.cp_to_tensor(CPD_estimate)
    return predicted_tensor, current_fval, iterations
#=============================================================================================


def tensorcomplete_TKD(np_array):
    tensor = tensorly.tensor(np_array)
    pass
