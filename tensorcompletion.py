import tensorly as tl
import numpy as np

#=================================================================================
#CP-WOPT when the tensor is treated as dense
def tensorcomplete_CP_WOPT_dense(np_array, known_indices, rank, stepsize=0.01, convergence_tolerance=1e-8, **kwargs):
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
    #Used to set an iteration limit
    iteration_condition = lambda i: False
    if 'iteration_limit' in kwargs.keys():
        iteration_condition = lambda i: i >= kwargs['iteration_limit']
    #The condition for convergence 
    def convergence_condition(prev_F, curr_F, tol):
        return abs(prev_F - curr_F)/(prev_F+tol) < tol
    
    predicted_tensor = None
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_fval = 1
    current_fval = 0
    while (not iteration_condition(iterations)) and (not convergence_condition(previous_fval, current_fval, convergence_tolerance)):
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
def tensorcomplete_CP_WOPT_sparse(np_array, known_indices, rank, stepsize=0.01, **kwargs):
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
    #Used to set an iteration limit
    iteration_condition = lambda i: False
    if 'iteration_limit' in kwargs.keys():
        iteration_condition = lambda i: i >= kwargs['iteration_limit']
    #The condition for convergence 
    def convergence_condition(prev_F, curr_F, tol=1e-8):
        return abs(prev_F - curr_F)/(prev_F+tol) < tol
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_fval = 1
    current_fval = 0
    while (not iteration_condition(iterations)) and (not convergence_condition(previous_fval, current_fval)):
        #Obtain v vectors for all ranks and known indices
        v_vectors = np.zeros(shape=(rank, Ndims, no_known_indices))
        for r in range(rank):                 
            for n in range(Ndims):
                factor_matrix = CPD_factors[n]
                for q in range(no_known_indices):
                    index = known_indices[q]
                    v_vectors[r,n,q] = factor_matrix[index[n], r]
        #Obtain vector z from original paper (changes across iterations)
        hadamard_prods = np.multiply.reduce(v_vectors, axis=1, keepdims=True)
        z = np.reshape(np.add.reduce(hadamard_prods, axis=0), newshape=(no_known_indices,))
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
            #leave one out continued vector Hadamard products for all ranks
            u_products_1 = np.multiply.reduce(v_vectors[:,0:mode,:], axis=1, keepdims=True)
            u_products_2 = np.multiply.reduce(v_vectors[:,mode+1:,:], axis=1, keepdims=True)
            u_products = np.multiply(u_products_1, u_products_2) 
            #For each of the r columns
            for r in range(rank):
                #u is as described in the paper - it holds the continued products
                u = np.multiply(t, u_products[r,0])
                #Find r^th column of mode^th gradient matrix
                for j in range(mode_size):
                    where_array = [j == known_indices[q][mode] for q in range(no_known_indices)]
                    gradient_matrix[j, r] = np.add.reduce(u, axis=0, where=where_array)        
            #Use gradient matrix to update factor matrix
            CPD_factors[mode] = CPD_factors[mode] + stepsize*gradient_matrix       
        iterations+=1
    #In this form of the CPD, the weights of all rank-1 components are 1
    CPD_weights = np.ones(shape=(rank,))
    CPD_estimate = (CPD_weights, CPD_factors)
    predicted_tensor = tl.cp_to_tensor(CPD_estimate)
    return predicted_tensor, current_fval, iterations
#=============================================================================================


#=============================================================================================
def tensorcomplete_TKD_Geng_Miles(np_array, known_indices, rank_list, hooi_tolerance, objective_tolerance=1e-8, **kwargs):
    #Generate tensor with unknown elements initialised to mean of known elements
    #Find elements corresponding to known indices and take their mean
    known_values = [np_array[index] for index in known_indices]
    no_known_values = len(known_indices)
    known_mean = np.mean(known_values)
    #First generate tensor with all known elements equal to mean
    initialisation = np.full(shape=np.shape(np_array), fill_value=known_mean)
    #Set known index positions to the corresponding known values
    for i in range(no_known_values):
        index = known_indices[i]
        initialisation[index] = known_values[i]
    #Generate tensor from initialisation
    target_tensor = tl.tensor(initialisation)
    #Perform initial HOOI to obtain core and factor matrices that form the initial prediction tensor
    core, factors = tl.decomposition.tucker(tensor=target_tensor, rank=rank_list, tol=hooi_tolerance)
    prediction_tensor = tl.tucker_tensor.tucker_to_tensor((core,factors))

    #Used to set an iteration limit
    iteration_condition = lambda i: False
    if 'iteration_limit' in kwargs.keys():
        iteration_condition = lambda i: i >= kwargs['iteration_limit']
    #The condition for convergence
    def convergence_condition(current_fval, tol):
        return current_fval < tol

    #Initial values allow the loop to progress
    prev_fval = 1
    current_fval = 1
    #Returned as readings
    iterations = 0
    while (not iteration_condition(iterations)) and (not convergence_condition(current_fval, objective_tolerance)):
        #Update target tensor according to values in predicted tensor corresponding to unknown values
        target_tensor = tl.copy(prediction_tensor)
        for i in range(no_known_values):
            index = known_indices[i]
            target_tensor[index] = known_values[i]
        core, factors = tl.decomposition.tucker(tensor=target_tensor, rank=rank_list, tol=hooi_tolerance)
        prediction_tensor = tl.tucker_tensor.tucker_to_tensor((core,factors))
        #Update function values
        prev_fval = current_fval
        current_fval = tl.norm(prediction_tensor - target_tensor)
        #Break in case of no decrease in fvalue (could be due to incorrect rank, too few elements)
        if iterations > 0 and current_fval > prev_fval:
            break
        iterations += 1
        
    converged = convergence_condition(current_fval, objective_tolerance)
    return prediction_tensor, current_fval, iterations, converged
#=============================================================================================


#=============================================================================================
def tensorcomplete_TKD_Gradient(np_array, known_indices, rank_list, stepsize=0.01, convergence_tolerance=1e-8, **kwargs):
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
    #Initialise factor matrices as left singular vectors of n-mode flattening and find core tensor through mode-n product with transpose
    #Essentially we are doing HOSVD.
    TKD_factors = []
    TKD_core = tl.copy(tensor)
    Ndims = len(np.shape(np_array))
    for mode in range(Ndims):
        unfolded_tensor = tl.unfold(tensor, mode=mode)
        u,_,_ = np.linalg.svd(unfolded_tensor, full_matrices=False)
        rank = rank_list[mode]
        factor_matrix_estimate = u[:,0:rank]
        TKD_factors.append(factor_matrix_estimate)
        TKD_core = tl.tenalg.mode_dot(TKD_core, factor_matrix_estimate.T, mode=mode)
    TKD_estimate = (TKD_core, TKD_factors)
        
    #ITERATIONS----------------------------
    #Used to set an iteration limit
    iteration_condition = lambda i: False
    if 'iteration_limit' in kwargs.keys():
        iteration_condition = lambda i: i >= kwargs['iteration_limit']
    #The condition for convergence
    def convergence_condition(prev_F, curr_F, tol):
        return abs(prev_F - curr_F)/(prev_F+tol) < tol
    predicted_tensor = None
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_fval = 1
    current_fval = 0
    while (not iteration_condition(iterations)) and (not convergence_condition(previous_fval, current_fval, convergence_tolerance)):
        #Obtain tensor Z from original paper (changes across iterations)
        predicted_tensor = tl.tucker_tensor.tucker_to_tensor(TKD_estimate)
        tensor_Z = tl.tensor(np.multiply(weighting_tensor, predicted_tensor))
        #Obtain squared norm of tensor Z
        Z_sq_norm = tl.tenalg.inner(tensor_Z, tensor_Z)
        #Obtain function value
        previous_fval = current_fval
        current_fval = 0.5*Y_sq_norm + 0.5*Z_sq_norm - tl.tenalg.inner(tensor_Y, tensor_Z)
        #Difference between tensors Y and Z
        tensor_T = tensor_Y - tensor_Z
        #Gradient update of each factor matrix wrt objective function.
        for mode in range(Ndims):
            leave_one_out_factors = TKD_factors[0:mode] + TKD_factors[mode+1:]
            continued_product = tl.tenalg.kronecker(leave_one_out_factors)
            gradient_intermediate = -np.matmul(tl.unfold(tensor_T, mode=mode), continued_product)
            gradient = np.matmul(gradient_intermediate, tl.unfold(TKD_core, mode=mode).T)
            TKD_factors[mode] = TKD_factors[mode] - stepsize*gradient
        #Gradient update of core tensor with respect to objective function
        factors_outer_product = tl.tenalg.outer(TKD_factors)
        #Reorder dimensions so that the first Ndims dimensions can be flattened into one
        new_axes_order = [2*i for i in range(Ndims)] + [2*i+1 for i in range(Ndims)]
        reordered_axes = np.transpose(factors_outer_product, axes=new_axes_order)
        partially_flattened = reordered_axes.reshape(-1, *reordered_axes.shape[-Ndims:])
        flattened_T = tensor_T.flatten()
        gradient = tl.tenalg.mode_dot(partially_flattened, flattened_T, mode=0)
        TKD_core = TKD_core - stepsize*gradient
        iterations+=1
    return predicted_tensor, current_fval, iterations
#=============================================================================================


#=============================================================================================
#Ket augmentation
def KA(image, child):
    dim1, dim2, dim3 = image.shape
    dimn = int(np.log(dim1) / np.log(child) + np.log(dim2) / np.log(child))
    newdim = []
    for i in range(dimn):
        newdim.append(child)
    newdim.append(dim3)
    highordertensor = np.zeros(newdim)
    d = int(child ** 0.5)
    indmat = np.arange(0, child, 1).reshape([d, d], order = 'F').astype(int)
    newind = np.zeros((dim1, dim2, dimn)).astype(int)
    for i in range(dim1):
        for j in range(dim2):
            x = i
            y = j
            for k in range(dimn):
                indx = int(x % d)
                indy = int(y % d)
                newind[i, j, dimn - k - 1] = indmat[indx, indy]
                x = x // d
                y = y // d 
            indtuple = tuple(map(tuple, newind[i, j, :].reshape(len(newind[0,0,:]),1)))
            highordertensor[indtuple] = image[i, j, :]
    return highordertensor, newind

#Reverse ket augmentation
def xind2mul(vec, child):
    newvec = np.zeros(len(vec))
    d = int(child ** 0.5)
    xind = np.zeros((d, d)).astype(int)
    for i in range(d):
        xind[:, i] = i
    xind = xind.reshape([child])
    for i in range(len(vec)):
        newvec[i] = xind[vec[i]]
    return newvec

def yind2mul(vec, child):
    newvec = np.zeros(len(vec))
    d = int(child ** 0.5)
    xind = np.zeros((d, d)).astype(int)
    for i in range(d):
        xind[i, :] = i
    xind = xind.reshape([child])
    for i in range(len(vec)):
        newvec[i] = xind[vec[i]]
    return newvec

def invKA(tensor, tind):
    lastdim = tensor.shape[-1]
    child = tensor.shape[0]
    dim1, dim2, dimn = tind.shape
    d = int(child ** 0.5)
    weightd = np.ones(dimn)
    for i in range(dimn):
        weightd[i] = d ** (dimn - 1 - i)

    image = np.zeros((dim1, dim2, lastdim)).astype(int)
    for i in range(dim1):
        for j in range(dim2):
            newind = tind[i, j, :]
            x = int(np.matmul(xind2mul(newind, child), weightd))
            y = int(np.matmul(yind2mul(newind, child), weightd))
            indtuple = tuple(map(tuple, newind.reshape(len(newind),1)))
            image[x, y, :] = tensor[indtuple]
    return image


def tensorcomplete_TMac_TT(np_array, known_indices, rank_list, convergence_tolerance=1e-8, **kwargs):
    #INITIALISATION-----------------------
    #Generate the alpha weights by generating the delta values. In the same loop, generate initial U and V matrices.
    dimension_tuple = np.shape(np_array)
    dimension_list = list(dimension_tuple)
    Ndims = len(dimension_list)
    deltas = [0]*(Ndims-1)
    U_matrices = []
    V_matrices = []
    X_unfoldings = []
    delta_sum = 0
    print('ITERATION 0---------------------------------------------------------------')
    for k in range(1, Ndims):
        print(f'K {k}---------------------------------------------------------------')
        array_k = k - 1
        dim1 = np.multiply.reduce(dimension_list[:array_k+1])
        dim2 = np.multiply.reduce(dimension_list[array_k+1:])
        rank = rank_list[array_k]
        X_k = np.reshape(np_array, newshape=(dim1, dim2))
        X_unfoldings.append(X_k)
        U_matrices.append(np.random.normal(size=(dim1,rank)))
        V_matrices.append(np.random.normal(size=(rank,dim2)))
        print(np.linalg.norm(X_k - (U_matrices[array_k] @ V_matrices[array_k])))
        deltas[array_k] = min(dim1, dim2)
        delta_sum += deltas[array_k]
    normalise = lambda a : a/delta_sum
    alphas = list(map(normalise, deltas))
    #ITERATIONS----------------------------
    #Used to set an iteration limit
    iteration_condition = lambda i: False
    if 'iteration_limit' in kwargs.keys():
        iteration_condition = lambda i: i >= kwargs['iteration_limit']
    #The condition for convergence
    norm_T = np.linalg.norm(np_array)
    def convergence_condition(prev_F, curr_F, tol):
        return abs(prev_F - curr_F)/(norm_T+tol) < tol
    predicted_tensor = None
    iterations = 0
    #Used to hold previous and current values of objective function
    previous_norm = norm_T
    current_norm = 0
    while (not iteration_condition(iterations)) and (not convergence_condition(previous_norm, current_norm, convergence_tolerance)):
        print(f'ITERATION {iterations + 1}---------------------------------------------------------------')
        #Update matricised tensors and matrices
        predicted_tensor = np.zeros(shape=dimension_tuple)
        for k in range(1, Ndims):
            array_k = k - 1
            #Obtain unfolded tensor X
            X = X_unfoldings[array_k]
            #Obtain U and V matrices
            U = U_matrices[array_k]
            V = V_matrices[array_k]
            # First matrix step
            new_U = X @ V.T
            #Second matrix step
            new_V = np.linalg.pinv((new_U.T @ new_U)) @ new_U.T @ X
            #Third matrix step
            new_X = new_U @ new_V
            #Update X unfoldings and U and V matrices
            U_matrices[array_k] = new_U
            V_matrices[array_k] = new_V
            #Fold X
            folded_X = np.reshape(new_X, newshape=dimension_tuple)
            alpha = alphas[array_k] 
            predicted_tensor += alpha*folded_X
        #Set the known elements
        for index in known_indices:
            predicted_tensor[index] = np_array[index]

        #Update objective function
        previous_norm = current_norm
        current_norm = np.linalg.norm(predicted_tensor)

        #Update X unfolding matrices
        for k in range(1, Ndims):
            print(f'K {k}---------------------------------------------------------------')
            array_k = k - 1
            dim1 = np.multiply.reduce(dimension_list[:array_k+1])
            dim2 = np.multiply.reduce(dimension_list[array_k+1:])
            X_k = np.reshape(predicted_tensor, newshape=(dim1, dim2))
            X_unfoldings.append(X_k)
            print(np.linalg.norm(X_k - (U_matrices[array_k] @ V_matrices[array_k])))
        
        iterations+=1

    objective = abs(current_norm - previous_norm)/norm_T 
    return predicted_tensor, objective, iterations
#=============================================================================================
