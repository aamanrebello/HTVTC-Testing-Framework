from tensorsearch import findBestValues
import tensorly as tl
import numpy as np
import random
import itertools

#SUPPORT FUNCTION====Convert tensor index to corresponding hyperparameter values based on range dict=============
def indexToHyperparameter(index, value_lists):
    hyperparameter_values = {}
    # Obtain the tensor index and the hyperparameter values to pass to eval_func
    dimension_index = 0
    for key in value_lists.keys():
        value_list = value_lists[key]
        if len(value_list) == 1:
            hyperparameter_values[key] = value_list[0]
        else:
            value_index = int(index[dimension_index])
            hyperparameter_values[key] = value_list[value_index]
            dimension_index += 1
    return hyperparameter_values

#MAIN FUNCTION====Uses the cross technique (Zhang, 2019) to generate the subtensors (body, arm, joint)========================
def generateCrossComponents(eval_func, ranges_dict, metric, **kwargs):

    # Obtain the evaluation mode for the machine learning model (prediction, probability or raw score)----------------------------------
    evaluation_mode = 'prediction'
    if 'evaluation_mode' in kwargs.keys():
        evaluation_mode = kwargs['evaluation_mode']
    # Obtain the number of evaluation function trials (default 1)------------------------------------------------------------------------
    eval_trials = 1
    if 'eval_trials' in kwargs.keys():
        eval_trials = kwargs['eval_trials']

    # Obtain the full lists of values for each hyperparameter from the provided ranges/provided list of values--------------------------
    hyperparameter_values = {}
    tensor_dimensions_list = []
    tensor_elements = 1

    for key in ranges_dict.keys():
        info = ranges_dict[key]
        value_list = None
        #If the values are already provided as a list
        if 'values' in info.keys():
            value_list = info['values']
        #If the values are provided as a range
        else:
            start = float(info['start'])
            end = float(info['end'])
            interval = float(info['interval'])
            value_list = np.linspace(start, end, int(round((end-start)/interval, 0))+1)
        # Update outer values
        hyperparameter_values[key] = value_list
        if len(value_list) == 1:
            continue
        tensor_dimensions_list.append(len(value_list))
        tensor_elements *= len(value_list)

    # Obtain the Tucker rank: if no rank is provided, the default is 1 for each dimension-------------------------------------------------
    tucker_rank_list = [1]*len(tensor_dimensions_list)
    if 'tucker_rank_list' in kwargs.keys():
        tucker_rank_list = kwargs['tucker_rank_list']
        
        if len(tucker_rank_list) != len(tensor_dimensions_list):
            raise(ValueError('The rank list length must be equal to the number of dimensions.'))
        
    tensor_dimensions_tuple = tuple(tensor_dimensions_list)

    # Generate body----------------------------------------------------------------------------------------------------------------------
    # Generate tensor indices of all elements within the body region
    value_lists = []
    for i in range(len(tucker_rank_list)):
        value_lists.append([el for el in range(tucker_rank_list[i])])
    body_indices = list(itertools.product(*value_lists))

    # Assign to each position in the body region using result of eval_func averaged over multiple trials
    body = np.zeros(tuple(tucker_rank_list))
    for tensor_index in body_indices:
        current_hyperparameter_values = indexToHyperparameter(tensor_index, hyperparameter_values)
        eval_result_avg = 0
        for trial in range(eval_trials):
            eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
        body[tuple(tensor_index)] = eval_result_avg

    # Generate arms and joints------------------------------------------------------------------------------------------------------------
    arms = []
    joints = []
    # Find the arms and joints along each dimension
    for dimension_index in range(len(tensor_dimensions_tuple)):
        dimension_rank = tucker_rank_list[dimension_index]
        dimension_size = tensor_dimensions_tuple[dimension_index]
        truncated_rank_list = tucker_rank_list[:dimension_index] + tucker_rank_list[dimension_index+1:]
        #Generate indices of all possible fibres along the dimension that could generate the arms and hinges
        value_lists = []
        for i in range(len(truncated_rank_list)):
            value_lists.append([el for el in range(truncated_rank_list[i])])
        joint_base_indices = list(itertools.product(*value_lists))
        #Randomly sample from the fibre indices to choose the arms and hinges
        no_samples = min(len(joint_base_indices), dimension_rank)
        sampled_indices = random.sample(joint_base_indices, no_samples)
        sorted_samples = sorted(sampled_indices)
        # Generate joint and arm matrices based on the sampled fibres
        joint_matrix = np.zeros((dimension_rank, no_samples))
        arm_matrix = np.zeros((dimension_size, no_samples))
        #col tracks the index of the column of the arm/joint matrix that is being written to
        col = 0
        for base_index in sorted_samples:
            # Generate joint and part of arm from body values
            for seq in range(dimension_rank):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                joint_matrix[seq, col] = body[new_index]
                arm_matrix[seq, col] = body[new_index]
            # Generate rest of the arm by evaluating the values
            for seq in range(dimension_rank, dimension_size):
                new_index = base_index[:dimension_index] + (seq,) + base_index[dimension_index:]
                current_hyperparameter_values = indexToHyperparameter(new_index, hyperparameter_values)
                # Assign to matrix using result of eval_func averaged over multiple trials
                eval_result_avg = 0
                for trial in range(eval_trials):
                    eval_result_avg += eval_func(**current_hyperparameter_values, metric=metric, evaluation_mode=evaluation_mode)/eval_trials
                arm_matrix[seq, col] = eval_result_avg
            col += 1

        arms.append(arm_matrix)
        joints.append(joint_matrix)
            
    return body, joints, arms

#MAIN FUNCTION====Noiseless reconstruction of the original tensor according to Zhang, 2019=============
def noiselessReconstruction(body, joint_matrices, arm_matrices):
    R_matrices = []
    for index in range(len(joint_matrices)):
        Y_arm_i = arm_matrices[index]
        Y_joint_i = joint_matrices[index]
        R_i = np.matmul(Y_arm_i, np.linalg.pinv(Y_joint_i))
        R_matrices.append(R_i)
    return tl.tucker_tensor.tucker_to_tensor((body, R_matrices))


#MAIN FUNCTION====Reconstruction accounting for noise according to Zhang, 2019==========================
def noisyReconstruction(body, joint_matrices, arm_matrices):
    R_matrices = []
    #Iterate over tensor dimensions
    for index in range(len(joint_matrices)):
        #Prepare matrices
        body_unfolding = tl.unfold(body, mode=index)
        joint_matrix = joint_matrices[index]
        arm_matrix = arm_matrices[index]
        #Calculate SVDs
        U, _, _ = np.linalg.svd(body_unfolding, full_matrices=True)
        _, _, Vh = np.linalg.svd(arm_matrix, full_matrices=True)
        #V contains the right singular vectors as columns
        V = Vh.T
        #Perform rotations
        A = np.matmul(arm_matrix, V)
        J = np.matmul(np.matmul(U.T, joint_matrix), V)
        #Identify value of r by varying s
        max_s = min(np.shape(J))
        r = 0
        product = None
        for s in range(max_s, 0, -1):
            #Condition 1
            trunc_J = J[:s, :s]
            if np.linalg.matrix_rank(trunc_J) < s:
                continue
            #Condition 2
            trunc_A = A[:, :s]
            # Lambda calculated using recommended c=3
            lambda_val = 3*(A.shape[0])/(J.shape[0])
            J_trunc_inv = np.linalg.inv(trunc_J)
            product = np.matmul(trunc_A, J_trunc_inv)
            #Calculate matrix spectral norm via svd (largest singular value)
            _, svals, _ = np.linalg.svd(product)
            spectral_norm = max(svals)
            if spectral_norm > lambda_val:
                continue
            r = s
        #Calculate R
        trunc_V = V[:r, :]
        R = np.matmul(product, trunc_V)
        R_matrices.append(R)
    return tl.tucker_tensor.tucker_to_tensor((body, R_matrices))
