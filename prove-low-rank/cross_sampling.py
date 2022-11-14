import numpy as np
import itertools
import random

#MAIN FUNCTION====Uses the cross technique (Zhang, 2019) to generate the subtensors (body, arm, joint)========================
def cross_sample_tensor(raw_tensor, **kwargs):
    NO_TENSOR_DIMS = len(np.shape(raw_tensor))
    tensor_dimensions_tuple = np.shape(raw_tensor)

    # Obtain the Tucker rank: if no rank is provided, the default is 1 for each dimension-------------------------------------------------
    tucker_rank_list = [1]*NO_TENSOR_DIMS
    if 'tucker_rank_list' in kwargs.keys():
        tucker_rank_list = kwargs['tucker_rank_list']
        
        if len(tucker_rank_list) != NO_TENSOR_DIMS:
            raise(ValueError('The rank list length must be equal to the number of dimensions.'))

    # Generate body----------------------------------------------------------------------------------------------------------------------
    # Generate tensor indices of all elements within the body region
    value_lists = []
    for i in range(len(tucker_rank_list)):
        value_lists.append([el for el in range(tucker_rank_list[i])])
    body_indices = list(itertools.product(*value_lists))

    # Assign to each position in the body region using result of eval_func averaged over multiple trials
    body = np.zeros(tuple(tucker_rank_list))
    for tensor_index in body_indices:
        body[tuple(tensor_index)] = raw_tensor[tuple(tensor_index)]

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
                # Assign to matrix using result of eval_func averaged over multiple trials
                arm_matrix[seq, col] = raw_tensor[new_index]
            col += 1

        arms.append(arm_matrix)
        joints.append(joint_matrix)

    #Calculate the number of elements
    arm_size = 0
    for arm in arms:
        arm_size += arm.size
    joint_size = 0
    for joint in joints:
        joint_size += joint.size

    no_elements = np.size(body) + arm_size - joint_size
            
    return body, joints, arms, no_elements