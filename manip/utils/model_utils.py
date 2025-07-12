import torch
import pytorch3d.transforms as transforms


def wxyz_to_xyzw(input_quat):
    # input_quat: 1 X w X 22 X 4 
    w = input_quat[..., 0:1]
    x = input_quat[..., 1:2]
    y = input_quat[..., 2:3]
    z = input_quat[..., 3:4]

    out_quat = torch.cat((x, y, z, w), dim=-1) # 1 X w X 22 X 4 

    return out_quat 


def xyzw_to_wxyz(input_quat):
    # input_quat: 1 X w X 22 X 4 
    x = input_quat[..., 0:1]
    y = input_quat[..., 1:2]
    z = input_quat[..., 2:3]
    w = input_quat[..., 3:4]

    out_quat = torch.cat((w, x, y, z), dim=-1) # 1 X w X 22 X 4 

    return out_quat


def apply_rotation_to_data(ds, trans2joint, cano_rot_mat, new_obj_rot_mat, curr_x):
    # cano_rot_mat:BS X 3 X 3, convert from the coodinate frame which canonicalize the first frame of a sequence to 
    # the frame that canonicalize the first frame of a window.
    # trans2joint: BS X 3 
    # new_obj_rot_mat: BS X 10(overlapped -length) X 3 X 3
    # curr_x: BS X window_size X D  
    # This function is to convert window data to sequence data. 
    bs, timesteps, _ = curr_x.shape 

    pred_obj_normalized_com_pos = curr_x[:, :, :3] # BS X window_size X 3 
    pred_obj_com_pos = ds.de_normalize_obj_pos_min_max(pred_obj_normalized_com_pos) 
    pred_obj_rel_rot_mat = curr_x[:, :, 3:12].reshape(bs, timesteps, 3, 3) # BS X window_size X 3 X 3, relative rotation wrt current window's first frames's cano.   
    pred_obj_rot_mat = ds.rel_rot_to_seq(pred_obj_rel_rot_mat, new_obj_rot_mat) # BS X window_size X 3 X 3 
    pred_human_normalized_jpos = curr_x[:, :, 12:12+24*3] # BS X window_size X (24*3)
    pred_human_jpos = ds.de_normalize_jpos_min_max(pred_human_normalized_jpos.reshape(bs, timesteps, 24, 3)) # BS X window_size X 24 X 3 
    pred_human_rot_6d = curr_x[:, :, 12+24*3:] # BS X window_size X (22*6) 

    pred_human_rot_mat = transforms.rotation_6d_to_matrix(pred_human_rot_6d.reshape(bs, timesteps, 22, 6)) # BS X T X 22 X 3 X 3 

    converted_obj_com_pos = torch.matmul(
        cano_rot_mat[:, None, :, :].repeat(1, timesteps, 1, 1).transpose(2, 3),
        pred_obj_com_pos[:, :, :, None]
    ).squeeze(-1) # BS X window_size X 3 

    converted_obj_rot_mat = torch.matmul(
        cano_rot_mat[:, None, :, :].repeat(1, timesteps,1, 1).transpose(2, 3),
        pred_obj_rot_mat
    ) # BS X window_size X 3 X 3 
    
    converted_human_jpos = torch.matmul(
        cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 24, 1, 1).transpose(3, 4),
        pred_human_jpos[:, :, :, :, None]
    ).squeeze(-1) # BS X T X 24 X 3 
    converted_rot_mat = torch.matmul(
        cano_rot_mat[:, None, None, :, :].repeat(1, timesteps, 22, 1, 1).transpose(3, 4),
        pred_human_rot_mat
    ) # BS X T X 22 X 3 X 3 

    converted_rot_6d = transforms.matrix_to_rotation_6d(converted_rot_mat) 

    return converted_obj_com_pos, converted_obj_rot_mat, converted_human_jpos, converted_rot_6d 


def calculate_obj_kpts_in_wrist(
    global_wrist_jpos,
    global_joint_rot_mat,
    seq_obj_kpts
):
    """
    global_wrist_jpos: BS X T X 2 X 3
    global_joint_rot_mat: BS X T X 2 X 3 X 3
    seq_obj_kpts: BS X T X K X 3
    
    pred_seq_obj_kpts_in_hand: BS X T X 2 X K X 3
    """
    K = seq_obj_kpts.shape[2]
    
    global_wrist_jpos_expand = global_wrist_jpos.unsqueeze(3).repeat(1, 1, 1, K, 1)  # BS X T X 2 X K X 3
    seq_obj_kpts_expand = seq_obj_kpts.unsqueeze(2).repeat(1, 1, 2, 1, 1) # BS X T X 2 X K X 3
    
    # (inv(R) * (p - t)^T)^T = (R^T * (p - t)^T)^T = (p - t) * R
    seq_obj_kpts_in_hand = torch.matmul(
        seq_obj_kpts_expand - global_wrist_jpos_expand, global_joint_rot_mat
    )
    
    return seq_obj_kpts_in_hand
    