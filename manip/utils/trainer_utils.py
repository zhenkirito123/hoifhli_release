import json
import os
import pickle
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
from pysdf import SDF
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from sklearn.cluster import DBSCAN

from manip.data.cano_traj_dataset import CanoObjectTrajDataset
from manip.data.humanml3d_dataset import quat_fk_torch
from manip.ik.smplx_ik import IK_Engine, SourceKeyPoints
from manip.inertialize.inert import (
    apply_inertialize,
)
from manip.lafan1.utils import (
    normalize,
    quat_between,
    quat_normalize,
    rotate_at_frame_w_obj,
)

# from manip.utils.smplx_to_phys import convert_smplx_to_phys


def determine_floor_height_and_contacts(body_joint_seq, fps=30):
    """
    Input: body_joint_seq N x 22 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    """
    FLOOR_VEL_THRESH = 0.005
    FLOOR_HEIGHT_OFFSET = 0.01

    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, 0, :]
    left_toe_seq = body_joint_seq[:, 10, :]
    right_toe_seq = body_joint_seq[:, 11, :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(
        left_static_foot_heights, right_static_foot_heights
    )
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(
            all_static_foot_heights.reshape(-1, 1)
        )
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)

        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )  # inds in the original sequence that correspond to this cluster

            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median
        offset_floor_height = (
            floor_height - FLOOR_HEIGHT_OFFSET
        )  # toe joint is actually inside foot mesh a bit

    else:
        floor_height = offset_floor_height = 0.0

    return floor_height


def export_to_ply(points, filename="output.ply"):
    # Open the file in write mode
    with open(filename, "w") as ply_file:
        # Write the PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Created by YourProgram\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        # Write the points data
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")


def zip_folder(folder_path, zip_path):
    try:
        subprocess.run(["rm", "-rf", zip_path], check=True)
        subprocess.run(
            ["zip", "-r", zip_path, folder_path[folder_path.rfind("/") + 1 :]],
            check=True,
            cwd=os.path.join(folder_path, ".."),
        )
        print(
            f"Folder '{folder_path}' has been compressed to '{zip_path}' successfully."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def run_smplx_model(
    root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True
):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(
            aa_rot_rep.device
        )  # BS X T X 30 X 3
        aa_rot_rep = torch.cat(
            (aa_rot_rep, padding_zeros_hand), dim=2
        )  # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(bs * num_steps, -1, 3)  # (BS*T) X n_joints X 3

    betas = (
        betas[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)
    )  # (BS*T) X 16
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3
    smpl_betas = betas  # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90

    B = smpl_trans.shape[0]  # (BS*T)

    smpl_vals = [
        smpl_trans,
        smpl_root_orient,
        smpl_betas,
        smpl_pose_body,
        smpl_pose_hand,
    ]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ["male", "female", "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64) * -1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(
            prev_nbidx, prev_nbidx + nbidx, dtype=np.int64
        )
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue

        # reconstruct SMPL
        (
            cur_pred_trans,
            cur_pred_orient,
            cur_betas,
            cur_pred_pose,
            cur_pred_pose_hand,
        ) = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(
            pose_body=cur_pred_pose,
            pose_hand=cur_pred_pose_hand,
            betas=cur_betas,
            root_orient=cur_pred_orient,
            trans=cur_pred_trans,
        )

        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0)  # () X 52 X 3
        lmiddle_index = 28
        rmiddle_index = 43
        x_pred_smpl_joints = torch.cat(
            (
                x_pred_smpl_joints_all[:, :22, :],
                x_pred_smpl_joints_all[:, lmiddle_index : lmiddle_index + 1, :],
                x_pred_smpl_joints_all[:, rmiddle_index : rmiddle_index + 1, :],
            ),
            dim=1,
        )
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]

    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]  # (BS*T) X 22 X 3

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map]  # (BS*T) X 6890 X 3

    x_pred_smpl_joints = x_pred_smpl_joints.reshape(
        bs, num_steps, -1, 3
    )  # BS X T X 22 X 3/BS X T X 24 X 3
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(
        bs, num_steps, -1, 3
    )  # BS X T X 6890 X 3

    mesh_faces = pred_body.f

    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces


def cycle(dl):
    while True:
        for data in dl:
            yield data


def calc_rot_diff_in_aa(rot_mat, target_rot_mat):
    """
    rot_mat: T X 3 X 3
    target_rot_mat: 1 X 3 X 3
    """
    target_rot_mat = transforms.quaternion_to_matrix(
        transforms.matrix_to_quaternion(target_rot_mat)
    )
    # target_rot_mat = target_rot_mat.repeat(rot_mat.shape[0], 1, 1)
    aa = transforms.matrix_to_axis_angle(
        rot_mat.transpose(-1, -2) @ target_rot_mat
    )  # T X 3
    return aa / np.pi * 180.0


def calc_wrist_object_static_error(
    obj_com_pos,
    obj_rot_mat,
    left_wrist_pos,
    left_wrist_rot_mat,
    right_wrist_pos,
    right_wrist_rot_mat,
    target_left_wrist_pos=None,
    target_left_wrist_rot_mat=None,
    target_right_wrist_pos=None,
    target_right_wrist_rot_mat=None,
    left_contact_label=None,
    right_contact_label=None,
    show_error=True,
):
    """
    obj_com_pos: T X 3, obj_rot_mat: T X 3 X 3, left_wrist_pos: T X 3, left_wrist_rot_mat: T X 3 X 3, right_wrist_pos: T X 3, right_wrist_rot_mat: T X 3 X 3
    target_left_wrist_pos: 1 X 3, target_left_wrist_rot_mat: 1 X 3 X 3
    left_contact_label: T X 1, right_contact_label: T X 1
    """
    left_wrist_pos_in_obj = (
        obj_rot_mat.transpose(1, 2) @ (left_wrist_pos - obj_com_pos).unsqueeze(-1)
    )[..., 0].cpu()  # T X 3
    right_wrist_pos_in_obj = (
        obj_rot_mat.transpose(1, 2) @ (right_wrist_pos - obj_com_pos).unsqueeze(-1)
    )[..., 0].cpu()  # T X 3
    left_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ left_wrist_rot_mat
    ).cpu()  # T X 3 X 3
    right_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ right_wrist_rot_mat
    ).cpu()  # T X 3 X 3

    if target_left_wrist_pos is None:
        target_left_wrist_pos = torch.mean(
            left_wrist_pos_in_obj, dim=0, keepdim=True
        )  # 1 X 3
    if target_right_wrist_pos is None:
        target_right_wrist_pos = torch.mean(
            right_wrist_pos_in_obj, dim=0, keepdim=True
        )  # 1 X 3
    if target_left_wrist_rot_mat is None:
        target_left_wrist_rot_mat = torch.mean(
            left_wrist_rot_mat_in_obj, dim=0, keepdim=True
        )  # 1 X 3 X 3
    if target_right_wrist_rot_mat is None:
        target_right_wrist_rot_mat = torch.mean(
            right_wrist_rot_mat_in_obj, dim=0, keepdim=True
        )  # 1 X 3 X 3

    if left_contact_label is not None:
        left_wrist_pos_err = torch.mean(
            torch.norm(
                left_contact_label * (left_wrist_pos_in_obj - target_left_wrist_pos),
                dim=-1,
            )
        )  # 1
        left_wrist_rot_err = torch.mean(
            torch.norm(
                left_contact_label
                * calc_rot_diff_in_aa(
                    left_wrist_rot_mat_in_obj, target_left_wrist_rot_mat
                ),
                dim=-1,
            )
        )
    else:
        left_wrist_pos_err = torch.mean(
            torch.norm(left_wrist_pos_in_obj - target_left_wrist_pos, dim=-1)
        )  # 1
        left_wrist_rot_err = torch.mean(
            torch.norm(
                calc_rot_diff_in_aa(
                    left_wrist_rot_mat_in_obj, target_left_wrist_rot_mat
                ),
                dim=-1,
            )
        )
    if right_contact_label is not None:
        right_wrist_pos_err = torch.mean(
            torch.norm(
                right_contact_label * (right_wrist_pos_in_obj - target_right_wrist_pos),
                dim=-1,
            )
        )
        right_wrist_rot_err = torch.mean(
            torch.norm(
                right_contact_label
                * calc_rot_diff_in_aa(
                    right_wrist_rot_mat_in_obj, target_right_wrist_rot_mat
                ),
                dim=-1,
            )
        )
    else:
        right_wrist_pos_err = torch.mean(
            torch.norm(right_wrist_pos_in_obj - target_right_wrist_pos, dim=-1)
        )
        right_wrist_rot_err = torch.mean(
            torch.norm(
                calc_rot_diff_in_aa(
                    right_wrist_rot_mat_in_obj, target_right_wrist_rot_mat
                ),
                dim=-1,
            )
        )

    if show_error:
        print(
            "Left wrist pos err: {0}, Right wrist pos err: {1}".format(
                left_wrist_pos_err, right_wrist_pos_err
            )
        )
        print(
            "Left wrist rot err: {0}, Right wrist rot err: {1}".format(
                left_wrist_rot_err, right_wrist_rot_err
            )
        )
    return (
        left_wrist_pos_err,
        right_wrist_pos_err,
        left_wrist_rot_err,
        right_wrist_rot_err,
    )


def calc_object_static_error(
    obj_com_pos,
    obj_rot_mat,
    target_obj_com_pos,
    target_obj_rot_mat,
    object_static_flag=None,
    show_error=True,
):
    """
    obj_com_pos: T X 3
    obj_rot_mat: T X 3 X 3
    target_obj_com_pos: T X 3
    target_obj_rot_mat: T X 3 X 3
    """
    if object_static_flag is not None:
        obj_pos_error = torch.mean(
            torch.norm(object_static_flag * (obj_com_pos - target_obj_com_pos), dim=-1)
        )
        obj_rot_error = torch.mean(
            torch.norm(
                object_static_flag
                * calc_rot_diff_in_aa(obj_rot_mat, target_obj_rot_mat),
                dim=-1,
            )
        )
    else:
        obj_pos_error = torch.mean(torch.norm(obj_com_pos - target_obj_com_pos, dim=-1))
        obj_rot_error = torch.mean(
            torch.norm(calc_rot_diff_in_aa(obj_rot_mat, target_obj_rot_mat), dim=-1)
        )
    if show_error:
        print(
            "Object pos error: {0}, Object rot error: {1}".format(
                obj_pos_error, obj_rot_error
            )
        )
    return obj_pos_error, obj_rot_error


def mirror_rot_6d(rot_6d):
    """
    rot_6d: BS X (T * 6)
    """
    rot_6d[..., 1::6] *= -1
    rot_6d[..., 2::6] *= -1
    rot_6d[..., 3::6] *= -1
    return rot_6d


def canonicalize_first_human_and_waypoints(
    first_human_pose,
    seq_waypoints_pos,
    trans2joint,
    parents,
    trainer,
    is_interaction: bool = False,
):
    # first_human_pose: BS X 1 X D
    # seq_waypoints_pos: BS X T X 3
    # trans2joint: BS X 3
    # parents:
    human_jpos = first_human_pose[:, :, : 24 * 3].reshape(-1, 1, 24, 3)

    human_rot_6d = first_human_pose[:, :, 24 * 3 :].reshape(-1, 1, 22, 6)
    human_rot_mat = transforms.rotation_6d_to_matrix(
        human_rot_6d
    )  # BS X 1 X 22 X 3 X 3
    human_rot_q = transforms.matrix_to_quaternion(human_rot_mat)  # BS X 1 X 22 X 4

    new_human_jpos, new_human_rot_q, _, _ = rotate_at_frame_w_obj(
        human_jpos.detach().cpu().numpy(),
        human_rot_q.detach().cpu().numpy(),
        human_jpos[:, :, 0, :].detach().cpu().numpy(),
        human_rot_q[:, :, 0, :].detach().cpu().numpy(),
        trans2joint.detach().cpu().numpy(),
        parents,
        n_past=1,
        floor_z=True,
        use_global_human=True,
    )
    # BS X 1 X 24 X 3, BS X 1 X 22 X 4

    new_human_rot_q = (
        torch.from_numpy(new_human_rot_q).to(first_human_pose.device).float()
    )
    new_human_jpos = (
        torch.from_numpy(new_human_jpos).to(first_human_pose.device).float()
    )

    new_human_rot_mat = transforms.quaternion_to_matrix(
        new_human_rot_q
    )  # BS X 1 X 22 X 3 X 3
    new_human_rot_6d = transforms.matrix_to_rotation_6d(new_human_rot_mat)

    # Move the first root trans to zero.
    global_human_root_jpos = new_human_jpos[:, :, 0, :].clone()  # BS X T(1) X 3
    global_human_root_trans = global_human_root_jpos + trans2joint[:, None, :].to(
        global_human_root_jpos.device
    )  # BS X T(1) X 3

    move_to_zero_trans = global_human_root_trans[
        :,
        0:1,
    ].clone()  # Move the first frame's root joint x, y to 0,  BS X 1 X 3
    move_to_zero_trans[:, :, 2] = 0  # BS X 1 X 3

    global_human_root_trans -= move_to_zero_trans
    global_human_root_jpos -= move_to_zero_trans

    new_human_jpos -= move_to_zero_trans[:, :, None, :]

    new_human_jpos_normalized = trainer.ds.normalize_jpos_min_max(new_human_jpos)

    # Need to normalize human joint positions.
    cano_first_human_pose = torch.cat(
        (
            new_human_jpos_normalized.reshape(-1, 1, 24 * 3),
            new_human_rot_6d.reshape(-1, 1, 22 * 6),
        ),
        dim=-1,
    )  # BS X 1 X (24*3+22*6)

    # The matrix that convert the orientation to canonicalized direction.
    cano_rot_mat = torch.matmul(
        new_human_rot_mat[:, 0, 0, :, :], human_rot_mat[:, 0, 0, :, :].transpose(1, 2)
    )  # BS X 3 X 3

    cano_seq_waypoints_pos = torch.matmul(
        cano_rot_mat[:, None, :, :].repeat(1, seq_waypoints_pos.shape[1], 1, 1),
        seq_waypoints_pos[:, :, :, None],
    ).squeeze(-1)  # BS X T X 3
    cano_seq_waypoints_pos -= move_to_zero_trans

    # Need to normalize waypoints xy
    if is_interaction:  # For interaction, waypoints should decide obj_com_pos
        cano_seq_waypoints_pos = trainer.ds.normalize_obj_pos_min_max(
            cano_seq_waypoints_pos
        )
    else:  # For navigation, waypoints should decide human's root xy
        cano_seq_waypoints_pos = trainer.ds.normalize_specific_jpos_min_max(
            cano_seq_waypoints_pos, 0
        )

    return cano_first_human_pose, cano_seq_waypoints_pos, cano_rot_mat
    # BS X 1 X D, BS X T X 3, BS X 3 X 3


def find_contact_frames(
    interaction_trainer,
    object_name,
    obj_com_pos,
    obj_rot_mat,
    left_wrist_pos,
    left_wrist_rot_mat,
    right_wrist_pos,
    right_wrist_rot_mat,
    async_hand=False,
    contact_labels: Optional[torch.Tensor] = None,
):
    if async_hand:
        return find_contact_frames_async_hand(
            interaction_trainer,
            object_name,
            obj_com_pos,
            obj_rot_mat,
            left_wrist_pos,
            left_wrist_rot_mat,
            right_wrist_pos,
            right_wrist_rot_mat,
            contact_labels=contact_labels,
        )

    left_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (left_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    left_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ left_wrist_rot_mat
    ).cpu()  # T X 3 X 3
    right_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (right_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    right_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ right_wrist_rot_mat
    ).cpu()  # T X 3 X 3
    v_threshold = 0.20
    obj_v = obj_com_pos[1:] - obj_com_pos[:-1]  # (T-1) X 3
    obj_v_norm = torch.norm(obj_v, dim=1) * 30  # T-1
    obj_v_norm_smooth = medfilt(obj_v_norm.detach().cpu().numpy(), kernel_size=9)

    contact_begin_frame = 0
    contact_end_frame = obj_v_norm_smooth.shape[0]
    for i in range(obj_v_norm_smooth.shape[0]):
        if obj_v_norm_smooth[i] > v_threshold:
            contact_begin_frame = i
            break
    for i in range(obj_v_norm_smooth.shape[0] - 1, -1, -1):
        if obj_v_norm_smooth[i] > v_threshold:
            contact_end_frame = i
            break
    # NOTE: more elegent way to find the contact frames??
    WRIST_STATIC_FRAME = 0
    contact_begin_frame = max(2, contact_begin_frame - WRIST_STATIC_FRAME)
    contact_end_frame = min(
        obj_v_norm_smooth.shape[0] - 3, contact_end_frame + WRIST_STATIC_FRAME
    )

    # load obj mesh
    obj_rest_verts, obj_mesh_faces = (
        interaction_trainer.ds.load_rest_pose_object_geometry(object_name)
    )
    obj_rest_verts = torch.from_numpy(obj_rest_verts)  # V X 3

    dis_threshold = 0.15
    left_cnt, right_cnt = 0, 0

    left_dises = []
    right_dises = []
    left_begin_frame, right_begin_frame, left_end_frame, right_end_frame = (
        -1,
        -1,
        -1,
        -1,
    )
    for i in range(left_wrist_pos_in_obj.shape[0]):
        left_dis = torch.min(
            torch.norm(obj_rest_verts - left_wrist_pos_in_obj[i], dim=1)
        )
        right_dis = torch.min(
            torch.norm(obj_rest_verts - right_wrist_pos_in_obj[i], dim=1)
        )
        if left_dis < dis_threshold:
            left_cnt += 1
            if left_begin_frame == -1:
                left_begin_frame = i
        if right_dis < dis_threshold:
            right_cnt += 1
            if right_begin_frame == -1:
                right_begin_frame = i
        left_dises.append(left_dis.item())
        right_dises.append(right_dis.item())
    for i in range(left_wrist_pos_in_obj.shape[0] - 1, -1, -1):
        left_dis = left_dises[i]
        right_dis = right_dises[i]
        if left_dis < dis_threshold:
            if left_end_frame == -1:
                left_end_frame = i
        if right_dis < dis_threshold:
            if right_end_frame == -1:
                right_end_frame = i

    # assert left_end_frame - left_begin_frame > 30
    # assert right_end_frame - right_begin_frame > 30
    left_contact = left_cnt > 15
    right_contact = right_cnt > 15

    if left_contact:
        left_wrist_pos_in_obj_mean = left_wrist_pos_in_obj[
            left_begin_frame + 5 : left_end_frame - 5
        ].mean(dim=0)
        left_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                left_wrist_rot_mat_in_obj[
                    left_begin_frame + 5 : left_end_frame - 5
                ].mean(dim=0)
            )
        )
    else:
        left_wrist_pos_in_obj_mean, left_wrist_rot_mat_in_obj_mean = None, None
    if right_contact:
        right_wrist_pos_in_obj_mean = right_wrist_pos_in_obj[
            right_begin_frame + 5 : right_end_frame - 5
        ].mean(dim=0)
        right_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                right_wrist_rot_mat_in_obj[
                    right_begin_frame + 5 : right_end_frame - 5
                ].mean(dim=0)
            )
        )
    else:
        right_wrist_pos_in_obj_mean, right_wrist_rot_mat_in_obj_mean = None, None

    left_wrist = {
        "wrist_pos": left_wrist_pos_in_obj_mean,
        "wrist_rot": left_wrist_rot_mat_in_obj_mean,
    }
    right_wrist = {
        "wrist_pos": right_wrist_pos_in_obj_mean,
        "wrist_rot": right_wrist_rot_mat_in_obj_mean,
    }

    # pickle.dump(left_wrist, open("left_wrist_contact.pkl", "wb"))
    # pickle.dump(right_wrist, open("right_wrist_contact.pkl", "wb"))

    return (
        left_contact,
        right_contact,
        contact_begin_frame,
        contact_end_frame,
        left_begin_frame,
        left_end_frame,
        right_begin_frame,
        right_end_frame,
        left_wrist,
        right_wrist,
    )


def find_contact_frames_async_hand(
    interaction_trainer,
    object_name,
    obj_com_pos,
    obj_rot_mat,
    left_wrist_pos,
    left_wrist_rot_mat,
    right_wrist_pos,
    right_wrist_rot_mat,
    contact_labels: Optional[torch.Tensor] = None,
):
    left_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (left_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    left_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ left_wrist_rot_mat
    ).cpu()  # T X 3 X 3
    right_wrist_pos_in_obj = (
        (obj_rot_mat.transpose(1, 2) @ (right_wrist_pos - obj_com_pos).unsqueeze(-1))[
            ..., 0
        ]
    ).cpu()  # T X 3
    right_wrist_rot_mat_in_obj = (
        obj_rot_mat.transpose(1, 2) @ right_wrist_rot_mat
    ).cpu()  # T X 3 X 3

    left_contact_labels = contact_labels[0, :, 0]
    right_contact_labels = contact_labels[0, :, 1]
    left_contact = len(left_contact_labels[left_contact_labels > 0.95]) > 15
    right_contact = len(right_contact_labels[right_contact_labels > 0.95]) > 15

    contact_begin_frame, contact_end_frame = 121, -1

    if left_contact:
        left_begin_frame = torch.where(left_contact_labels > 0.95)[0][0].item()
        left_end_frame = torch.where(left_contact_labels > 0.95)[0][-1].item()
        contact_begin_frame = min(left_begin_frame, contact_begin_frame)
        contact_end_frame = max(left_end_frame, contact_end_frame)
    else:
        left_begin_frame, left_end_frame = -1, -1

    if right_contact:
        right_begin_frame = torch.where(right_contact_labels > 0.95)[0][0].item()
        right_end_frame = torch.where(right_contact_labels > 0.95)[0][-1].item()
        contact_begin_frame = min(right_begin_frame, contact_begin_frame)
        contact_end_frame = max(right_end_frame, contact_end_frame)
    else:
        right_begin_frame, right_end_frame = -1, -1

    if left_contact:
        left_wrist_pos_in_obj_mean = left_wrist_pos_in_obj[
            left_contact_labels > 0.95
        ].mean(dim=0)
        left_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                left_wrist_rot_mat_in_obj[left_contact_labels > 0.95].mean(dim=0)
            )
        )
    else:
        left_wrist_pos_in_obj_mean, left_wrist_rot_mat_in_obj_mean = None, None
    if right_contact:
        right_wrist_pos_in_obj_mean = right_wrist_pos_in_obj[
            right_contact_labels > 0.95
        ].mean(dim=0)
        right_wrist_rot_mat_in_obj_mean = transforms.quaternion_to_matrix(
            transforms.matrix_to_quaternion(
                right_wrist_rot_mat_in_obj[right_contact_labels > 0.95].mean(dim=0)
            )
        )
    else:
        right_wrist_pos_in_obj_mean, right_wrist_rot_mat_in_obj_mean = None, None

    left_wrist = {
        "wrist_pos": left_wrist_pos_in_obj_mean,
        "wrist_rot": left_wrist_rot_mat_in_obj_mean,
    }
    right_wrist = {
        "wrist_pos": right_wrist_pos_in_obj_mean,
        "wrist_rot": right_wrist_rot_mat_in_obj_mean,
    }

    return (
        left_contact,
        right_contact,
        contact_begin_frame,
        contact_end_frame,
        left_begin_frame,
        left_end_frame,
        right_begin_frame,
        right_end_frame,
        left_wrist,
        right_wrist,
    )


def smplx_ik(
    target_human_jnts: torch.Tensor,
    human_root_trans: torch.Tensor,
    human_local_rot_aa_reps: torch.Tensor,
    betas: torch.Tensor,
    rest_human_offsets: torch.Tensor,
    right_wrist: bool = False,
    left_wrist: bool = False,
    feet: bool = False,
    comp_device="cuda",
    gender="male",
):
    # human_jnts: global joint position, T X 24 X 3
    # human_root_trans: T X 3
    # human_local_rot_aa_reps: T X 22 X 3
    # betas: 1 X 16
    T = target_human_jnts.shape[0]
    root_orient = human_local_rot_aa_reps[:, 0]  # T X 3
    pose_body = human_local_rot_aa_reps[:, 1:].reshape(-1, 63)  # T X 63
    betas = betas.repeat(T, 1)  # T X 16

    def build_ik_engine(right_wrist, left_wrist):
        data_loss = torch.nn.MSELoss(reduction="mean")
        stepwise_weights = [
            {"data": 10.0, "poZ_body": 0.0, "betas": 0.0},
        ]
        optimizer_args = {
            "type": "ADAM",
            "max_iter": 600,
            "lr": 0.1,
            "tolerance_change": 1e-4,
        }
        ik_engine = IK_Engine(
            verbosity=0,
            display_rc=(2, 2),
            data_loss=data_loss,
            stepwise_weights=stepwise_weights,
            optimizer_args=optimizer_args,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            feet=feet,
        ).to(comp_device)
        return ik_engine

    n_joints = 22
    if gender == "male":
        bm_fname = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/smpl_all_models/smplx/SMPLX_MALE.npz",
        )
    elif gender == "female":
        bm_fname = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/smpl_all_models/smplx/SMPLX_FEMALE.npz",
        )
    elif gender == "neutral":
        bm_fname = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/smpl_all_models/smplx/SMPLX_NEUTRAL.npz",
        )
    else:
        raise ValueError("Invalid gender")

    ik_engine = build_ik_engine(right_wrist, left_wrist)
    source_pts = SourceKeyPoints(
        bm=bm_fname,
        n_joints=n_joints,
    ).to(comp_device)
    target_pts = target_human_jnts[:, :n_joints].detach().to(comp_device)

    initial_body_params = {
        "pose_body": pose_body.detach()
        .type(torch.float)
        .to(comp_device)
        .requires_grad_(False),
        "root_orient": root_orient.detach()
        .type(torch.float)
        .to(comp_device)
        .requires_grad_(False),
        "trans": human_root_trans.detach()
        .type(torch.float)
        .to(comp_device)
        .requires_grad_(False),
        "betas": betas.detach().type(torch.float).to(comp_device).requires_grad_(False),
    }
    print("Doing IK...")
    new_root_orient, new_pose_body, global_pos = ik_engine(
        source_pts, target_pts, initial_body_params
    )  # global_pos: T X 24 X 3

    local_joint_rot_mat = transforms.axis_angle_to_matrix(
        torch.cat((new_root_orient, new_pose_body), dim=1).reshape(T, 22, 3)
    )  # T X 22 X 3 X 3
    curr_seq_local_jpos = rest_human_offsets.repeat(T, 1, 1)  # T X 24 X 3
    curr_seq_local_jpos[:, 0] = human_root_trans
    global_quat, _ = quat_fk_torch(
        local_joint_rot_mat.cuda(), curr_seq_local_jpos.cuda()
    )  # T X 22 X 4
    global_rot_mat = transforms.quaternion_to_matrix(global_quat)
    global_6d = transforms.matrix_to_rotation_6d(global_rot_mat)  # T X 22 X 6

    del ik_engine, source_pts, target_pts, initial_body_params
    torch.cuda.empty_cache()

    return global_6d, global_pos


def navigation_to_interaction_smooth_transition(
    prev_navigation_motion: torch.Tensor,
    all_res_list: torch.Tensor,
    interaction_trainer,
    navigation_trainer,
):
    prev_jpos = navigation_trainer.ds.de_normalize_jpos_min_max(
        prev_navigation_motion[:, :, : 24 * 3].clone().reshape(1, -1, 24, 3)
    )
    prev_rot_6d = (
        prev_navigation_motion[:, :, 24 * 3 : 24 * 3 + 22 * 6]
        .clone()
        .reshape(1, -1, 22, 6)
    )
    converted_human_jpos = interaction_trainer.ds.de_normalize_jpos_min_max(
        all_res_list[:, :, 12 : 12 + 24 * 3].clone().reshape(1, -1, 24, 3)
    )
    converted_rot_6d = (
        all_res_list[:, :, 12 + 24 * 3 : 12 + 24 * 3 + 22 * 6]
        .clone()
        .reshape(1, -1, 22, 6)
    )

    converted_human_jpos, converted_rot_6d, _, _ = apply_inertialize(
        prev_jpos=prev_jpos,
        prev_rot_6d=prev_rot_6d,
        window_jpos=converted_human_jpos,
        window_rot_6d=converted_rot_6d,
        ratio=1.0,
    )

    all_res_list[:, :, 12 : 12 + 24 * 3] = (
        interaction_trainer.ds.normalize_jpos_min_max(converted_human_jpos).reshape(
            1, -1, 24 * 3
        )
    )
    all_res_list[:, :, 12 + 24 * 3 : 12 + 24 * 3 + 22 * 6] = converted_rot_6d.reshape(
        1, -1, 22 * 6
    )
    return all_res_list


def interaction_to_navigation_smooth_transition(
    prev_interaction_motion: torch.Tensor,
    all_res_list: torch.Tensor,
    interaction_trainer,
    navigation_trainer,
):
    prev_jpos = interaction_trainer.ds.de_normalize_jpos_min_max(
        prev_interaction_motion[:, :, 12 : 12 + 24 * 3].clone().reshape(1, -1, 24, 3)
    )
    prev_rot_6d = (
        prev_interaction_motion[:, :, 12 + 24 * 3 : 12 + 24 * 3 + 22 * 6]
        .clone()
        .reshape(1, -1, 22, 6)
    )
    converted_human_jpos = navigation_trainer.ds.de_normalize_jpos_min_max(
        all_res_list[:, :, : 24 * 3].clone().reshape(1, -1, 24, 3)
    )
    converted_rot_6d = (
        all_res_list[:, :, 24 * 3 : 24 * 3 + 22 * 6].clone().reshape(1, -1, 22, 6)
    )

    converted_human_jpos, converted_rot_6d, _, _ = apply_inertialize(
        prev_jpos=prev_jpos,
        prev_rot_6d=prev_rot_6d,
        window_jpos=converted_human_jpos,
        window_rot_6d=converted_rot_6d,
        ratio=1.0,
        prev_blend_time=0.2,
        window_blend_time=0.2,
        zero_velocity=True,
    )

    all_res_list[:, :, : 24 * 3] = navigation_trainer.ds.normalize_jpos_min_max(
        converted_human_jpos
    ).reshape(1, -1, 24 * 3)
    all_res_list[:, :, 24 * 3 : 24 * 3 + 22 * 6] = converted_rot_6d.reshape(
        1, -1, 22 * 6
    )
    return all_res_list


def finger_smooth_transition(
    prev_finger_motion: torch.Tensor,
    finger_all_res_list: torch.Tensor,
):
    prev_rot_6d = prev_finger_motion.reshape(1, -1, 30, 6)  # 1 X T X 180
    converted_rot_6d = finger_all_res_list.reshape(1, -1, 30, 6)  # 1 X T X 180
    _, converted_rot_6d, _, _ = apply_inertialize(
        prev_jpos=None,
        prev_rot_6d=prev_rot_6d,
        window_jpos=None,
        window_rot_6d=converted_rot_6d,
        ratio=1.0,
    )
    finger_all_res_list = converted_rot_6d.reshape(-1, 180)
    return finger_all_res_list


def build_omomo_sdf():
    mesh_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/processed_data/rest_object_geo/",
    )

    object_names = [
        "clothesstand",
        "floorlamp",
        "largetable",
        "mop",
        "plasticbox",
        "suitcase",
        "tripod",
        "vacuum",
        "monitor",
        "trashcan",
        "woodchair",
        "smalltable",
        "whitechair",
        "largebox",
        "smallbox",
    ]
    sdf_dict = {}
    for name in object_names:
        path = os.path.join(mesh_dir, name + ".ply")
        mesh = trimesh.load(path)
        sdf = SDF(mesh.vertices, mesh.faces)
        sdf_dict[name] = sdf
    return sdf_dict


def smooth_res(all_res_list):
    device = all_res_list.device
    num_smaples = all_res_list.shape[0]

    for i in range(num_smaples):
        obj_human_motion = all_res_list[i, :, : 12 + 24 * 3 + 22 * 6]

        obj_rot_mat = obj_human_motion[:, 3:12].reshape(-1, 3, 3)  # T X 3 X 3
        obj_6d = transforms.matrix_to_rotation_6d(obj_rot_mat)  # T X 6
        obj_6d_new = np.apply_along_axis(
            lambda x: gaussian_filter1d(x, sigma=2), 0, obj_6d.detach().cpu().numpy()
        )
        obj_6d_new = torch.from_numpy(obj_6d_new).to(device)
        obj_rot_mat_new = transforms.rotation_6d_to_matrix(obj_6d_new)  # T X 3 X 3

        obj_human_motion = np.apply_along_axis(
            lambda x: gaussian_filter1d(x, sigma=2),
            0,
            obj_human_motion.detach().cpu().numpy(),
        )
        obj_human_motion = torch.from_numpy(obj_human_motion).to(device)

        obj_human_motion[:, 3:12] = obj_rot_mat_new.reshape(-1, 9)

        all_res_list[i, :, : 12 + 24 * 3 + 22 * 6] = obj_human_motion

    return all_res_list


def check_contact_style(contact_labels):
    flag = True
    if torch.any(contact_labels[..., 2:]):
        flag = False
    else:
        cnt = 0
        for i in range(1, contact_labels.shape[0]):
            if contact_labels[i, 0] == 1 and contact_labels[i - 1, 0] == 0:
                cnt += 1
        if cnt > 1:
            flag = False
        cnt = 0
        for i in range(1, contact_labels.shape[0]):
            if contact_labels[i, 1] == 1 and contact_labels[i - 1, 1] == 0:
                cnt += 1
        if cnt > 1:
            flag = False
    return flag


def load_palm_vertex_ids():
    data = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/smpl_all_models/palm_sample_indices.pkl",
            ),
            "rb",
        )
    )

    left_hand_vids = data["left_hand"]
    right_hand_vids = data["right_hand"]

    return left_hand_vids, right_hand_vids


def load_hand_face_ids():
    data = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/smpl_all_models/MANO_SMPLX_face_ids.pkl",
            ),
            "rb",
        )
    )

    left_hand_fids = data["left_hand"]
    right_hand_fids = data["right_hand"]

    return left_hand_fids, right_hand_fids


def decide_no_force_closure_from_objects(object_name):
    if object_name in [
        "largebox",
        "plasticbox",
        "trashcan",
        "smallbox",
        "suitcase",
    ]:
        no_fc = True
    elif object_name in [
        "clothesstand",
        "largetable",
        "whitechair",
        "floorlamp",
        "monitor",
        "smalltable",
        "tripod",
        "woodchair",
    ]:
        no_fc = False
    else:
        raise ValueError(f"Unknown object name: {object_name}")
    return no_fc


def build_object_static_flag(
    contact_begin_frame_list,
    contact_end_frame_list,
    window_size: int = 120,
):
    n = len(contact_begin_frame_list)
    object_static_flag = torch.ones((n, window_size, 1))
    for i in range(n):
        begin_frame = contact_begin_frame_list[i]
        end_frame = contact_end_frame_list[i]
        object_static_flag[i, begin_frame:end_frame] = 0
    return object_static_flag


def build_wrist_relative(
    right_wrist_pos_in_obj_all: torch.Tensor,
    right_wrist_rot_mat_in_obj_all: torch.Tensor,
    left_wrist_pos_in_obj_all: torch.Tensor,
    left_wrist_rot_mat_in_obj_all: torch.Tensor,
    reference_obj_rot_mat_list: torch.Tensor,
    left_contact_list: List[bool],
    right_contact_list: List[bool],
    left_begin_frame_list: List[int],
    left_end_frame_list: List[int],
    right_begin_frame_list: List[int],
    right_end_frame_list: List[int],
    dataset: CanoObjectTrajDataset,
    window_size: int = 120,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build wrist relative pose and mask.

    Args:
        right_wrist_pos_in_obj_all: (N, 3)
        right_wrist_rot_mat_in_obj_all: (N, 3, 3)
        left_wrist_pos_in_obj_all: (N, 3)
        left_wrist_rot_mat_in_obj_all: (N, 3, 3)
        reference_obj_rot_mat: (N, 3, 3)
        window_size: int

    Returns:
        wrist_relative: (N, window_size, 18)
        wrist_relative_mask: (N, window_size, 18)
    """
    n = right_wrist_pos_in_obj_all.shape[0]
    device = right_wrist_pos_in_obj_all.device

    wrist_relative = torch.zeros(n, window_size, 18)
    wrist_relative_mask = torch.ones(n, window_size, 18)

    for i in range(n):
        ref_rot_mat = reference_obj_rot_mat_list[i][0].to(device)  # 3 X 3

        left_wrist_pos_in_obj = left_wrist_pos_in_obj_all[i]  # 3
        left_wrist_rot_mat_in_obj = left_wrist_rot_mat_in_obj_all[i]  # 3 X 3
        right_wrist_pos_in_obj = right_wrist_pos_in_obj_all[i]  # 3
        right_wrist_rot_mat_in_obj = right_wrist_rot_mat_in_obj_all[i]  # 3 X 3

        new_left_wrist_pos_in_obj = ref_rot_mat @ left_wrist_pos_in_obj  # 3
        new_left_wrist_rot_mat_in_obj = ref_rot_mat @ left_wrist_rot_mat_in_obj  # 3 X 3
        new_right_wrist_pos_in_obj = ref_rot_mat @ right_wrist_pos_in_obj  # 3
        new_right_wrist_rot_mat_in_obj = (
            ref_rot_mat @ right_wrist_rot_mat_in_obj
        )  # 3 X 3

        new_wrist_relative = torch.cat(
            (
                dataset.normalize_wrist_relative_pos(new_left_wrist_pos_in_obj),
                transforms.matrix_to_rotation_6d(new_left_wrist_rot_mat_in_obj),
                dataset.normalize_wrist_relative_pos(new_right_wrist_pos_in_obj),
                transforms.matrix_to_rotation_6d(new_right_wrist_rot_mat_in_obj),
            ),
            dim=-1,
        )  # 18

        wrist_relative[i, :] = new_wrist_relative

        # Build mask.
        left_contact = left_contact_list[i]
        right_contact = right_contact_list[i]
        left_begin_frame = left_begin_frame_list[i]
        left_end_frame = left_end_frame_list[i]
        right_begin_frame = right_begin_frame_list[i]
        right_end_frame = right_end_frame_list[i]

        if left_contact:
            wrist_relative_mask[i, left_begin_frame:left_end_frame, :9] = 0

        if right_contact:
            wrist_relative_mask[i, right_begin_frame:right_end_frame, 9:] = 0

    return wrist_relative, wrist_relative_mask


def build_finger_pose() -> torch.Tensor:
    pass


def canonizalize_planned_path(
    planned_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # We want to canonicalize the direction of the init planned path to align with human's forward direction.
    # planned_points: T X 3
    num_steps = planned_points.shape[0]

    forward = normalize(
        planned_points[2, :].data.cpu().numpy()
        - planned_points[1, :].data.cpu().numpy()
    )
    forward[2] = 0.0
    forward = normalize(forward)
    if abs(forward[1]) < 1e-3:
        if forward[0] > 0:
            yrot = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            yrot = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        yrot = quat_normalize(
            quat_between(forward, np.array([1, 0, 0]))
        )  # 4-dim, from current direction to canonicalized direction
    # yrot = quat_normalize(quat_between(forward, np.array([0, -1, 0]))) # 4-dim, from current direction to canonicalized direction
    cano_quat = torch.from_numpy(yrot).float()[None, :].repeat(num_steps, 1)  # T X 4

    # Apply rotation to the original path.
    canonicalized_path_pts = transforms.quaternion_apply(
        cano_quat, planned_points
    )  # T X 3

    return cano_quat, canonicalized_path_pts


def sample_dense_waypoints_navigation(
    waypoints: np.ndarray, distance: float = 1.0
) -> np.ndarray:
    dis = 0.8
    dense_waypoints = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        last_waypoints = dense_waypoints[-1]
        segment_length = np.linalg.norm(waypoints[i + 1] - last_waypoints)
        if i == 0:
            length = 0.25
            while length < segment_length:
                dense_waypoints.append(
                    last_waypoints
                    + length * (waypoints[i + 1] - last_waypoints) / segment_length
                )
                length += dis
        else:
            length = dis
            while length < segment_length:
                dense_waypoints.append(
                    last_waypoints
                    + length * (waypoints[i + 1] - last_waypoints) / segment_length
                )
                length += dis
        dense_waypoints.append(waypoints[i + 1])

    # Adjust number of waypoints to ensure it's of the desired form
    v = dense_waypoints[-1] - dense_waypoints[-2]
    dense_waypoints.append(dense_waypoints[-1] + v * 0.01)
    dense_waypoints.append(dense_waypoints[-1] + v * 0.01)
    dense_waypoints.append(dense_waypoints[-1] + v * 0.01)
    return np.array(dense_waypoints)


def sample_dense_waypoints_interaction(
    waypoints: np.ndarray,
    distance_range: Tuple[float, float] = (0.6, 0.8),
    remainder: int = 1,
):
    # Validate the remainder value
    assert remainder in [0, 1, 2, 3], "Remainder must be one of [0, 1, 2, 3]."

    # Compute the distances between each consecutive pair of waypoints
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=-1)

    # For each segment, compute the number of intermediate points to insert
    num_points_per_segment = np.ceil(segment_lengths / distance_range[0]) - 1
    num_points_per_segment = np.maximum(
        num_points_per_segment, np.floor(segment_lengths / distance_range[1]) - 1
    ).astype(int)

    # Create dense waypoints by interpolating for each segment
    dense_waypoints = [waypoints[0]]
    for i, num_points in enumerate(num_points_per_segment):
        for j in range(num_points):
            t = (j + 1) / (num_points + 1)
            interpolated_point = (1 - t) * waypoints[i] + t * waypoints[i + 1]
            dense_waypoints.append(interpolated_point)
        dense_waypoints.append(waypoints[i + 1])

    # Adjust number of waypoints to ensure it's of the desired form
    # while len(dense_waypoints) % 4 != remainder:
    #     dense_waypoints.append(dense_waypoints[-1])
    return np.array(dense_waypoints)


def load_planned_path_as_waypoints(
    npy_data,
    use_canonicalization=True,
    load_for_nav=False,
    start_waypoint=None,
):
    # Assunme a human's walking speed is 1.1m/s~1.7m/s.
    # 30 frames (1 second) --- moving distance should be in range [1.1m, 1.7m]
    # In the first 30 frame, usually interact with object, so moving distance could be < 0.5m.
    # In the last 30 frame, usually release the object, so moving distance could be < 0.5m.

    x_data = torch.from_numpy(npy_data[:, 0].copy()).float()[:, None]  # K X 1
    y_data = torch.from_numpy(npy_data[:, 1].copy()).float()[:, None]  # K X 1
    z_data = torch.from_numpy(npy_data[:, 2].copy()).float()[:, None]  # K X 1

    xy_data = torch.cat((x_data, y_data), dim=-1)  # K X 2

    if start_waypoint is not None:
        # For navigation sequence
        # 1. Use previous interaction sequence's end pose to replace 1st waypoint.
        # 2. Remove the last waypoint since next interaction sequence use it as first object com.
        xy_data[0:1, :] = start_waypoint

    if load_for_nav:
        dense_xy_data = sample_dense_waypoints_navigation(
            xy_data.detach().cpu().numpy()
        )
        dense_xy_data = torch.from_numpy(dense_xy_data).float()
    else:
        # For interaction sequence
        # 1st interaction sequence: start waypoint, end waypoint should be repeated once.
        # Since at the begining, the human need to approach the object. At the end, the human need to release the object.
        dense_xy_data = sample_dense_waypoints_interaction(
            xy_data.detach().cpu().numpy(), distance_range=(0.6, 0.8), remainder=2
        )
        dense_xy_data = torch.from_numpy(dense_xy_data).float()
        dense_xy_data = torch.cat(
            (
                dense_xy_data[0:1, :],
                dense_xy_data,
                dense_xy_data[-1:, :],
                dense_xy_data[-1:, :],
            ),
            dim=0,
        )

    new_x_data = dense_xy_data[:, 0:1]
    new_y_data = dense_xy_data[:, 1:2]

    # Note that in navigation, waypoints represent the root translaton xy.
    # In interaction, waypoints represent the object com xy.

    # Tmp manually assign a floor height value
    new_z_data = z_data[0:1].repeat(new_x_data.shape[0], 1)
    new_z_data[-1] = z_data[-1]

    planned_data = torch.cat((new_x_data, new_y_data, new_z_data), dim=-1)  # T X 3

    if use_canonicalization:
        cano_quat, cano_planned_data = canonizalize_planned_path(planned_data)  # T X 3
    else:
        cano_planned_data = planned_data

    if use_canonicalization:
        return cano_quat, cano_planned_data
    else:
        return cano_planned_data


def generate_T_pose(
    rest_human_offsets: torch.Tensor,
    planned_obj_path: torch.Tensor,
) -> torch.Tensor:
    """Generate T pose.

    Navigation mudule uses T pose as the start pose.

    Args:
        rest_human_offsets (torch.Tensor): rest offsets of human joints, the shape is (1, 24, 3).
        planned_obj_path (torch.Tensor): planned object path, the shape is (N, 3).

    Returns:
        torch.Tensor: global joint positions and 6d orientation, the shape is (1, 1, 24*3 + 22*6).
    """
    curr_seq_local_jpos = rest_human_offsets  # 1 X 24 X 3
    curr_seq_local_jpos[0, 0, :2] = planned_obj_path[0, :2]
    curr_seq_local_jpos[0, 0, 2] = 0.905
    local_joint_rot_mat = pickle.load(open("local_joint_rot_mat.pkl", "rb"))[
        0:1
    ]  # 1 X 22 X 3 X 3
    global_quat, global_pos = quat_fk_torch(
        local_joint_rot_mat.cuda(), curr_seq_local_jpos.cuda()
    )
    global_rot_mat = transforms.quaternion_to_matrix(global_quat)
    global_6d = transforms.matrix_to_rotation_6d(global_rot_mat)
    return torch.cat((global_pos.reshape(-1), global_6d.reshape(-1))).reshape(1, 1, -1)


def find_tf_start_indices(bool_tensor):
    t_starts = bool_tensor[1:] & ~bool_tensor[:-1]
    t_starts = torch.cat((bool_tensor[:1], t_starts))
    t_start_indices = torch.nonzero(t_starts).squeeze().reshape(-1)

    f_starts = ~bool_tensor[1:] & bool_tensor[:-1]
    f_starts = torch.cat((~bool_tensor[:1], f_starts))
    f_start_indices = torch.nonzero(f_starts).squeeze().reshape(-1)

    t_indices = [(index.item(), "T") for index in t_start_indices]
    f_indices = [(index.item(), "F") for index in f_start_indices]
    combined_indices = t_indices + f_indices

    combined_indices.sort()

    return combined_indices


def fix_feet(
    pred_feet_contact: Optional[torch.Tensor],
    target_human_jnts: torch.Tensor,
    fix_feet_floor_penetration: bool = True,
    fix_feet_sliding: bool = False,
) -> torch.Tensor:
    """Fix feet penetration and sliding.

    For penetration, we set the z value of the feet joints to be at least 0.008.
    For sliding, we use inertialize to smooth the sliding motion.
    We then use IK to generate the final human motion.

    Args:
        all_res_list (torch.Tensor): the generated human motion.
        pred_feet_contact (torch.Tensor): the predicted feet contact label.
        ref_data_dict (Dict[str, torch.Tensor]): the reference data dictionary.
        rest_human_offsets (torch.Tensor): the rest human offsets.
        navigation_trainer (Trainer): the navigation trainer.
        fix_feet_floor_penetration (bool, optional): whether to fix feet floor penetration.
        fix_feet_sliding (bool, optional): whether to fix feet sliding.

    Raises:
        ValueError: if fix_feet_sliding is True but pred_feet_contact is None.

    Returns:
        torch.Tensor: the target human joint positions, used by IK.
    """

    if fix_feet_sliding and not pred_feet_contact:
        raise ValueError("fix_feet_sliding is True but pred_feet_contact is None.")

    feet_contact_label = (pred_feet_contact > 0.95)[0]  # T X 4
    T = feet_contact_label.shape[0]

    # Fix feet floor penetration.
    if fix_feet_floor_penetration:
        target_human_jnts[:, [7, 8, 10, 11], 2] = torch.maximum(
            target_human_jnts[:, [7, 8, 10, 11], 2], torch.tensor(0.02)
        )

    # Fix feet sliding, use inertialize. # NOTE: bad, not use now
    if fix_feet_sliding:
        for feet_idx, smplx_idx in enumerate([7, 8, 10, 11]):
            tf_start_indices = find_tf_start_indices(feet_contact_label[:, feet_idx])
            for index, flag in tf_start_indices:
                if flag == "T":
                    ptr = index
                    while ptr < T and feet_contact_label[ptr, feet_idx]:
                        target_human_jnts[ptr, smplx_idx, :2] = target_human_jnts[
                            index, smplx_idx, :2
                        ]
                        ptr += 1
                elif flag == "F":
                    if index != 0 and index != 1:
                        ptr = index
                        while ptr < T and not feet_contact_label[ptr, feet_idx]:
                            ptr += 1
                        if ptr < index + 2:
                            continue

                        # new_pos, _ = apply_linear_offset(
                        #     original_jpos=target_human_jnts[index:ptr, smplx_idx][None], # 1 X T X 3
                        #     new_target_jpos=target_human_jnts[index, smplx_idx][None], # 1 X 3
                        #     reversed=True,
                        # )
                        prev_jpos = target_human_jnts[
                            max(0, index - 5) : index, smplx_idx
                        ][None]  # 1 X T X 3
                        window_jpos = target_human_jnts[index:ptr, smplx_idx][
                            None
                        ]  # 1 X T X 3
                        new_pos, _, _, _ = apply_inertialize(
                            prev_jpos=prev_jpos,
                            prev_rot_6d=None,
                            window_jpos=window_jpos,
                            window_rot_6d=None,
                            ratio=0.5,
                        )

                        target_human_jnts[index:ptr, smplx_idx] = new_pos

    return target_human_jnts
