import sys

sys.path.append("../../")

import json
import os
import random

import joblib
import numpy as np
import pytorch3d.transforms as transforms
import torch
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import Dataset
from tqdm import tqdm

from manip.lafan1.utils import rotate_at_frame_w_obj


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    norm = torch.norm(x, dim=axis, keepdim=True)
    res = x / (norm + eps)
    return res


def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vectors
    :return: tensor of quaternions
    """
    res = torch.cat(
        [
            torch.sqrt(torch.sum(x * x, dim=-1)).unsqueeze(-1)
            * torch.sqrt(torch.sum(y * y, dim=-1)).unsqueeze(-1)
            + torch.sum(x * y, dim=-1).unsqueeze(-1),
            torch.cross(x, y),
        ],
        dim=-1,
    )
    return res


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def rotate(points, R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape) > 3:
        points = points.squeeze()
    if len(shape) < 3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1, 2))
    return r_points.reshape(shape)


def get_smpl_parents(use_joints24=True):
    smplh_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data/smpl_all_models/smplh_amass",
    )
    bm_path = os.path.join(smplh_path, "male/model.npz")
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data["kintree_table"]  # 2 X 52

    if use_joints24:
        parents = ori_kintree_table[0, :23]  # 23
        parents[0] = -1  # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list)  # 24
    else:
        parents = ori_kintree_table[0, :22]  # 22
        parents[0] = -1  # Assign -1 for the root joint's parent idx.

    return parents


def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3
    kintree = get_smpl_parents(use_joints24=False)

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(
                global_pose[:, parent_id], global_pose[:, jId]
            )

    return global_pose  # T X J X 3 X 3


def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3
    parents = get_smpl_parents(use_joints24=False)

    grot = transforms.matrix_to_quaternion(grot_mat)  # T X J X 4

    res = torch.cat(
        [
            grot[..., :1, :],
            transforms.quaternion_multiply(
                transforms.quaternion_invert(grot[..., parents[1:], :]),
                grot[..., 1:, :],
            ),
        ],
        dim=-2,
    )  # T X J X 4

    res_mat = transforms.quaternion_to_matrix(res)  # T X J X 3 X 3

    return res_mat


def quat_fk_torch(lrot_mat, lpos=None, use_joints24=True):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents()

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    if lpos is None:
        gr = [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            if i < lrot.shape[-2]:
                gr.append(
                    transforms.quaternion_multiply(
                        gr[parents[i]], lrot[..., i : i + 1, :]
                    )
                )

        res = torch.cat(gr, dim=-2), None
    else:
        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :])
                + gp[parents[i]]
            )
            if i < lrot.shape[-2]:
                gr.append(
                    transforms.quaternion_multiply(
                        gr[parents[i]], lrot[..., i : i + 1, :]
                    )
                )

        res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res


class HumanML3DDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        load_ds=True,
    ):
        self.load_ds = load_ds
        self.train = train

        self.window = window

        self.parents = get_smpl_parents()  # 24/22

        self.data_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/HumanML3D/processed_data/",
        )

        keep_same_len_window = False
        self.keep_same_len_window = keep_same_len_window

        train_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/HumanML3D/humanml3d_train_seq_names.json",
        )
        test_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/HumanML3D/humanml3d_test_seq_names.json",
        )

        if self.train:
            seq_names = json.load(open(train_json_path, "r"))["k_idx"]
        else:
            seq_names = json.load(open(test_json_path, "r"))["k_idx"]

        amass_npz_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/amass_smplx_humanml3d_processed",
        )

        if keep_same_len_window:
            if self.train:
                seq_data_path = os.path.join(
                    self.data_root_folder, "train_diffusion_humanml3d_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    self.data_root_folder,
                    "cano_train_diffusion_humanml3d_window_"
                    + str(self.window)
                    + "_joints24_same_len_window.p",
                )
            else:
                seq_data_path = os.path.join(
                    self.data_root_folder, "test_diffusion_humanml3d_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    self.data_root_folder,
                    "cano_test_diffusion_humanml3d_window_"
                    + str(self.window)
                    + "_joints24_same_len_window.p",
                )

            min_max_mean_std_data_path = os.path.join(
                self.data_root_folder,
                "cano_humanml3d_min_max_mean_std_data_window_"
                + str(self.window)
                + "_joints24_same_len_window.p",
            )
        else:
            if self.train:
                seq_data_path = os.path.join(
                    self.data_root_folder, "train_diffusion_humanml3d_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    self.data_root_folder,
                    "cano_train_diffusion_humanml3d_window_"
                    + str(self.window)
                    + "_joints24.p",
                )
                root_traj_xy_ori_path = os.path.join(
                    self.data_root_folder,
                    "train_diffusion_humanml3d_root_traj_xy_ori_joints24.p",
                )
            else:
                seq_data_path = os.path.join(
                    self.data_root_folder, "test_diffusion_humanml3d_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    self.data_root_folder,
                    "cano_test_diffusion_humanml3d_window_"
                    + str(self.window)
                    + "_joints24.p",
                )
                root_traj_xy_ori_path = os.path.join(
                    self.data_root_folder,
                    "test_diffusion_humanml3d_root_traj_xy_ori_joints24.p",
                )

            min_max_mean_std_data_path = os.path.join(
                self.data_root_folder,
                "cano_humanml3d_min_max_mean_std_data_window_"
                + str(self.window)
                + "_joints24.p",
            )

        if self.load_ds:
            if os.path.exists(processed_data_path):
                self.window_data_dict = joblib.load(processed_data_path)
            else:
                if os.path.exists(seq_data_path):
                    self.data_dict = joblib.load(seq_data_path)
                else:
                    self.load_npz_data(amass_npz_folder, seq_names, seq_data_path)

                self.cal_normalize_data_input()
                joblib.dump(self.window_data_dict, processed_data_path)

            if os.path.exists(root_traj_xy_ori_path):
                self.root_traj_xy_ori_dict = joblib.load(root_traj_xy_ori_path)
            else:
                self.cal_root_traj_xy_ori()
                joblib.dump(self.root_traj_xy_ori_dict, root_traj_xy_ori_path)
            assert len(self.window_data_dict) == len(self.root_traj_xy_ori_dict)
        else:
            self.window_data_dict = {}

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)

        self.global_jpos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_min"])
            .float()
            .reshape(24, 3)[None]
        )
        self.global_jpos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_max"])
            .float()
            .reshape(24, 3)[None]
        )

        # Get train and validation statistics.
        if self.train:
            print(
                "Total number of windows for training:{0}".format(
                    len(self.window_data_dict)
                )
            )
        else:
            print(
                "Total number of windows for validation:{0}".format(
                    len(self.window_data_dict)
                )
            )

        # Prepare SMPLX model
        soma_work_base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/smpl_all_models/",
        )
        support_base_dir = soma_work_base_dir
        surface_model_type = "smplx"
        surface_model_male_fname = os.path.join(
            support_base_dir, surface_model_type, "SMPLX_MALE.npz"
        )
        surface_model_female_fname = os.path.join(
            support_base_dir, surface_model_type, "SMPLX_FEMALE.npz"
        )
        surface_model_neutral_fname = os.path.join(
            support_base_dir, surface_model_type, "SMPLX_NEUTRAL.npz"
        )
        dmpl_fname = None
        num_dmpls = None
        num_expressions = None
        num_betas = 16

        self.male_bm = BodyModel(
            bm_fname=surface_model_male_fname,
            num_betas=num_betas,
            num_expressions=num_expressions,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
        self.female_bm = BodyModel(
            bm_fname=surface_model_female_fname,
            num_betas=num_betas,
            num_expressions=num_expressions,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
        self.neutral_bm = BodyModel(
            bm_fname=surface_model_neutral_fname,
            num_betas=num_betas,
            num_expressions=num_expressions,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False
        for p in self.neutral_bm.parameters():
            p.requires_grad = False

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        self.neutral_bm = self.neutral_bm.cuda()

        self.bm_dict = {
            "male": self.male_bm,
            "female": self.female_bm,
            "neutral": self.neutral_bm,
        }

    def load_npz_data(self, npz_data_folder, seq_names_list, seq_data_path):
        self.data_dict = {}
        cnt = 0

        dataset_names = os.listdir(npz_data_folder)
        for dataset_name in dataset_names:
            dataset_folder_path = os.path.join(npz_data_folder, dataset_name)

            sub_names = os.listdir(dataset_folder_path)
            for sub_name in sub_names:
                subject_folder_path = os.path.join(dataset_folder_path, sub_name)

                npz_files = os.listdir(subject_folder_path)
                for npz_name in npz_files:
                    npz_path = os.path.join(subject_folder_path, npz_name)

                    k_idx = npz_name.split("_stageii")[0].split("_")[-1]
                    if k_idx in seq_names_list:
                        npz_data = np.load(npz_path)

                        self.data_dict[cnt] = {}

                        curr_seq_name = npz_name.replace(".npz", "")

                        self.data_dict[cnt]["seq_name"] = curr_seq_name
                        self.data_dict[cnt]["betas"] = npz_data["betas"]  # 1 X 16
                        self.data_dict[cnt]["gender"] = npz_data["gender"]
                        self.data_dict[cnt]["trans2joint"] = npz_data["tran2joint"]  # 3
                        self.data_dict[cnt]["rest_offsets"] = npz_data[
                            "rest_offsets"
                        ]  # 22 X 3/24 X 3

                        self.data_dict[cnt]["trans"] = npz_data["trans"]  # T X 3
                        self.data_dict[cnt]["root_orient"] = npz_data[
                            "root_orient"
                        ]  # T X 3
                        self.data_dict[cnt]["pose_body"] = npz_data[
                            "pose_body"
                        ]  # T X (21*3)

                        self.data_dict[cnt]["texts"] = npz_data["texts"]

                        cnt += 1

        joblib.dump(self.data_dict, seq_data_path)

    def cal_root_traj_xy_ori(self):
        self.root_traj_xy_ori_dict = {}
        for index, _ in tqdm(self.window_data_dict.items()):
            data_input = self.window_data_dict[index]["motion"]

            num_joints = 24
            global_joint_rot = data_input[:, 2 * num_joints * 3 :]  # T X (22*6)
            root_rot = torch.from_numpy(global_joint_rot[:, :6]).float()  # T X 6
            root_rot = transforms.rotation_6d_to_matrix(root_rot)  # T X 3 X 3

            z_axis = torch.zeros(root_rot.shape[0], 3)  # T X 3
            z_axis[:, 2] = 1.0
            z_axis = z_axis.reshape(root_rot.shape[0], 3, 1)  # T X 3 X 1

            rotated_z_axis = torch.matmul(root_rot.float(), z_axis.float()).reshape(
                root_rot.shape[0], 3
            )  # T X 3
            rotated_z_axis[:, 2] = 0.0  # T X 3

            forward = normalize(rotated_z_axis)  # T X 3
            x_axis = torch.zeros(root_rot.shape[0], 3)  # T X 3
            x_axis[:, 0] = 1.0
            yrot = normalize(quat_between(x_axis, forward))  # T X 4

            yrot = transforms.matrix_to_rotation_6d(
                transforms.quaternion_to_matrix(yrot)
            )  # T X 6
            self.root_traj_xy_ori_dict[index] = yrot.detach().cpu().numpy()

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in self.data_dict:
            seq_name = self.data_dict[index]["seq_name"]

            betas = self.data_dict[index]["betas"]  # 1 X 16
            gender = self.data_dict[index]["gender"]

            seq_root_trans = self.data_dict[index]["trans"]  # T X 3
            seq_root_orient = self.data_dict[index]["root_orient"]  # T X 3
            seq_pose_body = self.data_dict[index]["pose_body"].reshape(
                -1, 21, 3
            )  # T X 21 X 3

            rest_human_offsets = self.data_dict[index]["rest_offsets"]  # 22 X 3/24 X 3
            trans2joint = self.data_dict[index]["trans2joint"]  # 3

            texts_list = self.data_dict[index]["texts"]

            num_steps = seq_root_trans.shape[0]
            for start_t_idx in range(0, num_steps, self.window // 6):
                end_t_idx = start_t_idx + self.window - 1

                if (
                    self.keep_same_len_window and end_t_idx >= num_steps
                ):  # This is the setting that each window has the same number of frames.
                    continue

                self.window_data_dict[s_idx] = {}

                window_seq_root_trans = seq_root_trans[start_t_idx : end_t_idx + 1]
                window_seq_root_orient = seq_root_orient[start_t_idx : end_t_idx + 1]
                window_seq_pose_body = seq_pose_body[start_t_idx : end_t_idx + 1]

                joint_aa_rep = torch.cat(
                    (
                        torch.from_numpy(
                            seq_root_orient[start_t_idx : end_t_idx + 1]
                        ).float()[:, None, :],
                        torch.from_numpy(
                            seq_pose_body[start_t_idx : end_t_idx + 1]
                        ).float(),
                    ),
                    dim=1,
                )  # T X J X 3
                X = (
                    torch.from_numpy(rest_human_offsets)
                    .float()[None]
                    .repeat(joint_aa_rep.shape[0], 1, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )  # T X J X 3
                X[:, 0, :] = seq_root_trans[start_t_idx : end_t_idx + 1]
                local_rot_mat = transforms.axis_angle_to_matrix(
                    joint_aa_rep
                )  # T X J X 3 X 3
                Q = (
                    transforms.matrix_to_quaternion(local_rot_mat)
                    .detach()
                    .cpu()
                    .numpy()
                )  # T X J X 4

                # Canonicalize based on the first human pose's orientation.
                X, Q, _, _ = rotate_at_frame_w_obj(
                    X[np.newaxis],
                    Q[np.newaxis],
                    X[np.newaxis],
                    Q[np.newaxis],
                    trans2joint[np.newaxis],
                    self.parents,
                    n_past=1,
                    floor_z=True,
                )
                # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4

                new_seq_root_trans = X[0, :, 0, :]  # T X 3
                new_local_rot_mat = transforms.quaternion_to_matrix(
                    torch.from_numpy(Q[0]).float()
                )  # T X J X 3 X 3
                new_local_aa_rep = transforms.matrix_to_axis_angle(
                    new_local_rot_mat
                )  # T X J X 3
                new_seq_root_orient = new_local_aa_rep[:, 0, :]  # T X 3
                new_seq_pose_body = new_local_aa_rep[:, 1:, :]  # T X 21 X 3

                query = self.process_window_data(
                    rest_human_offsets,
                    trans2joint,
                    new_seq_root_trans,
                    new_seq_root_orient.detach().cpu().numpy(),
                    new_seq_pose_body.detach().cpu().numpy(),
                )

                curr_global_jpos = query["global_jpos"].detach().cpu().numpy()
                curr_global_jvel = query["global_jvel"].detach().cpu().numpy()
                curr_global_rot_6d = query["global_rot_6d"].detach().cpu().numpy()

                self.window_data_dict[s_idx]["motion"] = np.concatenate(
                    (
                        curr_global_jpos.reshape(-1, 24 * 3),
                        curr_global_jvel.reshape(-1, 24 * 3),
                        curr_global_rot_6d.reshape(-1, 22 * 6),
                    ),
                    axis=1,
                )  # T X (24*3+24*3+22*6)

                self.window_data_dict[s_idx]["seq_name"] = seq_name
                self.window_data_dict[s_idx]["start_t_idx"] = start_t_idx
                self.window_data_dict[s_idx]["end_t_idx"] = end_t_idx

                self.window_data_dict[s_idx]["betas"] = betas
                self.window_data_dict[s_idx]["gender"] = gender

                self.window_data_dict[s_idx]["trans2joint"] = trans2joint

                self.window_data_dict[s_idx]["rest_human_offsets"] = rest_human_offsets

                self.window_data_dict[s_idx]["texts"] = texts_list

                s_idx += 1

            # if s_idx > 32:
            #     break

    def extract_min_max_mean_std_from_data(self):
        all_global_jpos_data = []
        all_global_jvel_data = []

        for s_idx in self.window_data_dict:
            curr_window_data = self.window_data_dict[s_idx]["motion"]  # T X D

            all_global_jpos_data.append(curr_window_data[:, : 24 * 3])
            all_global_jvel_data.append(curr_window_data[:, 24 * 3 : 2 * 24 * 3])

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(
            -1, 72
        )  # (N*T) X 72
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 72)

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        stats_dict = {}
        stats_dict["global_jpos_min"] = min_jpos
        stats_dict["global_jpos_max"] = max_jpos
        stats_dict["global_jvel_min"] = min_jvel
        stats_dict["global_jvel_max"] = max_jvel

        return stats_dict

    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X 22/24 X 3
        # or BS X T X J X 3
        if ori_jpos.dim() == 4:
            normalized_jpos = (
                ori_jpos - self.global_jpos_min.to(ori_jpos.device)[None]
            ) / (
                self.global_jpos_max.to(ori_jpos.device)[None]
                - self.global_jpos_min.to(ori_jpos.device)[None]
            )
        else:
            normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device)) / (
                self.global_jpos_max.to(ori_jpos.device)
                - self.global_jpos_min.to(ori_jpos.device)
            )
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        return normalized_jpos  # (BS X) T X 22/24 X 3

    def normalize_specific_jpos_min_max(self, ori_jpos, j_idx):
        # ori_jpos: T X 3
        # or BS X T X 3
        if ori_jpos.dim() == 3:
            normalized_jpos = (
                ori_jpos - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)[None]
            ) / (
                self.global_jpos_max[:, j_idx, :].to(ori_jpos.device)[None]
                - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)[None]
            )
        else:
            normalized_jpos = (
                ori_jpos - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)
            ) / (
                self.global_jpos_max[:, j_idx, :].to(ori_jpos.device)
                - self.global_jpos_min[:, j_idx, :].to(ori_jpos.device)
            )
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        return normalized_jpos  # (BS X) T X 22/24 X 3

    def de_normalize_jpos_min_max(self, normalized_jpos):
        # normalized_jpos: T X 22/24 X 3
        # or BS X T X J X 3
        normalized_jpos = (normalized_jpos + 1) * 0.5  # [0, 1] range

        if normalized_jpos.dim() == 4:
            de_jpos = (
                normalized_jpos
                * (
                    self.global_jpos_max.to(normalized_jpos.device)[None]
                    - self.global_jpos_min.to(normalized_jpos.device)[None]
                )
                + self.global_jpos_min.to(normalized_jpos.device)[None]
            )
        else:
            de_jpos = normalized_jpos * (
                self.global_jpos_max.to(normalized_jpos.device)
                - self.global_jpos_min.to(normalized_jpos.device)
            ) + self.global_jpos_min.to(normalized_jpos.device)

        return de_jpos  # (BS X) T X 22/24 X 3

    def de_normalize_specific_jpos_min_max(self, normalized_jpos, j_idx):
        # normalized_jpos: T X 3
        # or BS X T X 3
        normalized_jpos = (normalized_jpos + 1) * 0.5  # [0, 1] range

        if normalized_jpos.dim() == 3:
            de_jpos = (
                normalized_jpos
                * (
                    self.global_jpos_max[:, j_idx, :].to(normalized_jpos.device)[None]
                    - self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)[None]
                )
                + self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)[None]
            )
        else:
            de_jpos = normalized_jpos * (
                self.global_jpos_max[:, j_idx, :].to(normalized_jpos.device)
                - self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)
            ) + self.global_jpos_min[:, j_idx, :].to(normalized_jpos.device)

        return de_jpos  # (BS X) T X 3

    def process_window_data(
        self,
        rest_human_offsets,
        trans2joint,
        seq_root_trans,
        seq_root_orient,
        seq_pose_body,
    ):
        random_t_idx = 0
        end_t_idx = seq_root_trans.shape[0] - 1

        window_root_trans = torch.from_numpy(
            seq_root_trans[random_t_idx : end_t_idx + 1]
        ).cuda()
        window_root_orient = (
            torch.from_numpy(seq_root_orient[random_t_idx : end_t_idx + 1])
            .float()
            .cuda()
        )
        window_pose_body = (
            torch.from_numpy(seq_pose_body[random_t_idx : end_t_idx + 1]).float().cuda()
        )

        # Move thr first frame's human position to zero.
        move_to_zero_trans = window_root_trans[0:1, :].clone()  # 1 X 3
        move_to_zero_trans[:, 2] = 0

        window_root_trans = window_root_trans - move_to_zero_trans

        window_root_rot_mat = transforms.axis_angle_to_matrix(
            window_root_orient
        )  # T' X 3 X 3
        window_pose_rot_mat = transforms.axis_angle_to_matrix(
            window_pose_body
        )  # T' X 21 X 3 X 3

        # Generate global joint rotation
        local_joint_rot_mat = torch.cat(
            (window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1
        )  # T' X 22 X 3 X 3
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat)  # T' X 22 X 3 X 3

        curr_seq_pose_aa = torch.cat(
            (window_root_orient[:, None, :], window_pose_body), dim=1
        )  # T' X 22 X 3/T' X 24 X 3
        rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None]
        curr_seq_local_jpos = rest_human_offsets.repeat(
            curr_seq_pose_aa.shape[0], 1, 1
        ).cuda()  # T' X 22 X 3/T' X 24 X 3
        curr_seq_local_jpos[:, 0, :] = (
            window_root_trans - torch.from_numpy(trans2joint).cuda()[None]
        )  # T' X 22/24 X 3

        local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)

        global_jpos = human_jnts  # T' X 22/24 X 3
        global_jvel = global_jpos[1:] - global_jpos[:-1]  # (T'-1) X 22/24 X 3

        global_joint_rot_mat = local2global_pose(local_joint_rot_mat)  # T' X 22 X 3 X 3

        local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        query = {}

        query["local_rot_mat"] = local_joint_rot_mat  # T' X 22 X 3 X 3
        query["local_rot_6d"] = local_rot_6d  # T' X 22 X 6

        query["global_jpos"] = global_jpos  # T' X 22/24 X 3
        query["global_jvel"] = torch.cat(
            (
                global_jvel,
                torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device),
            ),
            dim=0,
        )  # T' X 22/24 X 3

        query["global_rot_mat"] = global_joint_rot_mat  # T' X 22 X 3 X 3
        query["global_rot_6d"] = global_rot_6d  # T' X 22 X 6

        return query

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        root_traj_xy_ori = torch.from_numpy(self.root_traj_xy_ori_dict[index]).float()

        seq_name = self.window_data_dict[index]["seq_name"]

        window_s_idx = self.window_data_dict[index]["start_t_idx"]
        window_e_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        rest_human_offsets = self.window_data_dict[index]["rest_human_offsets"]

        num_joints = 24
        normalized_jpos = self.normalize_jpos_min_max(
            data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)
        )  # T X 22 X 3

        global_joint_rot = data_input[:, 2 * num_joints * 3 :]  # T X (22*6)

        new_data_input = torch.cat(
            (normalized_jpos.reshape(-1, num_joints * 3), global_joint_rot), dim=1
        )
        ori_data_input = torch.cat(
            (data_input[:, : num_joints * 3], global_joint_rot), dim=1
        )

        # Compute foot contact
        positions = data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)[
            None
        ]  # 1 X T X 24 X 3
        feet = positions[:, :, (7, 8, 10, 11)]  # BS X T X 4 X 3
        feetv = torch.zeros(feet.shape[:3])  # BS X T X 4
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = feetv < 0.01
        contacts = contacts.squeeze(0)

        # Add padding.
        actual_steps = new_data_input.shape[0]
        if actual_steps < self.window:
            paded_new_data_input = torch.cat(
                (
                    new_data_input,
                    torch.zeros(self.window - actual_steps, new_data_input.shape[-1]),
                ),
                dim=0,
            )
            paded_ori_data_input = torch.cat(
                (
                    ori_data_input,
                    torch.zeros(self.window - actual_steps, ori_data_input.shape[-1]),
                ),
                dim=0,
            )
            paded_root_traj_xy_ori = torch.cat(
                (
                    root_traj_xy_ori,
                    torch.zeros(self.window - actual_steps, root_traj_xy_ori.shape[-1]),
                ),
                dim=0,
            )
            paded_contacts = torch.cat(
                (contacts, torch.zeros(self.window - actual_steps, 4)), dim=0
            )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input
            paded_root_traj_xy_ori = root_traj_xy_ori
            paded_contacts = contacts

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        data_input_dict["rest_human_offsets"] = rest_human_offsets

        data_input_dict["text"] = random.sample(
            self.window_data_dict[index]["texts"].tolist(), 1
        )[0]

        data_input_dict["feet_contact"] = paded_contacts.float()

        data_input_dict["root_traj_xy_ori"] = paded_root_traj_xy_ori  # T X 6

        return data_input_dict
        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3
