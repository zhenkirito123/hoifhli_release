import sys

sys.path.append("../../")

import json
import os
import pickle
import random
from collections import defaultdict

import joblib
import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import Dataset
from tqdm import tqdm

from manip.lafan1.utils import rotate_at_frame_w_obj
from manip.utils.visualize.tools.utils import contact_ids

left_parts = [
    contact_ids["L_Hand"],
    contact_ids["L_Index1"],
    contact_ids["L_Index2"],
    contact_ids["L_Index3"],
    contact_ids["L_Middle1"],
    contact_ids["L_Middle2"],
    contact_ids["L_Middle3"],
    contact_ids["L_Pinky1"],
    contact_ids["L_Pinky2"],
    contact_ids["L_Pinky3"],
    contact_ids["L_Ring1"],
    contact_ids["L_Ring2"],
    contact_ids["L_Ring3"],
    contact_ids["L_Thumb1"],
    contact_ids["L_Thumb2"],
    contact_ids["L_Thumb3"],
]
right_parts = [
    contact_ids["R_Hand"],
    contact_ids["R_Index1"],
    contact_ids["R_Index2"],
    contact_ids["R_Index3"],
    contact_ids["R_Middle1"],
    contact_ids["R_Middle2"],
    contact_ids["R_Middle3"],
    contact_ids["R_Pinky1"],
    contact_ids["R_Pinky2"],
    contact_ids["R_Pinky3"],
    contact_ids["R_Ring1"],
    contact_ids["R_Ring2"],
    contact_ids["R_Ring3"],
    contact_ids["R_Thumb1"],
    contact_ids["R_Thumb2"],
    contact_ids["R_Thumb3"],
]


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
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data/smpl_all_models/smplh_amass",
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


def quat_fk_torch(lrot_mat, lpos, use_joints24=True):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J/(J+2) X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    if use_joints24:
        parents = get_smpl_parents(use_joints24=True)
    else:
        parents = get_smpl_parents()

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :])
            + gp[parents[i]]
        )
        if i < lrot.shape[-2]:
            gr.append(
                transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :])
            )

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res


def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3
        part_verts = verts_list[p_idx]  # T X Nv X 3
        part_faces = torch.from_numpy(faces_list[p_idx])  # T X Nf X 3

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces + verts_num)

        verts_num += part_verts.shape[1]

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy()

    return merged_verts, merged_faces


class CanoObjectTrajDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=False,
        input_language_condition=False,
        use_first_frame_bps=False,
        use_random_frame_bps=False,
        use_object_keypoints=False,
        load_ds=True,
    ):
        self.load_ds = load_ds
        self.train = train

        self.window = window

        self.use_object_splits = use_object_splits
        self.train_objects = [
            "largetable",
            "woodchair",
            "plasticbox",
            "largebox",
            "smallbox",
            "trashcan",
            "monitor",
            "floorlamp",
            "clothesstand",
        ]  # 10 objects
        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod"]

        self.input_language_condition = input_language_condition

        self.use_first_frame_bps = use_first_frame_bps

        self.use_random_frame_bps = use_random_frame_bps

        self.use_object_keypoints = use_object_keypoints

        self.parents = get_smpl_parents()  # 24/22

        self.data_root_folder = data_root_folder
        self.obj_geo_root_folder = os.path.join(
            self.data_root_folder, "captured_objects"
        )

        self.rest_object_geo_folder = os.path.join(
            self.data_root_folder, "rest_object_geo"
        )
        if not os.path.exists(self.rest_object_geo_folder):
            os.makedirs(self.rest_object_geo_folder)

        self.bps_path = "./bps.pt"

        self.language_anno_folder = os.path.join(
            self.data_root_folder, "omomo_text_anno_json_data"
        )

        # self.contact_npy_folder = os.path.join(self.data_root_folder, "contact_labels_npy_files")
        self.contact_npy_folder = os.path.join(
            self.data_root_folder, "contact_labels_w_semantics_npy_files"
        )

        train_subjects = []
        test_subjects = []
        num_subjects = 17
        for s_idx in range(1, num_subjects + 1):
            if s_idx >= 16:
                test_subjects.append("sub" + str(s_idx))
            else:
                train_subjects.append("sub" + str(s_idx))

        keep_same_len_window = False
        self.keep_same_len_window = keep_same_len_window

        if keep_same_len_window:
            dest_obj_bps_npy_folder = os.path.join(
                self.data_root_folder,
                "cano_object_bps_npy_files_joints24_same_len_window_"
                + str(self.window),
            )
            dest_obj_bps_npy_folder_for_test = os.path.join(
                self.data_root_folder,
                "cano_object_bps_npy_files_for_test_joints24_same_len_window_"
                + str(self.window),
            )
        else:
            dest_obj_bps_npy_folder = os.path.join(
                self.data_root_folder,
                "cano_object_bps_npy_files_joints24_" + str(self.window),
            )
            dest_obj_bps_npy_folder_for_test = os.path.join(
                self.data_root_folder,
                "cano_object_bps_npy_files_for_test_joints24_" + str(self.window),
            )

        if not os.path.exists(dest_obj_bps_npy_folder):
            os.makedirs(dest_obj_bps_npy_folder)

        if not os.path.exists(dest_obj_bps_npy_folder_for_test):
            os.makedirs(dest_obj_bps_npy_folder_for_test)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test

        if keep_same_len_window:
            if self.train:
                seq_data_path = os.path.join(
                    data_root_folder, "train_diffusion_manip_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    data_root_folder,
                    "cano_train_diffusion_manip_window_"
                    + str(self.window)
                    + "_joints24_same_len_window.p",
                )
            else:
                seq_data_path = os.path.join(
                    data_root_folder, "test_diffusion_manip_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    data_root_folder,
                    "cano_test_diffusion_manip_window_"
                    + str(self.window)
                    + "_joints24_same_len_window.p",
                )

            min_max_mean_std_data_path = os.path.join(
                data_root_folder,
                "cano_min_max_mean_std_data_window_"
                + str(self.window)
                + "_joints24_same_len_window.p",
            )
        else:
            if self.train:
                seq_data_path = os.path.join(
                    data_root_folder, "train_diffusion_manip_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    data_root_folder,
                    "cano_train_diffusion_manip_window_"
                    + str(self.window)
                    + "_joints24.p",
                )
                standing_flag_path = os.path.join(
                    self.data_root_folder, "train_standing_flag_joints24.p"
                )
                wrist_relative_path = os.path.join(
                    self.data_root_folder, "train_wrist_relative_joints24.p"
                )
                object_static_flag_path = os.path.join(
                    self.data_root_folder, "train_object_static_flag_joints24.p"
                )
                root_traj_xy_ori_path = os.path.join(
                    self.data_root_folder,
                    "train_interaction_root_traj_xy_ori_joints24.p",
                )
            else:
                seq_data_path = os.path.join(
                    data_root_folder, "test_diffusion_manip_seq_joints24.p"
                )
                processed_data_path = os.path.join(
                    data_root_folder,
                    "cano_test_diffusion_manip_window_"
                    + str(self.window)
                    + "_joints24.p",
                )
                standing_flag_path = os.path.join(
                    self.data_root_folder, "test_standing_flag_joints24.p"
                )
                wrist_relative_path = os.path.join(
                    self.data_root_folder, "test_wrist_relative_joints24.p"
                )
                object_static_flag_path = os.path.join(
                    self.data_root_folder, "test_object_static_flag_joints24.p"
                )
                root_traj_xy_ori_path = os.path.join(
                    self.data_root_folder,
                    "test_interaction_root_traj_xy_ori_joints24.p",
                )
            min_max_mean_std_data_path = os.path.join(
                data_root_folder,
                "cano_min_max_mean_std_data_window_" + str(self.window) + "_joints24.p",
            )

        self.prep_bps_data()

        if self.load_ds:
            if os.path.exists(processed_data_path):
                self.window_data_dict = joblib.load(processed_data_path)

                # if not self.train:
                # Mannually enable this. For testing data (discarded some testing sequences)
                # self.get_bps_from_window_data_dict()
            else:
                self.data_dict = joblib.load(seq_data_path)

                self.extract_rest_pose_object_geometry_and_rotation()

                self.cal_normalize_data_input()
                joblib.dump(self.window_data_dict, processed_data_path)

            if os.path.exists(object_static_flag_path):
                self.object_static_flag_dict = joblib.load(object_static_flag_path)
            else:
                self.calc_object_static_flag()
                joblib.dump(self.object_static_flag_dict, object_static_flag_path)

            if os.path.exists(root_traj_xy_ori_path):
                self.root_traj_xy_ori_dict = joblib.load(root_traj_xy_ori_path)
            else:
                self.calc_root_traj_xy_ori()
                joblib.dump(self.root_traj_xy_ori_dict, root_traj_xy_ori_path)

            if os.path.exists(wrist_relative_path):
                self.wrist_relative_dict = joblib.load(wrist_relative_path)
            else:
                self.calc_wrist_relative()
                joblib.dump(self.wrist_relative_dict, wrist_relative_path)

            if os.path.exists(standing_flag_path):
                self.standing_flag_dict = joblib.load(standing_flag_path)
            else:
                self.calc_standing_flag()
                joblib.dump(self.standing_flag_dict, standing_flag_path)
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

        self.obj_pos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["obj_com_pos_min"])
            .float()
            .reshape(1, 3)
        )
        self.obj_pos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["obj_com_pos_max"])
            .float()
            .reshape(1, 3)
        )

        if self.use_object_splits:
            (
                self.window_data_dict,
                self.wrist_relative_dict,
                self.standing_flag_dict,
                self.object_static_flag_dict,
                self.root_traj_xy_ori_dict,
            ) = self.filter_out_object_split()

        if self.input_language_condition:
            (
                self.window_data_dict,
                self.wrist_relative_dict,
                self.standing_flag_dict,
                self.object_static_flag_dict,
                self.root_traj_xy_ori_dict,
            ) = self.filter_out_seq_wo_text()

        if not self.train:
            (
                self.window_data_dict,
                self.wrist_relative_dict,
                self.standing_flag_dict,
                self.object_static_flag_dict,
                self.root_traj_xy_ori_dict,
            ) = self.filter_out_short_sequences()

        # Get train and validation statistics.
        if self.train:
            print(
                "Total number of windows for training:{0}".format(
                    len(self.window_data_dict)
                )
            )  # all, Total number of windows for training:28859
        else:
            print(
                "Total number of windows for validation:{0}".format(
                    len(self.window_data_dict)
                )
            )  # all, 3224

        min_max_obj_pts_data_path = os.path.join(
            data_root_folder,
            "cano_min_max_obj_pts_data_window_" + str(self.window) + "_joints24.p",
        )
        if os.path.exists(min_max_obj_pts_data_path):
            obj_pts_min_max_data = joblib.load(min_max_obj_pts_data_path)
        else:
            if self.train:
                obj_pts_min_max_data = self.compute_object_keypoints_min_max()
                joblib.dump(obj_pts_min_max_data, min_max_obj_pts_data_path)

        self.obj_keypoints_min = torch.from_numpy(
            obj_pts_min_max_data["obj_pts_min"]
        ).float()
        self.obj_keypoints_max = torch.from_numpy(
            obj_pts_min_max_data["obj_pts_max"]
        ).float()

        # Prepare SMPLX model
        soma_work_base_dir = os.path.join(
            self.data_root_folder, "..", "smpl_all_models"
        )
        support_base_dir = soma_work_base_dir
        surface_model_type = "smplx"
        # surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "male", 'model.npz')
        # surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
        surface_model_male_fname = os.path.join(
            support_base_dir, surface_model_type, "SMPLX_MALE.npz"
        )
        surface_model_female_fname = os.path.join(
            support_base_dir, surface_model_type, "SMPLX_FEMALE.npz"
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

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()

        self.bm_dict = {"male": self.male_bm, "female": self.female_bm}

        assert len(self.window_data_dict) == len(self.wrist_relative_dict)
        assert len(self.window_data_dict) == len(self.standing_flag_dict)
        assert len(self.window_data_dict) == len(self.object_static_flag_dict)
        assert len(self.window_data_dict) == len(self.root_traj_xy_ori_dict)

        # self.generate_init_grasp_pose(self.train)

    def generate_init_grasp_pose(self, is_train):
        from sklearn.cluster import KMeans
        from tqdm import tqdm

        init_grasp_pose = {
            "left_hand": defaultdict(list),
            "right_hand": defaultdict(list),
        }
        for index, _ in tqdm(self.window_data_dict.items()):
            seq_name = self.window_data_dict[index]["seq_name"]
            object_name = seq_name.split("_")[1]
            window_s_idx = self.window_data_dict[index]["start_t_idx"]
            window_e_idx = self.window_data_dict[index]["end_t_idx"]
            contact_npy_path = os.path.join(self.contact_npy_folder, seq_name + ".npy")
            contact_npy_data = np.load(
                contact_npy_path
            )  # T X 4 (lhand, rhand, lfoot, rfoot)
            contact_labels = contact_npy_data[window_s_idx : window_e_idx + 1]  # W
            contact_labels = torch.from_numpy(contact_labels).float()

            wrist_relative = self.wrist_relative_dict[index].float()  # 18
            # set the wrist relative to zero if the not in contact.
            # wrist_relative[:, 0:9] *= contact_labels[:, 0:1] # left hand
            # wrist_relative[:, 9:18] *= contact_labels[:, 1:2] # right hand
            init_grasp_pose["left_hand"][object_name].append(
                wrist_relative[(contact_labels[:, 0:1] > 0.9).reshape(-1), 0:9]
            )
            init_grasp_pose["right_hand"][object_name].append(
                wrist_relative[(contact_labels[:, 1:2] > 0.9).reshape(-1), 9:18]
            )

        path = "init_grasp_pose_{}.pkl".format("train" if is_train else "test")

        pickle.dump(init_grasp_pose, open(path, "wb"))
        print("Init grasp pose saved to {}".format(path))
        # for hand in init_grasp_pose:
        #     for obj_name in init_grasp_pose[hand]:
        #         data = torch.cat(init_grasp_pose[hand][obj_name]) # T X 9
        #         pos = data[:, :3]
        #         rot_6d = data[:, 3:]
        #         # k-means

    def find_static_move_switch(self, object_static_flag):
        # object_static_flag: BS X T / T
        # Find the positions where 0 and 1 switch
        switch_positions = torch.where(
            torch.abs(torch.diff(object_static_flag, dim=-1)) > 1e-3
        )

        # Create a mask indicating switch positions
        switch_mask = torch.zeros_like(object_static_flag)
        switch_mask[switch_positions] = 1

        return switch_mask

    def calc_object_static_flag(self):
        self.object_static_flag_dict = {}
        for index, _ in tqdm(self.window_data_dict.items()):
            obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()  # T X 3
            obj_v = obj_com_pos[1:, :3] - obj_com_pos[:-1, :3]  # T X 3
            obj_v_norm = torch.norm(obj_v, dim=-1) * 30  # T-1
            obj_v_norm = torch.cat((obj_v_norm, obj_v_norm[-1:]))
            if obj_com_pos.shape[0] == 1:
                obj_v_norm = torch.zeros(1).to(obj_com_pos.device)

            seq_name = self.window_data_dict[index]["seq_name"]

            window_s_idx = self.window_data_dict[index]["start_t_idx"]
            window_e_idx = self.window_data_dict[index]["end_t_idx"]
            contact_npy_path = os.path.join(self.contact_npy_folder, seq_name + ".npy")
            contact_npy_data = np.load(
                contact_npy_path
            )  # T X 4 (lhand, rhand, lfoot, rfoot)
            contact_labels = contact_npy_data[window_s_idx : window_e_idx + 1]  # W
            contact_labels = torch.from_numpy(contact_labels).float()
            left_contact_label = contact_labels[:, 0]
            right_contact_label = contact_labels[:, 1]

            static_flag = (
                (left_contact_label < 1e-5)
                * (right_contact_label < 1e-5)
                * (obj_v_norm < 0.2)
            )
            assert static_flag.shape[0] == obj_com_pos.shape[0]

            switch_mask = self.find_static_move_switch(static_flag.float())  # T
            left_switch_mask = switch_mask.clone()
            right_switch_mask = switch_mask.clone()
            for i in range(switch_mask.shape[0]):
                pre_left_contact_label = (
                    left_contact_label[max(0, i - 15) : i + 1].mean() > 0.5
                )
                post_left_contact_label = (
                    left_contact_label[
                        i : min(i + 15, left_contact_label.shape[0])
                    ].mean()
                    > 0.5
                )
                left_switch_mask[i] = switch_mask[i] * (
                    pre_left_contact_label != post_left_contact_label
                )
                pre_right_contact_label = (
                    right_contact_label[max(0, i - 15) : i + 1].mean() > 0.5
                )
                post_right_contact_label = (
                    right_contact_label[
                        i : min(i + 15, right_contact_label.shape[0])
                    ].mean()
                    > 0.5
                )
                right_switch_mask[i] = switch_mask[i] * (
                    pre_right_contact_label != post_right_contact_label
                )

            self.object_static_flag_dict[index] = {
                "static_flag": static_flag[..., None],
                "left_switch_mask": left_switch_mask[..., None],
                "right_switch_mask": right_switch_mask[..., None],
            }

    def calc_root_traj_xy_ori(self):
        from tqdm import tqdm

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

    def calc_wrist_relative(self):
        self.wrist_relative_dict = {}
        for index, _ in tqdm(self.window_data_dict.items()):
            data_input = self.window_data_dict[index]["motion"]
            num_joints = 24
            human_jnts = torch.from_numpy(
                data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)
            ).float()  # T X 24 X 3
            huamn_rot_6d = torch.from_numpy(
                data_input[:, 2 * num_joints * 3 :].reshape(-1, num_joints - 2, 6)
            ).float()  # T X 22 X 6

            left_wrist_pos = human_jnts[:, 20]
            right_wrist_pos = human_jnts[:, 21]
            left_wrist_rot_mat = transforms.rotation_6d_to_matrix(huamn_rot_6d[:, 20])
            right_wrist_rot_mat = transforms.rotation_6d_to_matrix(huamn_rot_6d[:, 21])

            obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()  # T X 3
            obj_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_rot_mat"]
            ).float()  # T X 3 X 3

            left_wrist_pos_in_obj = (
                obj_rot_mat.transpose(1, 2)
                @ (left_wrist_pos - obj_com_pos).unsqueeze(-1)
            )[..., 0]  # T X 3
            right_wrist_pos_in_obj = (
                obj_rot_mat.transpose(1, 2)
                @ (right_wrist_pos - obj_com_pos).unsqueeze(-1)
            )[..., 0]  # T X 3
            left_wrist_rot_mat_in_obj = (
                obj_rot_mat.transpose(1, 2) @ left_wrist_rot_mat
            )  # T X 3 X 3
            right_wrist_rot_mat_in_obj = (
                obj_rot_mat.transpose(1, 2) @ right_wrist_rot_mat
            )  # T X 3 X 3

            # normalize the position, assume each axis of the position should be in 0.5m.
            wrist_relative = torch.cat(
                (
                    self.normalize_wrist_relative_pos(left_wrist_pos_in_obj),
                    transforms.matrix_to_rotation_6d(left_wrist_rot_mat_in_obj),
                    self.normalize_wrist_relative_pos(right_wrist_pos_in_obj),
                    transforms.matrix_to_rotation_6d(right_wrist_rot_mat_in_obj),
                ),
                dim=-1,
            )  # T X 18

            self.wrist_relative_dict[index] = wrist_relative
            assert wrist_relative.shape[0] == data_input.shape[0]

    def prep_rel_wrist_relative(self, wrist_relative, ref_rot_mat):
        # wrist_relative: T X 18
        # ref_rot_mat: 1 X 3 X 3
        # pos' = R_ref @ pos
        # rot' = R_ref @ rot
        left_wrist_pos_in_obj = self.de_normalize_wrist_relative_pos(
            wrist_relative[:, :3]
        )
        left_wrist_rot_mat_in_obj = transforms.rotation_6d_to_matrix(
            wrist_relative[:, 3:9]
        )
        right_wrist_pos_in_obj = self.de_normalize_wrist_relative_pos(
            wrist_relative[:, 9:12]
        )
        right_wrist_rot_mat_in_obj = transforms.rotation_6d_to_matrix(
            wrist_relative[:, 12:18]
        )

        new_left_wrist_pos_in_obj = (ref_rot_mat @ left_wrist_pos_in_obj.unsqueeze(-1))[
            ..., 0
        ]  # T X 3
        new_left_wrist_rot_mat_in_obj = (
            ref_rot_mat @ left_wrist_rot_mat_in_obj
        )  # T X 3 X 3
        new_right_wrist_pos_in_obj = (
            ref_rot_mat @ right_wrist_pos_in_obj.unsqueeze(-1)
        )[..., 0]  # T X 3
        new_right_wrist_rot_mat_in_obj = (
            ref_rot_mat @ right_wrist_rot_mat_in_obj
        )  # T X 3 X 3

        new_wrist_relative = torch.cat(
            (
                self.normalize_wrist_relative_pos(new_left_wrist_pos_in_obj),
                transforms.matrix_to_rotation_6d(new_left_wrist_rot_mat_in_obj),
                self.normalize_wrist_relative_pos(new_right_wrist_pos_in_obj),
                transforms.matrix_to_rotation_6d(new_right_wrist_rot_mat_in_obj),
            ),
            dim=-1,
        )  # T X 18
        return new_wrist_relative

    def normalize_wrist_relative_pos(self, normalize_wrist_relative_pos):
        return normalize_wrist_relative_pos / 0.5

    def de_normalize_wrist_relative_pos(self, de_normalize_wrist_relative_pos):
        return de_normalize_wrist_relative_pos * 0.5

    def calc_standing_flag(self):
        self.standing_flag_dict = {}
        for index, _ in self.window_data_dict.items():
            data_input = self.window_data_dict[index]["motion"]
            num_joints = 24
            human_jnts = data_input[-1, : num_joints * 3].reshape(
                num_joints, 3
            )  # 24 X 3

            root_bone1 = human_jnts[6] - human_jnts[3]
            root_bone1 /= np.linalg.norm(root_bone1)
            left_arm_bone1 = human_jnts[20] - human_jnts[18]
            left_arm_bone1 /= np.linalg.norm(left_arm_bone1)
            left_arm_bone2 = human_jnts[18] - human_jnts[16]
            left_arm_bone2 /= np.linalg.norm(left_arm_bone2)
            right_arm_bone1 = human_jnts[21] - human_jnts[19]
            right_arm_bone1 /= np.linalg.norm(right_arm_bone1)
            right_arm_bone2 = human_jnts[19] - human_jnts[17]
            right_arm_bone2 /= np.linalg.norm(right_arm_bone2)

            flag = (
                root_bone1[2] > 0.9
                and left_arm_bone1[2] < -0.8
                and left_arm_bone2[2] < -0.9
                and right_arm_bone1[2] < -0.8
                and right_arm_bone2[2] < -0.9
            )

            print(
                self.window_data_dict[index]["seq_name"],
                self.window_data_dict[index]["start_t_idx"],
                self.window_data_dict[index]["start_t_idx"]
                + self.window_data_dict[index]["motion"].shape[0],
                flag,
            )
            self.standing_flag_dict[index] = flag

    def load_language_annotation(self, seq_name):
        # seq_name: sub16_clothesstand_000, etc.
        json_path = os.path.join(self.language_anno_folder, seq_name + ".json")
        json_data = json.load(open(json_path, "r"))

        text_anno = json_data[seq_name]

        return text_anno

    def filter_out_short_sequences(self):
        # Remove short sequences from window_data_dict.
        new_cnt = 0
        new_window_data_dict = {}
        new_wrist_relative_dict = {}
        new_standing_flag_dict = {}
        new_object_static_flag_dict = {}
        new_root_traj_xy_ori_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data["seq_name"]
            object_name = seq_name.split("_")[1]

            curr_seq_len = window_data["motion"].shape[0]

            if curr_seq_len < self.window:
                continue

            if self.window_data_dict[k]["start_t_idx"] != 0:
                continue

            new_window_data_dict[new_cnt] = self.window_data_dict[k]
            new_wrist_relative_dict[new_cnt] = self.wrist_relative_dict[k]
            new_standing_flag_dict[new_cnt] = self.standing_flag_dict[k]
            new_object_static_flag_dict[new_cnt] = self.object_static_flag_dict[k]
            new_root_traj_xy_ori_dict[new_cnt] = self.root_traj_xy_ori_dict[k]
            if "ori_w_idx" in self.window_data_dict[k]:
                new_window_data_dict[new_cnt]["ori_w_idx"] = self.window_data_dict[k][
                    "ori_w_idx"
                ]
            else:
                new_window_data_dict[new_cnt]["ori_w_idx"] = k

            new_cnt += 1

        return (
            new_window_data_dict,
            new_wrist_relative_dict,
            new_standing_flag_dict,
            new_object_static_flag_dict,
            new_root_traj_xy_ori_dict,
        )

    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing.
        new_cnt = 0
        new_window_data_dict = {}
        new_wrist_relative_dict = {}
        new_standing_flag_dict = {}
        new_object_static_flag_dict = {}
        new_root_traj_xy_ori_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data["seq_name"]
            object_name = seq_name.split("_")[1]
            if self.train and object_name in self.train_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_wrist_relative_dict[new_cnt] = self.wrist_relative_dict[k]
                new_standing_flag_dict[new_cnt] = self.standing_flag_dict[k]
                new_object_static_flag_dict[new_cnt] = self.object_static_flag_dict[k]
                new_root_traj_xy_ori_dict[new_cnt] = self.root_traj_xy_ori_dict[k]
                new_window_data_dict[new_cnt]["ori_w_idx"] = k
                new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_wrist_relative_dict[new_cnt] = self.wrist_relative_dict[k]
                new_standing_flag_dict[new_cnt] = self.standing_flag_dict[k]
                new_object_static_flag_dict[new_cnt] = self.object_static_flag_dict[k]
                new_root_traj_xy_ori_dict[new_cnt] = self.root_traj_xy_ori_dict[k]
                new_window_data_dict[new_cnt]["ori_w_idx"] = k
                new_cnt += 1

        return (
            new_window_data_dict,
            new_wrist_relative_dict,
            new_standing_flag_dict,
            new_object_static_flag_dict,
            new_root_traj_xy_ori_dict,
        )

    def filter_out_seq_wo_text(self):
        new_cnt = 0
        new_window_data_dict = {}
        new_wrist_relative_dict = {}
        new_standing_flag_dict = {}
        new_object_static_flag_dict = {}
        new_root_traj_xy_ori_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            seq_name = window_data["seq_name"]
            text_json_path = os.path.join(self.language_anno_folder, seq_name + ".json")
            if os.path.exists(text_json_path):
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_wrist_relative_dict[new_cnt] = self.wrist_relative_dict[k]
                new_standing_flag_dict[new_cnt] = self.standing_flag_dict[k]
                new_object_static_flag_dict[new_cnt] = self.object_static_flag_dict[k]
                new_root_traj_xy_ori_dict[new_cnt] = self.root_traj_xy_ori_dict[k]
                if (
                    "ori_w_idx" in self.window_data_dict[k]
                ):  # Based on filtered results split by objects.
                    new_window_data_dict[new_cnt]["ori_w_idx"] = self.window_data_dict[
                        k
                    ]["ori_w_idx"]
                else:  # Based on the original window_daia_dict.
                    new_window_data_dict[new_cnt]["ori_w_idx"] = k
                new_cnt += 1

        return (
            new_window_data_dict,
            new_wrist_relative_dict,
            new_standing_flag_dict,
            new_object_static_flag_dict,
            new_root_traj_xy_ori_dict,
        )

    def apply_transformation_to_obj_geometry(
        self, obj_mesh_path, obj_scale, obj_rot, obj_trans
    ):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices)  # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces)  # Nf X 3

        ori_obj_verts = (
            torch.from_numpy(obj_mesh_verts)
            .float()[None]
            .repeat(obj_trans.shape[0], 1, 1)
        )  # T X Nv X 3

        if torch.is_tensor(obj_scale):
            seq_scale = obj_scale.float()
        else:
            seq_scale = torch.from_numpy(obj_scale).float()  # T

        if torch.is_tensor(obj_rot):
            seq_rot_mat = obj_rot.float()
        else:
            seq_rot_mat = torch.from_numpy(obj_rot).float()  # T X 3 X 3

        if obj_trans.shape[-1] != 1:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()[:, :, None]
            else:
                seq_trans = torch.from_numpy(obj_trans).float()[:, :, None]  # T X 3 X 1
        else:
            if torch.is_tensor(obj_trans):
                seq_trans = obj_trans.float()
            else:
                seq_trans = torch.from_numpy(obj_trans).float()  # T X 3 X 1

        transformed_obj_verts = (
            seq_scale.unsqueeze(-1).unsqueeze(-1)
            * seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2).to(seq_trans.device))
            + seq_trans
        )
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3

        return transformed_obj_verts, obj_mesh_faces

    def load_rest_pose_object_geometry(self, object_name):
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name + ".ply")
        if not os.path.exists(rest_obj_path):  # new object geometry
            rest_obj_path = os.path.join(
                self.rest_object_geo_folder,
                "../new_objects",
                object_name + ".obj",
            )

        mesh = trimesh.load_mesh(rest_obj_path)
        rest_verts = np.asarray(mesh.vertices)  # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces)  # Nf X 3

        return rest_verts, obj_mesh_faces

    def convert_rest_pose_obj_geometry(
        self, object_name, obj_scale, obj_trans, obj_rot
    ):
        # obj_scale: T, obj_trans: T X 3, obj_rot: T X 3 X 3
        # obj_mesh_verts: T X Nv X 3
        rest_obj_path = os.path.join(self.rest_object_geo_folder, object_name + ".ply")
        rest_obj_json_path = os.path.join(
            self.rest_object_geo_folder, object_name + ".json"
        )

        if os.path.exists(rest_obj_path):
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices)  # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces)  # Nf X 3

            rest_verts = torch.from_numpy(rest_verts)

            json_data = json.load(open(rest_obj_json_path, "r"))
            rest_pose_ori_obj_rot = np.asarray(
                json_data["rest_pose_ori_obj_rot"]
            )  # 3 X 3
            rest_pose_ori_obj_com_pos = np.asarray(
                json_data["rest_pose_ori_com_pos"]
            )  # 1 X 3
            obj_trans_to_com_pos = np.asarray(
                json_data["obj_trans_to_com_pos"]
            )  # 1 X 3
            # import pdb
            # pdb.set_trace()
        else:
            obj_mesh_verts, obj_mesh_faces = self.load_object_geometry(
                object_name, obj_scale, obj_trans, obj_rot
            )
            com_pos = obj_mesh_verts[0].mean(dim=0)[None]  # 1 X 3
            obj_trans_to_com_pos = (
                obj_trans[0:1] - com_pos.detach().cpu().numpy()
            )  # 1 X 3
            tmp_verts = obj_mesh_verts[0] - com_pos  # Nv X 3
            obj_rot = torch.from_numpy(obj_rot)
            tmp_verts = tmp_verts.to(obj_rot.device)
            # tmp_verts /= obj_scale[0] # Nv X 3

            # rest_verts = torch.matmul(obj_rot[0:1].repeat(tmp_verts.shape[0], 1, 1).transpose(1, 2), \
            #         tmp_verts[:, :, None]) # Nv X 3 X 1
            # rest_verts = rest_verts.squeeze(-1) # Nv X 3
            rest_verts = tmp_verts.clone()  # Nv X 3

            dest_mesh = trimesh.Trimesh(
                vertices=rest_verts.detach().cpu().numpy(),
                faces=obj_mesh_faces,
                process=False,
            )

            result = trimesh.exchange.ply.export_ply(dest_mesh, encoding="ascii")
            output_file = open(rest_obj_path, "wb+")
            output_file.write(result)
            output_file.close()

            rest_pose_ori_obj_rot = obj_rot[0].detach().cpu().numpy()  # 3 X 3
            rest_pose_ori_obj_com_pos = com_pos.detach().cpu().numpy()  # 1 X 3

            dest_data_dict = {}
            dest_data_dict["rest_pose_ori_obj_rot"] = rest_pose_ori_obj_rot.tolist()
            dest_data_dict["rest_pose_ori_com_pos"] = rest_pose_ori_obj_com_pos.tolist()
            dest_data_dict["obj_trans_to_com_pos"] = obj_trans_to_com_pos.tolist()

            json.dump(dest_data_dict, open(rest_obj_json_path, "w"))

            # import pdb
            # pdb.set_trace()

        # Compute object's BPS representation in rest pose.
        dest_obj_bps_npy_path = os.path.join(
            self.rest_object_geo_folder, object_name + ".npy"
        )

        if not os.path.exists(dest_obj_bps_npy_path):
            center_verts = torch.zeros(1, 3).to(rest_verts.device)
            object_bps = self.compute_object_geo_bps(
                rest_verts[None], center_verts
            )  # 1 X 1024 X 3
            np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

        return (
            rest_verts,
            obj_mesh_faces,
            rest_pose_ori_obj_rot,
            rest_pose_ori_obj_com_pos,
            obj_trans_to_com_pos,
        )

    def load_object_geometry_w_rest_geo(self, obj_rot, obj_com_pos, rest_verts):
        # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_veerts: Nv X 3
        # rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        # transformed_obj_verts = obj_scale.unsqueeze(-1).unsqueeze(-1) * \
        # obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        # transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3

        rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
        transformed_obj_verts = (
            obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
        )
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3

        return transformed_obj_verts

    def load_object_geometry(
        self,
        object_name,
        obj_scale,
        obj_trans,
        obj_rot,
        obj_bottom_scale=None,
        obj_bottom_trans=None,
        obj_bottom_rot=None,
    ):
        obj_mesh_path = os.path.join(
            self.obj_geo_root_folder, object_name + "_cleaned_simplified.obj"
        )

        obj_mesh_verts, obj_mesh_faces = self.apply_transformation_to_obj_geometry(
            obj_mesh_path, obj_scale, obj_rot, obj_trans
        )  # T X Nv X 3

        return obj_mesh_verts, obj_mesh_faces

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(
            x=obj_verts,
            feature_type=["deltas"],
            custom_basis=self.obj_bps.repeat(obj_trans.shape[0], 1, 1)
            + obj_trans[:, None, :],
        )["deltas"]  # T X N X 3

        return bps_object_geo

    def prep_bps_data(self):
        # R_bps = torch.tensor(
        #         [[1., 0., 0.],
        #         [0., 0., -1.],
        #         [0., 1., 0.]]).reshape(1, 3, 3)
        # R_bps = torch.tensor(
        #         [[1., 0., 0.],
        #         [0., 1., 0.],
        #         [0., 0., 1.]]).reshape(1, 3, 3)

        n_obj = 1024
        r_obj = 1.0  # Previous 0.6, cannot cover long objects.
        # n_sbj = 1024
        # r_sbj = 0.6
        # h_sbj = 2.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(
                1, -1, 3
            )
            # bps_sbj = rotate(sample_uniform_cylinder(n_points=n_sbj, radius=r_sbj, \
            #         height=h_sbj).reshape(1, -1, 3), R_bps.transpose(1, 2).cuda())

            bps = {
                "obj": bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            torch.save(bps, self.bps_path)

        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps["obj"]

    def extract_rest_pose_object_geometry_and_rotation(self):
        self.rest_pose_object_dict = {}

        for seq_idx in self.data_dict:
            seq_name = self.data_dict[seq_idx]["seq_name"]
            object_name = seq_name.split("_")[1]
            if object_name in ["vacuum", "mop"]:
                continue

            if object_name not in self.rest_pose_object_dict:
                obj_trans = self.data_dict[seq_idx]["obj_trans"][:, :, 0]  # T X 3
                obj_rot = self.data_dict[seq_idx]["obj_rot"]  # T X 3 X 3
                obj_scale = self.data_dict[seq_idx]["obj_scale"]  # T

                (
                    rest_verts,
                    obj_mesh_faces,
                    rest_pose_ori_rot,
                    rest_pose_ori_com_pos,
                    obj_trans_to_com_pos,
                ) = self.convert_rest_pose_obj_geometry(
                    object_name, obj_scale, obj_trans, obj_rot
                )

                self.rest_pose_object_dict[object_name] = {}
                self.rest_pose_object_dict[object_name]["ori_rotation"] = (
                    rest_pose_ori_rot  # 3 X 3
                )
                self.rest_pose_object_dict[object_name]["ori_trans"] = (
                    rest_pose_ori_com_pos  # 1 X 3
                )
                self.rest_pose_object_dict[object_name]["obj_trans_to_com_pos"] = (
                    obj_trans_to_com_pos  # 1 X 3
                )

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in self.data_dict:
            seq_name = self.data_dict[index]["seq_name"]

            object_name = seq_name.split("_")[1]

            # Skip vacuum, mop for now since they consist of two object parts.
            if object_name in ["vacuum", "mop"]:
                continue

            rest_pose_obj_data = self.rest_pose_object_dict[object_name]
            rest_pose_rot_mat = rest_pose_obj_data["ori_rotation"]  # 3 X 3
            # rest_pose_com_pos = rest_pose_obj_data['ori_trans'] # 1 X 3
            # obj_trans_to_com_pos = rest_pose_obj_data['obj_trans_to_com_pos'] # 1 X 3

            rest_obj_path = os.path.join(
                self.rest_object_geo_folder, object_name + ".ply"
            )
            mesh = trimesh.load_mesh(rest_obj_path)
            rest_verts = np.asarray(mesh.vertices)  # Nv X 3
            obj_mesh_faces = np.asarray(mesh.faces)  # Nf X 3
            rest_verts = torch.from_numpy(rest_verts).float()  # Nv X 3

            betas = self.data_dict[index]["betas"]  # 1 X 16
            gender = self.data_dict[index]["gender"]

            seq_root_trans = self.data_dict[index]["trans"]  # T X 3
            seq_root_orient = self.data_dict[index]["root_orient"]  # T X 3
            seq_pose_body = self.data_dict[index]["pose_body"].reshape(
                -1, 21, 3
            )  # T X 21 X 3

            rest_human_offsets = self.data_dict[index]["rest_offsets"]  # 22 X 3/24 X 3
            trans2joint = self.data_dict[index]["trans2joint"]  # 3

            # Used in old version without defining rest object geometry.
            seq_obj_trans = self.data_dict[index]["obj_trans"][:, :, 0]  # T X 3
            seq_obj_rot = self.data_dict[index]["obj_rot"]  # T X 3 X 3
            seq_obj_scale = self.data_dict[index]["obj_scale"]  # T

            seq_obj_verts, tmp_obj_faces = self.load_object_geometry(
                object_name, seq_obj_scale, seq_obj_trans, seq_obj_rot
            )  # T X Nv X 3, tensor
            seq_obj_com_pos = seq_obj_verts.mean(dim=1)  # T X 3

            # Convert the original rotation and translation to be relative to the rest pose object geometry.
            # obj_trans = self.data_dict[index]['obj_trans'][:, :, 0] - obj_trans_to_com_pos

            obj_trans = seq_obj_com_pos.clone().detach().cpu().numpy()

            rest_pose_rot_mat_rep = torch.from_numpy(rest_pose_rot_mat).float()[
                None, :, :
            ]  # 1 X 3 X 3
            obj_rot = torch.from_numpy(self.data_dict[index]["obj_rot"])  # T X 3 X 3
            obj_rot = torch.matmul(
                obj_rot,
                rest_pose_rot_mat_rep.repeat(obj_rot.shape[0], 1, 1).transpose(1, 2),
            )  # T X 3 X 3
            obj_rot = obj_rot.detach().cpu().numpy()

            num_steps = seq_root_trans.shape[0]
            # for start_t_idx in range(0, num_steps, self.window//2):
            for start_t_idx in range(0, num_steps, self.window // 4):
                end_t_idx = start_t_idx + self.window - 1

                if (
                    self.keep_same_len_window and end_t_idx >= num_steps
                ):  # This is the setting that each window has the same number of frames.
                    continue

                # if end_t_idx >= num_steps:
                #     # Discard short sequences, keep all the training data the same sequence length.
                #     end_t_idx = num_steps

                # Skip the segment that has a length < 30
                if end_t_idx - start_t_idx < 30:
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

                obj_x = obj_trans[start_t_idx : end_t_idx + 1].copy()  # T X 3
                obj_rot_mat = torch.from_numpy(
                    obj_rot[start_t_idx : end_t_idx + 1]
                ).float()  # T X 3 X 3
                obj_q = (
                    transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy()
                )  # T X 4

                # curr_obj_scale = torch.from_numpy(obj_scale[start_t_idx:end_t_idx+1]).float() # T

                # Canonicalize based on the first human pose's orientation.
                X, Q, new_obj_x, new_obj_q = rotate_at_frame_w_obj(
                    X[np.newaxis],
                    Q[np.newaxis],
                    obj_x[np.newaxis],
                    obj_q[np.newaxis],
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

                new_obj_rot_mat = transforms.quaternion_to_matrix(
                    torch.from_numpy(new_obj_q[0]).float()
                )  # T X 3 X 3 \

                cano_obj_mat = torch.matmul(
                    new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)
                )  # 3 X 3

                # obj_verts, tmp_obj_faces = self.load_object_geometry(object_name, curr_obj_scale.detach().cpu().numpy(), \
                #         new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy()) # T X Nv X 3, tensor

                obj_verts = self.load_object_geometry_w_rest_geo(
                    new_obj_rot_mat,
                    torch.from_numpy(new_obj_x[0]).float().to(new_obj_rot_mat.device),
                    rest_verts,
                )

                center_verts = obj_verts.mean(dim=1)  # T X 3

                # query = self.process_window_data(rest_human_offsets, trans2joint, \
                #     new_seq_root_trans, new_seq_root_orient.detach().cpu().numpy(), \
                #     new_seq_pose_body.detach().cpu().numpy(),  \
                #     new_obj_x[0], new_obj_rot_mat.detach().cpu().numpy(), \
                #     curr_obj_scale.detach().cpu().numpy(), center_verts)

                query = self.process_window_data(
                    rest_human_offsets,
                    trans2joint,
                    new_seq_root_trans,
                    new_seq_root_orient.detach().cpu().numpy(),
                    new_seq_pose_body.detach().cpu().numpy(),
                    new_obj_x[0],
                    new_obj_rot_mat.detach().cpu().numpy(),
                    center_verts,
                )

                # Compute BPS representation for this window
                # Save to numpy file
                dest_obj_bps_npy_path = os.path.join(
                    self.dest_obj_bps_npy_folder, seq_name + "_" + str(s_idx) + ".npy"
                )

                if not os.path.exists(dest_obj_bps_npy_path):
                    # object_bps = self.compute_object_geo_bps(obj_verts[0:1], center_verts[0:1]) # For the setting that only computes the first frame.
                    object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                    np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

                curr_global_jpos = query["global_jpos"].detach().cpu().numpy()
                curr_global_jvel = query["global_jvel"].detach().cpu().numpy()
                curr_global_rot_6d = query["global_rot_6d"].detach().cpu().numpy()

                self.window_data_dict[s_idx]["cano_obj_mat"] = (
                    cano_obj_mat.detach().cpu().numpy()
                )

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

                self.window_data_dict[s_idx]["obj_rot_mat"] = (
                    query["obj_rot_mat"].detach().cpu().numpy()
                )

                self.window_data_dict[s_idx]["window_obj_com_pos"] = (
                    query["window_obj_com_pos"].detach().cpu().numpy()
                )

                self.window_data_dict[s_idx]["rest_human_offsets"] = rest_human_offsets

                s_idx += 1

            # if s_idx > 32:
            #     break

    def extract_min_max_mean_std_from_data(self):
        all_global_jpos_data = []
        all_global_jvel_data = []

        all_obj_com_pos_data = []

        for s_idx in self.window_data_dict:
            curr_window_data = self.window_data_dict[s_idx]["motion"]  # T X D

            all_global_jpos_data.append(curr_window_data[:, : 24 * 3])
            all_global_jvel_data.append(curr_window_data[:, 24 * 3 : 2 * 24 * 3])

            curr_com_pos = self.window_data_dict[s_idx]["window_obj_com_pos"]  # T X 3

            all_obj_com_pos_data.append(curr_com_pos)

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(
            -1, 72
        )  # (N*T) X 72
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 72)

        all_obj_com_pos_data = np.vstack(all_obj_com_pos_data).reshape(
            -1, 3
        )  # (N*T) X 3

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        min_com_pos = all_obj_com_pos_data.min(axis=0)
        max_com_pos = all_obj_com_pos_data.max(axis=0)

        stats_dict = {}
        stats_dict["global_jpos_min"] = min_jpos
        stats_dict["global_jpos_max"] = max_jpos
        stats_dict["global_jvel_min"] = min_jvel
        stats_dict["global_jvel_max"] = max_jvel

        stats_dict["obj_com_pos_min"] = min_com_pos
        stats_dict["obj_com_pos_max"] = max_com_pos

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

    def normalize_obj_pos_min_max(self, ori_obj_pos):
        # ori_jpos: T X 3
        if ori_obj_pos.dim() == 3:  # BS X T X 3
            normalized_jpos = (
                ori_obj_pos - self.obj_pos_min.to(ori_obj_pos.device)[None]
            ) / (
                self.obj_pos_max.to(ori_obj_pos.device)[None]
                - self.obj_pos_min.to(ori_obj_pos.device)[None]
            )
        else:
            normalized_jpos = (
                ori_obj_pos - self.obj_pos_min.to(ori_obj_pos.device)
            ) / (
                self.obj_pos_max.to(ori_obj_pos.device)
                - self.obj_pos_min.to(ori_obj_pos.device)
            )

        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        return normalized_jpos  # T X 3 /BS X T X 3

    def de_normalize_obj_pos_min_max(self, normalized_obj_pos):
        normalized_obj_pos = (normalized_obj_pos + 1) * 0.5  # [0, 1] range
        if normalized_obj_pos.dim() == 3:
            de_jpos = (
                normalized_obj_pos
                * (
                    self.obj_pos_max.to(normalized_obj_pos.device)[None]
                    - self.obj_pos_min.to(normalized_obj_pos.device)[None]
                )
                + self.obj_pos_min.to(normalized_obj_pos.device)[None]
            )
        else:
            de_jpos = normalized_obj_pos * (
                self.obj_pos_max.to(normalized_obj_pos.device)
                - self.obj_pos_min.to(normalized_obj_pos.device)
            ) + self.obj_pos_min.to(normalized_obj_pos.device)

        return de_jpos  # T X 3

    def normalize_jpos_min_max_hand_foot(self, ori_jpos, hand_only=True):
        # ori_jpos: BS X T X 2 X 3
        lhand_idx = 22
        rhand_idx = 23

        lfoot_idx = 10
        rfoot_idx = 11

        bs = ori_jpos.shape[0]
        num_steps = ori_jpos.shape[1]
        ori_jpos = ori_jpos.reshape(bs, num_steps, -1)  # BS X T X (2*3)

        if hand_only:
            hand_foot_jpos_max = torch.cat(
                (
                    self.global_jpos_max[0, lhand_idx],
                    self.global_jpos_max[0, rhand_idx],
                ),
                dim=0,
            )  # (3*4)

            hand_foot_jpos_min = torch.cat(
                (
                    self.global_jpos_min[0, lhand_idx],
                    self.global_jpos_min[0, rhand_idx],
                ),
                dim=0,
            )
        else:
            hand_foot_jpos_max = torch.cat(
                (
                    self.global_jpos_max[0, lhand_idx],
                    self.global_jpos_max[0, rhand_idx],
                    self.global_jpos_max[0, lfoot_idx],
                    self.global_jpos_max[0, rfoot_idx],
                ),
                dim=0,
            )  # (3*4)

            hand_foot_jpos_min = torch.cat(
                (
                    self.global_jpos_min[0, lhand_idx],
                    self.global_jpos_min[0, rhand_idx],
                    self.global_jpos_min[0, lfoot_idx],
                    self.global_jpos_min[0, rfoot_idx],
                ),
                dim=0,
            )

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]
        normalized_jpos = (ori_jpos - hand_foot_jpos_min.to(ori_jpos.device)) / (
            hand_foot_jpos_max.to(ori_jpos.device)
            - hand_foot_jpos_min.to(ori_jpos.device)
        )
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        normalized_jpos = normalized_jpos.reshape(bs, num_steps, -1, 3)

        return normalized_jpos  # BS X T X 2 X 3

    def de_normalize_jpos_min_max_hand_foot(self, normalized_jpos, hand_only=True):
        # normalized_jpos: BS X T X (3*4)
        lhand_idx = 22
        rhand_idx = 23

        lfoot_idx = 10
        rfoot_idx = 11

        bs, num_steps, _ = normalized_jpos.shape

        normalized_jpos = (normalized_jpos + 1) * 0.5  # [0, 1] range

        if hand_only:
            hand_foot_jpos_max = torch.cat(
                (
                    self.global_jpos_max[0, lhand_idx],
                    self.global_jpos_max[0, rhand_idx],
                ),
                dim=0,
            )  # (3*4)

            hand_foot_jpos_min = torch.cat(
                (
                    self.global_jpos_min[0, lhand_idx],
                    self.global_jpos_min[0, rhand_idx],
                ),
                dim=0,
            )
        else:
            hand_foot_jpos_max = torch.cat(
                (
                    self.global_jpos_max[0, lhand_idx],
                    self.global_jpos_max[0, rhand_idx],
                    self.global_jpos_max[0, lfoot_idx],
                    self.global_jpos_max[0, rfoot_idx],
                ),
                dim=0,
            )  # (3*4)

            hand_foot_jpos_min = torch.cat(
                (
                    self.global_jpos_min[0, lhand_idx],
                    self.global_jpos_min[0, rhand_idx],
                    self.global_jpos_min[0, lfoot_idx],
                    self.global_jpos_min[0, rfoot_idx],
                ),
                dim=0,
            )

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]

        de_jpos = normalized_jpos * (
            hand_foot_jpos_max.to(normalized_jpos.device)
            - hand_foot_jpos_min.to(normalized_jpos.device)
        ) + hand_foot_jpos_min.to(normalized_jpos.device)

        return de_jpos.reshape(bs, num_steps, -1, 3)  # BS X T X 4(2) X 3

    def process_window_data(
        self,
        rest_human_offsets,
        trans2joint,
        seq_root_trans,
        seq_root_orient,
        seq_pose_body,
        obj_trans,
        obj_rot,
        center_verts,
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

        # window_obj_scale = torch.from_numpy(obj_scale[random_t_idx:end_t_idx+1]).float().cuda() # T
        window_obj_rot_mat = (
            torch.from_numpy(obj_rot[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X 3 X 3
        window_obj_trans = (
            torch.from_numpy(obj_trans[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X 3

        window_center_verts = center_verts[random_t_idx : end_t_idx + 1].to(
            window_obj_trans.device
        )

        # Move the first frame's object position to zero.
        # move_to_zero_trans = window_center_verts[0:1, :].clone() # 1 X 3
        # move_to_zero_trans[:, 2] = 0

        # Move thr first frame's human position to zero.
        move_to_zero_trans = window_root_trans[0:1, :].clone()  # 1 X 3
        move_to_zero_trans[:, 2] = 0

        # Move motion and object translation to make the initial pose trans 0.
        window_root_trans = window_root_trans - move_to_zero_trans
        window_obj_trans = window_obj_trans - move_to_zero_trans
        window_center_verts = window_center_verts - move_to_zero_trans

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

        query["obj_trans"] = window_obj_trans  # T' X 3
        query["obj_rot_mat"] = window_obj_rot_mat  # T' X 3 X 3
        # query['obj_scale'] = window_obj_scale # T'

        query["window_obj_com_pos"] = window_center_verts  # T X 3

        # import pdb
        # pdb.set_trace()

        return query

    def __len__(self):
        return len(self.window_data_dict)

    def prep_rel_obj_rot_mat(self, obj_rot_mat):
        # obj_rot_mat: T X 3 X 3
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = obj_rot_mat[:, 0:1].repeat(
                1, timesteps, 1, 1
            )  # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(
                obj_rot_mat, init_obj_rot_mat.transpose(2, 3)
            )  # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry.
            init_obj_rot_mat = obj_rot_mat[0:1].repeat(timesteps, 1, 1)  # T X 3 X 3
            rel_rot_mat = torch.matmul(
                obj_rot_mat, init_obj_rot_mat.transpose(1, 2)
            )  # T X 3 X 3

        return rel_rot_mat

    def prep_rel_obj_rot_mat_w_reference_mat(self, obj_rot_mat, ref_rot_mat):
        # obj_rot_mat: T X 3 X 3 / BS X T X 3 X 3
        # ref_rot_mat: BS X 1 X 3 X 3/ 1 X 3 X 3
        if obj_rot_mat.dim() == 4:
            timesteps = obj_rot_mat.shape[1]

            init_obj_rot_mat = ref_rot_mat.repeat(1, timesteps, 1, 1)  # BS X T X 3 X 3
            rel_rot_mat = torch.matmul(
                obj_rot_mat, init_obj_rot_mat.transpose(2, 3)
            )  # BS X T X 3 X 3
        else:
            timesteps = obj_rot_mat.shape[0]

            # Compute relative rotation matrix with respect to the first frame's object geometry.
            init_obj_rot_mat = ref_rot_mat.repeat(timesteps, 1, 1)  # T X 3 X 3
            rel_rot_mat = torch.matmul(
                obj_rot_mat, init_obj_rot_mat.transpose(1, 2)
            )  # T X 3 X 3

        return rel_rot_mat

    def rel_rot_to_seq(self, rel_rot_mat, obj_rot_mat):
        # rel_rot_mat: BS X T X 3 X 3
        # obj_rot_mat: BS X T X 3 X 3 (only use the first frame's rotation)
        timesteps = rel_rot_mat.shape[1]

        # Compute relative rotation matrix with respect to the first frame's object geometry.
        init_obj_rot_mat = obj_rot_mat[:, 0:1].repeat(
            1, timesteps, 1, 1
        )  # BS X T X 3 X 3
        obj_rot_mat = torch.matmul(rel_rot_mat, init_obj_rot_mat.to(rel_rot_mat.device))

        return obj_rot_mat

    def com_to_obj_trans(self, seq_com_pos, first_frame_obj_com2trans):
        # seq_com_pos: BS X T X 3
        # first_frame_obj_com2trans: BS X 1 X 3

        obj_trans_for_vis = seq_com_pos - first_frame_obj_com2trans.to(
            seq_com_pos.device
        )

        return obj_trans_for_vis

    def get_nn_pts(self, object_name, window_obj_rot_mat, obj_com_pos):
        # window_obj_rot_mat: T X 3 X 3
        # obj_com_pos: T X 3
        window_obj_rot_mat = torch.from_numpy(window_obj_rot_mat).float()
        obj_com_pos = torch.from_numpy(obj_com_pos).float()

        rest_obj_bps_npy_path = os.path.join(
            self.rest_object_geo_folder, object_name + ".npy"
        )
        rest_obj_bps_data = np.load(rest_obj_bps_npy_path)  # 1 X 1024 X 3
        nn_pts_on_mesh = self.obj_bps + torch.from_numpy(rest_obj_bps_data).float().to(
            self.obj_bps.device
        )  # 1 X 1024 X 3
        nn_pts_on_mesh = nn_pts_on_mesh.squeeze(0)  # 1024 X 3

        # Compute point positions for each frame
        sampled_nn_pts_on_mesh = nn_pts_on_mesh[None].repeat(
            window_obj_rot_mat.shape[0], 1, 1
        )
        transformed_obj_nn_pts = (
            window_obj_rot_mat.bmm(sampled_nn_pts_on_mesh.transpose(1, 2))
            + obj_com_pos[:, :, None]
        )
        transformed_obj_nn_pts = transformed_obj_nn_pts.transpose(1, 2)  # T X K X 3

        return transformed_obj_nn_pts

    def compute_object_keypoints_min_max(self):
        min_obj_pts = None
        max_obj_pts = None
        for idx in self.window_data_dict:
            window_obj_rot_mat = self.window_data_dict[idx]["obj_rot_mat"]
            window_obj_com_pos = self.window_data_dict[idx]["window_obj_com_pos"]

            seq_name = self.window_data_dict[idx]["seq_name"]
            object_name = seq_name.split("_")[1]

            transformed_obj_pts = self.get_nn_pts(
                object_name, window_obj_rot_mat, window_obj_com_pos
            )  # T X 1024 X 3

            transformed_obj_pts = transformed_obj_pts.reshape(-1, 3)  # (T*1024) X 3

            if min_obj_pts is None:
                min_obj_pts = transformed_obj_pts.min(dim=0)[0][None]
            else:
                min_obj_pts = torch.cat((min_obj_pts, transformed_obj_pts), dim=0).min(
                    dim=0
                )[0][None]

            if max_obj_pts is None:
                max_obj_pts = transformed_obj_pts.max(dim=0)[0][None]
            else:
                max_obj_pts = torch.cat((max_obj_pts, transformed_obj_pts), dim=0).max(
                    dim=0
                )[0][None]

        stats_dict = {}
        stats_dict["obj_pts_min"] = min_obj_pts.detach().cpu().numpy()  # 1 X 3
        stats_dict["obj_pts_max"] = max_obj_pts.detach().cpu().numpy()  # 1 X 3

        return stats_dict

    def normalize_obj_keypoints_min_max(self, ori_obj_pts):
        # ori_obj_pts: BS X T X K X 3
        normalized_obj_pts = (
            ori_obj_pts - self.obj_keypoints_min.to(ori_obj_pts.device)[None, None]
        ) / (
            self.obj_keypoints_max.to(ori_obj_pts.device)[None, None]
            - self.obj_keypoints_min.to(ori_obj_pts.device)[None, None]
        )

        normalized_obj_pts = normalized_obj_pts * 2 - 1  # [-1, 1] range

        return normalized_obj_pts  # BS X T X K X 3

    def de_normalize_obj_keypoints_min_max(self, normalized_obj_pts):
        # ori_obj_pts: BS X T X K X 3
        normalized_obj_pts = (normalized_obj_pts + 1) * 0.5  # [0, 1] range

        de_obj_pts = (
            normalized_obj_pts
            * (
                self.obj_keypoints_max.to(normalized_obj_pts.device)[None, None]
                - self.obj_keypoints_min.to(normalized_obj_pts.device)[None, None]
            )
            + self.obj_keypoints_min.to(normalized_obj_pts.device)[None, None]
        )

        return de_obj_pts  # BS X T X K X 3

    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        standing_flag = self.standing_flag_dict[index]
        object_static_flag = self.object_static_flag_dict[index]["static_flag"].float()
        left_switch_mask = self.object_static_flag_dict[index][
            "left_switch_mask"
        ].float()
        right_switch_mask = self.object_static_flag_dict[index][
            "right_switch_mask"
        ].float()
        root_traj_xy_ori = torch.from_numpy(self.root_traj_xy_ori_dict[index]).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        if "GRAB_contact_labels" in self.window_data_dict[index]:
            seq_name = seq_name.replace("/", "_")[:-4]  # like: 's1_binoculars_lift'
        object_name = seq_name.split("_")[1]

        window_s_idx = self.window_data_dict[index]["start_t_idx"]
        window_e_idx = self.window_data_dict[index]["end_t_idx"]
        if "GRAB_contact_labels" in self.window_data_dict[index]:
            contact_labels = self.window_data_dict[index]["GRAB_contact_labels"]
        else:
            contact_npy_path = os.path.join(self.contact_npy_folder, seq_name + ".npy")
            contact_npy_data = np.load(
                contact_npy_path
            )  # T X 4 (lhand, rhand, lfoot, rfoot)
            contact_labels = contact_npy_data[window_s_idx : window_e_idx + 1]  # W
        contact_labels = torch.from_numpy(contact_labels).float()

        trans2joint = self.window_data_dict[index]["trans2joint"]

        rest_human_offsets = self.window_data_dict[index]["rest_human_offsets"]

        if self.use_first_frame_bps or self.use_random_frame_bps:
            if (
                (not self.train)
                or self.use_object_splits
                or self.input_language_condition
            ):
                ori_w_idx = self.window_data_dict[index]["ori_w_idx"]
                obj_bps_npy_path = os.path.join(
                    self.dest_obj_bps_npy_folder,
                    seq_name + "_" + str(ori_w_idx) + ".npy",
                )
            else:
                obj_bps_npy_path = os.path.join(
                    self.dest_obj_bps_npy_folder,
                    seq_name + "_" + str(index) + ".npy",
                )
        else:
            obj_bps_npy_path = os.path.join(
                self.rest_object_geo_folder, object_name + ".npy"
            )

        obj_bps_data = np.load(obj_bps_npy_path)  # T X N X 3

        if self.use_first_frame_bps:
            random_sampled_t_idx = 0
            obj_bps_data = obj_bps_data[0:1]  # 1 X N X 3
        elif self.use_random_frame_bps:
            random_sampled_t_idx = random.sample(list(range(obj_bps_data.shape[0])), 1)[
                0
            ]
            obj_bps_data = obj_bps_data[
                random_sampled_t_idx : random_sampled_t_idx + 1
            ]  # 1 X N X 3

        obj_bps_data = torch.from_numpy(obj_bps_data)

        obj_com_pos = torch.from_numpy(
            self.window_data_dict[index]["window_obj_com_pos"]
        ).float()

        normalized_obj_com_pos = self.normalize_obj_pos_min_max(obj_com_pos)

        # Prepare object motion information
        window_obj_rot_mat = torch.from_numpy(
            self.window_data_dict[index]["obj_rot_mat"]
        ).float()

        # Prepare relative rotation
        if self.use_first_frame_bps:
            reference_obj_rot_mat = window_obj_rot_mat[0:1]
            window_rel_obj_rot_mat = self.prep_rel_obj_rot_mat(window_obj_rot_mat)
        elif self.use_random_frame_bps:
            reference_obj_rot_mat = window_obj_rot_mat[
                random_sampled_t_idx : random_sampled_t_idx + 1
            ]
            window_rel_obj_rot_mat = self.prep_rel_obj_rot_mat_w_reference_mat(
                window_obj_rot_mat,
                window_obj_rot_mat[random_sampled_t_idx : random_sampled_t_idx + 1],
            )

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

        wrist_relative = self.wrist_relative_dict[index].float()  # 18
        wrist_relative = self.prep_rel_wrist_relative(
            wrist_relative,
            window_obj_rot_mat[random_sampled_t_idx : random_sampled_t_idx + 1],
        )
        # set the wrist relative to zero if the not in contact.
        wrist_relative[:, 0:9] *= contact_labels[:, 0:1]  # left hand
        wrist_relative[:, 9:18] *= contact_labels[:, 1:2]  # right hand

        # Compute foot contact
        positions = data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)[
            None
        ]  # 1 X T X 24 X 3
        feet = positions[:, :, (7, 8, 10, 11)]  # BS X T X 4 X 3
        feetv = torch.zeros(feet.shape[:3])  # BS X T X 4
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = feetv < 0.01
        contacts = contacts.squeeze(0)

        # Prepare object keypoints for each frame.
        if self.use_object_keypoints:
            # Load rest pose BPS and compute nn points on the object.
            rest_obj_bps_npy_path = os.path.join(
                self.rest_object_geo_folder, object_name + ".npy"
            )
            rest_obj_bps_data = np.load(rest_obj_bps_npy_path)  # 1 X 1024 X 3
            nn_pts_on_mesh = self.obj_bps + torch.from_numpy(
                rest_obj_bps_data
            ).float().to(self.obj_bps.device)  # 1 X 1024 X 3
            nn_pts_on_mesh = nn_pts_on_mesh.squeeze(0)  # 1024 X 3

            # Random sample 100 points used for training
            sampled_vidxs = random.sample(list(range(1024)), 100)
            sampled_nn_pts_on_mesh = nn_pts_on_mesh[sampled_vidxs]  # K X 3

            # During inference, use all 1024 points for penetration loss, contact loss?
            # sampled_nn_pts_on_mesh = nn_pts_on_mesh # K X 3

            rest_pose_obj_nn_pts = sampled_nn_pts_on_mesh.clone()

            # Compute point positions for each frame
            sampled_nn_pts_on_mesh = sampled_nn_pts_on_mesh[None].repeat(
                window_obj_rot_mat.shape[0], 1, 1
            )
            transformed_obj_nn_pts = (
                window_obj_rot_mat.bmm(sampled_nn_pts_on_mesh.transpose(1, 2))
                + obj_com_pos[:, :, None]
            )
            transformed_obj_nn_pts = transformed_obj_nn_pts.transpose(1, 2)  # T X K X 3

            # normalized_obj_keypoints = self.normalize_obj_keypoints_min_max(transformed_obj_nn_pts[None]).squeeze(0) # T X K X 3

            if transformed_obj_nn_pts.shape[0] < self.window:
                paded_transformed_obj_nn_pts = torch.cat(
                    (
                        transformed_obj_nn_pts,
                        torch.zeros(
                            self.window - transformed_obj_nn_pts.shape[0],
                            transformed_obj_nn_pts.shape[1],
                            transformed_obj_nn_pts.shape[2],
                        ),
                    ),
                    dim=0,
                )

                # paded_normalized_obj_nn_pts = torch.cat((normalized_obj_keypoints, \
                #                 torch.zeros(self.window-normalized_obj_keypoints.shape[0], \
                #                 normalized_obj_keypoints.shape[1], normalized_obj_keypoints.shape[2])), dim=0)
            else:
                paded_transformed_obj_nn_pts = transformed_obj_nn_pts
                # paded_normalized_obj_nn_pts = normalized_obj_keypoints

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

            paded_normalized_obj_com_pos = torch.cat(
                (normalized_obj_com_pos, torch.zeros(self.window - actual_steps, 3)),
                dim=0,
            )
            paded_obj_com_pos = torch.cat(
                (
                    torch.from_numpy(
                        self.window_data_dict[index]["window_obj_com_pos"]
                    ).float(),
                    torch.zeros(self.window - actual_steps, 3),
                ),
                dim=0,
            )

            paded_obj_rot_mat = torch.cat(
                (window_obj_rot_mat, torch.zeros(self.window - actual_steps, 3, 3)),
                dim=0,
            )

            if self.use_random_frame_bps or self.use_first_frame_bps:
                paded_rel_obj_rot_mat = torch.cat(
                    (
                        window_rel_obj_rot_mat,
                        torch.zeros(self.window - actual_steps, 3, 3),
                    ),
                    dim=0,
                )

            paded_contacts = torch.cat(
                (contacts, torch.zeros(self.window - actual_steps, 4)), dim=0
            )

            paded_contact_labels = torch.cat(
                (contact_labels, torch.zeros(self.window - actual_steps, 4)), dim=0
            )
            paded_wrist_relative = torch.cat(
                (wrist_relative, torch.zeros(self.window - actual_steps, 18)), dim=0
            )
            paded_object_static_flag = torch.cat(
                (object_static_flag, torch.zeros(self.window - actual_steps, 1)), dim=0
            )
            paded_left_switch_mask = torch.cat(
                (left_switch_mask, torch.zeros(self.window - actual_steps, 1)), dim=0
            )
            paded_right_switch_mask = torch.cat(
                (right_switch_mask, torch.zeros(self.window - actual_steps, 1)), dim=0
            )
            paded_root_traj_xy_ori = torch.cat(
                (
                    root_traj_xy_ori,
                    torch.zeros(self.window - actual_steps, root_traj_xy_ori.shape[-1]),
                ),
                dim=0,
            )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input

            paded_normalized_obj_com_pos = normalized_obj_com_pos
            paded_obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()

            paded_obj_rot_mat = window_obj_rot_mat

            if self.use_random_frame_bps or self.use_first_frame_bps:
                paded_rel_obj_rot_mat = window_rel_obj_rot_mat

            paded_contacts = contacts

            paded_contact_labels = contact_labels
            paded_wrist_relative = wrist_relative
            paded_object_static_flag = object_static_flag
            paded_left_switch_mask = left_switch_mask
            paded_right_switch_mask = right_switch_mask
            paded_root_traj_xy_ori = root_traj_xy_ori

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        if self.use_first_frame_bps or self.use_random_frame_bps:
            data_input_dict["ori_obj_motion"] = torch.cat(
                (paded_obj_com_pos, paded_rel_obj_rot_mat.reshape(-1, 9)), dim=-1
            )  # T X (3+9)
            data_input_dict["obj_motion"] = torch.cat(
                (paded_normalized_obj_com_pos, paded_rel_obj_rot_mat.reshape(-1, 9)),
                dim=-1,
            )  # T X (3+9)

            data_input_dict["input_obj_bps"] = obj_bps_data  # 1 X 1024 X 3
            # data_input_dict['first_obj_com_pos'] = obj_com_pos[0:1] # 1 X 3
        else:
            data_input_dict["ori_obj_motion"] = torch.cat(
                (paded_obj_com_pos, paded_obj_rot_mat.reshape(-1, 9)), dim=-1
            )  # T X (3+9)
            data_input_dict["obj_motion"] = torch.cat(
                (paded_normalized_obj_com_pos, paded_obj_rot_mat.reshape(-1, 9)), dim=-1
            )  # T X (3+9)
            data_input_dict["input_obj_bps"] = obj_bps_data[0:1]  # 1 X 1024 X 3
            # data_input_dict['first_obj_com_pos'] = torch.zeros(1, 3).float() # 1 X 3

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat  # T X 3 X 3
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["betas"] = torch.from_numpy(
            self.window_data_dict[index]["betas"]
        ).float()
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        data_input_dict["rest_human_offsets"] = rest_human_offsets

        # data_input_dict['first_frame_obj_com2trans'] = first_frame_obj_com2trans # 1 X 3

        data_input_dict["contact_labels"] = paded_contact_labels.float()  # T X 4

        if self.use_first_frame_bps or self.use_random_frame_bps:
            data_input_dict["reference_obj_rot_mat"] = reference_obj_rot_mat

        data_input_dict["s_idx"] = window_s_idx
        data_input_dict["e_idx"] = window_e_idx

        if self.input_language_condition:
            # Load language annotation
            seq_text_anno = self.load_language_annotation(seq_name)
            data_input_dict["text"] = seq_text_anno  # a string
            if standing_flag:
                data_input_dict["text"] += " Return to standing pose."

        if self.use_object_keypoints:
            # data_input_dict['ori_obj_keypoints'] = paded_transformed_obj_nn_pts.reshape(self.window, -1) # T X K X 3 -> T X (K*3)
            # data_input_dict['obj_keypoints'] = paded_normalized_obj_nn_pts.reshape(self.window, -1)
            data_input_dict["ori_obj_keypoints"] = (
                paded_transformed_obj_nn_pts  # T X K X 3
            )

            data_input_dict["rest_pose_obj_pts"] = rest_pose_obj_nn_pts  # K X 3

        data_input_dict["feet_contact"] = paded_contacts.float()

        data_input_dict["wrist_relative"] = paded_wrist_relative
        data_input_dict["object_static_flag"] = paded_object_static_flag
        data_input_dict["left_switch_mask"] = paded_left_switch_mask
        data_input_dict["right_switch_mask"] = paded_right_switch_mask
        data_input_dict["root_traj_xy_ori"] = paded_root_traj_xy_ori  # T X 6

        return data_input_dict
        # data_input_dict['motion']: T X (22*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3
