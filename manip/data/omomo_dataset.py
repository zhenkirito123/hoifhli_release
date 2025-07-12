import json
import os
import pickle
import sys
import time
from collections import defaultdict

import joblib
import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
from bps_torch.bps import bps_torch
from tqdm import tqdm

from manip.data.hand_foot_dataset import (
    GraspDataset,
    HandFootManipDataset,
    get_smpl_parents,
)


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


class OMOMODataset(HandFootManipDataset):
    def __init__(
        self,
        train,
        window=120,
        use_window_bps=False,
        use_object_splits=False,
        use_joints24=False,
        load_ds=True,
    ):
        self.load_ds = load_ds
        self.parents_wholebody = np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/smpl_all_models/smplx_parents_52.npy",
            )
        )

        self.train = train

        self.window = window

        # self.use_window_bps = True

        self.use_joints24 = True
        self.joint_num = 24
        self.lhand_idx = 20
        self.rhand_idx = 21

        self.parents = get_smpl_parents()  # 52

        self.build_paths()

        dest_obj_bps_npy_folder = os.path.join(
            self.data_root_folder, "object_bps_npy_files_joints24"
        )
        dest_obj_bps_npy_folder_for_test = os.path.join(
            self.data_root_folder, "object_bps_npy_files_for_eval_joints24"
        )

        dest_ambient_sensor_npy_folder = os.path.join(
            self.data_root_folder, "ambient_sensor_npy_files"
        )
        dest_ambient_sensor_npy_folder_for_test = os.path.join(
            self.data_root_folder, "ambient_sensor_npy_files_for_eval"
        )

        dest_proximity_sensor_npy_folder = os.path.join(
            self.data_root_folder, "proximity_sensor_npy_files"
        )
        dest_proximity_sensor_npy_folder_for_test = os.path.join(
            self.data_root_folder, "proximity_sensor_npy_files_for_eval"
        )

        for path in [
            dest_obj_bps_npy_folder,
            dest_obj_bps_npy_folder_for_test,
            dest_ambient_sensor_npy_folder,
            dest_ambient_sensor_npy_folder_for_test,
            dest_proximity_sensor_npy_folder,
            dest_proximity_sensor_npy_folder_for_test,
        ]:
            if not os.path.exists(path):
                os.makedirs(path)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder
            self.dest_ambient_sensor_npy_folder = dest_ambient_sensor_npy_folder
            self.dest_proximity_sensor_npy_folder = dest_proximity_sensor_npy_folder
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test
            self.dest_ambient_sensor_npy_folder = (
                dest_ambient_sensor_npy_folder_for_test
            )
            self.dest_proximity_sensor_npy_folder = (
                dest_proximity_sensor_npy_folder_for_test
            )

        if self.train:
            processed_data_path = os.path.join(
                self.data_root_folder,
                "train_diffusion_manip_window_" + str(self.window) + "_cano_joints24.p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "train_contact_label.p"
            )
        else:
            processed_data_path = os.path.join(
                self.data_root_folder,
                "test_diffusion_manip_window_"
                + str(self.window)
                + "_processed_joints24.p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "test_contact_label.p"
            )

        min_max_mean_std_data_path = os.path.join(
            self.data_root_folder,
            "min_max_mean_std_data_window_" + str(120) + "_cano_joints24.p",
        )

        self.prep_bps_data()
        self.prep_sensor_data()
        self.prep_obj_indices()

        if self.load_ds:
            if os.path.exists(processed_data_path):
                self.window_data_dict = joblib.load(processed_data_path)

                if len(os.listdir(self.dest_ambient_sensor_npy_folder)) == 0:
                    print("Compute ambient sensor for all windows...")
                    self.compute_ambient_sensor_all()

                if len(os.listdir(self.dest_proximity_sensor_npy_folder)) == 0:
                    print("Compute proximity sensor for all windows...")
                    self.compute_proximity_sensor_all()

                # if not self.train:
                # Mannually enable this. For testing data (discarded some sequences)
                # self.get_bps_from_window_data_dict()
            else:
                raise ValueError(
                    "Cannot find processed_data_path:{0}, run process_\{dataset\}.py".format(
                        processed_data_path
                    )
                )
        else:
            self.window_data_dict = {}

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            raise ValueError(
                "Cannot find min_max_mean_std_data_path:{0}, run process_\{dataset\}.py".format(
                    min_max_mean_std_data_path
                )
            )

        self.global_jpos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_min"])
            .float()
            .reshape(-1, 3)[None]
        )
        self.global_jpos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_max"])
            .float()
            .reshape(-1, 3)[None]
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

    def get_contact_label(self):
        self.contact_label_dict = {}
        for s_idx in tqdm(range(len(self.window_data_dict))):
            real_contact_label = self.get_contact_label_single(s_idx)
            self.contact_label_dict[s_idx] = real_contact_label

    def get_contact_label_single(self, s_idx):
        window_data = self.window_data_dict[s_idx]
        seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
        obj_name = seq_name.split("_")[1]

        global_jpos = torch.from_numpy(
            window_data["motion"][:, : self.joint_num * 3]
        ).reshape(-1, self.joint_num, 3)  # T X J X 3
        # global_rot_6d = torch.from_numpy(window_data['motion'][:, 2*self.joint_num*3:]).reshape(-1, self.joint_num-2, 6) # T X J X 6
        # global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # T X J X 3 X 3
        left_wrist_pos = global_jpos[:, 20, :]  # T X 3
        right_wrist_pos = global_jpos[:, 21, :]  # T X 3

        obj_scale = window_data["obj_scale"]
        obj_trans = window_data["obj_trans"]
        obj_rot = window_data["obj_rot_mat"]
        if obj_name in ["mop", "vacuum"]:
            obj_bottom_scale = window_data["obj_bottom_scale"]
            obj_bottom_trans = window_data["obj_bottom_trans"]
            obj_bottom_rot = window_data["obj_bottom_rot_mat"]
        else:
            obj_bottom_scale = None
            obj_bottom_trans = None
            obj_bottom_rot = None
        obj_verts_, _ = self.load_object_geometry(
            obj_name,
            obj_scale,
            obj_trans,
            obj_rot,
            obj_bottom_scale,
            obj_bottom_trans,
            obj_bottom_rot,
        )  # T X Nv X 3, tensor

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
        for i in range(left_wrist_pos.shape[0]):
            left_dis = torch.min(torch.norm(obj_verts_[i] - left_wrist_pos[i], dim=1))
            right_dis = torch.min(torch.norm(obj_verts_[i] - right_wrist_pos[i], dim=1))
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
        for i in range(left_wrist_pos.shape[0] - 1, -1, -1):
            left_dis = left_dises[i]
            right_dis = right_dises[i]
            if left_dis < dis_threshold:
                if left_end_frame == -1:
                    left_end_frame = i
            if right_dis < dis_threshold:
                if right_end_frame == -1:
                    right_end_frame = i

        left_contact = left_cnt > 15
        right_contact = right_cnt > 15

        real_contact_label = np.zeros(
            (left_wrist_pos.shape[0], 2)
        )  # T X 2, 0: left hand, 1: right hand
        if left_contact:
            real_contact_label[left_begin_frame : left_end_frame + 1, 0] = 1
        if right_contact:
            real_contact_label[right_begin_frame : right_end_frame + 1, 1] = 1

        return real_contact_label

    def prep_obj_indices(
        self,
    ):
        if self.train:
            file_name = "train_obj_indices.pkl"
        else:
            file_name = "test_obj_indices.pkl"
        data_path = os.path.join(self.data_root_folder, file_name)
        if os.path.exists(data_path):
            return
        self.object_lists = defaultdict(list)
        # train: ['floorlamp', 'suitcase', 'tripod', 'clothesstand', 'mop', 'plasticbox', 'smalltable', 'monitor', 'smallbox', 'largebox', 'whitechair', 'trashcan', 'largetable', 'vacuum', 'woodchair']
        # test: ['suitcase', 'monitor', 'largebox', 'smallbox', 'mop', 'smalltable', 'trashcan', 'largetable', 'woodchair', 'clothesstand', 'plasticbox', 'floorlamp', 'vacuum', 'tripod', 'whitechair']
        # "clothesstand",
        # "floorlamp",
        # "largetable",
        # "mop",
        # "plasticbox",
        # "suitcase",
        # "tripod",
        # "vacuum",
        # "monitor",
        # "trashcan",
        # "woodchair",
        # "smalltable",
        # "whitechair",
        # "largebox",
        # "smallbox",

        for path in os.listdir(self.dest_ambient_sensor_npy_folder):
            if path.endswith(".npy"):
                obj_name = path[:-4].split("_")[1]
                idx = path[:-4].split("_")[-1]

                self.object_lists[obj_name].append(int(idx))
        print("Object lists:", self.object_lists.keys())
        pickle.dump(self.object_lists, open(data_path, "wb"))

    def build_paths(self):
        self.bps_radius = 1.0
        self.bps_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/processed_omomo/bps{}.pt",
        ).format(int(self.bps_radius * 100))

        self.data_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/omomo_data/processed_omomo",
        )
        self.obj_geo_root_folder = os.path.join(
            self.data_root_folder, "captured_objects"
        )

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
        if object_name == "vacuum" or object_name == "mop":
            two_parts = True
        else:
            two_parts = False

        if two_parts:
            obj_mesh_path = os.path.join(
                self.obj_geo_root_folder, object_name + "_cleaned_simplified_top.obj"
            )
            obj_mesh_verts, obj_mesh_faces = self.apply_transformation_to_obj_geometry(
                obj_mesh_path, obj_scale, obj_rot, obj_trans
            )  # T X Nv X 3
            return obj_mesh_verts, obj_mesh_faces

        if two_parts:
            top_obj_mesh_path = os.path.join(
                self.obj_geo_root_folder, object_name + "_cleaned_simplified_top.obj"
            )
            bottom_obj_mesh_path = os.path.join(
                self.obj_geo_root_folder, object_name + "_cleaned_simplified_bottom.obj"
            )

            top_obj_mesh_verts, top_obj_mesh_faces = (
                self.apply_transformation_to_obj_geometry(
                    top_obj_mesh_path, obj_scale, obj_rot, obj_trans
                )
            )
            bottom_obj_mesh_verts, bottom_obj_mesh_faces = (
                self.apply_transformation_to_obj_geometry(
                    bottom_obj_mesh_path,
                    obj_bottom_scale,
                    obj_bottom_rot,
                    obj_bottom_trans,
                )
            )

            obj_mesh_verts, obj_mesh_faces = merge_two_parts(
                [top_obj_mesh_verts, bottom_obj_mesh_verts],
                [top_obj_mesh_faces, bottom_obj_mesh_faces],
            )
        else:
            obj_mesh_verts, obj_mesh_faces = self.apply_transformation_to_obj_geometry(
                obj_mesh_path, obj_scale, obj_rot, obj_trans
            )  # T X Nv X 3

        return obj_mesh_verts, obj_mesh_faces

    def apply_transformation_inverse(self, vertices, obj_trans, obj_rot):
        # obj_trans: T X 3, obj_rot: T X 3 X 3
        # vertices: T X Nv X 3
        transformed_vertices = (
            obj_rot.transpose(1, 2)
            .bmm((vertices - obj_trans[:, None]).transpose(1, 2))
            .transpose(1, 2)
        )  # T X Nv X 3
        return transformed_vertices

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

        seq_scale = torch.from_numpy(obj_scale).float()  # T
        seq_rot_mat = torch.from_numpy(obj_rot).float()  # T X 3 X 3
        if obj_trans.shape[-1] != 1:
            seq_trans = torch.from_numpy(obj_trans).float()[:, :, None]  # T X 3 X 1
        else:
            seq_trans = torch.from_numpy(obj_trans).float()  # T X 3 X 1
        transformed_obj_verts = (
            seq_scale.unsqueeze(-1).unsqueeze(-1)
            * seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2))
            + seq_trans
        )
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3

        return transformed_obj_verts, obj_mesh_faces

    def compute_ambient_sensor_all(self):
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
            obj_name = seq_name.split("_")[1]

            dest_ambient_sensor_npy_path = os.path.join(
                self.dest_ambient_sensor_npy_folder,
                seq_name + "_" + str(s_idx) + ".npy",
            )
            if not os.path.exists(dest_ambient_sensor_npy_path):
                global_jpos = torch.from_numpy(
                    window_data["motion"][:, : self.joint_num * 3]
                ).reshape(-1, self.joint_num, 3)  # T X J X 3
                global_rot_6d = torch.from_numpy(
                    window_data["motion"][:, 2 * self.joint_num * 3 :]
                ).reshape(-1, self.joint_num - 2, 6)  # T X J X 6
                global_rot_mat = transforms.rotation_6d_to_matrix(
                    global_rot_6d
                )  # T X J X 3 X 3
                left_wrist_pos = global_jpos[:, 20, :]  # T X 3
                right_wrist_pos = global_jpos[:, 21, :]  # T X 3
                left_middle_finger_ori = global_rot_mat[:, 20]  # T X 3 X 3
                right_middle_finger_ori = global_rot_mat[:, 21]  # T X 3 X 3

                left_offset = (
                    torch.Tensor([0.1, 0.0, 0.0])
                    .reshape(1, 3, 1)
                    .repeat(left_middle_finger_ori.shape[0], 1, 1)
                )
                right_offset = (
                    torch.Tensor([-0.1, 0.0, 0.0])
                    .reshape(1, 3, 1)
                    .repeat(left_middle_finger_ori.shape[0], 1, 1)
                )
                left_middle_finger_pos = (
                    left_wrist_pos + left_middle_finger_ori.bmm(left_offset)[..., 0]
                )  # T X 3
                right_middle_finger_pos = (
                    right_wrist_pos + right_middle_finger_ori.bmm(right_offset)[..., 0]
                )  # T X 3

                obj_scale = window_data["obj_scale"]
                obj_trans = window_data["obj_trans"]
                obj_rot = window_data["obj_rot_mat"]
                if obj_name in ["mop", "vacuum"]:
                    obj_bottom_scale = window_data["obj_bottom_scale"]
                    obj_bottom_trans = window_data["obj_bottom_trans"]
                    obj_bottom_rot = window_data["obj_bottom_rot_mat"]
                else:
                    obj_bottom_scale = None
                    obj_bottom_trans = None
                    obj_bottom_rot = None
                obj_verts_, _ = self.load_object_geometry(
                    obj_name,
                    obj_scale,
                    obj_trans,
                    obj_rot,
                    obj_bottom_scale,
                    obj_bottom_trans,
                    obj_bottom_rot,
                )  # T X Nv X 3, tensor
                ambient_sensor = self.compute_ambient_sensor(
                    obj_verts_,
                    left_middle_finger_pos,
                    right_middle_finger_pos,
                    left_middle_finger_ori,
                    right_middle_finger_ori,
                )

                np.save(dest_ambient_sensor_npy_path, ambient_sensor.cpu().numpy())

    def compute_proximity_sensor_all(self):
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
            obj_name = seq_name.split("_")[1]

            dest_proximity_sensor_npy_path = os.path.join(
                self.dest_proximity_sensor_npy_folder,
                seq_name + "_" + str(s_idx) + ".npy",
            )
            if not os.path.exists(dest_proximity_sensor_npy_path):
                global_jpos = torch.from_numpy(
                    window_data["motion"][:, : self.joint_num * 3]
                ).reshape(-1, self.joint_num, 3)  # T X J X 3
                global_rot_6d = torch.from_numpy(
                    window_data["motion"][:, 2 * self.joint_num * 3 :]
                ).reshape(-1, self.joint_num - 2, 6)  # T X J X 6
                global_rot_mat = transforms.rotation_6d_to_matrix(
                    global_rot_6d
                )  # T X J X 3 X 3
                left_wrist_pos = global_jpos[:, 20, :]  # T X 3
                right_wrist_pos = global_jpos[:, 21, :]  # T X 3
                left_wrist_ori = global_rot_mat[:, 20]  # T X 3 X 3
                right_wrist_ori = global_rot_mat[:, 21]  # T X 3 X 3

                obj_scale = window_data["obj_scale"]
                obj_trans = window_data["obj_trans"]
                obj_rot = window_data["obj_rot_mat"]
                if obj_name in ["mop", "vacuum"]:
                    obj_bottom_scale = window_data["obj_bottom_scale"]
                    obj_bottom_trans = window_data["obj_bottom_trans"]
                    obj_bottom_rot = window_data["obj_bottom_rot_mat"]
                else:
                    obj_bottom_scale = None
                    obj_bottom_trans = None
                    obj_bottom_rot = None
                obj_verts_, _ = self.load_object_geometry(
                    obj_name,
                    obj_scale,
                    obj_trans,
                    obj_rot,
                    obj_bottom_scale,
                    obj_bottom_trans,
                    obj_bottom_rot,
                )  # T X Nv X 3, tensor
                proximity_sensor = self.compute_proximity_sensor(
                    obj_verts_,
                    left_wrist_pos,
                    right_wrist_pos,
                    left_wrist_ori,
                    right_wrist_ori,
                )

                np.save(dest_proximity_sensor_npy_path, proximity_sensor.cpu().numpy())

    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        object_name = seq_name.split("_")[1]

        start_t_idx = self.window_data_dict[index]["start_t_idx"]
        end_t_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        obj_bps_npy_path = os.path.join(
            self.dest_obj_bps_npy_folder, seq_name + "_" + str(index) + ".npy"
        )
        obj_bps_data = np.load(obj_bps_npy_path)  # T X N X 3
        obj_bps_data = torch.from_numpy(obj_bps_data)

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

            paded_obj_bps = torch.cat(
                (
                    obj_bps_data.reshape(actual_steps, -1),
                    torch.zeros(
                        self.window - actual_steps,
                        obj_bps_data.reshape(actual_steps, -1).shape[1],
                    ),
                ),
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
                (
                    torch.from_numpy(
                        self.window_data_dict[index]["obj_rot_mat"]
                    ).float(),
                    torch.zeros(self.window - actual_steps, 3, 3),
                ),
                dim=0,
            )
            paded_obj_scale = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_scale"]).float(),
                    torch.zeros(
                        self.window - actual_steps,
                    ),
                ),
                dim=0,
            )
            paded_obj_trans = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_trans"]).float(),
                    torch.zeros(self.window - actual_steps, 3),
                ),
                dim=0,
            )

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_rot_mat"]
                        ).float(),
                        torch.zeros(self.window - actual_steps, 3, 3),
                    ),
                    dim=0,
                )
                paded_obj_bottom_scale = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_scale"]
                        ).float(),
                        torch.zeros(
                            self.window - actual_steps,
                        ),
                    ),
                    dim=0,
                )
                paded_obj_bottom_trans = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_trans"]
                        )
                        .float()
                        .squeeze(-1),
                        torch.zeros(self.window - actual_steps, 3),
                    ),
                    dim=0,
                )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input

            paded_obj_bps = obj_bps_data.reshape(new_data_input.shape[0], -1)
            paded_obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()

            paded_obj_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_rot_mat"]
            ).float()
            paded_obj_scale = torch.from_numpy(
                self.window_data_dict[index]["obj_scale"]
            ).float()
            paded_obj_trans = torch.from_numpy(
                self.window_data_dict[index]["obj_trans"]
            ).float()

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_rot_mat"]
                ).float()
                paded_obj_bottom_scale = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_scale"]
                ).float()
                paded_obj_bottom_trans = (
                    torch.from_numpy(self.window_data_dict[index]["obj_bottom_trans"])
                    .float()
                    .squeeze(-1)
                )

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["obj_bps"] = paded_obj_bps
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        if object_name in ["mop", "vacuum"]:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_bottom_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_bottom_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_bottom_trans
        else:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_trans

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        return data_input_dict


class OMOMOAmbientSensorDataset(OMOMODataset):
    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        object_name = seq_name.split("_")[1]

        start_t_idx = self.window_data_dict[index]["start_t_idx"]
        end_t_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder, seq_name + "_" + str(index) + ".npy"
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)

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

            paded_ambient_sensor = torch.cat(
                (
                    ambient_sensor_data,
                    torch.zeros(
                        self.window - actual_steps, ambient_sensor_data.shape[1]
                    ),
                ),
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
                (
                    torch.from_numpy(
                        self.window_data_dict[index]["obj_rot_mat"]
                    ).float(),
                    torch.zeros(self.window - actual_steps, 3, 3),
                ),
                dim=0,
            )
            paded_obj_scale = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_scale"]).float(),
                    torch.zeros(
                        self.window - actual_steps,
                    ),
                ),
                dim=0,
            )
            paded_obj_trans = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_trans"]).float(),
                    torch.zeros(self.window - actual_steps, 3),
                ),
                dim=0,
            )

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_rot_mat"]
                        ).float(),
                        torch.zeros(self.window - actual_steps, 3, 3),
                    ),
                    dim=0,
                )
                paded_obj_bottom_scale = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_scale"]
                        ).float(),
                        torch.zeros(
                            self.window - actual_steps,
                        ),
                    ),
                    dim=0,
                )
                paded_obj_bottom_trans = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_trans"]
                        )
                        .float()
                        .squeeze(-1),
                        torch.zeros(self.window - actual_steps, 3),
                    ),
                    dim=0,
                )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input

            paded_ambient_sensor = ambient_sensor_data
            paded_obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()

            paded_obj_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_rot_mat"]
            ).float()
            paded_obj_scale = torch.from_numpy(
                self.window_data_dict[index]["obj_scale"]
            ).float()
            paded_obj_trans = torch.from_numpy(
                self.window_data_dict[index]["obj_trans"]
            ).float()

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_rot_mat"]
                ).float()
                paded_obj_bottom_scale = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_scale"]
                ).float()
                paded_obj_bottom_trans = (
                    torch.from_numpy(self.window_data_dict[index]["obj_bottom_trans"])
                    .float()
                    .squeeze(-1)
                )

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["obj_bps"] = paded_ambient_sensor
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        if object_name in ["mop", "vacuum"]:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_bottom_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_bottom_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_bottom_trans
        else:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_trans

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        return data_input_dict


class OMOMOProximitySensorDataset(OMOMODataset):
    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        object_name = seq_name.split("_")[1]

        start_t_idx = self.window_data_dict[index]["start_t_idx"]
        end_t_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        proximity_sensor_npy_path = os.path.join(
            self.dest_proximity_sensor_npy_folder, seq_name + "_" + str(index) + ".npy"
        )
        proximity_sensor_data = np.load(proximity_sensor_npy_path)  # T X (N*2)
        proximity_sensor_data = torch.from_numpy(proximity_sensor_data)

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

            paded_proximity_sensor = torch.cat(
                (
                    proximity_sensor_data,
                    torch.zeros(
                        self.window - actual_steps, proximity_sensor_data.shape[1]
                    ),
                ),
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
                (
                    torch.from_numpy(
                        self.window_data_dict[index]["obj_rot_mat"]
                    ).float(),
                    torch.zeros(self.window - actual_steps, 3, 3),
                ),
                dim=0,
            )
            paded_obj_scale = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_scale"]).float(),
                    torch.zeros(
                        self.window - actual_steps,
                    ),
                ),
                dim=0,
            )
            paded_obj_trans = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_trans"]).float(),
                    torch.zeros(self.window - actual_steps, 3),
                ),
                dim=0,
            )

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_rot_mat"]
                        ).float(),
                        torch.zeros(self.window - actual_steps, 3, 3),
                    ),
                    dim=0,
                )
                paded_obj_bottom_scale = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_scale"]
                        ).float(),
                        torch.zeros(
                            self.window - actual_steps,
                        ),
                    ),
                    dim=0,
                )
                paded_obj_bottom_trans = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_trans"]
                        )
                        .float()
                        .squeeze(-1),
                        torch.zeros(self.window - actual_steps, 3),
                    ),
                    dim=0,
                )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input

            paded_proximity_sensor = proximity_sensor_data
            paded_obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()

            paded_obj_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_rot_mat"]
            ).float()
            paded_obj_scale = torch.from_numpy(
                self.window_data_dict[index]["obj_scale"]
            ).float()
            paded_obj_trans = torch.from_numpy(
                self.window_data_dict[index]["obj_trans"]
            ).float()

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_rot_mat"]
                ).float()
                paded_obj_bottom_scale = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_scale"]
                ).float()
                paded_obj_bottom_trans = (
                    torch.from_numpy(self.window_data_dict[index]["obj_bottom_trans"])
                    .float()
                    .squeeze(-1)
                )

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["obj_bps"] = paded_proximity_sensor
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        if object_name in ["mop", "vacuum"]:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_bottom_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_bottom_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_bottom_trans
        else:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_trans

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        return data_input_dict


class OMOMOBothSensorDataset(OMOMODataset):
    def __getitem__(self, index):
        # index = 0 # For debug
        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        object_name = seq_name.split("_")[1]

        start_t_idx = self.window_data_dict[index]["start_t_idx"]
        end_t_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder, seq_name + "_" + str(index) + ".npy"
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)
        ambient_sensor_data = self.normalize_clip_sensor(ambient_sensor_data)

        proximity_sensor_npy_path = os.path.join(
            self.dest_proximity_sensor_npy_folder, seq_name + "_" + str(index) + ".npy"
        )
        proximity_sensor_data = np.load(proximity_sensor_npy_path)  # T X (N*2)
        proximity_sensor_data = torch.from_numpy(proximity_sensor_data)
        proximity_sensor_data = self.normalize_clip_sensor(proximity_sensor_data)

        contact_label_data = self.get_contact_label_single(index)  # T X 2

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

            paded_ambient_sensor = torch.cat(
                (
                    ambient_sensor_data,
                    torch.zeros(
                        self.window - actual_steps, ambient_sensor_data.shape[1]
                    ),
                ),
                dim=0,
            )
            paded_proximity_sensor = torch.cat(
                (
                    proximity_sensor_data,
                    torch.zeros(
                        self.window - actual_steps, proximity_sensor_data.shape[1]
                    ),
                ),
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
            paded_contact_label_data = torch.cat(
                (
                    torch.from_numpy(contact_label_data).float(),
                    torch.zeros(self.window - actual_steps, 2),
                ),
                dim=0,
            )

            paded_obj_rot_mat = torch.cat(
                (
                    torch.from_numpy(
                        self.window_data_dict[index]["obj_rot_mat"]
                    ).float(),
                    torch.zeros(self.window - actual_steps, 3, 3),
                ),
                dim=0,
            )
            paded_obj_scale = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_scale"]).float(),
                    torch.zeros(
                        self.window - actual_steps,
                    ),
                ),
                dim=0,
            )
            paded_obj_trans = torch.cat(
                (
                    torch.from_numpy(self.window_data_dict[index]["obj_trans"]).float(),
                    torch.zeros(self.window - actual_steps, 3),
                ),
                dim=0,
            )

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_rot_mat"]
                        ).float(),
                        torch.zeros(self.window - actual_steps, 3, 3),
                    ),
                    dim=0,
                )
                paded_obj_bottom_scale = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_scale"]
                        ).float(),
                        torch.zeros(
                            self.window - actual_steps,
                        ),
                    ),
                    dim=0,
                )
                paded_obj_bottom_trans = torch.cat(
                    (
                        torch.from_numpy(
                            self.window_data_dict[index]["obj_bottom_trans"]
                        )
                        .float()
                        .squeeze(-1),
                        torch.zeros(self.window - actual_steps, 3),
                    ),
                    dim=0,
                )
        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input

            paded_ambient_sensor = ambient_sensor_data
            paded_proximity_sensor = proximity_sensor_data
            paded_obj_com_pos = torch.from_numpy(
                self.window_data_dict[index]["window_obj_com_pos"]
            ).float()
            paded_contact_label_data = torch.from_numpy(contact_label_data).float()

            paded_obj_rot_mat = torch.from_numpy(
                self.window_data_dict[index]["obj_rot_mat"]
            ).float()
            paded_obj_scale = torch.from_numpy(
                self.window_data_dict[index]["obj_scale"]
            ).float()
            paded_obj_trans = torch.from_numpy(
                self.window_data_dict[index]["obj_trans"]
            ).float()

            if object_name in ["mop", "vacuum"]:
                paded_obj_bottom_rot_mat = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_rot_mat"]
                ).float()
                paded_obj_bottom_scale = torch.from_numpy(
                    self.window_data_dict[index]["obj_bottom_scale"]
                ).float()
                paded_obj_bottom_trans = (
                    torch.from_numpy(self.window_data_dict[index]["obj_bottom_trans"])
                    .float()
                    .squeeze(-1)
                )

        data_input_dict = {}
        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input

        data_input_dict["obj_bps"] = torch.cat(
            (paded_ambient_sensor, paded_proximity_sensor), dim=-1
        )
        data_input_dict["obj_com_pos"] = paded_obj_com_pos
        data_input_dict["contact_label"] = paded_contact_label_data

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        if object_name in ["mop", "vacuum"]:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_bottom_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_bottom_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_bottom_trans
        else:
            data_input_dict["obj_bottom_rot_mat"] = paded_obj_rot_mat
            data_input_dict["obj_bottom_scale"] = paded_obj_scale
            data_input_dict["obj_bottom_trans"] = paded_obj_trans

        data_input_dict["betas"] = self.window_data_dict[index]["betas"]
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["seq_name"] = seq_name
        data_input_dict["obj_name"] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint
        data_input_dict["s_idx"] = index

        return data_input_dict


class OMOMOGraspDataset(OMOMODataset):
    def __init__(
        self,
        train,
        window=30,
        use_window_bps=False,
        use_object_splits=False,
        use_joints24=False,
        load_ds=True,
    ):
        self.load_ds = load_ds
        self.parents_wholebody = os.path.join(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../data/smpl_all_models/smplx_parents_52.npy",
            )
        )

        self.train = train

        self.window = window

        # self.use_window_bps = True

        self.use_joints24 = True
        self.joint_num = 24
        self.lhand_idx = 20
        self.rhand_idx = 21

        self.parents = get_smpl_parents()  # 52

        self.build_paths()

        dest_obj_bps_npy_folder = os.path.join(
            self.data_root_folder, "grasp_object_bps_npy_files_joints24"
        )
        dest_obj_bps_npy_folder_for_test = os.path.join(
            self.data_root_folder, "grasp_object_bps_npy_files_for_eval_joints24"
        )

        dest_ambient_sensor_npy_folder = os.path.join(
            self.data_root_folder, "grasp_ambient_sensor_npy_files"
        )
        dest_ambient_sensor_npy_folder_for_test = os.path.join(
            self.data_root_folder, "grasp_ambient_sensor_npy_files_for_eval"
        )

        dest_proximity_sensor_npy_folder = os.path.join(
            self.data_root_folder, "grasp_proximity_sensor_npy_files"
        )
        dest_proximity_sensor_npy_folder_for_test = os.path.join(
            self.data_root_folder, "grasp_proximity_sensor_npy_files_for_eval"
        )

        for path in [
            dest_obj_bps_npy_folder,
            dest_obj_bps_npy_folder_for_test,
            dest_ambient_sensor_npy_folder,
            dest_ambient_sensor_npy_folder_for_test,
            dest_proximity_sensor_npy_folder,
            dest_proximity_sensor_npy_folder_for_test,
        ]:
            if not os.path.exists(path):
                os.makedirs(path)

        if self.train:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder
            self.dest_ambient_sensor_npy_folder = dest_ambient_sensor_npy_folder
            self.dest_proximity_sensor_npy_folder = dest_proximity_sensor_npy_folder
        else:
            self.dest_obj_bps_npy_folder = dest_obj_bps_npy_folder_for_test
            self.dest_ambient_sensor_npy_folder = (
                dest_ambient_sensor_npy_folder_for_test
            )
            self.dest_proximity_sensor_npy_folder = (
                dest_proximity_sensor_npy_folder_for_test
            )

        if self.train:
            ori_processed_data_path = os.path.join(
                self.data_root_folder,
                "train_diffusion_manip_window_" + str(self.window) + "_cano_joints24.p",
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "grasp_train_diffusion_manip_window_"
                + str(self.window)
                + "_cano_joints24.p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "grasp_train_contact_label.p"
            )
        else:
            ori_processed_data_path = os.path.join(
                self.data_root_folder,
                "test_diffusion_manip_window_"
                + str(self.window)
                + "_processed_joints24.p",
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "grasp_test_diffusion_manip_window_"
                + str(self.window)
                + "_processed_joints24.p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "grasp_test_contact_label.p"
            )

        min_max_mean_std_data_path = os.path.join(
            self.data_root_folder,
            "grasp_min_max_mean_std_data_window_" + str(120) + "_cano_joints24.p",
        )

        self.prep_bps_data()
        self.prep_sensor_data()
        self.prep_obj_indices()

        if self.load_ds:
            if os.path.exists(processed_data_path):
                self.window_data_dict = joblib.load(processed_data_path)

                # if len(os.listdir(self.dest_ambient_sensor_npy_folder)) == 0:
                #     print("Compute ambient sensor for all windows...")
                #     self.compute_ambient_sensor_all()

                # if len(os.listdir(self.dest_proximity_sensor_npy_folder)) == 0:
                #     print("Compute proximity sensor for all windows...")
                #     self.compute_proximity_sensor_all()

                # if not self.train:
                # Mannually enable this. For testing data (discarded some sequences)
                # self.get_bps_from_window_data_dict()
            else:
                if os.path.exists(ori_processed_data_path):
                    self.data_dict = joblib.load(ori_processed_data_path)

                self.cal_normalize_grasp_data_input()
                joblib.dump(self.window_data_dict, processed_data_path)

        else:
            self.window_data_dict = {}

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            raise ValueError(
                "Cannot find min_max_mean_std_data_path:{0}, run process_\{dataset\}.py".format(
                    min_max_mean_std_data_path
                )
            )

        self.global_jpos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_min"])
            .float()
            .reshape(-1, 3)[None]
        )
        self.global_jpos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_max"])
            .float()
            .reshape(-1, 3)[None]
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

    def cal_normalize_grasp_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in tqdm(self.data_dict):
            contact_label_data = self.get_contact_label_single(index)  # T X 2

            seq_name = self.data_dict[index][
                "seq_name"
            ]  # like: 's1/binoculars_lift.npz'
            seq_name_new = seq_name.replace("/", "_")[:-4]  # like: 's1_binoculars_lift'

            # betas = self.data_dict[index]['betas'] # 1 X 16
            gender = self.data_dict[index]["gender"]

            seq_root_trans = self.data_dict[index]["trans"]  # T X 3
            seq_root_orient = self.data_dict[index]["root_orient"]  # T X 3
            seq_pose_body = self.data_dict[index]["pose_body"].reshape(
                -1, self.joint_num, 3
            )  # T X 52 X 3

            rest_human_offsets = self.data_dict[index]["rest_offsets"]  # 52 X 3
            trans2joint = self.data_dict[index]["trans2joint"]  # 3

            obj_trans = self.data_dict[index]["obj_trans"]  # T X 3
            obj_rot = self.data_dict[index]["obj_rot"]  # T X 3 X 3

            obj_scale = self.data_dict[index]["obj_scale"]  # T

            obj_com_pos = self.data_dict[index]["obj_com_pos"]  # T X 3

            num_steps = seq_root_trans.shape[0]

            right_grasp_starts = []
            left_grasp_starts = []

            seq_data_path = os.path.join(self.obj_geo_root_folder, seq_name)
            seq_data = parse_npz(seq_data_path)

            contact_label = seq_data["contact"]["object"][::4]  # 120fps / 30fps = 4
            real_contact_label = np.zeros(
                (contact_label.shape[0], 2)
            )  # T X 2, 0: left hand, 1: right hand
            assert real_contact_label.shape[0] == num_steps
            for i in range(contact_label.shape[0]):
                contact_part = set(contact_label[i])
                left_flag = 0
                right_flag = 0
                for left_part in left_parts:
                    if left_part in contact_part:
                        left_flag = 1
                        break
                for right_part in right_parts:
                    if right_part in contact_part:
                        right_flag = 1
                        break
                real_contact_label[i, 0] = left_flag
                real_contact_label[i, 1] = right_flag

            for i in range(self.window, real_contact_label.shape[0]):
                if (
                    real_contact_label[i - self.window : i - 3, 0].sum() == 0
                    and real_contact_label[i, 0] == 1
                ):
                    left_grasp_starts.append(i - self.window + 1)
                if (
                    real_contact_label[i - self.window : i - 3, 1].sum() == 0
                    and real_contact_label[i, 1] == 1
                ):
                    right_grasp_starts.append(i - self.window + 1)

            # TODO: now only support left hand
            for start_t_idx in left_grasp_starts:
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps

                # Skip the segment that has a length < 5
                if end_t_idx - start_t_idx < 5:
                    continue

                self.window_data_dict[s_idx] = {}

                # Canonicalize the first frame's orientation.
                # J = 52
                joint_aa_rep = torch.from_numpy(
                    seq_pose_body[start_t_idx : end_t_idx + 1]
                ).float()  # T X J X 3
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

                curr_obj_scale = torch.from_numpy(
                    obj_scale[start_t_idx : end_t_idx + 1]
                ).float()  # T

                _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(
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

                window_obj_com_pos = obj_com_pos[
                    start_t_idx : end_t_idx + 1
                ].copy()  # T X 3

                X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj(
                    X[np.newaxis],
                    Q[np.newaxis],
                    window_obj_com_pos[np.newaxis],
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
                new_seq_pose_body = new_local_aa_rep[:, 1:, :]  # T X (J-1) X 3

                new_obj_rot_mat = transforms.quaternion_to_matrix(
                    torch.from_numpy(new_obj_q[0]).float()
                )  # T X 3 X 3 \

                cano_obj_mat = torch.matmul(
                    new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1)
                )  # 3 X 3

                obj_verts, _ = self.load_object_geometry(
                    self.data_dict[index]["object_mesh_path"],
                    curr_obj_scale.detach().cpu().numpy(),
                    new_obj_x[0],
                    new_obj_rot_mat.detach().cpu().numpy(),
                )  # T X Nv X 3, tensor

                center_verts = obj_verts.mean(dim=1)  # T X 3

                query = self.process_window_data(
                    rest_human_offsets,
                    trans2joint,
                    new_seq_root_trans,
                    new_seq_root_orient.detach().cpu().numpy(),
                    new_seq_pose_body.detach().cpu().numpy(),
                    new_obj_x[0],
                    new_obj_rot_mat.detach().cpu().numpy(),
                    curr_obj_scale.detach().cpu().numpy(),
                    new_obj_com_pos[0],
                    center_verts,
                )

                # Compute BPS representation for this window
                # Save to numpy file
                # dest_obj_bps_npy_path = os.path.join(self.dest_obj_bps_npy_folder, seq_name_new+"_"+str(s_idx)+".npy")

                # if not os.path.exists(dest_obj_bps_npy_path):
                #     object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                #     np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

                # Compute ambient and proximity sensor
                # Save to numpy file
                dest_ambient_sensor_npy_path = os.path.join(
                    self.dest_ambient_sensor_npy_folder,
                    seq_name_new + "_" + str(s_idx) + ".npy",
                )
                if not os.path.exists(dest_ambient_sensor_npy_path):
                    left_middle_finger_pos = query["global_jpos"][:, 28, :]  # T X 3
                    right_middle_finger_pos = query["global_jpos"][:, 43, :]  # T X 3
                    left_middle_finger_ori = query["global_rot_mat"][:, 20]  # T X 3 X 3
                    right_middle_finger_ori = query["global_rot_mat"][
                        :, 21
                    ]  # T X 3 X 3
                    obj_verts_, _ = self.load_object_geometry(
                        self.data_dict[index]["object_mesh_path"],
                        query["obj_scale"].detach().cpu().numpy(),
                        query["obj_trans"].detach().cpu().numpy(),
                        query["obj_rot_mat"].detach().cpu().numpy(),
                    )  # T X Nv X 3, tensor

                    ambient_sensor = self.compute_ambient_sensor(
                        obj_verts_.cuda(),
                        left_middle_finger_pos.cuda(),
                        right_middle_finger_pos.cuda(),
                        left_middle_finger_ori.cuda(),
                        right_middle_finger_ori.cuda(),
                    )
                    np.save(dest_ambient_sensor_npy_path, ambient_sensor.cpu().numpy())

                dest_proximity_sensor_npy_path = os.path.join(
                    self.dest_proximity_sensor_npy_folder,
                    seq_name_new + "_" + str(s_idx) + ".npy",
                )
                if not os.path.exists(dest_proximity_sensor_npy_path):
                    left_wrist_pos = query["global_jpos"][:, 20, :]  # T X 3
                    right_wrist_pos = query["global_jpos"][:, 21, :]  # T X 3
                    left_wrist_ori = query["global_rot_mat"][:, 20]  # T X 3 X 3
                    right_wrist_ori = query["global_rot_mat"][:, 21]  # T X 3 X 3
                    obj_verts_, _ = self.load_object_geometry(
                        self.data_dict[index]["object_mesh_path"],
                        query["obj_scale"].detach().cpu().numpy(),
                        query["obj_trans"].detach().cpu().numpy(),
                        query["obj_rot_mat"].detach().cpu().numpy(),
                    )  # T X Nv X 3, tensor

                    proximity_sensor = self.compute_proximity_sensor(
                        obj_verts_.cuda(),
                        left_wrist_pos.cuda(),
                        right_wrist_pos.cuda(),
                        left_wrist_ori.cuda(),
                        right_wrist_ori.cuda(),
                    )
                    np.save(
                        dest_proximity_sensor_npy_path, proximity_sensor.cpu().numpy()
                    )

                self.window_data_dict[s_idx]["cano_obj_mat"] = (
                    cano_obj_mat.detach().cpu().numpy()
                )

                curr_global_jpos = query["global_jpos"].detach().cpu().numpy()
                curr_global_jvel = query["global_jvel"].detach().cpu().numpy()
                curr_global_rot_6d = query["global_rot_6d"].detach().cpu().numpy()
                curr_local_rot_6d = query["local_rot_6d"].detach().cpu().numpy()

                self.window_data_dict[s_idx]["object_mesh_path"] = self.data_dict[
                    index
                ]["object_mesh_path"]
                self.window_data_dict[s_idx]["vtemp_path"] = self.data_dict[index][
                    "vtemp_path"
                ]

                # ['motion]: J * 3 + J * 3 + J * 6: pos + vel + rot_6d
                self.window_data_dict[s_idx]["motion"] = np.concatenate(
                    (
                        curr_global_jpos.reshape(-1, self.joint_num * 3),
                        curr_global_jvel.reshape(-1, self.joint_num * 3),
                        curr_global_rot_6d.reshape(-1, self.joint_num * 6),
                    ),
                    axis=1,
                )  # T X (52*3+52*3+52*6)
                self.window_data_dict[s_idx]["local_rot"] = curr_local_rot_6d.reshape(
                    -1, self.joint_num * 6
                )

                self.window_data_dict[s_idx]["seq_name"] = seq_name
                self.window_data_dict[s_idx]["start_t_idx"] = start_t_idx
                self.window_data_dict[s_idx]["end_t_idx"] = end_t_idx

                # self.window_data_dict[s_idx]['betas'] = betas
                self.window_data_dict[s_idx]["gender"] = gender

                self.window_data_dict[s_idx]["trans2joint"] = trans2joint

                self.window_data_dict[s_idx]["obj_trans"] = (
                    query["obj_trans"].detach().cpu().numpy()
                )
                self.window_data_dict[s_idx]["obj_rot_mat"] = (
                    query["obj_rot_mat"].detach().cpu().numpy()
                )
                self.window_data_dict[s_idx]["obj_scale"] = (
                    query["obj_scale"].detach().cpu().numpy()
                )

                self.window_data_dict[s_idx]["obj_com_pos"] = (
                    query["obj_com_pos"].detach().cpu().numpy()
                )
                self.window_data_dict[s_idx]["window_obj_com_pos"] = (
                    query["window_obj_com_pos"].detach().cpu().numpy()
                )

                self.window_data_dict[s_idx]["is_left"] = True
                s_idx += 1
