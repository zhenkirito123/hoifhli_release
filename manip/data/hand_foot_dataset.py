import json
import os
import pickle
import sys
import time

import joblib
import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform, sample_uniform_cylinder
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import Dataset
from tqdm import tqdm

from manip.lafan1.utils import rotate_at_frame_w_obj
from manip.utils.visualize.tools.utils import contact_ids, parse_npz

SMPLH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "smpl_all_models",
)

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


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def get_smpl_parents():
    parents_path = os.path.join(SMPLH_PATH, "smplx_parents_52.npy")
    parents = np.load(parents_path)

    return parents


def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3
    kintree = get_smpl_parents()

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
    pass


def quat_ik_wholebody(grot_mat, parents):
    # grot: T X J X 3 X 3
    # parent: J
    assert parents[0] == -1 and np.all(parents[1:] >= 0)
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


def quat_ik_hand(grot_mat, parents):
    # grot: T X J X 3 X 3
    # parent: J
    assert parents[0] == -1 and parents[1] == -1 and np.all(parents[2:] >= 0)
    grot = transforms.matrix_to_quaternion(grot_mat)  # T X J X 4

    res = torch.cat(
        [
            grot[..., :2, :],
            transforms.quaternion_multiply(
                transforms.quaternion_invert(grot[..., parents[2:], :]),
                grot[..., 2:, :],
            ),
        ],
        dim=-2,
    )  # T X J X 4

    res_mat = transforms.quaternion_to_matrix(res)  # T X J X 3 X 3

    return res_mat


def quat_fk_torch(
    lrot_mat,
    lpos,
):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    parents = get_smpl_parents()

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :])
            + gp[parents[i]]
        )
        gr.append(
            transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :])
        )

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res


class HandFootManipDataset(Dataset):
    def __init__(
        self,
        train,
        window=120,
        use_window_bps=False,
        use_object_splits=False,
        use_joints24=False,
    ):
        self.parents_wholebody = np.load(
            os.path.join(SMPLH_PATH, "smplx_parents_52.npy")
        )

        self.train = train

        self.window = window

        # self.use_window_bps = True

        self.use_joints24 = True
        self.joint_num = 52
        self.lhand_idx = 20
        self.rhand_idx = 21

        self.parents = get_smpl_parents()  # 52

        self.build_paths()

        dest_obj_bps_npy_folder = os.path.join(
            self.data_root_folder, "object_bps_npy_files"
        )
        dest_obj_bps_npy_folder_for_test = os.path.join(
            self.data_root_folder, "object_bps_npy_files_for_eval"
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
            seq_data_path = os.path.join(
                self.data_root_folder, "train_diffusion_manip_seq.p"
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "train_diffusion_manip_window_" + str(self.window) + ".p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "train_contact_label.p"
            )
        else:
            seq_data_path = os.path.join(
                self.data_root_folder, "test_diffusion_manip_seq.p"
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "test_diffusion_manip_window_" + str(self.window) + ".p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "test_contact_label.p"
            )

        min_max_mean_std_data_path = os.path.join(
            self.data_root_folder,
            "min_max_mean_std_data_window_" + str(self.window) + ".p",
        )

        self.prep_bps_data()
        self.prep_sensor_data()

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)

            if len(os.listdir(self.dest_ambient_sensor_npy_folder)) == 0:
                print("Compute ambient sensor for all windows...")
                self.compute_ambient_sensor_all()

            if len(os.listdir(self.dest_proximity_sensor_npy_folder)) == 0:
                print("Compute proximity sensor for all windows...")
                self.compute_proximity_sensor_all()

        else:
            if os.path.exists(seq_data_path):
                self.data_dict = joblib.load(seq_data_path)
            else:
                raise ValueError(
                    "Cannot find seq_data_path:{0}, run process_\{dataset\}.py".format(
                        seq_data_path
                    )
                )

            self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)

        if os.path.exists(contact_label_path):
            self.contact_label_dict = joblib.load(contact_label_path)
        else:
            self.get_contact_label()
            joblib.dump(self.contact_label_dict, contact_label_path)

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
            else:
                raise ValueError(
                    "Cannot find min_max_mean_std_data_path:{0}, run process_\{dataset\}.py".format(
                        min_max_mean_std_data_path
                    )
                )

        self.global_jpos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_min"])
            .float()
            .reshape(self.joint_num, 3)[None]
        )
        self.global_jpos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_max"])
            .float()
            .reshape(self.joint_num, 3)[None]
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

    def build_paths(self):
        self.obj_geo_root_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "grab_data",
            "grab",
        )

        self.bps_radius = 0.14
        self.bps_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "grab_data",
            "processed_omomo",
            "bps{}.pt".format(int(self.bps_radius * 100)),
        )

        self.data_root_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "grab_data",
            "processed_omomo",
        )

    def apply_transformation_to_obj_geometry(
        self,
        obj_mesh_path,
        obj_scale,
        obj_trans,
        obj_rot,
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

    def load_object_geometry(
        self,
        object_mesh_path,
        obj_scale,
        obj_trans,
        obj_rot,
        obj_bottom_scale=None,
        obj_bottom_trans=None,
        obj_bottom_rot=None,
    ):
        # obj_trans: T X 3, obj_rot: T X 3 X 3

        obj_mesh_verts, obj_mesh_faces = self.apply_transformation_to_obj_geometry(
            object_mesh_path,
            obj_scale,
            obj_trans,
            obj_rot,
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
        n_obj = 1024
        r_obj = self.bps_radius
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(
                1, -1, 3
            )

            bps = {
                "obj": bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)

        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps["obj"]

    def prep_sensor_data(self):
        n_ambient_sensor = 1024
        r_ambient_sensor = 0.15
        self.sensor_bps_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "grab_data",
            "processed_omomo",
            "sensor_bps{}.pt".format(int(r_ambient_sensor * 100)),
        )
        if not os.path.exists(self.sensor_bps_path):
            while True:
                bps_obj = sample_sphere_uniform(
                    n_points=n_ambient_sensor * 10, radius=r_ambient_sensor
                )
                bps_obj = bps_obj[bps_obj[:, 1] <= 0][:n_ambient_sensor]
                if bps_obj.shape[0] == n_ambient_sensor:
                    break

            bps_obj = bps_obj.reshape(1, n_ambient_sensor, 3)

            bps = {
                "obj": bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new sensor bps data to:{0}".format(self.sensor_bps_path))
            torch.save(bps, self.sensor_bps_path)

        self.ambient_bps = torch.load(self.sensor_bps_path)["obj"]  # 1 X N X 3

        self.bps_torch = bps_torch()

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "smpl_all_models",
            "palm_sample_vertices.pkl",
        )
        self.proximity_bps = pickle.load(open(path, "rb"))
        self.left_proximity_bps = torch.from_numpy(self.proximity_bps["left_hand"])[
            None
        ]  # 1 X N X 3
        self.right_proximity_bps = torch.from_numpy(self.proximity_bps["right_hand"])[
            None
        ]  # 1 X N X 3

    def compute_ambient_sensor_all(self):
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
            seq_name_new = seq_name.replace("/", "_")[:-4]  # like: 's1_binoculars_lift'

            dest_ambient_sensor_npy_path = os.path.join(
                self.dest_ambient_sensor_npy_folder,
                seq_name_new + "_" + str(s_idx) + ".npy",
            )
            if not os.path.exists(dest_ambient_sensor_npy_path):
                global_jpos = torch.from_numpy(
                    window_data["motion"][:, : self.joint_num * 3]
                ).reshape(-1, self.joint_num, 3)  # T X J X 3
                global_rot_6d = torch.from_numpy(
                    window_data["motion"][:, 2 * self.joint_num * 3 :]
                ).reshape(-1, self.joint_num, 6)  # T X J X 6
                global_rot_mat = transforms.rotation_6d_to_matrix(
                    global_rot_6d
                )  # T X J X 3 X 3
                left_middle_finger_pos = global_jpos[:, 28, :]  # T X 3
                right_middle_finger_pos = global_jpos[:, 43, :]  # T X 3
                left_middle_finger_ori = global_rot_mat[:, 20]  # T X 3 X 3
                right_middle_finger_ori = global_rot_mat[:, 21]  # T X 3 X 3

                obj_verts_, _ = self.load_object_geometry(
                    window_data["object_mesh_path"],
                    window_data["obj_scale"],
                    window_data["obj_trans"],
                    window_data["obj_rot_mat"],
                )  # T X Nv X 3, tensor
                ambient_sensor = self.compute_ambient_sensor(
                    obj_verts_,
                    left_middle_finger_pos,
                    right_middle_finger_pos,
                    left_middle_finger_ori,
                    right_middle_finger_ori,
                )

                np.save(dest_ambient_sensor_npy_path, ambient_sensor.cpu().numpy())

    def compute_ambient_sensor(
        self,
        obj_verts,
        left_middle_finger_pos,
        right_middle_finger_pos,
        left_middle_finger_ori,
        right_middle_finger_ori,
    ):
        # obj_verts: T X Nv X 3,
        # left_middle_finger_pos: T X 3, right_middle_finger_pos: T X 3, left_middle_finger_ori: T X 3 X 3, right_middle_finger_ori: T X 3 X 3
        self.ambient_bps = self.ambient_bps.to(obj_verts.device)

        T = obj_verts.shape[0]
        left_bps_points = (
            left_middle_finger_ori.bmm(self.ambient_bps.repeat(T, 1, 1).transpose(1, 2))
            + left_middle_finger_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        left_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=left_bps_points
        )["dists"]  # T X N
        right_bps_points = (
            right_middle_finger_ori.bmm(
                self.ambient_bps.repeat(T, 1, 1).transpose(1, 2)
            )
            + right_middle_finger_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        right_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=right_bps_points
        )["dists"]  # T X N

        return torch.cat((left_sensor, right_sensor), dim=-1)  # T X (N*2)

    def compute_proximity_sensor_all(self):
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
            seq_name_new = seq_name.replace("/", "_")[:-4]  # like: 's1_binoculars_lift'

            dest_proximity_sensor_npy_path = os.path.join(
                self.dest_proximity_sensor_npy_folder,
                seq_name_new + "_" + str(s_idx) + ".npy",
            )
            if not os.path.exists(dest_proximity_sensor_npy_path):
                global_jpos = torch.from_numpy(
                    window_data["motion"][:, : self.joint_num * 3]
                ).reshape(-1, self.joint_num, 3)  # T X J X 3
                global_rot_6d = torch.from_numpy(
                    window_data["motion"][:, 2 * self.joint_num * 3 :]
                ).reshape(-1, self.joint_num, 6)  # T X J X 6
                global_rot_mat = transforms.rotation_6d_to_matrix(
                    global_rot_6d
                )  # T X J X 3 X 3
                left_wrist_pos = global_jpos[:, 20, :]  # T X 3
                right_wrist_pos = global_jpos[:, 21, :]  # T X 3
                left_wrist_ori = global_rot_mat[:, 20]  # T X 3 X 3
                right_wrist_ori = global_rot_mat[:, 21]  # T X 3 X 3

                obj_verts_, _ = self.load_object_geometry(
                    window_data["object_mesh_path"],
                    window_data["obj_scale"],
                    window_data["obj_trans"],
                    window_data["obj_rot_mat"],
                )  # T X Nv X 3, tensor
                proximity_sensor = self.compute_proximity_sensor(
                    obj_verts_,
                    left_wrist_pos,
                    right_wrist_pos,
                    left_wrist_ori,
                    right_wrist_ori,
                )

                np.save(dest_proximity_sensor_npy_path, proximity_sensor.cpu().numpy())

    def compute_proximity_sensor(
        self,
        obj_verts,
        left_wrist_pos,
        right_wrist_pos,
        left_wrist_ori,
        right_wrist_ori,
    ):
        # obj_verts: T X Nv X 3,
        # left_middle_finger_pos: T X 3, right_middle_finger_pos: T X 3, left_middle_finger_ori: T X 3 X 3, right_middle_finger_ori: T X 3 X 3
        self.left_proximity_bps = self.left_proximity_bps.to(obj_verts.device)
        self.right_proximity_bps = self.right_proximity_bps.to(obj_verts.device)

        T = obj_verts.shape[0]
        left_bps_points = (
            left_wrist_ori.bmm(self.left_proximity_bps.repeat(T, 1, 1).transpose(1, 2))
            + left_wrist_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        left_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=left_bps_points
        )["dists"]  # T X N
        right_bps_points = (
            right_wrist_ori.bmm(
                self.right_proximity_bps.repeat(T, 1, 1).transpose(1, 2)
            )
            + right_wrist_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        right_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=right_bps_points
        )["dists"]  # T X N

        return torch.cat((left_sensor, right_sensor), dim=-1)  # T X (N*2)

    def compute_mirror_proximity_sensor(
        self,
        obj_verts,
        left_wrist_pos,
        right_wrist_pos,
        left_wrist_ori,
        right_wrist_ori,
    ):
        # obj_verts: T X Nv X 3,
        # left_middle_finger_pos: T X 3, right_middle_finger_pos: T X 3, left_middle_finger_ori: T X 3 X 3, right_middle_finger_ori: T X 3 X 3
        self.left_proximity_bps = self.left_proximity_bps.to(obj_verts.device)
        self.right_proximity_bps = self.right_proximity_bps.to(obj_verts.device)

        left_proximity_bps = self.right_proximity_bps.clone()
        left_proximity_bps[:, :, 0] = -left_proximity_bps[:, :, 0]
        right_proximity_bps = self.left_proximity_bps.clone()
        right_proximity_bps[:, :, 0] = -right_proximity_bps[:, :, 0]

        T = obj_verts.shape[0]
        left_bps_points = (
            left_wrist_ori.bmm(left_proximity_bps.repeat(T, 1, 1).transpose(1, 2))
            + left_wrist_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        left_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=left_bps_points
        )["dists"]  # T X N
        right_bps_points = (
            right_wrist_ori.bmm(right_proximity_bps.repeat(T, 1, 1).transpose(1, 2))
            + right_wrist_pos[..., None]
        ).transpose(1, 2)  # T X N X 3
        right_sensor = self.bps_torch.encode(
            x=obj_verts, feature_type=["dists"], custom_basis=right_bps_points
        )["dists"]  # T X N

        return torch.cat((left_sensor, right_sensor), dim=-1)  # T X (N*2)

    def get_contact_label(self):
        self.contact_label_dict = {}
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            seq_name = window_data["seq_name"]  # like: 's1/binoculars_lift.npz'
            start_t_idx = window_data["start_t_idx"]
            end_t_idx = window_data["end_t_idx"]

            seq_data_path = os.path.join(self.obj_geo_root_folder, seq_name)
            seq_data = parse_npz(seq_data_path)

            contact_label = seq_data["contact"]["object"][::4][
                start_t_idx : end_t_idx + 1
            ]  # 120fps / 30fps = 4, see process_grab.py
            real_contact_label = np.zeros(
                (contact_label.shape[0], 2)
            )  # T X 2, 0: left hand, 1: right hand
            assert real_contact_label.shape[0] == window_data["local_rot"].shape[0]
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
            self.contact_label_dict[s_idx] = real_contact_label

    def cal_normalize_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in tqdm(self.data_dict):
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
            for start_t_idx in range(0, num_steps, self.window // 2):
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps

                # Skip the segment that has a length < 30
                if end_t_idx - start_t_idx < 30:
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
                dest_obj_bps_npy_path = os.path.join(
                    self.dest_obj_bps_npy_folder,
                    seq_name_new + "_" + str(s_idx) + ".npy",
                )

                if not os.path.exists(dest_obj_bps_npy_path):
                    object_bps = self.compute_object_geo_bps(obj_verts, center_verts)
                    np.save(dest_obj_bps_npy_path, object_bps.data.cpu().numpy())

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
                        obj_verts_,
                        left_middle_finger_pos,
                        right_middle_finger_pos,
                        left_middle_finger_ori,
                        right_middle_finger_ori,
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
                        obj_verts_,
                        left_wrist_pos,
                        right_wrist_pos,
                        left_wrist_ori,
                        right_wrist_ori,
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

                s_idx += 1

            # break

    def extract_min_max_mean_std_from_data(self):
        all_global_jpos_data = []
        all_global_jvel_data = []

        for s_idx in self.window_data_dict:
            curr_window_data = self.window_data_dict[s_idx]["motion"]  # T X D

            all_global_jpos_data.append(curr_window_data[:, : self.joint_num * 3])
            all_global_jvel_data.append(
                curr_window_data[:, self.joint_num * 3 : 2 * self.joint_num * 3]
            )

            start_t_idx = self.window_data_dict[s_idx]["start_t_idx"]
            end_t_idx = self.window_data_dict[s_idx]["end_t_idx"]
            curr_seq_name = self.window_data_dict[s_idx]["seq_name"]

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(
            -1, self.joint_num * 3
        )  # (N*T) X (J*3)
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(
            -1, self.joint_num * 3
        )

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
        # ori_jpos: T X J X 3
        normalized_jpos = (ori_jpos - self.global_jpos_min.to(ori_jpos.device)) / (
            self.global_jpos_max.to(ori_jpos.device)
            - self.global_jpos_min.to(ori_jpos.device)
        )
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        return normalized_jpos  # T X J X 3

    def de_normalize_jpos_min_max(self, normalized_jpos):
        normalized_jpos = (normalized_jpos + 1) * 0.5  # [0, 1] range
        de_jpos = normalized_jpos * (
            self.global_jpos_max.to(normalized_jpos.device)
            - self.global_jpos_min.to(normalized_jpos.device)
        ) + self.global_jpos_min.to(normalized_jpos.device)

        return de_jpos  # T X J X 3

    def normalize_jpos_min_max_hand_foot(self, ori_jpos, hand_only=True):
        # ori_jpos: BS X T X 2 X 3
        lhand_idx = self.lhand_idx
        rhand_idx = self.rhand_idx

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
        lhand_idx = self.lhand_idx
        rhand_idx = self.rhand_idx

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

    def normalize_clip_sensor(self, sensor):
        # sensor: T X N
        SENSOR_MIN = 0
        SENSOR_MAX = 0.15
        sensor = torch.clamp(sensor, min=SENSOR_MIN, max=SENSOR_MAX)
        sensor = (sensor - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN)
        sensor = sensor * 2 - 1

        return sensor  # T X N

    def process_window_data(
        self,
        rest_human_offsets,
        trans2joint,
        seq_root_trans,
        seq_root_orient,
        seq_pose_body,
        obj_trans,
        obj_rot,
        obj_scale,
        obj_com_pos,
        center_verts,
        obj_bottom_trans=None,
        obj_bottom_rot=None,
        obj_bottom_scale=None,
    ):
        # J = 52
        random_t_idx = 0
        end_t_idx = seq_root_trans.shape[0] - 1

        window_root_trans = torch.from_numpy(
            seq_root_trans[random_t_idx : end_t_idx + 1]
        ).cuda()
        window_root_orient = (
            torch.from_numpy(seq_root_orient[random_t_idx : end_t_idx + 1])
            .float()
            .cuda()
        )  # T X 3
        window_pose_body = (
            torch.from_numpy(seq_pose_body[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X (J-1) X 3

        window_obj_scale = (
            torch.from_numpy(obj_scale[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T

        window_obj_rot_mat = (
            torch.from_numpy(obj_rot[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X 3 X 3
        window_obj_trans = (
            torch.from_numpy(obj_trans[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X 3
        if obj_bottom_trans is not None:
            window_obj_bottom_scale = (
                torch.from_numpy(obj_bottom_scale[random_t_idx : end_t_idx + 1])
                .float()
                .cuda()
            )  # T

            window_obj_bottom_rot_mat = (
                torch.from_numpy(obj_bottom_rot[random_t_idx : end_t_idx + 1])
                .float()
                .cuda()
            )  # T X 3 X 3
            window_obj_bottom_trans = (
                torch.from_numpy(obj_bottom_trans[random_t_idx : end_t_idx + 1])
                .float()
                .cuda()
            )  # T X 3

        window_obj_com_pos = (
            torch.from_numpy(obj_com_pos[random_t_idx : end_t_idx + 1]).float().cuda()
        )  # T X 3
        window_center_verts = center_verts[random_t_idx : end_t_idx + 1].to(
            window_obj_com_pos.device
        )

        move_to_zero_trans = window_root_trans[0:1, :].clone()  # 1 X 3
        move_to_zero_trans[:, 2] = 0

        # Move motion and object translation to make the initial pose trans 0.
        window_root_trans = window_root_trans - move_to_zero_trans
        window_obj_trans = window_obj_trans - move_to_zero_trans
        window_obj_com_pos = window_obj_com_pos - move_to_zero_trans
        window_center_verts = window_center_verts - move_to_zero_trans
        if obj_bottom_trans is not None:
            window_obj_bottom_trans = window_obj_bottom_trans - move_to_zero_trans

        window_root_rot_mat = transforms.axis_angle_to_matrix(
            window_root_orient
        )  # T' X 3 X 3
        window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        window_pose_rot_mat = transforms.axis_angle_to_matrix(
            window_pose_body
        )  # T' X (J-1) X 3 X 3

        # Generate global joint rotation
        local_joint_rot_mat = torch.cat(
            (window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1
        )  # T' X J X 3 X 3
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat)  # T' X J X 3 X 3
        global_joint_rot_quat = transforms.matrix_to_quaternion(
            global_joint_rot_mat
        )  # T' X J X 4

        curr_seq_pose_aa = torch.cat(
            (window_root_orient[:, None, :], window_pose_body), dim=1
        )  # T' X J X 3
        rest_human_offsets = torch.from_numpy(rest_human_offsets).float()[None]
        curr_seq_local_jpos = rest_human_offsets.repeat(
            curr_seq_pose_aa.shape[0], 1, 1
        ).cuda()  # T' X J X 3
        curr_seq_local_jpos[:, 0, :] = (
            window_root_trans - torch.from_numpy(trans2joint).cuda()[None]
        )  # T' X J X 3

        local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
        _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)

        global_jpos = human_jnts  # T' X J X 3
        global_jvel = global_jpos[1:] - global_jpos[:-1]  # (T'-1) X J X 3

        global_joint_rot_mat = local2global_pose(local_joint_rot_mat)  # T' X J X 3 X 3

        local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
        global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

        query = {}

        query["local_rot_mat"] = local_joint_rot_mat  # T' X J X 3 X 3
        query["local_rot_6d"] = local_rot_6d  # T' X J X 6

        query["global_jpos"] = global_jpos  # T' X J X 3
        query["global_jvel"] = torch.cat(
            (
                global_jvel,
                torch.zeros(1, global_jvel.shape[1], 3).to(global_jvel.device),
            ),
            dim=0,
        )  # T' X J X 3

        query["global_rot_mat"] = global_joint_rot_mat  # T' X J X 3 X 3
        query["global_rot_6d"] = global_rot_6d  # T' X J X 6

        query["obj_trans"] = window_obj_trans  # T' X 3
        query["obj_rot_mat"] = window_obj_rot_mat  # T' X 3 X 3

        query["obj_scale"] = window_obj_scale  # T'

        query["obj_com_pos"] = window_obj_com_pos  # T' X 3

        query["window_obj_com_pos"] = window_center_verts  # T X 3

        if obj_bottom_trans is not None:
            query["obj_bottom_trans"] = window_obj_bottom_trans
            query["obj_bottom_rot_mat"] = window_obj_bottom_rot_mat

            query["obj_bottom_scale"] = window_obj_bottom_scale  # T'

        return query

    # def apply_transformation_to_obj_point_clouds(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
    #     mesh = trimesh.load_mesh(obj_mesh_path)
    #     obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3

    #     ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3

    #     seq_scale = torch.from_numpy(obj_scale).float() # T
    #     seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3
    #     seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1

    #     transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
    #     seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
    #     transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3

    #     return transformed_obj_verts

    # def load_object_point_clouds(self, object_name, obj_scale, obj_trans, obj_rot):
    #     obj_mesh_path = os.path.join(self.obj_point_cloud_root_folder, object_name+"_cleaned_simplified.ply")
    #     if object_name == "vacuum" or object_name == "mop":
    #         obj_mesh_path = os.path.join(self.obj_point_cloud_root_folder, object_name+"_cleaned_simplified_top.ply")

    #     obj_mesh_verts =self.apply_transformation_to_obj_point_clouds(obj_mesh_path, \
    #     obj_scale, obj_rot, obj_trans) # T X Nv X 3

    #     return obj_mesh_verts

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        # index = 0 # For debug
        object_mesh_path = self.window_data_dict[index]["object_mesh_path"]
        vtemp_path = self.window_data_dict[index]["vtemp_path"]

        data_input = self.window_data_dict[index]["motion"]
        data_input = torch.from_numpy(data_input).float()
        local_rot = self.window_data_dict[index]["local_rot"]
        local_rot = torch.from_numpy(local_rot).float()

        seq_name = self.window_data_dict[index]["seq_name"]
        seq_name_new = seq_name.replace("/", "_")[:-4]  # like: 's1_binoculars_lift'

        start_t_idx = self.window_data_dict[index]["start_t_idx"]
        end_t_idx = self.window_data_dict[index]["end_t_idx"]

        trans2joint = self.window_data_dict[index]["trans2joint"]

        obj_bps_npy_path = os.path.join(
            self.dest_obj_bps_npy_folder, seq_name_new + "_" + str(index) + ".npy"
        )
        obj_bps_data = np.load(obj_bps_npy_path)  # T X N X 3
        obj_bps_data = torch.from_numpy(obj_bps_data)

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)

        proximity_sensor_npy_path = os.path.join(
            self.dest_proximity_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        proximity_sensor_data = np.load(proximity_sensor_npy_path)  # T X (N*2)
        proximity_sensor_data = torch.from_numpy(proximity_sensor_data)

        num_joints = self.joint_num

        normalized_jpos = self.normalize_jpos_min_max(
            data_input[:, : num_joints * 3].reshape(-1, num_joints, 3)
        )  # T X J X 3

        global_joint_rot = data_input[:, 2 * num_joints * 3 :]  # T X (J*6)

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
            paded_local_rot = torch.cat(
                (
                    local_rot,
                    torch.zeros(self.window - actual_steps, local_rot.shape[-1]),
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

        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input
            paded_local_rot = local_rot

            paded_obj_bps = obj_bps_data.reshape(new_data_input.shape[0], -1)
            paded_ambient_sensor = ambient_sensor_data
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

        data_input_dict = {}

        data_input_dict["start_t_idx"] = start_t_idx
        data_input_dict["end_t_idx"] = end_t_idx

        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input
        data_input_dict["local_rot"] = paded_local_rot

        data_input_dict["obj_bps"] = paded_obj_bps
        data_input_dict["ambient_sensor"] = paded_ambient_sensor
        data_input_dict["proximity"] = paded_proximity_sensor
        data_input_dict["obj_com_pos"] = paded_obj_com_pos

        data_input_dict["obj_rot_mat"] = paded_obj_rot_mat
        data_input_dict["obj_scale"] = paded_obj_scale
        data_input_dict["obj_trans"] = paded_obj_trans

        # data_input_dict['betas'] = self.window_data_dict[index]['betas']
        data_input_dict["gender"] = str(self.window_data_dict[index]["gender"])

        data_input_dict["object_mesh_path"] = object_mesh_path
        data_input_dict["vtemp_path"] = vtemp_path
        data_input_dict["seq_name"] = seq_name
        # data_input_dict['obj_name'] = seq_name.split("_")[1]

        data_input_dict["seq_len"] = actual_steps

        data_input_dict["trans2joint"] = trans2joint

        return data_input_dict


class GraspDataset(HandFootManipDataset):
    def __init__(
        self,
        train,
        window=30,
        use_window_bps=False,
        use_object_splits=False,
        use_joints24=False,
        random_length=False,
    ):
        self.parents_wholebody = np.load(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "smpl_all_models",
                "smplx_parents_52.npy",
            )
        )

        self.train = train

        self.window = window

        self.random_length = random_length

        # self.use_window_bps = True

        self.use_joints24 = True
        self.joint_num = 52
        self.lhand_idx = 20
        self.rhand_idx = 21

        self.parents = get_smpl_parents()  # 52

        self.build_paths()

        dest_obj_bps_npy_folder = os.path.join(
            self.data_root_folder, "grasp_object_bps_npy_files"
        )
        dest_obj_bps_npy_folder_for_test = os.path.join(
            self.data_root_folder, "grasp_object_bps_npy_files_for_eval"
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
            seq_data_path = os.path.join(
                self.data_root_folder, "train_diffusion_manip_seq.p"
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "grasp_train_diffusion_window_" + str(self.window) + ".p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "grasp_train_contact_label.p"
            )
            left_wrist_reverse_sequence_path = os.path.join(
                self.data_root_folder, "grasp_train_left_wrist_reverse_sequence.p"
            )
        else:
            seq_data_path = os.path.join(
                self.data_root_folder, "test_diffusion_manip_seq.p"
            )
            processed_data_path = os.path.join(
                self.data_root_folder,
                "grasp_test_diffusion_window_" + str(self.window) + ".p",
            )
            contact_label_path = os.path.join(
                self.data_root_folder, "grasp_test_contact_label.p"
            )
            left_wrist_reverse_sequence_path = os.path.join(
                self.data_root_folder, "grasp_test_left_wrist_reverse_sequence.p"
            )

        min_max_mean_std_data_path = os.path.join(
            self.data_root_folder,
            "grasp_min_max_mean_std_data_window_" + str(self.window) + ".p",
        )
        left_wrist_reverse_sequence_min_max_mean_std_data_path = os.path.join(
            self.data_root_folder,
            "grasp_left_wrist_reverse_sequence_min_max_mean_std_data_window_"
            + str(self.window)
            + ".p",
        )

        self.prep_bps_data()
        self.prep_sensor_data()

        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)

            # if len(os.listdir(self.dest_ambient_sensor_npy_folder)) == 0:
            #     print("Compute ambient sensor for all windows...")
            #     self.compute_ambient_sensor_all()

            # if len(os.listdir(self.dest_proximity_sensor_npy_folder)) == 0:
            #     print("Compute proximity sensor for all windows...")
            #     self.compute_proximity_sensor_all()

        else:
            if os.path.exists(seq_data_path):
                self.data_dict = joblib.load(seq_data_path)
            else:
                raise ValueError(
                    "Cannot find seq_data_path:{0}, run process_\{dataset\}.py".format(
                        seq_data_path
                    )
                )

            self.cal_normalize_grasp_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)

        if os.path.exists(contact_label_path):
            self.contact_label_dict = joblib.load(contact_label_path)
        else:
            self.get_contact_label()
            joblib.dump(self.contact_label_dict, contact_label_path)

        if os.path.exists(left_wrist_reverse_sequence_path):
            self.left_wrist_reverse_sequence_dict = joblib.load(
                left_wrist_reverse_sequence_path
            )
        else:
            self.get_left_wrist_reverse_sequence()
            joblib.dump(
                self.left_wrist_reverse_sequence_dict, left_wrist_reverse_sequence_path
            )

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
            else:
                raise ValueError(
                    "Cannot find min_max_mean_std_data_path:{0}, run process_\{dataset\}.py".format(
                        min_max_mean_std_data_path
                    )
                )
        self.global_jpos_min = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_min"])
            .float()
            .reshape(self.joint_num, 3)[None]
        )
        self.global_jpos_max = (
            torch.from_numpy(min_max_mean_std_jpos_data["global_jpos_max"])
            .float()
            .reshape(self.joint_num, 3)[None]
        )

        if os.path.exists(left_wrist_reverse_sequence_min_max_mean_std_data_path):
            left_wrist_reverse_sequence_min_max_mean_std_data = joblib.load(
                left_wrist_reverse_sequence_min_max_mean_std_data_path
            )
        else:
            if self.train:
                left_wrist_reverse_sequence_min_max_mean_std_data = (
                    self.extract_left_wrist_min_max_mean_std_from_data()
                )
                joblib.dump(
                    left_wrist_reverse_sequence_min_max_mean_std_data,
                    left_wrist_reverse_sequence_min_max_mean_std_data_path,
                )
            else:
                raise ValueError(
                    "Cannot find min_max_mean_std_data_path:{0}, run process_\{dataset\}.py".format(
                        left_wrist_reverse_sequence_min_max_mean_std_data_path
                    )
                )
        self.left_wrist_reverse_sequence_jpos_min = torch.from_numpy(
            left_wrist_reverse_sequence_min_max_mean_std_data["min_jpos"]
        ).float()
        self.left_wrist_reverse_sequence_jpos_max = torch.from_numpy(
            left_wrist_reverse_sequence_min_max_mean_std_data["max_jpos"]
        ).float()

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

        assert len(self.window_data_dict) == len(self.contact_label_dict)
        assert len(self.window_data_dict) == len(self.left_wrist_reverse_sequence_dict)

    def cal_normalize_grasp_data_input(self):
        self.window_data_dict = {}
        s_idx = 0
        for index in tqdm(self.data_dict):
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
            right_grasp_ends = []
            left_grasp_starts = []
            left_grasp_ends = []

            seq_data_path = os.path.join(self.obj_geo_root_folder, seq_name)
            seq_data = parse_npz(seq_data_path)

            contact_label = seq_data["contact"]["object"][
                ::4
            ]  # 120fps / 30fps = 4, see process_grab.py
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

            for i in range(0, real_contact_label.shape[0] - self.window):
                if (
                    real_contact_label[i + 3 : i + self.window, 0].sum() == 0
                    and real_contact_label[i, 0] == 1
                ):
                    left_grasp_ends.append(i)
                if (
                    real_contact_label[i + 3 : i + self.window, 1].sum() == 0
                    and real_contact_label[i, 1] == 1
                ):
                    right_grasp_ends.append(i)

            # TODO: now only support left hand
            grasp_dict = {
                "left_grasp_starts": left_grasp_starts,
                "left_grasp_ends": left_grasp_ends,
            }
            for key, grasp_indices in grasp_dict.items():
                for start_t_idx in grasp_indices:
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
                        transforms.matrix_to_quaternion(obj_rot_mat)
                        .detach()
                        .cpu()
                        .numpy()
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
                        right_middle_finger_pos = query["global_jpos"][
                            :, 43, :
                        ]  # T X 3
                        left_middle_finger_ori = query["global_rot_mat"][
                            :, 20
                        ]  # T X 3 X 3
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
                        np.save(
                            dest_ambient_sensor_npy_path, ambient_sensor.cpu().numpy()
                        )

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
                            dest_proximity_sensor_npy_path,
                            proximity_sensor.cpu().numpy(),
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
                    self.window_data_dict[s_idx]["local_rot"] = (
                        curr_local_rot_6d.reshape(-1, self.joint_num * 6)
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
                    if "right" in key:
                        self.window_data_dict[s_idx]["is_left"] = False

                    self.window_data_dict[s_idx]["is_reach"] = True
                    if "end" in key:
                        self.window_data_dict[s_idx]["is_reach"] = False
                    s_idx += 1

    def normalize_left_wrist_reverse_sequence_jpos_min_max(
        self, left_wrist_reverse_sequence
    ):
        jpos = left_wrist_reverse_sequence[..., :3]
        jrot = left_wrist_reverse_sequence[..., 3:]

        normalized_jpos = (
            jpos - self.left_wrist_reverse_sequence_jpos_min.to(jpos.device)
        ) / (
            self.left_wrist_reverse_sequence_jpos_max.to(jpos.device)
            - self.left_wrist_reverse_sequence_jpos_min.to(jpos.device)
        )
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range

        return torch.cat((normalized_jpos, jrot), dim=-1)

    def de_normalize_left_wrist_reverse_sequence_jpos_min_max(
        self, normalized_left_wrist_reverse_sequence
    ):
        normalized_jpos = normalized_left_wrist_reverse_sequence[..., :3]
        jrot = normalized_left_wrist_reverse_sequence[..., 3:]

        jpos = (normalized_jpos + 1) / 2 * (
            self.left_wrist_reverse_sequence_jpos_max.to(normalized_jpos.device)
            - self.left_wrist_reverse_sequence_jpos_min.to(normalized_jpos.device)
        ) + self.left_wrist_reverse_sequence_jpos_min.to(normalized_jpos.device)

        return torch.cat((jpos, jrot), dim=-1)

    def get_left_wrist_reverse_sequence(self):
        self.left_wrist_reverse_sequence_dict = {}
        for s_idx in tqdm(range(len(self.window_data_dict))):
            window_data = self.window_data_dict[s_idx]
            global_jpos = torch.from_numpy(
                window_data["motion"][:, : self.joint_num * 3]
            ).reshape(-1, self.joint_num, 3)  # T X J X 3
            global_rot_6d = torch.from_numpy(
                window_data["motion"][:, 2 * self.joint_num * 3 :]
            ).reshape(-1, self.joint_num, 6)  # T X J X 6
            global_rot_mat = transforms.rotation_6d_to_matrix(
                global_rot_6d
            )  # T X J X 3 X 3
            left_wrist_pos = global_jpos[:, 20, :]  # T X 3
            right_wrist_pos = global_jpos[:, 21, :]  # T X 3
            left_wrist_ori = global_rot_mat[:, 20]  # T X 3 X 3
            right_wrist_ori = global_rot_mat[:, 21]  # T X 3 X 3

            reverse_cano_left_wrist_pos = (
                left_wrist_ori[-1:]
                .transpose(1, 2)
                .matmul((left_wrist_pos - left_wrist_pos[-1]).unsqueeze(-1))
                .squeeze(-1)
            )  # T X 3
            reverse_cano_left_wrist_ori = (
                left_wrist_ori[-1:].transpose(1, 2).matmul(left_wrist_ori)
            )  # T X 3 X 3
            reverse_cano_left_wrist_rot_6d = transforms.matrix_to_rotation_6d(
                reverse_cano_left_wrist_ori
            )  # T X 6

            reverse_seq = (
                torch.cat(
                    (reverse_cano_left_wrist_pos, reverse_cano_left_wrist_rot_6d), dim=1
                )
                .detach()
                .cpu()
                .numpy()
            )  # T X 9

            self.left_wrist_reverse_sequence_dict[s_idx] = reverse_seq

    def extract_left_wrist_min_max_mean_std_from_data(self):
        norm = {}
        t = []
        for i in range(len(self.left_wrist_reverse_sequence_dict)):
            reverse_seq = self.left_wrist_reverse_sequence_dict[i]
            t.append(reverse_seq[:, :3])
        t = np.concatenate(t, axis=0)
        min_jpos = t.min(axis=0)
        max_jpos = t.max(axis=0)
        norm["min_jpos"] = min_jpos
        norm["max_jpos"] = max_jpos
        return norm
