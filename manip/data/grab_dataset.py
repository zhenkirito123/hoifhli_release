import json
import os
import sys
import time

import joblib
import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh

from manip.data.hand_foot_dataset import GraspDataset, HandFootManipDataset


def apply_transformation_to_obj_geometry(
    obj_mesh_path,
    obj_scale,
    obj_trans,
    obj_rot,
):
    mesh = trimesh.load_mesh(obj_mesh_path)
    obj_mesh_verts = np.asarray(mesh.vertices)  # Nv X 3
    obj_mesh_faces = np.asarray(mesh.faces)  # Nf X 3

    ori_obj_verts = (
        torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1)
    )  # T X Nv X 3

    seq_scale = torch.from_numpy(obj_scale).float()  # T
    seq_rot_mat = torch.from_numpy(obj_rot).float()  # T X 3 X 3
    if obj_trans.shape[-1] != 1:
        seq_trans = torch.from_numpy(obj_trans).float()[:, :, None]  # T X 3 X 1
    else:
        seq_trans = torch.from_numpy(obj_trans).float()  # T X 3 X 1
    transformed_obj_verts = (
        seq_scale.unsqueeze(-1).unsqueeze(-1)
        * seq_rot_mat.transpose(1, 2).bmm(ori_obj_verts.transpose(1, 2))
        + seq_trans
    )  # NOTE: seq_rot_mat.transpose(1, 2)! Refer to https://github.com/otaheri/GRAB/blob/284cba757bd10364fd38eb883c33d4490c4d98f5/tools/objectmodel.py#L97
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2)  # T X Nv X 3

    return transformed_obj_verts, obj_mesh_faces


def load_object_geometry(
    object_mesh_path,
    obj_scale,
    obj_trans,
    obj_rot,
    obj_bottom_scale=None,
    obj_bottom_trans=None,
    obj_bottom_rot=None,
):
    # obj_trans: T X 3, obj_rot: T X 3 X 3

    obj_mesh_verts, obj_mesh_faces = apply_transformation_to_obj_geometry(
        object_mesh_path,
        obj_scale,
        obj_trans,
        obj_rot,
    )

    return obj_mesh_verts, obj_mesh_faces  # T X Nv X 3, Nf X 3


class GrabDataset(HandFootManipDataset):
    def build_paths(self):
        self.obj_geo_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/grab",
        )

        self.bps_radius = 0.14
        self.bps_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/processed_omomo/bps{}.pt".format(
                int(self.bps_radius * 100)
            ),
        )

        self.data_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/processed_omomo",
        )

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

        return load_object_geometry(
            object_mesh_path,
            obj_scale,
            obj_trans,
            obj_rot,
        )

    def apply_transformation_inverse(self, vertices, obj_trans, obj_rot):
        # obj_trans: T X 3, obj_rot: T X 3 X 3
        # vertices: T X Nv X 3
        transformed_vertices = obj_rot.bmm(
            (vertices - obj_trans[:, None]).transpose(1, 2)
        ).transpose(1, 2)  # T X Nv X 3
        return transformed_vertices


class GrabAmbientSensorDataset(GrabDataset):
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

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)

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

        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input
            paded_local_rot = local_rot

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

        data_input_dict = {}

        data_input_dict["start_t_idx"] = start_t_idx
        data_input_dict["end_t_idx"] = end_t_idx

        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input
        data_input_dict["local_rot"] = paded_local_rot

        # paded_ambient_sensor = torch.clip(paded_ambient_sensor, -0.15, 0.15)
        data_input_dict["obj_bps"] = paded_ambient_sensor
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


class GrabProximitySensorDataset(GrabDataset):
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

        # paded_proximity_sensor = torch.clip(paded_proximity_sensor, -0.15, 0.15)
        data_input_dict["obj_bps"] = paded_proximity_sensor
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


class GrabBothSensorDataset(GrabDataset):
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

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)
        ambient_sensor_data = self.normalize_clip_sensor(ambient_sensor_data)

        proximity_sensor_npy_path = os.path.join(
            self.dest_proximity_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        proximity_sensor_data = np.load(proximity_sensor_npy_path)  # T X (N*2)
        proximity_sensor_data = torch.from_numpy(proximity_sensor_data)
        proximity_sensor_data = self.normalize_clip_sensor(proximity_sensor_data)

        contact_label_data = self.contact_label_dict[index]  # T X 2

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

        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input
            paded_local_rot = local_rot

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

        data_input_dict = {}

        data_input_dict["start_t_idx"] = start_t_idx
        data_input_dict["end_t_idx"] = end_t_idx

        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input
        data_input_dict["local_rot"] = paded_local_rot

        # paded_proximity_sensor = torch.clip(paded_proximity_sensor, -0.15, 0.15)
        data_input_dict["obj_bps"] = torch.cat(
            (paded_ambient_sensor, paded_proximity_sensor), dim=-1
        )
        data_input_dict["obj_com_pos"] = paded_obj_com_pos
        data_input_dict["contact_label"] = paded_contact_label_data

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


class GrabGraspDataset(GraspDataset):
    def build_paths(self):
        self.obj_geo_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/grab",
        )

        self.bps_radius = 0.14
        self.bps_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/processed_omomo/bps{}.pt".format(
                int(self.bps_radius * 100)
            ),
        )

        self.data_root_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/grab_data/processed_omomo",
        )

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

        return load_object_geometry(
            object_mesh_path,
            obj_scale,
            obj_trans,
            obj_rot,
        )

    def apply_transformation_inverse(self, vertices, obj_trans, obj_rot):
        # obj_trans: T X 3, obj_rot: T X 3 X 3
        # vertices: T X Nv X 3
        transformed_vertices = obj_rot.bmm(
            (vertices - obj_trans[:, None]).transpose(1, 2)
        ).transpose(1, 2)  # T X Nv X 3
        return transformed_vertices

    def __getitem__(self, index):
        # index = 0 # For debug
        is_reach = self.window_data_dict[index]["is_reach"]
        is_left = self.window_data_dict[index]["is_left"]
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

        ambient_sensor_npy_path = os.path.join(
            self.dest_ambient_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        ambient_sensor_data = np.load(ambient_sensor_npy_path)  # T X (N*2)
        ambient_sensor_data = torch.from_numpy(ambient_sensor_data)
        ambient_sensor_data = self.normalize_clip_sensor(ambient_sensor_data)

        proximity_sensor_npy_path = os.path.join(
            self.dest_proximity_sensor_npy_folder,
            seq_name_new + "_" + str(index) + ".npy",
        )
        proximity_sensor_data = np.load(proximity_sensor_npy_path)  # T X (N*2)
        proximity_sensor_data = torch.from_numpy(proximity_sensor_data)
        proximity_sensor_data = self.normalize_clip_sensor(proximity_sensor_data)

        contact_label_data = torch.from_numpy(
            self.contact_label_dict[index]
        ).float()  # T X 2
        left_wrist_reverse_sequence = torch.from_numpy(
            self.left_wrist_reverse_sequence_dict[index]
        ).float()  # T X 9
        left_wrist_reverse_sequence = (
            self.normalize_left_wrist_reverse_sequence_jpos_min_max(
                left_wrist_reverse_sequence
            )
        )

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

        obj_com_pos = torch.from_numpy(
            self.window_data_dict[index]["window_obj_com_pos"]
        ).float()
        obj_rot_mat = torch.from_numpy(
            self.window_data_dict[index]["obj_rot_mat"]
        ).float()
        obj_scale = torch.from_numpy(self.window_data_dict[index]["obj_scale"]).float()
        obj_trans = torch.from_numpy(self.window_data_dict[index]["obj_trans"]).float()

        if self.random_length and np.random.rand() < 0.5:
            new_length = np.random.randint(5, self.window + 1)
            if is_reach:
                new_data_input = new_data_input[-new_length:]
                ori_data_input = ori_data_input[-new_length:]
                local_rot = local_rot[-new_length:]

                ambient_sensor_data = ambient_sensor_data[-new_length:]
                proximity_sensor_data = proximity_sensor_data[-new_length:]
                obj_com_pos = obj_com_pos[-new_length:]
                contact_label_data = contact_label_data[-new_length:]
                left_wrist_reverse_sequence = left_wrist_reverse_sequence[-new_length:]

                obj_rot_mat = obj_rot_mat[-new_length:]
                obj_scale = obj_scale[-new_length:]
                obj_trans = obj_trans[-new_length:]
            else:
                new_data_input = new_data_input[:new_length]
                ori_data_input = ori_data_input[:new_length]
                local_rot = local_rot[:new_length]

                ambient_sensor_data = ambient_sensor_data[:new_length]
                proximity_sensor_data = proximity_sensor_data[:new_length]
                obj_com_pos = obj_com_pos[:new_length]
                contact_label_data = contact_label_data[:new_length]
                left_wrist_reverse_sequence = left_wrist_reverse_sequence[:new_length]

                obj_rot_mat = obj_rot_mat[:new_length]
                obj_scale = obj_scale[:new_length]
                obj_trans = obj_trans[:new_length]

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
                (obj_com_pos, torch.zeros(self.window - actual_steps, 3)), dim=0
            )
            paded_contact_label_data = torch.cat(
                (contact_label_data, torch.zeros(self.window - actual_steps, 2)), dim=0
            )
            paded_left_wrist_reverse_sequence = torch.cat(
                (
                    left_wrist_reverse_sequence,
                    torch.zeros(self.window - actual_steps, 9),
                ),
                dim=0,
            )

            paded_obj_rot_mat = torch.cat(
                (obj_rot_mat, torch.zeros(self.window - actual_steps, 3, 3)), dim=0
            )
            paded_obj_scale = torch.cat(
                (
                    obj_scale,
                    torch.zeros(
                        self.window - actual_steps,
                    ),
                ),
                dim=0,
            )
            paded_obj_trans = torch.cat(
                (obj_trans, torch.zeros(self.window - actual_steps, 3)), dim=0
            )

        else:
            paded_new_data_input = new_data_input
            paded_ori_data_input = ori_data_input
            paded_local_rot = local_rot

            paded_ambient_sensor = ambient_sensor_data
            paded_proximity_sensor = proximity_sensor_data
            paded_obj_com_pos = obj_com_pos
            paded_contact_label_data = contact_label_data
            paded_left_wrist_reverse_sequence = left_wrist_reverse_sequence

            paded_obj_rot_mat = obj_rot_mat
            paded_obj_scale = obj_scale
            paded_obj_trans = obj_trans

        data_input_dict = {}

        data_input_dict["start_t_idx"] = start_t_idx
        data_input_dict["end_t_idx"] = end_t_idx

        data_input_dict["motion"] = paded_new_data_input
        data_input_dict["ori_motion"] = paded_ori_data_input
        data_input_dict["local_rot"] = paded_local_rot

        data_input_dict["obj_bps"] = torch.cat(
            (paded_ambient_sensor, paded_proximity_sensor), dim=-1
        )
        data_input_dict["obj_com_pos"] = paded_obj_com_pos
        data_input_dict["contact_label"] = paded_contact_label_data
        data_input_dict["left_wrist_reverse_sequence"] = (
            paded_left_wrist_reverse_sequence
        )

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
        data_input_dict["is_left"] = is_left

        return data_input_dict
