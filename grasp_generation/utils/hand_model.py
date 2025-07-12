"""
Modified from https://github.com/PKU-EPIC/DexGraspNet
"""

import json
import os
import pickle

import numpy as np
import plotly.graph_objects as go
import smplx
import torch
import trimesh as tm
from pytorch3d import transforms
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes


class HandModel:
    def __init__(
        self,
        mano_root,
        contact_indices_path,
        pose_distrib_path,
        beta=[
            0.8882,
            0.0634,
            0.7364,
            -2.1568,
            -1.0418,
            -0.5665,
            4.1727,
            1.4160,
            2.1836,
            2.5980,
            -2.3136,
            -0.6962,
            1.7863,
            0.0176,
            0.7098,
            1.5602,
        ],
        gender="male",
        device="cpu",
        left_hand=False,
        batch_size=1,
        no_fc=False,
    ):
        """
        Create a Hand Model for MANO

        Parameters
        ----------
        mano_root: str
            base directory of MANO_RIGHT.pkl
        contact_indices_path: str
            path to hand-selected contact candidates
        pose_distrib_path: str
            path to a multivariate gaussian distribution of the `thetas` of MANO
        device: str | torch.Device
            device for torch tensors
        """
        self.left_hand = left_hand  # NOTE: only support all batch left or right
        # load SMPL-X
        smplx_model_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
            "smpl_all_models",
        )
        self.beta = torch.tensor([beta]).to(device=device)
        self.sbj_m = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            gender=gender,
            batch_size=batch_size,
            flat_hand_mean=True,
            use_pca=False,
        ).to(device=device)
        data = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    ),
                    "data",
                    "smpl_all_models",
                    "MANO_SMPLX_vertex_ids.pkl",
                ),
                "rb",
            )
        )
        self.lhand_verts = torch.from_numpy(data["left_hand"]).to(device=device)
        self.rhand_verts = torch.from_numpy(data["right_hand"]).to(device=device)
        data = pickle.load(
            open(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    ),
                    "data",
                    "smpl_all_models",
                    "MANO_SMPLX_face_ids.pkl",
                ),
                "rb",
            )
        )
        self.lhand_faces = torch.from_numpy(data["left_hand"]).to(device=device)
        self.rhand_faces = torch.from_numpy(data["right_hand"]).to(device=device)
        self.hand_faces = self.lhand_faces if self.left_hand else self.rhand_faces

        self.device = device

        # load contact points and pose distribution

        with open(contact_indices_path, "r") as f:
            self.contact_indices = json.load(f)
        self.contact_indices = torch.tensor(
            self.contact_indices, dtype=torch.long, device=self.device
        )
        self.n_contact_candidates = len(self.contact_indices)

        self.thumb_contact_indices = torch.tensor(
            [699, 700, 753, 754, 714, 741, 755, 757, 739, 756, 760, 740, 762, 763],
            dtype=torch.long,
            device=self.device,
        )
        self.index_contact_indices = torch.tensor(
            [
                194,
                195,
                165,
                48,
                49,
                166,
                46,
                47,
                280,
                237,
                238,
                340,
                341,
                330,
                342,
                328,
                343,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.middle_contact_indices = torch.tensor(
            [
                375,
                386,
                387,
                358,
                359,
                376,
                356,
                357,
                402,
                396,
                397,
                452,
                453,
                440,
                454,
                438,
                455,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.ring_contact_indices = torch.tensor(
            [
                485,
                496,
                497,
                470,
                471,
                486,
                468,
                469,
                513,
                506,
                507,
                563,
                564,
                551,
                565,
                549,
                566,
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.pinky_contact_indices = torch.tensor(
            [614, 615, 582, 583, 580, 581, 681, 681, 625, 666, 683],
            dtype=torch.long,
            device=self.device,
        )
        self.palm_contact_indices = torch.tensor(
            [73, 96, 98, 99, 772, 774, 775, 777], dtype=torch.long, device=self.device
        )
        self.other_contact_indices = [
            self.index_contact_indices,
            self.middle_contact_indices,
            self.ring_contact_indices,
            self.pinky_contact_indices,
            self.palm_contact_indices,
        ]

        self.pose_distrib = torch.load(pose_distrib_path, map_location=device)
        if self.left_hand:
            self.pose_distrib[0][1::3] *= -1
            self.pose_distrib[0][2::3] *= -1
        if no_fc:
            self.pose_distrib[0] *= 1e-2
        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.vertices = None
        self.keypoints = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, thetas, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+3+45) torch.FloatTensor
            translation, rotation in axis angles, and `thetas`
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()

        batch = hand_pose.shape[0]
        global_orient = torch.zeros((batch, 3), dtype=torch.float32, device=self.device)
        zero_hand = torch.zeros((batch, 45), dtype=torch.float32, device=self.device)
        root_trans = torch.zeros((batch, 3), dtype=torch.float32, device=self.device)
        if self.left_hand:
            body_pose = torch.cat(
                (
                    torch.zeros((batch, 57), dtype=torch.float32, device=self.device),
                    hand_pose[:, 3:6],
                    torch.zeros((batch, 3), dtype=torch.float32, device=self.device),
                ),
                dim=-1,
            )
            output = self.sbj_m(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=hand_pose[:, -45:],
                right_hand_pose=zero_hand,
                transl=root_trans,
                beta=self.beta.repeat(batch, 1),
            )
            self.keypoints = torch.cat(
                (
                    output.joints[:, 20:21],
                    output.joints[:, 25:40],
                    output.joints[:, -61:-56],
                ),
                dim=1,
            )
            self.vertices = output.vertices[:, self.lhand_verts]
        else:
            body_pose = torch.cat(
                (
                    torch.zeros((batch, 60), dtype=torch.float32, device=self.device),
                    hand_pose[:, 3:6],
                ),
                dim=-1,
            )
            output = self.sbj_m(
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=zero_hand,
                right_hand_pose=hand_pose[:, -45:],
                transl=root_trans,
                beta=self.beta.repeat(batch, 1),
            )
            self.keypoints = torch.cat(
                (
                    output.joints[:, 21:22],
                    output.joints[:, 40:55],
                    output.joints[:, -56:-51],
                ),
                dim=1,
            )
            self.vertices = output.vertices[:, self.rhand_verts]

        wrist_pos = self.keypoints[:, 0].detach().clone()
        true_wrist_pos = hand_pose[:, :3]
        self.keypoints += (true_wrist_pos - wrist_pos)[:, None]
        self.vertices += (true_wrist_pos - wrist_pos)[:, None]
        # print(self.keypoints)
        # print(self.keypoints.shape)
        self.contact_point_indices = contact_point_indices
        self.contact_points = self.vertices[
            torch.arange(len(hand_pose)).unsqueeze(1),
            self.contact_indices[self.contact_point_indices],
        ]

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use the inner product of the ObjectPoint-to-HandNearestNeighbour vector
        and the vertex normal of the HandNearestNeighbour to approximate the sdf

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Alternative 1: directly using Kaolin results in a time-consuming for-loop along the batch dimension
        # Alternative 2: discarding the inner product with the vertex normal will mess up the optimization severely
        # we reserve the implementation of the second alternative as comments below
        mesh = Meshes(
            verts=self.vertices, faces=self.hand_faces.unsqueeze(0).repeat(len(x), 1, 1)
        )
        normals = mesh.verts_normals_packed().view(-1, 778, 3)
        knn_result = knn_points(x, self.vertices, K=1)
        knn_idx = (torch.arange(len(x)).unsqueeze(1), knn_result.idx[:, :, 0])
        dis = -((x - self.vertices[knn_idx]) * normals[knn_idx].detach()).sum(dim=-1)
        # interior = ((x - self.vertices[knn_idx]) * normals[knn_idx]).sum(dim=-1) < 0
        # dis = torch.sqrt(knn_result.dists[:, :, 0] + 1e-8)
        # dis = torch.where(interior, dis, -dis)
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        dis = (
            (self.keypoints.unsqueeze(1) - self.keypoints.unsqueeze(2) + 1e-13)
            .square()
            .sum(3)
            .sqrt()
        )
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        loss = -torch.where(dis < 0.018, dis, torch.zeros_like(dis))
        return loss.sum((1, 2))

    def get_contact_candidates(self):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        return self.vertices[
            torch.arange(len(self.vertices)).unsqueeze(1),
            self.contact_indices.unsqueeze(0),
        ]

    def get_penetraion_keypoints(self):
        """
        Get MANO keypoints

        Returns
        -------
        points: (N, 21, 3) torch.Tensor
            MANO keypoints
        """
        return self.keypoints

    def get_plotly_data(
        self,
        i,
        opacity=0.5,
        color="lightblue",
        with_keypoints=False,
        with_contact_points=False,
        pose=None,
    ):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_keypoints: bool
            whether to visualize keypoints
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        v = self.vertices[i].detach().cpu().numpy()
        if pose is not None:
            v = v @ pose[:3, :3].T + pose[:3, 3]
        f = self.hand_faces
        hand_plotly = [
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                text=list(range(len(v))),
                color=color,
                opacity=opacity,
                hovertemplate="%{text}",
            )
        ]
        if with_keypoints:
            keypoints = self.keypoints[i].detach().cpu().numpy()
            if pose is not None:
                keypoints = keypoints @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(
                go.Scatter3d(
                    x=keypoints[:, 0],
                    y=keypoints[:, 1],
                    z=keypoints[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            )
            for penetration_keypoint in keypoints:
                mesh = tm.primitives.Capsule(radius=0.009, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                hand_plotly.append(
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.5,
                    )
                )
        if with_contact_points:
            contact_points = self.vertices[0, self.contact_indices].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            hand_plotly.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            )
        return hand_plotly

    def get_trimesh_data(self, i):
        """
        Get visualization data for trimesh

        Parameters
        ----------
        i: int
            index of data

        Returns
        -------
        data: trimesh.Trimesh
        """
        v = self.vertices[i].detach().cpu().numpy()
        f = self.hand_faces.detach().cpu().numpy()
        data = tm.Trimesh(v, f)
        return data
