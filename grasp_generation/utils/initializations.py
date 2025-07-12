"""
Modified from https://github.com/PKU-EPIC/DexGraspNet
"""

import math

import numpy as np
import pytorch3d.ops
import pytorch3d.structures
import pytorch3d.transforms
import torch
import torch.nn.functional
import transforms3d
import trimesh as tm


def initialize_convex_hull(
    hand_model, object_model, args, wrist_pos, wrist_rot, move_away=False, no_fc=False
):
    """
    Initialize grasp translation, rotation, thetas, and contact point indices

    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    if len(wrist_pos.shape) == 1:  # 3, 3 X 3
        hand_model.initial_translation = wrist_pos[None].repeat(total_batch_size, 1)
        translation[0] = wrist_pos
        rotation[0] = wrist_rot
        for i in range(1, total_batch_size):
            translation[i] = wrist_pos
            axis = torch.randn(3)
            axis /= torch.norm(axis)
            aa = axis * torch.rand(1) * np.pi / 6
            rotation[i] = wrist_rot.matmul(
                pytorch3d.transforms.axis_angle_to_matrix(aa).to(device)
            )
        if move_away:
            if hand_model.left_hand:
                offset = rotation.matmul(
                    torch.tensor([0.1, 0, 0], dtype=torch.float, device=device).reshape(
                        1, 3, 1
                    )
                ).squeeze(-1)
                translation = 1.2 * (wrist_pos + offset) - offset
            else:
                offset = rotation.matmul(
                    torch.tensor(
                        [-0.1, 0, 0], dtype=torch.float, device=device
                    ).reshape(1, 3, 1)
                ).squeeze(-1)
                translation = 1.2 * (wrist_pos + offset) - offset
    elif len(wrist_pos.shape) == 2:  # T X 3, T X 3 X 3
        hand_model.initial_translation = wrist_pos
        translation = wrist_pos
        rotation = wrist_rot
    else:
        raise ValueError("Invalid wrist_pos shape")

    # initialize thetas
    # thetas_mu: hand-crafted canonicalized hand articulation
    # use normal distribution to jitter the thetas

    thetas_mu = (
        torch.tensor(
            [
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                *(torch.pi / 2 * torch.tensor([1, 0.5, 0], dtype=torch.float)),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.float,
            device=device,
        )
        .unsqueeze(0)
        .repeat(total_batch_size, 1)
    )
    if hand_model.left_hand:
        thetas_mu = (
            torch.tensor(
                [
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *(torch.pi / 2 * torch.tensor([1, -0.5, 0], dtype=torch.float)),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.float,
                device=device,
            )
            .unsqueeze(0)
            .repeat(total_batch_size, 1)
        )
    thetas_sigma = args.jitter_strength * torch.ones(
        [total_batch_size, 45], dtype=torch.float, device=device
    )
    thetas = torch.normal(thetas_mu, thetas_sigma)
    if no_fc:
        thetas *= 1e-2

    rotation = pytorch3d.transforms.quaternion_to_axis_angle(
        pytorch3d.transforms.matrix_to_quaternion(rotation)
    )
    hand_pose = torch.cat(
        [
            translation,
            rotation,
            thetas,
        ],
        dim=1,
    )
    hand_pose.requires_grad_()

    contact_point_indices = torch.randint(
        hand_model.n_contact_candidates,
        size=[total_batch_size, args.n_contact],
        device=device,
    )
    hand_model.set_parameters(hand_pose, contact_point_indices)


def initialize_convex_hull_original(hand_model, object_model, args, no_fc=False):
    """
    Initialize grasp translation, rotation, thetas, and contact point indices

    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        # get inflated convex hull

        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(
            vertices.unsqueeze(0), faces.unsqueeze(0)
        )

        # sample points

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
            mesh_pytorch3d, num_samples=100 * batch_size_each
        )
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[
            0
        ][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # sample parameters

        distance = args.distance_lower + (
            args.distance_upper - args.distance_lower
        ) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        deviate_theta = args.theta_lower + (
            args.theta_upper - args.theta_lower
        ) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        process_theta = (
            2
            * math.pi
            * torch.rand([batch_size_each], dtype=torch.float, device=device)
        )
        rotate_theta = (
            2
            * math.pi
            * torch.rand([batch_size_each], dtype=torch.float, device=device)
        )

        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros(
            [batch_size_each, 3, 3], dtype=torch.float, device=device
        )
        rotation_global = torch.zeros(
            [batch_size_each, 3, 3], dtype=torch.float, device=device
        )
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    process_theta[j], deviate_theta[j], rotate_theta[j], axes="rzxz"
                ),
                dtype=torch.float,
                device=device,
            )
            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2,
                    -math.acos(n[j, 2]),
                    0,
                    axes="rzxz",
                ),
                dtype=torch.float,
                device=device,
            )
        translation[i * batch_size_each : (i + 1) * batch_size_each] = (
            p
            - distance.unsqueeze(1)
            * (
                rotation_global
                @ rotation_local
                @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(
                    1, -1, 1
                )
            ).squeeze(2)
        )
        rotation[i * batch_size_each : (i + 1) * batch_size_each] = (
            rotation_global @ rotation_local
        )

    translation_hand = torch.tensor([-0.1, -0.05, 0], dtype=torch.float, device=device)
    rotation_hand = torch.tensor(
        transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, np.pi / 6, axes="rzxz"),
        dtype=torch.float,
        device=device,
    )

    translation = translation + rotation @ translation_hand
    rotation = rotation @ rotation_hand

    # initialize thetas
    # thetas_mu: hand-crafted canonicalized hand articulation
    # use normal distribution to jitter the thetas

    thetas_mu = (
        torch.tensor(
            [
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                torch.pi / 6,
                0,
                0,
                0,
                0,
                0,
                0,
                *(torch.pi / 2 * torch.tensor([1, 0.5, 0], dtype=torch.float)),
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.float,
            device=device,
        )
        .unsqueeze(0)
        .repeat(total_batch_size, 1)
    )
    if hand_model.left_hand:
        thetas_mu = (
            torch.tensor(
                [
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -torch.pi / 6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    *(torch.pi / 2 * torch.tensor([1, -0.5, 0], dtype=torch.float)),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.float,
                device=device,
            )
            .unsqueeze(0)
            .repeat(total_batch_size, 1)
        )
    thetas_sigma = args.jitter_strength * torch.ones(
        [total_batch_size, 45], dtype=torch.float, device=device
    )
    thetas = torch.normal(thetas_mu, thetas_sigma)
    if no_fc:
        thetas *= 1e-2

    rotation = pytorch3d.transforms.quaternion_to_axis_angle(
        pytorch3d.transforms.matrix_to_quaternion(rotation)
    )
    hand_pose = torch.cat(
        [
            translation,
            rotation,
            thetas,
        ],
        dim=1,
    )
    hand_pose.requires_grad_()

    # initialize contact point indices

    contact_point_indices = torch.randint(
        hand_model.n_contact_candidates,
        size=[total_batch_size, args.n_contact],
        device=device,
    )

    hand_model.set_parameters(hand_pose, contact_point_indices)
