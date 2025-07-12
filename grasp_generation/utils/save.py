import os
import shutil
from typing import List, Tuple

import numpy as np
import trimesh


def save_hand_meshes(
    hand_model, object_model, result_path, args, successes, data_lists, lefthand
) -> Tuple[List[List[str]], List[str], List[np.ndarray]]:
    all_hand_mesh_paths = []
    all_object_mesh_paths = []
    all_hand_poses = []
    for i in range(len(args.object_code_list)):
        hand_poses = []
        hand_mesh_paths = []
        min_fail_hand_mesh_path = None
        min_fail_hand_pose = None
        min_fail_energy = float("inf")

        mesh_path = os.path.join(result_path, args.object_code_list[i] + f"_{i}")
        hand = "left_hand" if lefthand else "right_hand"
        os.makedirs(mesh_path, exist_ok=True)
        if os.path.exists(os.path.join(mesh_path, hand)):
            shutil.rmtree(os.path.join(mesh_path, hand))
        os.makedirs(os.path.join(mesh_path, hand), exist_ok=True)
        os.makedirs(os.path.join(mesh_path, hand, "succ"), exist_ok=True)
        os.makedirs(os.path.join(mesh_path, hand, "fail"), exist_ok=True)

        obj_mesh = trimesh.Trimesh(
            vertices=object_model.object_mesh_list[i].vertices,
            faces=object_model.object_mesh_list[i].faces,
        )
        obj_mesh.export(os.path.join(mesh_path, "object.obj"))
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            hand_mesh = trimesh.Trimesh(
                hand_model.vertices[idx].detach().cpu().numpy(),
                hand_model.hand_faces.detach().cpu().numpy(),
            )
            if successes[idx]:
                path = os.path.join(
                    mesh_path, hand, "succ", "{}_{}.obj".format(hand, j)
                )
                hand_mesh.export(path)
                hand_mesh_paths.append(path)
                hand_poses.append(hand_model.hand_pose[idx].detach().cpu().numpy())
            else:
                path = os.path.join(
                    mesh_path, hand, "fail", "{}_{}.obj".format(hand, j)
                )
                hand_mesh.export(path)
                if data_lists[i][j]["energy"] < min_fail_energy:
                    min_fail_energy = data_lists[i][j]["energy"]
                    min_fail_hand_mesh_path = path
                    min_fail_hand_pose = (
                        hand_model.hand_pose[idx].detach().cpu().numpy()
                    )

        data_list = data_lists[i]
        np.save(
            os.path.join(mesh_path, hand, args.object_code_list[i] + ".npy"),
            data_list,
            allow_pickle=True,
        )
        print("Save hand meshes to", os.path.join(mesh_path, hand))

        if min_fail_hand_mesh_path is not None:
            hand_mesh_paths.append(min_fail_hand_mesh_path)
            hand_poses.append(min_fail_hand_pose)

        all_object_mesh_paths.append(os.path.join(mesh_path, "object.obj"))
        all_hand_mesh_paths.append(hand_mesh_paths)
        all_hand_poses.append(hand_poses)

    return all_hand_mesh_paths, all_object_mesh_paths, all_hand_poses
