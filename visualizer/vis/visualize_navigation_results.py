# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys

sys.path.append(".")
sys.path.append("..")

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"

import trimesh

from visualizer.tools.cfg_parser import Config
from visualizer.tools.meshviewer import Mesh, MeshViewer, colors, points2sphere
from visualizer.tools.utils import euler, params2torch, parse_npz, to_cpu

USE_FLAT_HAND_MEAN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_sequences(cfg):
    result_path = cfg.result_path

    mv = MeshViewer(offscreen=cfg.offscreen)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
    camera_pose[:3, 3] = np.array([-0.5, -5.0, 1.5])
    mv.update_camera_pose(camera_pose)

    vis_sequence(cfg, result_path, mv, cfg.interaction_epoch, cfg.s_idx)
    mv.close_viewer()


def vis_sequence(cfg, result_path, mv, interaction_epoch, s_idx):
    # 1 pass, process all data
    ball_meshes = []
    human_meshes = []

    b_meshes = []
    s_meshes = []
    b_meshes_gt = []
    s_meshes_gt = []

    ball_path = os.path.join(
        result_path, "ball_objs_step_{}_bs_idx_0".format(interaction_epoch)
    )
    subject_path = os.path.join(
        result_path, "objs_step_{}_bs_idx_0".format(interaction_epoch)
    )

    ori_ball_files = os.listdir(ball_path)
    ori_ball_files.sort()

    for tmp_name in ori_ball_files:
        # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
        if ".ply" in tmp_name:
            if "start" in tmp_name:
                continue
            ball_meshes.append(trimesh.load(os.path.join(ball_path, tmp_name)))
            b_meshes.append(
                Mesh(
                    vertices=ball_meshes[-1].vertices,
                    faces=ball_meshes[-1].faces,
                    vc=colors["green"],
                )
            )
    ori_obj_files = os.listdir(subject_path)
    ori_obj_files.sort()
    for tmp_name in ori_obj_files:
        # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
        if ".ply" in tmp_name:
            if "object" not in tmp_name:
                human_meshes.append(trimesh.load(os.path.join(subject_path, tmp_name)))
                s_meshes.append(
                    Mesh(
                        vertices=human_meshes[-1].vertices,
                        faces=human_meshes[-1].faces,
                        vc=colors["pink"],
                        smooth=True,
                    )
                )

    # gt
    ball_path_gt = os.path.join(
        result_path, "ball_objs_step_{}_bs_idx_0_gt".format(interaction_epoch)
    )
    subject_path_gt = os.path.join(
        result_path, "objs_step_{}_bs_idx_0_gt".format(interaction_epoch)
    )
    vis_gt = os.path.exists(subject_path_gt) and len(os.listdir(subject_path_gt)) > 0
    if vis_gt:
        offset = np.array([2, 0, 0])
        ori_ball_files_gt = os.listdir(ball_path_gt)
        ori_ball_files_gt.sort()
        for tmp_name in ori_ball_files_gt:
            # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            if ".ply" in tmp_name:
                if "start" in tmp_name:
                    continue
                mesh = trimesh.load(os.path.join(ball_path_gt, tmp_name))
                new_v = mesh.vertices + offset
                b_meshes_gt.append(
                    Mesh(vertices=new_v, faces=mesh.faces, vc=colors["green"])
                )
        ori_obj_files_gt = os.listdir(subject_path_gt)
        ori_obj_files_gt.sort()
        for tmp_name in ori_obj_files_gt:
            # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            if ".ply" in tmp_name:
                if "object" not in tmp_name:
                    mesh = trimesh.load(os.path.join(subject_path_gt, tmp_name))
                    new_v = mesh.vertices + offset
                    s_meshes_gt.append(
                        Mesh(
                            vertices=new_v,
                            faces=mesh.faces,
                            vc=colors["yellow"],
                            smooth=True,
                        )
                    )

    # 2 pass, render the data
    skip_frame = 1
    if not cfg.offscreen:
        while True:
            import time

            T = len(s_meshes)
            for frame in range(0, T, skip_frame):
                start_time = time.time()

                s_mesh = s_meshes[frame]
                meshes = [s_mesh]

                meshes.extend(b_meshes)
                if vis_gt:
                    s_mesh_gt = s_meshes_gt[frame]
                    meshes.extend([s_mesh_gt])
                    meshes.extend(b_meshes_gt)

                mv.set_static_meshes(meshes)
                while time.time() - start_time < 0.03:
                    pass
    else:
        import imageio

        img_paths = []

        path1, path2 = result_path.split("/")[-2], result_path.split("/")[-1]
        save_dir = os.path.join(result_path, "../..", "vis")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, path1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_path = os.path.join(result_path, "vis")
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        T = len(s_meshes)
        for frame in tqdm(range(0, T, skip_frame), leave=False):
            s_mesh = s_meshes[frame]
            meshes = [s_mesh]

            meshes.extend(b_meshes)
            if vis_gt:
                s_mesh_gt = s_meshes_gt[frame]
                meshes.extend([s_mesh_gt])
                meshes.extend(b_meshes_gt)

            mv.set_static_meshes(meshes)

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
            camera_pose[:2, 3] = np.mean(s_mesh.vertices, axis=0)[:2] + np.array(
                [0.5, -3.0]
            )
            camera_pose[2, 3] = 1.3
            mv.update_camera_pose(camera_pose)

            mv.save_snapshot(os.path.join(img_path, "%05d.png" % frame))
            img_paths.append(os.path.join(img_path, "%05d.png" % frame))

        video_name = os.path.join(save_dir, "output_{}.mp4".format(s_idx))

        im_arr = []

        for image in img_paths:
            im = imageio.v2.imread(image)
            im_arr.append(im)
        im_arr = np.asarray(im_arr)
        imageio.mimwrite(video_name, im_arr, fps=30, quality=8)

        print("video saved to %s" % video_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize")

    parser.add_argument(
        "--result-path", required=True, type=str, help="The path to the results"
    )
    parser.add_argument("--s-idx", required=True, type=str)
    parser.add_argument("--interaction-epoch", required=True, type=int)
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="The path to the folder containing smplx models",
    )
    parser.add_argument("--offscreen", action="store_true")

    args = parser.parse_args()

    result_path = args.result_path
    model_path = args.model_path

    cfg = {
        "result_path": result_path,
        "model_path": model_path,
        "offscreen": args.offscreen,
        "s_idx": args.s_idx,
        "interaction_epoch": args.interaction_epoch,
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)
