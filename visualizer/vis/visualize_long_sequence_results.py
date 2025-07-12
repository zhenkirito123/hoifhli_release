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
import glob
import os

import numpy as np
import smplx
import torch
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"

import trimesh

from visualizer.tools.cfg_parser import Config
from visualizer.tools.meshviewer import Mesh, MeshViewer, colors, points2sphere
from visualizer.tools.objectmodel import ObjectModel
from visualizer.tools.utils import euler, params2torch, parse_npz, to_cpu

USE_FLAT_HAND_MEAN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_sequences(cfg):
    result_path = cfg.result_path

    mv = MeshViewer(offscreen=cfg.offscreen)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
    camera_pose[:3, 3] = np.array([-0.5, -3.0, 1.5])
    mv.update_camera_pose(camera_pose)

    vis_sequence(cfg, result_path, mv, cfg.interaction_epoch, cfg.s_idx, cfg.video_idx)
    mv.close_viewer()


def vis_sequence(cfg, result_path_all, mv, interaction_epoch, s_idx, video_idx):
    # process the initial object mesh
    initial_obj_meshes = []
    initial_obj_paths = cfg.initial_obj_path.split("&")
    for initial_obj_path in initial_obj_paths:
        if os.path.exists(initial_obj_path):
            initial_obj_mesh = trimesh.load(initial_obj_path)
            initial_obj_meshes.append(
                Mesh(
                    vertices=initial_obj_mesh.vertices,
                    faces=initial_obj_mesh.faces,
                    vc=colors["green"],
                    smooth=True,
                )
            )

    result_paths = result_path_all.split("&")
    # 1 pass, process all data
    ball_meshes = []
    human_meshes = []
    object_meshes = []

    b_meshes = []
    o_meshes = []
    s_meshes = []
    for result_path in result_paths:
        if "navi" in result_path:
            ball_path = os.path.join(result_path, "ball_objs_bs_idx_0_vis_no_scene")
            subject_path = os.path.join(result_path, "objs_bs_idx_0_vis_no_scene")
        else:
            ball_path = os.path.join(
                result_path,
                "ball_objs_step_{}_bs_idx_0_vis_no_scene".format(interaction_epoch),
            )
            subject_path = os.path.join(
                result_path,
                "objs_step_{}_bs_idx_0_vis_no_scene".format(interaction_epoch),
            )
        sub_ball_meshes = []
        sub_human_meshes = []
        sub_object_meshes = []

        sub_b_meshes = []
        sub_o_meshes = []
        sub_s_meshes = []
        ori_ball_files = os.listdir(ball_path)
        ori_ball_files.sort()
        for tmp_name in ori_ball_files:
            # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            if ".ply" in tmp_name:
                if "start" in tmp_name:
                    continue
                sub_ball_meshes.append(trimesh.load(os.path.join(ball_path, tmp_name)))
                sub_b_meshes.append(
                    Mesh(
                        vertices=sub_ball_meshes[-1].vertices,
                        faces=sub_ball_meshes[-1].faces,
                        vc=colors["green"],
                    )
                )
        ori_obj_files = os.listdir(subject_path)
        ori_obj_files.sort()
        for tmp_name in ori_obj_files:
            # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            if ".ply" in tmp_name:
                if "object" not in tmp_name:
                    sub_human_meshes.append(
                        trimesh.load(os.path.join(subject_path, tmp_name))
                    )
                    sub_human_mesh = Mesh(
                        vertices=sub_human_meshes[-1].vertices,
                        faces=sub_human_meshes[-1].faces,
                        vc=colors["pink"],
                        smooth=True,
                    )
                    sub_human_mesh.set_vertex_colors(
                        vc=colors["red"],
                        vertex_ids=sub_human_meshes[-1].vertices[:, 2] < 0,
                    )
                    sub_s_meshes.append(sub_human_mesh)
                else:
                    sub_object_meshes.append(
                        trimesh.load(os.path.join(subject_path, tmp_name))
                    )
                    sub_o_meshes.append(
                        Mesh(
                            vertices=sub_object_meshes[-1].vertices,
                            faces=sub_object_meshes[-1].faces,
                            vc=colors["yellow"],
                        )
                    )
        assert len(sub_object_meshes) == 0 or len(sub_human_meshes) == len(
            sub_object_meshes
        )

        ball_meshes.append(sub_ball_meshes)
        human_meshes.append(sub_human_meshes)
        object_meshes.append(sub_object_meshes)

        b_meshes.append(sub_b_meshes)
        o_meshes.append(sub_o_meshes)
        s_meshes.append(sub_s_meshes)

    # 2 pass, render the data
    skip_frame = 1
    if not cfg.offscreen:
        while True:
            import time

            for sub_b_meshes, sub_o_meshes, sub_s_meshes in zip(
                b_meshes, o_meshes, s_meshes
            ):
                T = len(sub_s_meshes)
                for frame in range(0, T, skip_frame):
                    start_time = time.time()

                    s_mesh = sub_s_meshes[frame]
                    meshes = [s_mesh]

                    if len(sub_o_meshes) > 0:
                        o_mesh = sub_o_meshes[frame]
                        meshes = [o_mesh, s_mesh]

                    meshes.extend(initial_obj_meshes)
                    # meshes.extend(sub_b_meshes)

                    mv.set_static_meshes(meshes)
                    while time.time() - start_time < 0.03:
                        pass
    else:
        import imageio

        save_dir = os.path.join("{}".format(cfg.save_dir_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, s_idx + "_" + cfg.use_guidance)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_paths = []
        for result_path, sub_b_meshes, sub_o_meshes, sub_s_meshes in tqdm(
            zip(result_paths, b_meshes, o_meshes, s_meshes)
        ):
            img_path = os.path.join(result_path, "vis")
            if not os.path.exists(img_path):
                os.makedirs(img_path)

            T = len(sub_s_meshes)
            for frame in tqdm(range(0, T, skip_frame), leave=False):
                s_mesh = sub_s_meshes[frame]
                meshes = [s_mesh]

                if len(sub_o_meshes) > 0:
                    o_mesh = sub_o_meshes[frame]
                    meshes = [o_mesh, s_mesh]

                meshes.extend(sub_b_meshes)
                meshes.extend(initial_obj_meshes)

                mv.set_static_meshes(meshes)

                camera_pose = np.eye(4)
                camera_pose[:3, :3] = euler([80, -15, 0], "xzx")
                camera_pose[:2, 3] = np.mean(s_mesh.vertices, axis=0)[:2] + np.array(
                    [0, -2.0]
                )
                camera_pose[2, 3] = 1.3
                mv.update_camera_pose(camera_pose)

                mv.save_snapshot(os.path.join(img_path, "%05d.png" % frame))
                img_paths.append(os.path.join(img_path, "%05d.png" % frame))

        video_name = os.path.join(save_dir, "output_{}.mp4".format(video_idx))

        im_arr = []

        for image in tqdm(img_paths, desc="loading images"):
            im = imageio.v2.imread(image)
            im_arr.append(im)
        im_arr = np.asarray(im_arr)
        imageio.mimwrite(video_name, im_arr, fps=30, quality=8)

        print("video saved to %s" % video_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAB-visualize")

    parser.add_argument(
        "--result-path", required=True, type=str, help="The path to the chois_results"
    )
    parser.add_argument(
        "--initial-obj-path",
        type=str,
    )
    parser.add_argument(
        "--save-dir-name",
        type=str,
    )

    parser.add_argument("--s-idx", required=True, type=str)
    parser.add_argument("--video-idx", required=True, type=str)
    parser.add_argument("--interaction-epoch", required=True, type=int)
    parser.add_argument("--use_guidance", required=True, type=str)

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

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        "result_path": result_path,
        "initial_obj_path": args.initial_obj_path,
        "model_path": model_path,
        "offscreen": args.offscreen,
        "s_idx": args.s_idx,
        "video_idx": args.video_idx,
        "interaction_epoch": args.interaction_epoch,
        "use_guidance": args.use_guidance,
        "save_dir_name": args.save_dir_name,
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)
