
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
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm
os.environ['PYOPENGL_PLATFORM'] = 'egl' 

from visualizer.tools.objectmodel import ObjectModel
from visualizer.tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from visualizer.tools.utils import parse_npz
from visualizer.tools.utils import params2torch
from visualizer.tools.utils import to_cpu
from visualizer.tools.utils import euler
from visualizer.tools.cfg_parser import Config

import trimesh

USE_FLAT_HAND_MEAN = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_sequences(cfg):

    result_path = cfg.result_path

    mv = MeshViewer(offscreen=cfg.offscreen)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -3., 1.5])
    mv.update_camera_pose(camera_pose)

    vis_sequence(cfg, result_path, mv)
    mv.close_viewer()


def vis_sequence(cfg, result_path, mv):
    human_meshes = []
    object_meshes = []
    human_meshes_gt = []
    object_meshes_gt = []
    
    ori_obj_files = os.listdir(result_path)
    ori_obj_files.sort()
    human_files = []
    object_files = []
    for tmp_name in ori_obj_files:
        # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
        if ".ply" in tmp_name:
            if "object" not in tmp_name:
                human_files.append(tmp_name)
            else:
                object_files.append(tmp_name)
                
    T = len(human_files)
    for i in range(T):
        human_meshes.append(trimesh.load(os.path.join(result_path, human_files[i])))
        object_meshes.append(trimesh.load(os.path.join(result_path, object_files[i])))

    idx_str = result_path[result_path.find("idx"):result_path.rfind("/")]
    gt_path = result_path.replace(idx_str, idx_str+"_gt")
    vis_gt = os.path.exists(gt_path) and len(os.listdir(gt_path)) > 0 
    if vis_gt:
        offset = np.array([1, 0, 0])
        ori_obj_files = os.listdir(gt_path)
        ori_obj_files.sort()
        human_files = []
        object_files = []
        for tmp_name in ori_obj_files:
            # if ".obj" in tmp_name or ".ply" in tmp_name and "object" not in tmp_name:
            if ".ply" in tmp_name:
                if "object" not in tmp_name:
                    human_files.append(tmp_name)
                else:
                    object_files.append(tmp_name)

        T = len(human_files)
        for i in range(T):
            human_meshes_gt.append(trimesh.load(os.path.join(gt_path, human_files[i])))
            object_meshes_gt.append(trimesh.load(os.path.join(gt_path, object_files[i])))

    skip_frame = 1
    o_meshes = []
    s_meshes = []
    o_meshes_gt = []
    s_meshes_gt = []
    for frame in range(0, T, skip_frame):
        o_mesh = Mesh(vertices=object_meshes[frame].vertices, faces=object_meshes[frame].faces, vc=colors['yellow'])
        s_mesh = Mesh(vertices=human_meshes[frame].vertices, faces=human_meshes[frame].faces, vc=colors['pink'], smooth=True)
        
        # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=abs(object_meshes[frame].vertices[:, 0] - (-0.22)) < 0.01)
        # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=abs(human_meshes[frame].vertices[:, 0] - (-0.22)) < 0.01)
        # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=abs(human_meshes[frame].vertices[:, 1] - (-0.522)) < 0.01)
        # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=abs(human_meshes[frame].vertices[:, 2] - (1)) < 0.01)
        
        o_meshes.append(o_mesh)
        s_meshes.append(s_mesh)
        if vis_gt:
            o_mesh_gt = Mesh(vertices=object_meshes_gt[frame].vertices + offset, faces=object_meshes_gt[frame].faces, vc=colors['green'])
            s_mesh_gt = Mesh(vertices=human_meshes_gt[frame].vertices + offset, faces=human_meshes_gt[frame].faces, vc=colors['pink'], smooth=True)
            o_meshes_gt.append(o_mesh_gt)
            s_meshes_gt.append(s_mesh_gt)
        
    if not cfg.offscreen:
        while True:
            import time
            for frame in range(0, T, skip_frame):
                start_time = time.time()
                o_mesh = o_meshes[frame]
                s_mesh = s_meshes[frame]
                
                if vis_gt:
                    o_mesh_gt = o_meshes_gt[frame]
                    s_mesh_gt = s_meshes_gt[frame]
                    mv.set_static_meshes([o_mesh, s_mesh, o_mesh_gt, s_mesh_gt])
                else:
                    mv.set_static_meshes([o_mesh, s_mesh])
                while time.time() - start_time < 0.03:
                    pass
    else:
        import imageio 

        img_path = os.path.join(result_path, "vis")
        if not os.path.exists(img_path):
            os.makedirs(img_path)
            
        path1, path2 = result_path.split("/")[-2], result_path.split("/")[-1]
        save_dir = os.path.join(result_path, "../../vis", path1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, path2)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for frame in tqdm(range(0, T, skip_frame)):
            o_mesh = o_meshes[frame]
            s_mesh = s_meshes[frame]
            
            if vis_gt:
                o_mesh_gt = o_meshes_gt[frame]
                s_mesh_gt = s_meshes_gt[frame]
                mv.set_static_meshes([o_mesh, s_mesh, o_mesh_gt, s_mesh_gt])
            else:
                mv.set_static_meshes([o_mesh, s_mesh])
                
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
            camera_pose[:2, 3] = np.mean(o_mesh.vertices, axis=0)[:2] + np.array([0, -2.])
            camera_pose[2, 3] = 1.3
            mv.update_camera_pose(camera_pose)
            
            mv.save_snapshot(os.path.join(img_path, "%05d.png"%frame))
        
        video_name = os.path.join(save_dir, 'output.mp4')
        
        images = [img for img in os.listdir(img_path) if img.endswith(".png")]
        images.sort()
        im_arr = []

        for image in images:
            path = os.path.join(img_path, image)
            im = imageio.v2.imread(path)
            im_arr.append(im)
        im_arr = np.asarray(im_arr)
        imageio.mimwrite(video_name, im_arr, fps=30, quality=8) 

        print("video saved to %s"%video_name)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-visualize')

    parser.add_argument('--result-path', required=True, type=str,
                        help='The path to the downloaded grab data')

    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    
    parser.add_argument("--offscreen", action="store_true")

    args = parser.parse_args()

    result_path = args.result_path
    model_path = args.model_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'result_path': result_path,
        'model_path': model_path,
        'offscreen': args.offscreen,
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg)

