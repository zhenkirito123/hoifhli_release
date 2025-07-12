import os
import pickle
import random
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scipy.signal import medfilt

from argument_parser import parse_opt
from grasp_generation.gen_grasp import run_grasp
from manip.data.humanml3d_dataset import normalize, quat_between
from manip.inertialize.inert import apply_inertialize, apply_linear_offset
from manip.model.transformer_hand_foot_manip_cond_diffusion_model import (
    CondGaussianDiffusionBothSensorNew,
)
from manip.utils.trainer_utils import (
    decide_no_force_closure_from_objects,
    finger_smooth_transition,
    fix_feet,
    generate_T_pose,
    interaction_to_navigation_smooth_transition,
    load_planned_path_as_waypoints,
    mirror_rot_6d,
    navigation_to_interaction_smooth_transition,
    smooth_res,
    smplx_ik,
)
from trainer_finger_diffusion import OMOMOBothSensorTrainer as FingerTrainer
from trainer_interaction_motion_diffusion import (
    Trainer as InteractionTrainer,
)
from trainer_interaction_motion_diffusion import (
    build_interaction_trainer,
    build_wrist_relative_conditions,
    find_static_frame_at_end,
    run_interaction_trainer,
)
from trainer_navigation_motion_diffusion import (
    NavigationCondGaussianDiffusion,
    calculate_navi_representation_dim,
)
from trainer_navigation_motion_diffusion import Trainer as NavigationTrainer

object_static_height_floor = {
    "monitor": 0.23,
    "trashcan": 0.16,
    "largebox": 0.175,
    "smallbox": 0.08,
    "plasticbox": 0.137,
    "suitcase": 0.265,
    "woodchair": 0.43,
    "whitechair": 0.45,
    "smalltable": 0.28,
    "largetable": 0.37,
    "floorlamp": 0.91,
    "floorlamp1": 0.88,
    "clothesstand": 0.52,
    "tripod": 0.50,
    "bottle": 0.28,
    "right_shoes1": 0.065,
    "toy1": 0,
    "toy2": 0.085,
    "laundrybasket": 0.34,
    "vase1_big": 0.23,
    "vase1_big2": 0.425,
    "box_a": 0.13,
    "box_b": 0.1,
    "box_c": 0.0686,
    "newobject": 0.425,
}


def get_object_static_height_floor(object_name):
    return object_static_height_floor[object_name]


def generate_object_orientation(
    object_name,
    forward,
):
    assert forward.shape[0] == 3 and len(forward.shape) == 1
    # when facing direction is [1, 0, 0]
    object_static_orientation = {
        "monitor": torch.Tensor(
            [
                [0.8932, 0.4497, 0.0000],
                [-0.4497, 0.8932, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),  # front
        "trashcan": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "largebox": torch.Tensor(
            [
                [0.9727, -0.2321, 0.0000],
                [0.2321, 0.9727, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),
        "box_a": torch.Tensor(
            [
                [0.9727, -0.2321, 0.0000],
                [0.2321, 0.9727, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),
        "box_b": torch.Tensor(
            [
                [0.9727, -0.2321, 0.0000],
                [0.2321, 0.9727, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),
        "box_c": torch.Tensor(
            [
                [0.9727, -0.2321, 0.0000],
                [0.2321, 0.9727, 0.0000],
                [0.0000, 0.0000, 1.0000],
            ]
        ),
        "smallbox": torch.Tensor(
            [
                [0.6000, 0.8000, 0.0000],
                [-0.8000, 0.6000, -0.0000],
                [-0.0000, 0.0000, 1.0000],
            ]
        ),
        "plasticbox": torch.Tensor(
            [
                [0.9765, 0.2157, 0.0000],
                [-0.2157, 0.9765, -0.0000],
                [-0.0000, 0.0000, 1.0000],
            ]
        ),
        "suitcase": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "woodchair": torch.Tensor(
            [
                [-0.9880, 0.1545, 0.0000],
                [-0.1545, -0.9880, -0.0000],
                [-0.0000, 0.0000, 1.0000],
            ]
        ),
        "whitechair": torch.Tensor(
            [
                [0.2905, 0.9569, 0.0000],
                [-0.9569, 0.2905, -0.0000],
                [-0.0000, 0.0000, 1.0000],
            ]
        ),
        "smalltable": torch.Tensor(
            [
                [1.0000, -0.0053, 0.0000],
                [0.0053, 0.9997, -0.0254],
                [0.0001, 0.0254, 0.9997],
            ]
        ),
        "largetable": torch.Tensor(
            [
                [0.6910, 0.7228, 0.0083],
                [-0.7098, 0.6806, -0.1816],
                [-0.1369, 0.1196, 0.9833],
            ]
        ),
        "floorlamp": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "floorlamp1": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "clothesstand": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "tripod": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "bottle": torch.Tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        "right_shoes1": torch.Tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        "toy1": torch.Tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "toy2": torch.Tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        "laundrybasket": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "vase1_big": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "vase1_big2": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        "newobject": torch.Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
    }

    x_axis = torch.Tensor([1, 0, 0]).float().to(forward.device)
    forward[2] = 0
    forward = forward / (torch.norm(forward) + 1e-8)
    forward = forward.float()
    if abs(forward[1]) < 1e-3:
        if forward[0] > 0:
            yrot = torch.Tensor([1.0, 0.0, 0.0, 0.0]).to(forward.device)
        else:
            yrot = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(forward.device)
    else:
        yrot = normalize(quat_between(x_axis, forward)).to(forward.device)
    ori_rot = object_static_orientation[object_name].to(forward.device)
    object_orientation = transforms.quaternion_to_matrix(yrot).matmul(ori_rot)

    return object_orientation


def random_paths(start_point, num=2):
    path_data = []
    for i in range(num):
        dis = random.uniform(1, 2)
        if i == num - 1:
            dis = random.uniform(1.5, 2.0)
        vec = np.random.randn(3)
        vec[2] = 0
        vec /= np.linalg.norm(vec)
        if i == 0:
            path_data.append(start_point + dis * vec)
        else:
            path_data.append(path_data[-1] + dis * vec)

    path_data = np.array(path_data)
    path_data[..., 2] = 0.5
    return path_data


def random_paths_object(start_point, prev_vec=None, num=2):
    path_data = []
    for i in range(num):
        if i == 0:
            dis = random.uniform(0.25, 0.5)
        else:
            dis = random.uniform(2, 3)
        vec = np.random.randn(3)
        vec[2] = 0
        vec /= np.linalg.norm(vec)
        if i == 0:
            if prev_vec is not None:
                vec = prev_vec / np.linalg.norm(prev_vec)
            path_data.append(start_point + dis * vec)
        else:
            path_data.append(path_data[-1] + dis * vec)

    path_data = np.array(path_data)
    path_data[0, 2] = random.uniform(0.25, 0.85)
    path_data[-1, 2] = random.uniform(0.25, 0.85)
    return path_data


def build_coarse_interaction_trainer(
    opt, device, milestone, load_ds=False
) -> InteractionTrainer:
    # (Object position and rotation) + (Human pose) + (Contact label)
    repr_dim = (3 + 9) + (24 * 3 + 22 * 6) + (4)

    vis_wdir = "./results/interaction/{}".format(opt.vis_wdir)
    interaction_wdir = os.path.join(opt.project, opt.cnet_save_dir, "weights")

    interaction_trainer = build_interaction_trainer(
        opt=opt,
        device=device,
        vis_wdir=vis_wdir,
        results_folder=interaction_wdir,
        repr_dim=repr_dim,
        loss_type="l1",
        use_feet_contact=False,
        use_wandb=False,
        load_ds=load_ds,
    )
    interaction_trainer.load(milestone)
    interaction_trainer.ema.ema_model.eval()
    print(f"Loaded coarse interaction model weight: {milestone}")
    return interaction_trainer


def build_fine_interaction_trainer(
    opt, device, milestone, load_ds=False
) -> InteractionTrainer:
    # (Object position and rotation) + (Human pose) + (Contact label) + (Relative wrist)
    repr_dim = (3 + 9) + (24 * 3 + 22 * 6) + (4) + (2 * (3 + 6))

    # Human root orientation in xy plane
    if opt.add_interaction_root_xy_ori:
        repr_dim += 6

    # Feet floor contact
    if opt.add_interaction_feet_contact:
        repr_dim += 4

    vis_wdir = None
    interaction_wdir = os.path.join(opt.project, opt.rnet_save_dir, "weights")

    interaction_trainer = build_interaction_trainer(
        opt=opt,
        device=device,
        vis_wdir=vis_wdir,
        results_folder=interaction_wdir,
        repr_dim=repr_dim,
        loss_type="l1",
        use_feet_contact=opt.add_interaction_feet_contact,
        use_wandb=False,
        load_ds=load_ds,
    )
    interaction_trainer.load(milestone)
    interaction_trainer.ema.ema_model.eval()
    print(f"Loaded fine interaction model weight: {milestone}")
    return interaction_trainer


def build_navi_trainer(opt, device, milestone) -> NavigationTrainer:
    vis_wdir = "./results/navigation/{}".format(opt.vis_wdir)
    wdir = os.path.join(opt.project, opt.navi_save_dir, "weights")

    repr_dim = calculate_navi_representation_dim(opt)

    if opt.use_l2_loss:
        loss_type = "l2"
    else:
        loss_type = "l1"

    diffusion_model = NavigationCondGaussianDiffusion(
        opt,
        d_feats=repr_dim,
        d_model=opt.d_model_nav,
        n_dec_layers=opt.n_dec_layers_nav,
        n_head=opt.n_head_nav,
        d_k=opt.d_k_nav,
        d_v=opt.d_v_nav,
        max_timesteps=opt.window + 1,
        out_dim=repr_dim,
        timesteps=1000,
        objective="pred_x0",
        loss_type=loss_type,
        input_first_human_pose=True,
        add_feet_contact=opt.add_feet_contact,
    )
    diffusion_model.to(device)

    trainer = NavigationTrainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,  # 32
        train_lr=opt.learning_rate,  # 1e-4
        train_num_steps=8000000,  # 700000, total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=str(wdir),
        vis_folder=vis_wdir,
        use_wandb=False,
        load_ds=False,
    )
    trainer.load(milestone)
    trainer.ema.ema_model.eval()
    print(f"Loaded navigation model weight: {milestone}")
    return trainer


def build_finger_opt(opt):
    finger_arg_dict = {}
    for key, value in opt.__dict__.items():
        if key[:7] == "finger_":
            finger_arg_dict[key[7:]] = value
    return SimpleNamespace(**finger_arg_dict)


def build_finger_trainer(finger_opt, device) -> FingerTrainer:
    repr_dim = 15 * 6
    loss_type = "l1"

    diffusion_model = CondGaussianDiffusionBothSensorNew(
        finger_opt,
        d_feats=repr_dim,
        d_model=finger_opt.d_model,
        n_dec_layers=finger_opt.n_dec_layers,
        n_head=finger_opt.n_head,
        d_k=finger_opt.d_k,
        d_v=finger_opt.d_v,
        max_timesteps=finger_opt.window + 1,
        out_dim=repr_dim,
        timesteps=1000,
        objective="pred_x0",
        loss_type=loss_type,
        batch_size=finger_opt.batch_size,
        bps_in_dim=1024,
        second_bps_in_dim=100,
        object_wrist_feats_dim=9,
    )
    diffusion_model.to(device)

    common_params = {
        "opt": finger_opt,
        "diffusion_model": diffusion_model,
        "train_batch_size": finger_opt.batch_size,  # 32
        "train_lr": finger_opt.learning_rate,  # 1e-4
        "train_num_steps": 400000,  # 700000, total training steps
        "gradient_accumulate_every": 2,  # gradient accumulation steps
        "ema_decay": 0.995,  # exponential moving average decay
        "amp": True,  # turn on mixed precision
        "results_folder": os.path.join(
            finger_opt.project, finger_opt.exp_name, "weights"
        ),
        "use_wandb": False,
        "load_ds": False,
    }
    trainer = FingerTrainer(**common_params)

    finger_milestone = str(finger_opt.milestone)
    trainer.load(finger_milestone)
    trainer.ema.ema_model.eval()
    print(f"Loaded finger model weight: {finger_milestone}")
    return trainer


def simple_random_setting(
    object_list,
    sub_num,
    num=100,
    default_action_name="lift",
):
    """
    This function creates random configurations for object manipulation scenarios,
    including object positions, orientations, and movement paths.

    This method assumes that each object can only be manipulated from specific directions.
    To accommodate this, it first generates a navigation path toward the object, and then
    places the object at the end of that path to ensure a valid initial pose for manipulation.

    Args:
        object_list (List[str]): List of object names to choose from randomly
        sub_num (int): Number of sub-scenarios to generate per main scenario
        num (int, optional): Number of main scenarios to generate. Defaults to 100
        default_action_name (str, optional): Type of action to perform. Must be one of
                                   "lift", "push", or "pull". Defaults to "lift"

    Returns:
        Tuple[List, List, List, List, List, List]: A tuple containing:
            - obj_initial_rot_mat_list: List of initial rotation matrices for objects
            - obj_end_rot_mat_list: List of final rotation matrices for objects
            - path_data_list: List of movement path data
            - object_names_list: List of object names used in each scenario
            - action_names_list: List of action names for each scenario
            - text_list: List of text descriptions for each scenario
    """
    (
        obj_initial_rot_mat_list,
        obj_end_rot_mat_list,
        path_data_list,
        object_names_list,
        action_names_list,
        text_list,
    ) = [], [], [], [], [], []
    table_height_list = []
    for i in range(num):
        (
            sub_obj_initial_rot_mat_list,
            sub_obj_end_rot_mat_list,
            sub_path_data_list,
            sub_object_names_list,
            sub_action_names_list,
            sub_text_list,
        ) = [], [], [], [], [], []
        sub_table_height_list = []
        for j in range(sub_num):
            if j % 2 == 1:
                object_name = random.choice(object_list)
                path = random_paths_object(
                    sub_path_data_list[-1][-1],
                    prev_vec=sub_path_data_list[-1][-1] - sub_path_data_list[-1][-2],
                )
                path[0][2] = get_object_static_height_floor(object_name)
                path[-1][2] = get_object_static_height_floor(object_name)
                initial_rot_mat = generate_object_orientation(
                    object_name,
                    forward=torch.from_numpy(path[0] - sub_path_data_list[-1][-1]),
                )
                end_rot_mat = generate_object_orientation(
                    object_name, forward=torch.from_numpy(path[-1] - path[-2])
                )
                action_name = default_action_name

                if action_name == "push":
                    text = f"Push the {object_name}, move the {object_name}. Return to standing pose."
                elif action_name == "pull":
                    text = f"Pull the {object_name}, move the {object_name}. Return to standing pose."
                else:
                    assert action_name == "lift", (
                        "action_name must be lift, not {}".format(action_name)
                    )
                    text = f"Lift the {object_name}, move the {object_name}, and put down the {object_name}. Return to standing pose."

                table_height = path[-1][2] - get_object_static_height_floor(object_name)
            else:
                if j == 0:
                    path = random_paths(np.array([0, 0, 0]))
                else:
                    path = random_paths(sub_path_data_list[-1][-1])
                initial_rot_mat = []
                end_rot_mat = []
                object_name = ""
                action_name = "walking"
                text = "A person walks, then standing still."
                table_height = 0.0
            sub_obj_initial_rot_mat_list.append(initial_rot_mat)
            sub_obj_end_rot_mat_list.append(end_rot_mat)
            sub_path_data_list.append(path)
            sub_object_names_list.append(object_name)
            sub_action_names_list.append(action_name)
            sub_text_list.append(text)
            sub_table_height_list.append(table_height)
        obj_initial_rot_mat_list.append(sub_obj_initial_rot_mat_list)
        obj_end_rot_mat_list.append(sub_obj_end_rot_mat_list)
        path_data_list.append(sub_path_data_list)
        object_names_list.append(sub_object_names_list)
        action_names_list.append(sub_action_names_list)
        text_list.append(sub_text_list)
        table_height_list.append(sub_table_height_list)
    return (
        obj_initial_rot_mat_list,
        obj_end_rot_mat_list,
        path_data_list,
        object_names_list,
        action_names_list,
        text_list,
        table_height_list,
    )


def run_grasp_generation(
    curr_object_name: str,
    left_contact: bool,
    right_contact: bool,
    left_wrist_init_pose: Optional[Dict],
    right_wrist_init_pose: Optional[Dict],
    no_force_closure: bool = False,
    grasp_batch: int = 1,
    grasp_iter: int = 5000,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Dict],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Dict],
]:
    if right_contact:
        (
            right_wrist_pos_in_obj,
            right_wrist_rot_mat_in_obj,
            right_finger_local_rot_6d,
        ) = run_grasp(
            lefthand=False,
            name="right",
            wrist_init_pose=right_wrist_init_pose,
            batch_size=grasp_batch,
            object_code_list=[curr_object_name],
            n_iter=grasp_iter,
            no_fc=no_force_closure,
        )
        right_wrist_pos_in_obj = right_wrist_pos_in_obj[0]
        right_wrist_rot_mat_in_obj = right_wrist_rot_mat_in_obj[0]
        right_finger_local_rot_6d = right_finger_local_rot_6d[0]

        right_wrist_pose = {
            "wrist_pos": right_wrist_pos_in_obj.cpu(),
            "wrist_rot": right_wrist_rot_mat_in_obj.cpu(),
        }
    else:
        right_wrist_pos_in_obj = None
        right_wrist_rot_mat_in_obj = None
        right_finger_local_rot_6d = None
        right_wrist_pose = None

    if left_contact:
        (
            left_wrist_pos_in_obj,
            left_wrist_rot_mat_in_obj,
            left_finger_local_rot_6d,
        ) = run_grasp(
            lefthand=True,
            name="left",
            wrist_init_pose=left_wrist_init_pose,
            batch_size=grasp_batch,
            object_code_list=[curr_object_name],
            n_iter=grasp_iter,
            no_fc=no_force_closure,
        )
        left_wrist_pos_in_obj = left_wrist_pos_in_obj[0]  # 3
        left_wrist_rot_mat_in_obj = left_wrist_rot_mat_in_obj[0]  # 3 X 3
        left_finger_local_rot_6d = left_finger_local_rot_6d[0]  # 90

        left_wrist_pose = {
            "wrist_pos": left_wrist_pos_in_obj.cpu(),
            "wrist_rot": left_wrist_rot_mat_in_obj.cpu(),
        }
    else:
        left_wrist_pos_in_obj = None
        left_wrist_rot_mat_in_obj = None
        left_finger_local_rot_6d = None
        left_wrist_pose = None

    return (
        right_wrist_pos_in_obj,
        right_wrist_rot_mat_in_obj,
        right_finger_local_rot_6d,
        right_wrist_pose,
        left_wrist_pos_in_obj,
        left_wrist_rot_mat_in_obj,
        left_finger_local_rot_6d,
        left_wrist_pose,
    )


def add_static_phase(
    all_res_list: torch.Tensor,
    coarse_all_res_list: Optional[torch.Tensor],
    fine_all_res_list: Optional[torch.Tensor],
    contact_begin_frame: int,
    contact_end_frame: int,
    finger_all_res_list: Optional[torch.Tensor] = None,
    static_phase_len: int = 3,
):
    all_res_list = torch.cat(
        (
            all_res_list[:, :contact_begin_frame],
            all_res_list[:, contact_begin_frame - 1 : contact_begin_frame].repeat(
                1, static_phase_len, 1
            ),
            all_res_list[:, contact_begin_frame:contact_end_frame],
            all_res_list[:, contact_end_frame : contact_end_frame + 1].repeat(
                1, static_phase_len, 1
            ),
            all_res_list[:, contact_end_frame:],
        ),
        dim=1,
    )

    if finger_all_res_list is not None:
        finger_all_res_list = torch.cat(
            (
                finger_all_res_list[:contact_begin_frame],
                finger_all_res_list[
                    contact_begin_frame - 1 : contact_begin_frame
                ].repeat(static_phase_len, 1),
                finger_all_res_list[contact_begin_frame:contact_end_frame],
                finger_all_res_list[contact_end_frame : contact_end_frame + 1].repeat(
                    static_phase_len, 1
                ),
                finger_all_res_list[contact_end_frame:],
            ),
            dim=0,
        )
    return all_res_list, coarse_all_res_list, fine_all_res_list, finger_all_res_list


def save_initial_end_object_meshes(
    path_data: List[np.ndarray],
    obj_initial_rot_mats: List[torch.Tensor],
    obj_end_rot_mats: List[torch.Tensor],
    object_names: List[str],
    num_sub_seq_path: int,
    p_idx: int,
    vis_wdir: str,
    interaction_trainer: InteractionTrainer,
) -> List[str]:
    obj_paths = []
    dest_mesh_vis_folder = "./results/initial_obj_vis/{}".format(vis_wdir)
    if not os.path.exists(dest_mesh_vis_folder):
        os.makedirs(dest_mesh_vis_folder)

    # Delete the previous results if exists, create a new folder.
    dest_mesh_vis_folder = os.path.join(dest_mesh_vis_folder, "p_idx_{}".format(p_idx))
    if os.path.exists(dest_mesh_vis_folder):
        shutil.rmtree(dest_mesh_vis_folder)
    if not os.path.exists(dest_mesh_vis_folder):
        os.makedirs(dest_mesh_vis_folder)

    for o_idx in range(num_sub_seq_path):
        if len(obj_initial_rot_mats[o_idx]) > 0:
            object_name = object_names[o_idx]
            obj_rest_verts, obj_mesh_faces = (
                interaction_trainer.ds.load_rest_pose_object_geometry(object_name)
            )
            obj_rest_verts = torch.from_numpy(obj_rest_verts)

            # The begin object mesh.
            rot_mat = obj_initial_rot_mats[o_idx][None]  # 1 X 3 X 3
            pos = torch.from_numpy(path_data[o_idx][0:1])  # 1 X 3

            obj_mesh_verts = interaction_trainer.ds.load_object_geometry_w_rest_geo(
                rot_mat, pos, obj_rest_verts.float().to(rot_mat.device)
            )
            mesh = trimesh.Trimesh(vertices=obj_mesh_verts[0], faces=obj_mesh_faces)
            curr_mesh_path = os.path.join(
                dest_mesh_vis_folder, "o_idx_begin_{}".format(o_idx) + ".ply"
            )
            mesh.export(curr_mesh_path)
            obj_paths.append(curr_mesh_path)

            # The end object mesh.
            rot_mat = obj_end_rot_mats[o_idx][None]  # 1 X 3 X 3
            pos = torch.from_numpy(path_data[o_idx][-1:])  # 1 X 3

            obj_mesh_verts = interaction_trainer.ds.load_object_geometry_w_rest_geo(
                rot_mat, pos, obj_rest_verts.float().to(rot_mat.device)
            )
            mesh = trimesh.Trimesh(vertices=obj_mesh_verts[0], faces=obj_mesh_faces)
            curr_mesh_path = os.path.join(
                dest_mesh_vis_folder, "o_idx_end_{}".format(o_idx) + ".ply"
            )
            mesh.export(curr_mesh_path)
            obj_paths.append(curr_mesh_path)

    print("Visualizing initial obj mesh for {}".format(dest_mesh_vis_folder))
    return obj_paths


def save_interaction_motion_meshes(
    interaction_trainer: InteractionTrainer,
    all_res_list,
    ref_obj_rot_mat,
    ref_data_dict: Dict,
    step: Union[str, int],
    planned_waypoints_pos: torch.Tensor,
    curr_object_name: str,
    vis_tag: Optional[str] = None,
    dest_mesh_vis_folder: Optional[str] = None,
    finger_all_res_list: Optional[torch.Tensor] = None,
):
    dest_mesh_vis_folder, *_, params_path = interaction_trainer.gen_vis_res_long_seq(
        all_res_list=all_res_list,
        ref_obj_rot_mat=ref_obj_rot_mat,
        ref_data_dict=ref_data_dict,
        step=step,
        planned_waypoints_pos=planned_waypoints_pos,
        curr_object_name=curr_object_name,
        vis_tag=vis_tag,
        vis_wo_scene=True,
        cano_quat=None,
        finger_all_res_list=finger_all_res_list,
    )

    return (
        dest_mesh_vis_folder,
        params_path,
    )


def render_motion_clip(
    mesh_save_folders: List[str],
    initial_end_obj_mesh_paths: List[str],
    p_idx: int,
    video_paths: List[str],
    use_guidance_str: str,
    interaction_checkpoint_epoch: str,
    video_save_dir_name: str,
) -> str:
    """Render a motion clip from the given mesh save folders.

    Args:
        mesh_save_folders (List[str]): the paths to the mesh save folders.
        initial_end_obj_mesh_paths (List[str]): the paths to the initial and end object meshes.
        p_idx (int): the index of the trajectory, used to name the output video.
        video_paths (List[str]): the paths to the motion clips to merge.
        use_guidance_str (str): whether to use guidance in the denoising process, used to name the
            output video.
        interaction_checkpoint_epoch (str): the epoch of the interaction model, used to name the
            output video.
        video_save_dir_name (str): the directory to save the output video.
        coarse_video_paths (Optional[List[str]], optional): the paths to the coarse motion clips.
        fine_video_paths (Optional[List[str]], optional): the paths to the fine motion clips.
        coarse_dest_mesh_vis_folder (Optional[str], optional): the directory to save the coarse
            mesh visualization.
        fine_dest_mesh_vis_folder (Optional[str], optional): the directory to save the fine mesh
            visualization.

    Returns:
        str: the output video path.
    """

    mesh_save_folders_str = "&".join(mesh_save_folders)
    initial_end_obj_mesh_paths_str = "&".join(initial_end_obj_mesh_paths)

    subprocess.run(
        [
            "python",
            "visualizer/vis/visualize_long_sequence_results.py",
            "--result-path",
            mesh_save_folders_str,
            "--initial-obj-path",
            initial_end_obj_mesh_paths_str,
            "--s-idx",
            str(p_idx),
            "--video-idx",
            str(len(video_paths)),
            "--save-dir-name",
            video_save_dir_name,
            "--interaction-epoch",
            str(interaction_checkpoint_epoch),
            "--model-path",
            "../data/smplx/models_smplx_v1_1/models/",
            "--use_guidance",
            use_guidance_str,
            "--offscreen",
        ]
    )

    return os.path.join(
        "{}".format(video_save_dir_name),
        "{}_{}".format(p_idx, use_guidance_str),
        "output_{}.mp4".format(len(video_paths)),
    )


def merge_motion_clips(
    final_video_path: str,
    video_paths: List[str],
):
    """Merge motion clips into a single video.

    Args:
        final_video_path (str): the path to save the final video.
        video_paths (List[str]): the paths to the motion clips to merge.
        coarse_video_paths (List[str], optional): the paths to the coarse motion clips.
        fine_video_paths (List[str], optional): the paths to the fine motion clips.
        final_coarse_video_path (str, optional): the path to save the final coarse motion clip.
        final_fine_video_path (str, optional): the path to save the final fine motion clip.
        compare_video_dir (str, optional): the directory to save the comparison video.
    """

    video_clips = [VideoFileClip(video_path) for video_path in video_paths]
    video_clips = concatenate_videoclips(video_clips)
    video_clips.write_videofile(final_video_path)


def set_obj_start_end(
    obj_com_pos: torch.Tensor,
    obj_rot_mat: torch.Tensor,
    contact_begin_frame: int,
    contact_end_frame: int,
    obj_initial_rot_mat: torch.Tensor,
    obj_end_rot_mat: torch.Tensor,
    path_data: List[np.ndarray],
    device: torch.device = torch.device("cuda:0"),
):
    if contact_begin_frame == 0:
        contact_begin_frame = 1

    obj_com_pos[:contact_begin_frame] = torch.from_numpy(path_data[0]).to(device)
    obj_rot_mat[:contact_begin_frame] = obj_initial_rot_mat.to(device)
    obj_com_pos[contact_end_frame:] = torch.from_numpy(path_data[-1]).to(device)
    obj_rot_mat[contact_end_frame:] = obj_end_rot_mat.to(device)

    obj_com_pos_contact, obj_rot_mat_contact = apply_linear_offset(
        original_jpos=obj_com_pos[None][
            :, contact_begin_frame : contact_end_frame - 10
        ],
        original_rot_6d=transforms.matrix_to_rotation_6d(obj_rot_mat)[None][
            :, contact_begin_frame : contact_end_frame - 10
        ],
        new_target_jpos=obj_com_pos[contact_begin_frame - 1][None],
        new_target_rot_6d=transforms.matrix_to_rotation_6d(
            obj_rot_mat[contact_begin_frame - 1][None]
        ),
        reversed=True,
    )
    obj_com_pos[contact_begin_frame : contact_end_frame - 10] = obj_com_pos_contact[0]
    obj_rot_mat[contact_begin_frame : contact_end_frame - 10] = (
        transforms.rotation_6d_to_matrix(obj_rot_mat_contact[0])
    )
    obj_com_pos_contact, obj_rot_mat_contact = apply_linear_offset(
        original_jpos=obj_com_pos[None][
            :, contact_begin_frame + 10 : contact_end_frame
        ],
        original_rot_6d=transforms.matrix_to_rotation_6d(obj_rot_mat)[None][
            :, contact_begin_frame + 10 : contact_end_frame
        ],
        new_target_jpos=obj_com_pos[contact_end_frame][None],
        new_target_rot_6d=transforms.matrix_to_rotation_6d(
            obj_rot_mat[contact_end_frame][None]
        ),
    )
    obj_com_pos[contact_begin_frame + 10 : contact_end_frame] = obj_com_pos_contact[0]
    obj_rot_mat[contact_begin_frame + 10 : contact_end_frame] = (
        transforms.rotation_6d_to_matrix(obj_rot_mat_contact[0])
    )
    return obj_com_pos, obj_rot_mat


def post_process(
    object_data_dict: Dict[str, Any],
    interaction_trainer: InteractionTrainer,
    path_data: List[np.ndarray],
    obj_initial_rot_mat: torch.Tensor,
    obj_end_rot_mat: torch.Tensor,
    obj_com_pos: torch.Tensor,
    obj_rot_mat: torch.Tensor,
    all_res_list: torch.Tensor,
    contact_begin_frame: int,
    contact_end_frame: int,
    human_jnts: torch.Tensor,
    left_wrist_pos: torch.Tensor,
    left_wrist_rot_mat: torch.Tensor,
    left_wrist_pos_in_obj: torch.Tensor,
    left_wrist_rot_mat_in_obj: torch.Tensor,
    right_wrist_pos: torch.Tensor,
    right_wrist_rot_mat: torch.Tensor,
    right_wrist_pos_in_obj: torch.Tensor,
    right_wrist_rot_mat_in_obj: torch.Tensor,
    left_contact: bool,
    left_begin_frame: int,
    left_end_frame: int,
    right_contact: bool,
    right_begin_frame: int,
    right_end_frame: int,
    pred_feet_contact: Optional[torch.Tensor],
    fix_feet_floor_penetration: bool = True,
    fix_feet_sliding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post process the generated human motion and object trajectory.

    1. Fix the object trajectory at the at the beginning and the end.
    2. [Optional] Smooth the whole object trajectory.
    3. Smooth the wrist trajectory, at the beginning and the end of the contact.
    4. Fix feet penetration and sliding.

    path_data: 2 X 3, the gt global obejct com position at the beginning and the end.
    obj_initial_rot_mat: 3 X 3, global rotation matrix of the object at the beginning.
    obj_end_rot_mat: 3 X 3, global rotation matrix of the object at the end.
    obj_com_pos: T X 3, the predicted global object com position.
    obj_rot_mat: T X 3 X 3, the predicted global object rotation matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    device = obj_com_pos.device

    # Smooth the object trajectory at the beginning and the end.
    if (left_contact or right_contact) and contact_end_frame - contact_begin_frame > 20:
        obj_com_pos, obj_rot_mat = set_obj_start_end(
            obj_com_pos,
            obj_rot_mat,
            contact_begin_frame,
            contact_end_frame,
            obj_initial_rot_mat,
            obj_end_rot_mat,
            path_data,
            device,
        )
    else:
        obj_com_pos, obj_rot_mat = set_obj_start_end(
            obj_com_pos,
            obj_rot_mat,
            1,
            119,
            obj_initial_rot_mat,
            obj_end_rot_mat,
            path_data,
            device,
        )

    # Update the object trajectory in all_res_list.
    all_res_list[0, :, :3] = interaction_trainer.ds.normalize_obj_pos_min_max(
        obj_com_pos.reshape(1, -1, 3)
    ).reshape(-1, 3)
    all_res_list[0, :, 3:12] = (
        interaction_trainer.ds.prep_rel_obj_rot_mat_w_reference_mat(
            obj_rot_mat.reshape(1, -1, 3, 3),
            object_data_dict["reference_obj_rot_mat"].to(device),
        ).reshape(-1, 9)
    )

    # Smooth the wrist trajectory, at the beginning and the end of the contact.
    target_human_jnts = human_jnts.clone()

    if right_contact:
        right_wrist_pos_contact = (
            obj_com_pos + (obj_rot_mat @ right_wrist_pos_in_obj.reshape(3, 1))[..., 0]
        )  # T X 3
        right_wrist_rot_mat_contact = (
            obj_rot_mat @ right_wrist_rot_mat_in_obj
        )  # T X 3 X 3

        new_right_wrist_pos = []
        new_right_wrist_rot_6d = []
        if right_begin_frame > 0:
            right_wrist_pos_prev_contact, right_wrist_rot_6d_pre_contact = (
                apply_linear_offset(
                    original_jpos=right_wrist_pos[None][:, :right_begin_frame],
                    original_rot_6d=transforms.matrix_to_rotation_6d(
                        right_wrist_rot_mat
                    )[None][:, :right_begin_frame],
                    new_target_jpos=right_wrist_pos_contact[right_begin_frame - 1][
                        None
                    ],
                    new_target_rot_6d=transforms.matrix_to_rotation_6d(
                        right_wrist_rot_mat_contact[right_begin_frame - 1][None]
                    ),
                )
            )
            new_right_wrist_pos.append(right_wrist_pos_prev_contact[0])
            new_right_wrist_rot_6d.append(right_wrist_rot_6d_pre_contact[0])

        new_right_wrist_pos.append(
            right_wrist_pos_contact[right_begin_frame:right_end_frame]
        )
        new_right_wrist_rot_6d.append(
            transforms.matrix_to_rotation_6d(
                right_wrist_rot_mat_contact[right_begin_frame:right_end_frame]
            )
        )

        if right_end_frame < right_wrist_pos.shape[0]:
            right_wrist_pos_post_contact, right_wrist_rot_6d_post_contact = (
                apply_linear_offset(
                    original_jpos=right_wrist_pos[None][:, right_end_frame:],
                    original_rot_6d=transforms.matrix_to_rotation_6d(
                        right_wrist_rot_mat
                    )[None][:, right_end_frame:],
                    new_target_jpos=right_wrist_pos_contact[right_end_frame][None],
                    new_target_rot_6d=transforms.matrix_to_rotation_6d(
                        right_wrist_rot_mat_contact[right_end_frame][None]
                    ),
                    reversed=True,
                )
            )
            new_right_wrist_pos.append(right_wrist_pos_post_contact[0])
            new_right_wrist_rot_6d.append(right_wrist_rot_6d_post_contact[0])

        new_right_wrist_pos = torch.cat((new_right_wrist_pos), dim=0)  # T X 3
        new_right_wrist_rot_6d = torch.cat((new_right_wrist_rot_6d), dim=0)  # T X 6

        target_human_jnts[:, 21] = new_right_wrist_pos
    else:
        new_right_wrist_rot_6d = transforms.matrix_to_rotation_6d(right_wrist_rot_mat)

    if left_contact:
        left_wrist_pos_contact = (
            obj_com_pos + (obj_rot_mat @ left_wrist_pos_in_obj.reshape(3, 1))[..., 0]
        )  # T X 3
        left_wrist_rot_mat_contact = (
            obj_rot_mat @ left_wrist_rot_mat_in_obj
        )  # T X 3 X 3

        new_left_wrist_pos = []
        new_left_wrist_rot_6d = []

        if left_begin_frame > 0:
            left_wrist_pos_prev_contact, left_wrist_rot_6d_pre_contact = (
                apply_linear_offset(
                    original_jpos=left_wrist_pos[None][:, :left_begin_frame],
                    original_rot_6d=transforms.matrix_to_rotation_6d(
                        left_wrist_rot_mat
                    )[None][:, :left_begin_frame],
                    new_target_jpos=left_wrist_pos_contact[left_begin_frame - 1][None],
                    new_target_rot_6d=transforms.matrix_to_rotation_6d(
                        left_wrist_rot_mat_contact[left_begin_frame - 1][None]
                    ),
                )
            )
            new_left_wrist_pos.append(left_wrist_pos_prev_contact[0])
            new_left_wrist_rot_6d.append(left_wrist_rot_6d_pre_contact[0])

        new_left_wrist_pos.append(
            left_wrist_pos_contact[left_begin_frame:left_end_frame]
        )
        new_left_wrist_rot_6d.append(
            transforms.matrix_to_rotation_6d(
                left_wrist_rot_mat_contact[left_begin_frame:left_end_frame]
            )
        )

        if left_end_frame < left_wrist_pos.shape[0]:
            left_wrist_pos_post_contact, left_wrist_rot_6d_post_contact = (
                apply_linear_offset(
                    original_jpos=left_wrist_pos[None][:, left_end_frame:],
                    original_rot_6d=transforms.matrix_to_rotation_6d(
                        left_wrist_rot_mat
                    )[None][:, left_end_frame:],
                    new_target_jpos=left_wrist_pos_contact[left_end_frame][None],
                    new_target_rot_6d=transforms.matrix_to_rotation_6d(
                        left_wrist_rot_mat_contact[left_end_frame][None]
                    ),
                    reversed=True,
                )
            )
            new_left_wrist_pos.append(left_wrist_pos_post_contact[0])
            new_left_wrist_rot_6d.append(left_wrist_rot_6d_post_contact[0])

        new_left_wrist_pos = torch.cat((new_left_wrist_pos), dim=0)  # T X 3
        new_left_wrist_rot_6d = torch.cat((new_left_wrist_rot_6d), dim=0)  # T X 6

        target_human_jnts[:, 20] = new_left_wrist_pos
    else:
        new_left_wrist_rot_6d = transforms.matrix_to_rotation_6d(left_wrist_rot_mat)

    # Fix feet penetration and sliding.
    if fix_feet_floor_penetration or fix_feet_sliding:
        target_human_jnts = fix_feet(
            pred_feet_contact=pred_feet_contact,
            fix_feet_floor_penetration=fix_feet_floor_penetration,
            fix_feet_sliding=fix_feet_sliding,
            target_human_jnts=target_human_jnts,
        )

    return (
        all_res_list.detach(),
        target_human_jnts.detach(),
        new_right_wrist_rot_6d.detach(),
        new_left_wrist_rot_6d.detach(),
    )


def call_grasp_model_long_seq(
    all_res_list: torch.Tensor,
    ref_obj_rot_mat: torch.Tensor,
    left_contact: bool,
    right_contact: bool,
    left_wrist_pos: torch.Tensor,
    right_wrist_pos: torch.Tensor,
    left_wrist_rot_mat: torch.Tensor,
    right_wrist_rot_mat: torch.Tensor,
    finger_trainer: FingerTrainer,
    interaction_trainer: InteractionTrainer,
    object_name: str,
    left_begin_frame: int,
    left_end_frame: int,
    right_begin_frame: int,
    right_end_frame: int,
    rest_left_hand_local_rot_6d: torch.Tensor,
    rest_right_hand_local_rot_6d: torch.Tensor,
    window_size: int = 30,
    left_palm_pos: Optional[torch.Tensor] = None,
    right_palm_pos: Optional[torch.Tensor] = None,
    left_finger_local_rot_6d: Optional[torch.Tensor] = None,
    right_finger_local_rot_6d: Optional[torch.Tensor] = None,
):
    """
    Generate reaching/leaving motion.

    # TODO: if with wrist traj input, need to implement the cano/de-cano step.
    # NOTE: current model only takes sensor as input, which has nothing to do with the global orientation, so can ignore the cano/de-cano step.

    # NOTE: Now we only support BS = 1.
    Args:
        all_res_list: BS X T X (12+24*3+22*6)
        ref_obj_rot_mat: BS X 1 X 3 X 3
        left_wrist_pos: T X 3
        right_wrist_pos: T X 3
        left_wrist_rot_mat: T X 3 X 3
        right_wrist_rot_mat: T X 3 X 3
        left_finger_local_rot_6d: 90
        right_finger_local_rot_6d: 90
    Returns:
        finger_all_res_list: T X (30*6)
    """

    if left_contact and left_finger_local_rot_6d is None:
        raise ValueError("left_finger_local_rot_6d is None.")
    if right_contact and right_finger_local_rot_6d is None:
        raise ValueError("right_finger_local_rot_6d is None.")

    assert all_res_list.shape[0] == 1
    num_seq = all_res_list.shape[0]
    T = all_res_list.shape[1]

    #################################################### calculate sensor value ####################################################
    obj_rest_verts, obj_mesh_faces = (
        interaction_trainer.ds.load_rest_pose_object_geometry(object_name)
    )
    obj_rest_verts = torch.from_numpy(obj_rest_verts)

    reference_obj_rot_mat = ref_obj_rot_mat.repeat(
        1, all_res_list.shape[1], 1, 1
    )  # N X T X 3 X 3
    pred_obj_rel_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
        num_seq, -1, 3, 3
    )  # N X T X 3 X 3
    pred_obj_rot_mat = interaction_trainer.ds.rel_rot_to_seq(
        pred_obj_rel_rot_mat, reference_obj_rot_mat
    )

    curr_obj_rot_mat = pred_obj_rot_mat[0]  # T X 3 X 3
    curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
    curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat)  # T X 3 X 3

    pred_normalized_obj_trans = all_res_list[:, :, :3]  # N X T X 3
    pred_seq_com_pos = interaction_trainer.ds.de_normalize_obj_pos_min_max(
        pred_normalized_obj_trans
    )[0]  # T X 3

    obj_mesh_verts = interaction_trainer.ds.load_object_geometry_w_rest_geo(
        curr_obj_rot_mat,
        pred_seq_com_pos,
        obj_rest_verts.float().to(curr_obj_rot_mat.device),
    )

    left_wrist_pos = left_wrist_pos  # T X 3
    right_wrist_pos = right_wrist_pos  # T X 3
    left_wrist_rot_mat = left_wrist_rot_mat  # T X 3 X 3
    right_wrist_rot_mat = right_wrist_rot_mat  # T X 3 X 3

    proximity_sensor = finger_trainer.ds.compute_proximity_sensor(
        obj_mesh_verts,
        left_wrist_pos,
        right_wrist_pos,
        left_wrist_rot_mat,
        right_wrist_rot_mat,
    )  # T X 200
    proximity_sensor = finger_trainer.ds.normalize_clip_sensor(proximity_sensor)
    mirror_proximity_sensor = finger_trainer.ds.compute_mirror_proximity_sensor(
        obj_mesh_verts,
        left_wrist_pos,
        right_wrist_pos,
        left_wrist_rot_mat,
        right_wrist_rot_mat,
    )  # T X 200
    mirror_proximity_sensor = finger_trainer.ds.normalize_clip_sensor(
        mirror_proximity_sensor
    )

    # NOTE: ambient sensor not used right now
    ambient_sensor = torch.zeros(T, 2048).to(proximity_sensor.device)  # T X 2048
    # if left_palm_pos is not None and right_palm_pos is not None:
    #     left_middle_finger_pos = left_palm_pos
    #     right_middle_finger_pos = right_palm_pos
    # else:
    #     left_offset = torch.Tensor([0.1142, -0.0059, -0.0040]).reshape(1, 3, 1).repeat(left_wrist_rot_mat.shape[0], 1, 1).to(left_wrist_rot_mat.device)
    #     right_offset = torch.Tensor([-0.1137, -0.0095, -0.0049]).reshape(1, 3, 1).repeat(left_wrist_rot_mat.shape[0], 1, 1).to(left_wrist_rot_mat.device)
    #     left_middle_finger_pos = left_wrist_pos + left_wrist_rot_mat.bmm(left_offset)[..., 0] # T X 3
    #     right_middle_finger_pos = right_wrist_pos + right_wrist_rot_mat.bmm(right_offset)[..., 0] # T X 3
    # ambient_sensor = finger_trainer.ds.compute_ambient_sensor(obj_mesh_verts, left_middle_finger_pos, right_middle_finger_pos, left_wrist_rot_mat, right_wrist_rot_mat) # T X 2048
    # ambient_sensor = finger_trainer.ds.normalize_clip_sensor(ambient_sensor)

    contact_label = torch.zeros(ambient_sensor.shape[0], 2).to(
        ambient_sensor.device
    )  # T X 2
    if left_contact:
        contact_label[left_begin_frame:left_end_frame, 0] = 1
    if right_contact:
        contact_label[right_begin_frame:right_end_frame, 1] = 1

    cond_mask = None
    #################################################### calculate sensor value ####################################################

    #################################################### create clip ####################################################
    ambient_sensors = []
    proximity_sensors = []
    start_finger_local_6ds = []
    end_finger_local_6ds = []
    if left_contact:
        left_reach_ambient_sensor = ambient_sensor[
            max(0, left_begin_frame - window_size + 1) : left_begin_frame + 1, 0:1024
        ]
        left_reach_proximity_sensor = proximity_sensor[
            max(0, left_begin_frame - window_size + 1) : left_begin_frame + 1, 0:100
        ]
        ambient_sensors.append(left_reach_ambient_sensor)
        proximity_sensors.append(left_reach_proximity_sensor)
        start_finger_local_6ds.append(rest_left_hand_local_rot_6d.cuda())  # 1 X 90
        end_finger_local_6ds.append(left_finger_local_rot_6d[None].cuda())  # 1 X 90

        left_leave_ambient_sensor = ambient_sensor[
            left_end_frame : min(left_end_frame + window_size, ambient_sensor.shape[0]),
            0:1024,
        ]
        left_leave_proximity_sensor = proximity_sensor[
            left_end_frame : min(left_end_frame + window_size, ambient_sensor.shape[0]),
            0:100,
        ]
        ambient_sensors.append(left_leave_ambient_sensor)
        proximity_sensors.append(left_leave_proximity_sensor)
        start_finger_local_6ds.append(left_finger_local_rot_6d[None].cuda())  # 1 X 90
        end_finger_local_6ds.append(rest_left_hand_local_rot_6d.cuda())  # 1 X 90

    if right_contact:
        right_reach_ambient_sensor = ambient_sensor[
            max(0, right_begin_frame - window_size + 1) : right_begin_frame + 1, 1024:
        ]
        right_reach_proximity_sensor = mirror_proximity_sensor[
            max(0, right_begin_frame - window_size + 1) : right_begin_frame + 1, 100:
        ]
        mirror_rest_finger_local_6ds = mirror_rot_6d(
            rest_right_hand_local_rot_6d.clone().cuda()
        )
        mirror_ref_finger_local_6ds = mirror_rot_6d(
            right_finger_local_rot_6d[None].clone().cuda()
        )
        ambient_sensors.append(right_reach_ambient_sensor)
        proximity_sensors.append(right_reach_proximity_sensor)
        start_finger_local_6ds.append(mirror_rest_finger_local_6ds)  # 1 X 90
        end_finger_local_6ds.append(mirror_ref_finger_local_6ds)  # 1 X 90

        right_leave_ambient_sensor = ambient_sensor[
            right_end_frame : min(
                right_end_frame + window_size, ambient_sensor.shape[0]
            ),
            1024:,
        ]
        right_leave_proximity_sensor = mirror_proximity_sensor[
            right_end_frame : min(
                right_end_frame + window_size, ambient_sensor.shape[0]
            ),
            100:,
        ]
        ambient_sensors.append(right_leave_ambient_sensor)
        proximity_sensors.append(right_leave_proximity_sensor)
        start_finger_local_6ds.append(mirror_ref_finger_local_6ds)  # 1 X 90
        end_finger_local_6ds.append(mirror_rest_finger_local_6ds)  # 1 X 90

    padded_ambient_sensors = []
    padded_proximity_sensors = []
    seq_lens = []
    for a_s, p_s in zip(ambient_sensors, proximity_sensors):
        if a_s.shape[0] < window_size:
            padded_a_s = torch.cat(
                (
                    a_s,
                    torch.zeros(window_size - a_s.shape[0], a_s.shape[1]).to(
                        a_s.device
                    ),
                ),
                dim=0,
            )
            padded_p_s = torch.cat(
                (
                    p_s,
                    torch.zeros(window_size - p_s.shape[0], p_s.shape[1]).to(
                        p_s.device
                    ),
                ),
                dim=0,
            )
        else:
            padded_a_s = a_s
            padded_p_s = p_s
        seq_lens.append(a_s.shape[0])
        padded_ambient_sensors.append(padded_a_s[None])  # 1 X window_size X 2048
        padded_proximity_sensors.append(padded_p_s[None])  # 1 X window_size X 100

    seq_lens = (
        torch.tensor(seq_lens).to(ambient_sensors[0].device).long()
    )  # num_windows
    padded_ambient_sensors = torch.cat(
        padded_ambient_sensors
    )  # num_windows X window_size X 1024
    padded_proximity_sensors = torch.cat(
        padded_proximity_sensors
    )  # num_windows X window_size X 100
    start_finger_local_6ds = torch.cat(start_finger_local_6ds)  # num_windows X 90
    end_finger_local_6ds = torch.cat(end_finger_local_6ds)  # num_windows X 90
    #################################################### create clip ####################################################

    #################################################### canonicalize all windows ####################################################
    #################################################### canonicalize all windows ####################################################

    #################################################### sample ####################################################
    with torch.no_grad():
        x_start = torch.randn(
            padded_ambient_sensors.shape[0], padded_ambient_sensors.shape[1], 90
        ).cuda()  # num_windows X window_size X 90
        if finger_trainer.ref_pose_condition:
            cond_mask = torch.ones_like(x_start).to(x_start.device)
            cond_mask[:, 0] = 0
            cond_mask[torch.arange(x_start.shape[0]), seq_lens - 1] = 0

            x_start[:, 0] = start_finger_local_6ds
            x_start[torch.arange(x_start.shape[0]), seq_lens - 1] = end_finger_local_6ds
        ori_data_cond = torch.cat(
            (
                torch.zeros(
                    padded_ambient_sensors.shape[0], padded_ambient_sensors.shape[1], 9
                ).to(padded_ambient_sensors.device),  # wrist pos & rot, all 0 now
                padded_ambient_sensors,
                padded_proximity_sensors,
            ),
            dim=-1,
        )

        # Generate padding mask
        actual_seq_len = (
            seq_lens + 1
        )  # BS, + 1 since we need additional timestep for noise level
        tmp_mask = torch.arange(window_size + 1).to(actual_seq_len.device).expand(
            padded_ambient_sensors.shape[0], window_size + 1
        ) < actual_seq_len[:, None].repeat(1, window_size + 1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(padded_ambient_sensors.device)
        grasp_all_res_list = finger_trainer.ema.ema_model.sample(
            x_start, ori_data_cond, cond_mask=cond_mask, padding_mask=padding_mask
        )  # num_windows X window_size X (15 * 6)
    #################################################### sample ####################################################

    #################################################### de-canonicalize all windows ####################################################
    #################################################### de-canonicalize all windows ####################################################

    #################################################### stitch all windows ####################################################
    finger_all_res_list = torch.zeros(T, 30 * 6).to(all_res_list.device)
    if left_contact:
        finger_all_res_list[..., 0:90] = left_finger_local_rot_6d.to(
            all_res_list.device
        )
    else:
        finger_all_res_list[..., 0:90] = rest_left_hand_local_rot_6d.to(
            all_res_list.device
        )
    if right_contact:
        finger_all_res_list[..., 90:180] = right_finger_local_rot_6d.to(
            all_res_list.device
        )
    else:
        finger_all_res_list[..., 90:180] = rest_right_hand_local_rot_6d.to(
            all_res_list.device
        )

    idx = 0
    if left_contact:
        if left_begin_frame > 0:
            finger_all_res_list[
                max(0, left_begin_frame - window_size + 1) : left_begin_frame + 1, 0:90
            ] = grasp_all_res_list[idx, : seq_lens[idx]].clone()
            finger_all_res_list[: max(0, left_begin_frame - window_size + 1), 0:90] = (
                grasp_all_res_list[idx, 0:1].clone()
            )
            prev_rot_6d = finger_all_res_list[: left_begin_frame + 1, 0:90].reshape(
                1, -1, 15, 6
            )
            window_rot_6d = finger_all_res_list[left_begin_frame + 1 :, 0:90].reshape(
                1, -1, 15, 6
            )
            _, new_rot_6d, _, new_prev_rot_6d = apply_inertialize(
                prev_jpos=None,
                prev_rot_6d=prev_rot_6d,
                window_jpos=None,
                window_rot_6d=window_rot_6d,
                ratio=0.0,
                prev_blend_time=0.2,
                window_blend_time=0.2,
            )
            finger_all_res_list[: left_begin_frame + 1, 0:90] = new_prev_rot_6d.reshape(
                -1, 90
            )
        idx += 1

        if left_end_frame < T - 1:
            finger_all_res_list[
                left_end_frame : min(
                    left_end_frame + window_size, finger_all_res_list.shape[0]
                ),
                0:90,
            ] = grasp_all_res_list[idx, : seq_lens[idx]].clone()
            finger_all_res_list[
                min(left_end_frame + window_size, finger_all_res_list.shape[0]) :, 0:90
            ] = grasp_all_res_list[idx, -1:].clone()
            prev_rot_6d = finger_all_res_list[:left_end_frame, 0:90].reshape(
                1, -1, 15, 6
            )
            window_rot_6d = finger_all_res_list[left_end_frame:, 0:90].reshape(
                1, -1, 15, 6
            )
            _, new_rot_6d, _, new_prev_rot_6d = apply_inertialize(
                prev_jpos=None,
                prev_rot_6d=prev_rot_6d,
                window_jpos=None,
                window_rot_6d=window_rot_6d,
                ratio=1.0,
                prev_blend_time=0.2,
                window_blend_time=0.2,
            )
            finger_all_res_list[left_end_frame:, 0:90] = new_rot_6d.reshape(-1, 90)
        idx += 1
    if right_contact:
        if right_begin_frame > 0:
            grasp_motion = mirror_rot_6d(
                grasp_all_res_list[idx, : seq_lens[idx]].clone()
            )  # seq_lens[idx] X 90
            finger_all_res_list[
                max(0, right_begin_frame - window_size + 1) : right_begin_frame + 1,
                90:180,
            ] = grasp_motion
            finger_all_res_list[
                : max(0, right_begin_frame - window_size + 1), 90:180
            ] = grasp_motion[0:1]
            prev_rot_6d = finger_all_res_list[: right_begin_frame + 1, 90:180].reshape(
                1, -1, 15, 6
            )
            window_rot_6d = finger_all_res_list[
                right_begin_frame + 1 :, 90:180
            ].reshape(1, -1, 15, 6)
            _, new_rot_6d, _, new_prev_rot_6d = apply_inertialize(
                prev_jpos=None,
                prev_rot_6d=prev_rot_6d,
                window_jpos=None,
                window_rot_6d=window_rot_6d,
                ratio=0.0,
                prev_blend_time=0.2,
                window_blend_time=0.2,
            )
            finger_all_res_list[: right_begin_frame + 1, 90:180] = (
                new_prev_rot_6d.reshape(-1, 90)
            )
        idx += 1

        if right_end_frame < T - 1:
            grasp_motion = mirror_rot_6d(
                grasp_all_res_list[idx, : seq_lens[idx]].clone()
            )  # seq_lens[idx] X 90
            finger_all_res_list[
                right_end_frame : min(
                    right_end_frame + window_size, finger_all_res_list.shape[0]
                ),
                90:180,
            ] = grasp_motion
            finger_all_res_list[
                min(right_end_frame + window_size, finger_all_res_list.shape[0]) :,
                90:180,
            ] = grasp_motion[-1:]
            prev_rot_6d = finger_all_res_list[:right_end_frame, 90:180].reshape(
                1, -1, 15, 6
            )
            window_rot_6d = finger_all_res_list[right_end_frame:, 90:180].reshape(
                1, -1, 15, 6
            )
            _, new_rot_6d, _, new_prev_rot_6d = apply_inertialize(
                prev_jpos=None,
                prev_rot_6d=prev_rot_6d,
                window_jpos=None,
                window_rot_6d=window_rot_6d,
                ratio=1.0,
                prev_blend_time=0.2,
                window_blend_time=0.2,
            )
            finger_all_res_list[right_end_frame:, 90:180] = new_rot_6d.reshape(-1, 90)
        idx += 1
    #################################################### stitch all windows ####################################################

    return finger_all_res_list.detach()


def generate_navigation_motion(
    interaction_trainer: InteractionTrainer,
    navigation_trainer: NavigationTrainer,
    ref_data_dict: Dict[str, torch.Tensor],
    prev_interaction_motion: Optional[torch.Tensor],
    prev_interaction_end_human_pose: Optional[torch.Tensor],
    trans2joint: torch.Tensor,
    rest_human_offsets: torch.Tensor,
    planned_obj_path: torch.Tensor,
    text_list: List[List[str]],
    p_idx: int,
    o_idx: int,
    overlap_frame_num_navi: int,
    step_dis: float = 0.9,
    use_cut_step: bool = True,
    fix_feet_floor_penetration: bool = True,
    fix_feet_sliding: bool = False,
    add_finger_motion: bool = True,
    rest_left_hand_local_rot_6d: Optional[torch.Tensor] = None,
    rest_right_hand_local_rot_6d: Optional[torch.Tensor] = None,
    prev_interaction_finger_motion: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    # First navigation starts from T_start_pose.
    if prev_interaction_end_human_pose is None:
        prev_interaction_end_human_pose = generate_T_pose(
            rest_human_offsets=rest_human_offsets,
            planned_obj_path=planned_obj_path,
        )

    # Sample navigation motion.
    all_res_list, whole_cond_mask, seq_human_root_pos, pred_feet_contact = (
        navigation_trainer.call_navigation_model_long_seq(
            trans2joint=trans2joint,
            rest_human_offsets=rest_human_offsets,
            prev_interaction_end_human_pose=prev_interaction_end_human_pose,
            planned_obj_path=planned_obj_path,
            text_list=text_list,
            p_idx=p_idx,
            o_idx=o_idx,
            overlap_frame_num=overlap_frame_num_navi,
            step_dis=step_dis,
            use_cut_step=use_cut_step,
        )
    )

    # Empirically, when both feet are in contact with the ground and have zero velocity,
    # it is easier to achieve a smooth transition between navigation and interaction.
    # To enforce this condition, we trim the last few frames of the navigation motion.
    left_feet_contact = medfilt(
        pred_feet_contact[0, :, 0].detach().cpu().numpy(), kernel_size=3
    )  # T
    right_feet_contact = medfilt(
        pred_feet_contact[0, :, 1].detach().cpu().numpy(), kernel_size=3
    )  # T
    left_feet_contact = left_feet_contact > 0.95
    right_feet_contact = right_feet_contact > 0.95
    flag = False
    cut_step = len(left_feet_contact) - 1
    for cut_step in reversed(range(len(left_feet_contact))):
        if flag and (
            (left_feet_contact[cut_step] == right_feet_contact[cut_step])
            or (left_feet_contact[cut_step] != left_feet_contact[cut_step + 1])
        ):
            break
        if left_feet_contact[cut_step] != right_feet_contact[cut_step]:
            flag = True
            continue

    all_res_list = all_res_list[:, :cut_step]
    pred_feet_contact = pred_feet_contact[:, :cut_step]
    # print("Navi cut_step: ", cut_step)

    # Fix feet penetration and sliding.
    if fix_feet_floor_penetration or fix_feet_sliding:
        target_human_jnts, human_root_trans, human_local_rot_aa_reps = (
            navigation_trainer.calc_jpos_from_navi_res(all_res_list, ref_data_dict)
        )  # T X 24 X 3, T X 3, T X 22 X 3
        target_human_jnts = fix_feet(
            pred_feet_contact=pred_feet_contact,
            fix_feet_floor_penetration=fix_feet_floor_penetration,
            fix_feet_sliding=fix_feet_sliding,
            target_human_jnts=target_human_jnts,
        )
        # Run IK.
        global_6d, global_pos = smplx_ik(
            target_human_jnts=target_human_jnts,
            human_root_trans=human_root_trans,
            human_local_rot_aa_reps=human_local_rot_aa_reps,
            betas=ref_data_dict["betas"][0],
            rest_human_offsets=rest_human_offsets,
            right_wrist=False,
            left_wrist=False,
            feet=(fix_feet_floor_penetration or fix_feet_sliding),
            gender=ref_data_dict["gender"][0],
        )
        all_res_list[0, :, 24 * 3 : 24 * 3 + 22 * 6] = global_6d.reshape(-1, 22 * 6)

    # Smooth the transition between navigation and previous interaction.
    if prev_interaction_motion is not None:
        all_res_list = interaction_to_navigation_smooth_transition(
            prev_interaction_motion,
            all_res_list,
            interaction_trainer,
            navigation_trainer,
        )

    # Add finger motion.
    if add_finger_motion:
        finger_all_res_list = torch.cat(
            (rest_left_hand_local_rot_6d, rest_right_hand_local_rot_6d), dim=1
        ).repeat(all_res_list.shape[1], 1)  # T X 180
        if prev_interaction_finger_motion is not None:
            finger_all_res_list = finger_smooth_transition(
                prev_interaction_finger_motion, finger_all_res_list
            )

    prev_navigation_motion = all_res_list.clone()
    prev_navigation_end_human_pose = all_res_list[:, -1:, :].clone()  # BS X 1 X D
    prev_navigation_finger_motion = (
        finger_all_res_list.clone() if finger_all_res_list is not None else None
    )

    # Recover the human jpos to its unnormalized version, since navigation and interaction models
    # have different min, max value.
    prev_navigation_end_human_pose[:, :, : 24 * 3] = (
        navigation_trainer.ds.de_normalize_jpos_min_max(
            prev_navigation_end_human_pose[:, :, : 24 * 3].reshape(-1, 1, 24, 3)
        ).reshape(-1, 1, 24 * 3)
    )

    return (
        all_res_list,
        finger_all_res_list,
        prev_navigation_motion,
        prev_navigation_end_human_pose,
        prev_navigation_finger_motion,
        whole_cond_mask,
    )


def generate_interaction_motion(
    coarse_interaction_trainer: InteractionTrainer,
    fine_interaction_trainer: InteractionTrainer,
    navigation_trainer: Optional[NavigationTrainer],
    finger_trainer: Optional[FingerTrainer],
    curr_object_name: str,
    action_name: str,
    text: str,
    all_object_data_dict: Dict[str, Any],
    ref_data_dict: Dict,
    planned_obj_path: torch.Tensor,
    path_data: List[np.ndarray],
    obj_initial_rot_mat: torch.Tensor,
    obj_end_rot_mat: torch.Tensor,
    overlap_frame_num: int,
    prev_navigation_motion: Optional[torch.Tensor],
    prev_navigation_end_human_pose: Optional[torch.Tensor],
    prev_navigation_finger_motion: Optional[torch.Tensor],
    rest_human_offsets: torch.Tensor,
    trans2joint: torch.Tensor,
    table_height: float,
    rest_left_hand_local_rot_6d: torch.Tensor,
    rest_right_hand_local_rot_6d: torch.Tensor,
    add_finger_motion: bool = True,
    add_interaction_root_xy_ori: bool = True,
    add_interaction_feet_contact: bool = True,
    fix_feet_floor_penetration: bool = True,
    fix_feet_sliding: bool = False,
    smooth_whole_traj: bool = False,
    no_force_closure: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Generate interaction motion.

    Raises:
        ValueError: if prev_navigation_motion is None but navigation_trainer is not None.

    Returns:
        all_res_list: the final human and object motion.
        finger_all_res_list: the generated finger motion.
        coarse_all_res_list: the generated motion from CoarseNet.
        fine_all_res_list: the generated motion from RefineNet, without post-processing.
        prev_interaction_motion: the generated human motion.
        prev_interaction_end_human_pose: the end human pose.
        prev_interaction_finger_motion: the generated finger motion.
        contact_begin_frame: the frame when the contact starts.
        contact_end_frame: the frame when the contact ends.
    """

    if navigation_trainer is None and prev_navigation_motion is not None:
        raise ValueError(
            "prev_navigation_motion is not None but navigation_trainer is None."
        )

    # Stage 1: Run CoarseNet.
    (
        coarse_all_res_list,
        _,
        (
            left_contact,
            right_contact,
            left_begin_frame,
            left_end_frame,
            right_begin_frame,
            right_end_frame,
            left_wrist_pose,
            right_wrist_pose,
            contact_begin_frame,
            contact_end_frame,
        ),
        *_,
    ) = run_interaction_trainer(
        trainer=coarse_interaction_trainer,
        object_data_dict=all_object_data_dict[curr_object_name],
        ref_data_dict=ref_data_dict,
        planned_obj_path=planned_obj_path,
        text=text,
        obj_initial_rot_mat=obj_initial_rot_mat,
        overlap_frame_num=overlap_frame_num,
        prev_navigation_end_human_pose=prev_navigation_end_human_pose,
        rest_human_offsets=rest_human_offsets,
        curr_object_name=curr_object_name,
        curr_action_name=action_name,
        table_height=table_height,
        trans2joint=trans2joint,
        obj_end_rot_mat=obj_end_rot_mat,
    )

    # Stage 2: Grasp generation.
    (
        right_wrist_pos_in_obj,
        right_wrist_rot_mat_in_obj,
        right_finger_local_rot_6d,
        right_wrist_pose,
        left_wrist_pos_in_obj,
        left_wrist_rot_mat_in_obj,
        left_finger_local_rot_6d,
        left_wrist_pose,
    ) = run_grasp_generation(
        curr_object_name=curr_object_name,
        left_contact=left_contact,
        right_contact=right_contact,
        left_wrist_init_pose=left_wrist_pose,
        right_wrist_init_pose=right_wrist_pose,
        no_force_closure=no_force_closure,
        # grasp_iter=1,  # FIXME:
    )

    # Stage 3: Run RefineNet.
    ## Build relative wrist conditions for RefineNet.
    available_conditions_wrist_relative = build_wrist_relative_conditions(
        object_data_dict=all_object_data_dict[curr_object_name],
        left_wrist_pose=left_wrist_pose,
        right_wrist_pose=right_wrist_pose,
        trainer=fine_interaction_trainer,
        left_contact=left_contact,
        right_contact=right_contact,
        left_begin_frame=left_begin_frame,
        left_end_frame=left_end_frame,
        right_begin_frame=right_begin_frame,
        right_end_frame=right_end_frame,
        contact_begin_frame=contact_begin_frame,
        contact_end_frame=contact_end_frame,
    )

    ## Generate interaction motion.
    (
        all_res_list,
        pred_feet_contact,
        (
            left_contact,
            right_contact,
            left_begin_frame,
            left_end_frame,
            right_begin_frame,
            right_end_frame,
            left_wrist_pose,
            right_wrist_pose,
            contact_begin_frame,
            contact_end_frame,
        ),
        (obj_com_pos, obj_rot_mat),
        (
            human_jnts,
            human_root_trans,
            human_local_rot_aa_reps,
            human_jnts_rot_mat_global,
            left_wrist_pos,
            left_wrist_rot_mat,
            right_wrist_pos,
            right_wrist_rot_mat,
        ),
    ) = run_interaction_trainer(
        trainer=fine_interaction_trainer,
        object_data_dict=all_object_data_dict[curr_object_name],
        ref_data_dict=ref_data_dict,
        planned_obj_path=planned_obj_path,
        text=text,
        obj_initial_rot_mat=obj_initial_rot_mat,
        overlap_frame_num=overlap_frame_num,
        prev_navigation_end_human_pose=prev_navigation_end_human_pose,
        rest_human_offsets=rest_human_offsets,
        trans2joint=trans2joint,
        curr_object_name=curr_object_name,
        curr_action_name=action_name,
        table_height=table_height,
        obj_end_rot_mat=obj_end_rot_mat,
        available_conditions_wrist_relative=available_conditions_wrist_relative,
        add_root_ori=add_interaction_root_xy_ori,
        add_feet_contact=add_interaction_feet_contact,
    )
    fine_all_res_list = all_res_list.clone()

    # Cut the static phase at the end of the motion.
    cut_frame = find_static_frame_at_end(human_jnts, contact_end_frame)

    all_res_list = all_res_list[:, : cut_frame + 1]
    coarse_all_res_list = coarse_all_res_list[:, : cut_frame + 1]
    fine_all_res_list = fine_all_res_list[:, : cut_frame + 1]
    obj_com_pos = obj_com_pos[: cut_frame + 1]
    obj_rot_mat = obj_rot_mat[: cut_frame + 1]
    human_jnts = human_jnts[: cut_frame + 1]
    human_root_trans = human_root_trans[: cut_frame + 1]
    human_local_rot_aa_reps = human_local_rot_aa_reps[: cut_frame + 1]
    human_jnts_rot_mat_global = human_jnts_rot_mat_global[: cut_frame + 1]
    left_wrist_pos = left_wrist_pos[: cut_frame + 1]
    left_wrist_rot_mat = left_wrist_rot_mat[: cut_frame + 1]
    right_wrist_pos = right_wrist_pos[: cut_frame + 1]
    right_wrist_rot_mat = right_wrist_rot_mat[: cut_frame + 1]
    pred_feet_contact = pred_feet_contact[:, : cut_frame + 1]
    # print(
    #     "Original frame is: {}, cut at frame: {}".format(
    #         human_jnts.shape[0] - 1, cut_frame
    #     )
    # )

    # Post processing.
    all_res_list, target_human_jnts, new_right_wrist_rot_6d, new_left_wrist_rot_6d = (
        post_process(
            object_data_dict=all_object_data_dict[curr_object_name],
            interaction_trainer=fine_interaction_trainer,
            path_data=path_data,
            obj_initial_rot_mat=obj_initial_rot_mat,
            obj_end_rot_mat=obj_end_rot_mat,
            obj_com_pos=obj_com_pos,
            obj_rot_mat=obj_rot_mat,
            all_res_list=all_res_list,
            contact_begin_frame=contact_begin_frame,
            contact_end_frame=contact_end_frame,
            human_jnts=human_jnts,
            left_wrist_pos=left_wrist_pos,
            left_wrist_rot_mat=left_wrist_rot_mat,
            left_wrist_pos_in_obj=left_wrist_pos_in_obj,
            left_wrist_rot_mat_in_obj=left_wrist_rot_mat_in_obj,
            right_wrist_pos=right_wrist_pos,
            right_wrist_rot_mat=right_wrist_rot_mat,
            right_wrist_pos_in_obj=right_wrist_pos_in_obj,
            right_wrist_rot_mat_in_obj=right_wrist_rot_mat_in_obj,
            left_contact=left_contact,
            right_contact=right_contact,
            left_begin_frame=left_begin_frame,
            left_end_frame=left_end_frame,
            right_begin_frame=right_begin_frame,
            right_end_frame=right_end_frame,
            pred_feet_contact=pred_feet_contact,
            fix_feet_floor_penetration=fix_feet_floor_penetration,
            fix_feet_sliding=fix_feet_sliding,
        )
    )
    # Run IK.
    if left_contact or right_contact or fix_feet_floor_penetration or fix_feet_sliding:
        global_6d, global_pos = smplx_ik(
            target_human_jnts=target_human_jnts,
            human_root_trans=human_root_trans,
            human_local_rot_aa_reps=human_local_rot_aa_reps,
            betas=ref_data_dict["betas"][0],
            rest_human_offsets=rest_human_offsets,
            right_wrist=right_contact,
            left_wrist=left_contact,
            feet=(fix_feet_floor_penetration or fix_feet_sliding),
            gender=ref_data_dict["gender"][0],
        )
        all_res_list[0, :, 12 + 24 * 3 : 12 + 24 * 3 + 22 * 6] = global_6d.reshape(
            -1, 22 * 6
        )
    all_res_list[0, :, 12 + 24 * 3 + 20 * 6 : 12 + 24 * 3 + 21 * 6] = (
        new_left_wrist_rot_6d
    )
    all_res_list[0, :, 12 + 24 * 3 + 21 * 6 : 12 + 24 * 3 + 22 * 6] = (
        new_right_wrist_rot_6d
    )

    if smooth_whole_traj:
        all_res_list = smooth_res(all_res_list)

    # Stage 4: Run FingerNet.
    # If the object is not in contact with the hand, use the default rest finger pose.
    if add_finger_motion and (left_contact or right_contact):
        finger_all_res_list = call_grasp_model_long_seq(
            all_res_list,
            ref_obj_rot_mat=all_object_data_dict[curr_object_name][
                "reference_obj_rot_mat"
            ],
            left_contact=left_contact,
            right_contact=right_contact,
            left_wrist_pos=global_pos[:, 20],
            right_wrist_pos=global_pos[:, 21],
            left_wrist_rot_mat=transforms.rotation_6d_to_matrix(new_left_wrist_rot_6d),
            right_wrist_rot_mat=transforms.rotation_6d_to_matrix(
                new_right_wrist_rot_6d
            ),
            rest_left_hand_local_rot_6d=rest_left_hand_local_rot_6d,
            rest_right_hand_local_rot_6d=rest_right_hand_local_rot_6d,
            finger_trainer=finger_trainer,
            interaction_trainer=fine_interaction_trainer,
            object_name=curr_object_name,
            left_begin_frame=left_begin_frame,
            left_end_frame=left_end_frame,
            right_begin_frame=right_begin_frame,
            right_end_frame=right_end_frame,
            left_finger_local_rot_6d=left_finger_local_rot_6d,
            right_finger_local_rot_6d=right_finger_local_rot_6d,
        )

    # Add small static phase when performing grasping to make the motion more natural.
    all_res_list, coarse_all_res_list, fine_all_res_list, finger_all_res_list = (
        add_static_phase(
            all_res_list=all_res_list,
            coarse_all_res_list=coarse_all_res_list,
            fine_all_res_list=fine_all_res_list,
            contact_begin_frame=contact_begin_frame,
            contact_end_frame=contact_end_frame,
            finger_all_res_list=finger_all_res_list,
        )
    )

    # Smooth the transition between interaction and previous navigation.
    if prev_navigation_motion is not None:
        all_res_list = navigation_to_interaction_smooth_transition(
            prev_navigation_motion,
            all_res_list,
            fine_interaction_trainer,
            navigation_trainer,
        )

        if (
            finger_all_res_list is not None
            and prev_navigation_finger_motion is not None
        ):
            finger_all_res_list = finger_smooth_transition(
                prev_navigation_finger_motion, finger_all_res_list
            )

    prev_interaction_motion = all_res_list.clone()
    prev_interaction_end_human_pose = all_res_list[
        :, -1:, 12 : 12 + 24 * 3 + 22 * 6
    ].clone()  # BS X 1 X D (24*3+22*6)
    prev_interaction_finger_motion = (
        finger_all_res_list.clone() if finger_all_res_list is not None else None
    )

    # Recover the human jpos to its unnormalized version, since navigation and interaction models
    # have different min, max value.
    prev_interaction_end_human_pose[:, :, : 24 * 3] = (
        fine_interaction_trainer.ds.de_normalize_jpos_min_max(
            prev_interaction_end_human_pose[:, :, : 24 * 3].reshape(-1, 1, 24, 3)
        ).reshape(-1, 1, 24 * 3)
    )
    return (
        all_res_list,
        finger_all_res_list,
        coarse_all_res_list,
        fine_all_res_list,
        prev_interaction_motion,
        prev_interaction_end_human_pose,
        prev_interaction_finger_motion,
        contact_begin_frame,
        contact_end_frame,
    )


def cond_sample_res_w_long_planned_path_for_multi_objects(
    opt,
    device,
    object_list: List[str] = ["largebox"],
    sub_num: int = 4,
    coarse_milestone: str = "10",
    fine_milestone: str = "10",
    navi_milestone: str = "10",
    overlap_frame_num: int = 60,
    overlap_frame_num_navi: int = 90,
    smooth_whole_traj: bool = False,
    VISUALIZE: bool = True,
    fix_feet_floor_penetration: bool = True,
    fix_feet_sliding: bool = False,
):
    #################################################### load model ####################################################
    navigation_trainer = build_navi_trainer(opt, device, navi_milestone)
    interaction_trainer = build_coarse_interaction_trainer(
        opt,
        device=device,
        milestone=coarse_milestone,
    )
    fine_interaction_trainer = build_fine_interaction_trainer(
        opt,
        device=device,
        milestone=fine_milestone,
    )
    finger_trainer = None
    if opt.add_finger_motion:
        finger_opt = build_finger_opt(opt)
        finger_trainer = build_finger_trainer(finger_opt, device)
    #################################################### load model ####################################################

    #################################################### load data ####################################################
    # Load rest hand pose
    rest_hand_pose = pickle.load(open("rest_hand_pose.pkl", "rb"))
    rest_left_hand_pose = rest_hand_pose["left_hand_pose"].cuda()  # 45
    rest_right_hand_pose = rest_hand_pose["right_hand_pose"].cuda()  # 45
    rest_left_hand_local_rot_6d = transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rest_left_hand_pose.reshape(-1, 3) * 0.8)
    ).reshape(1, 15 * 6)
    rest_right_hand_local_rot_6d = transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rest_right_hand_pose.reshape(-1, 3) * 0.8)
    ).reshape(1, 15 * 6)

    # Load all_object_data_dict from pickle file
    all_object_data_dict_data = pickle.load(
        open("all_object_data_dict_for_eval.pkl", "rb")
    )
    all_object_data_dict = all_object_data_dict_data["all_object_data_dict"]
    ref_data_dict = all_object_data_dict_data["ref_data_dict"]

    rest_human_offsets = ref_data_dict["rest_human_offsets"]
    trans2joint = ref_data_dict["trans2joint"]

    for key, val in all_object_data_dict.items():
        new_dict = {}
        new_dict["reference_obj_rot_mat"] = val["reference_obj_rot_mat"]
        new_dict["obj_name"] = val["obj_name"]
        new_dict["input_obj_bps"] = val["input_obj_bps"]

        new_dict["trans2joint"] = ref_data_dict[
            "trans2joint"
        ]  # use ref_dict's human skeleton. these should only be used in interaction guidance for now.
        new_dict["betas"] = ref_data_dict["betas"]
        new_dict["gender"] = ref_data_dict["gender"]

        all_object_data_dict[key] = new_dict

    # FIXME: fix this new object list
    if False:
        new_object_list = [
            "bottle",
            "floorlamp1",
            "right_shoes1",
            "toy2",
            "laundrybasket",
            "vase1",
            "vase1_big",
            "vase1_big2",
            "toy1",
            "laundrybasket",
            "newobject",
            "box_a",
            "box_b",
            "box_c",
        ]
        for object_name in new_object_list:
            new_dict = {}
            obj_rot_mat = torch.eye(3).reshape(1, 1, 3, 3)
            obj_com_pos = torch.zeros(1, 1, 3)
            new_dict["reference_obj_rot_mat"] = obj_rot_mat
            new_dict["obj_name"] = [object_name]
            mesh_dir = "/move/u/zhenwu/grasp_repo/DexGraspNet/data/raw_omomo_new/bps"
            if os.path.exists(os.path.join(mesh_dir, object_name + ".pkl")):
                bps_data = pickle.load(
                    open(os.path.join(mesh_dir, object_name + ".pkl"), "rb")
                )  # 1 X 1 X 1024 X 3
                new_dict["input_obj_bps"] = bps_data
            else:
                curr_obj_bps, _ = self.get_object_geometry_bps(
                    interaction_trainer.ds, [object_name], obj_rot_mat, obj_com_pos
                )
                new_dict["input_obj_bps"] = curr_obj_bps.cpu()
                pickle.dump(
                    new_dict["input_obj_bps"],
                    open(os.path.join(mesh_dir, object_name + ".pkl"), "wb"),
                )

            new_dict["trans2joint"] = ref_data_dict[
                "trans2joint"
            ]  # use ref_dict's human skeleton. these should only be used in interaction guidance for now.
            new_dict["betas"] = ref_data_dict["betas"]
            new_dict["gender"] = ref_data_dict["gender"]

            all_object_data_dict[object_name] = new_dict
    #################################################### load data ####################################################

    #################################################### generate long path ####################################################
    (
        obj_initial_rot_mat_list,
        obj_end_rot_mat_list,
        path_data_list,
        object_names_list,
        action_names_list,
        text_list,
        table_height_list,
    ) = simple_random_setting(object_list=object_list, sub_num=sub_num)
    num_planned_path = len(path_data_list)
    #################################################### generate long path ####################################################

    prev_interaction_motion_last = None
    prev_interaction_end_human_pose_last = None
    prev_interaction_finger_motion_last = None

    for p_idx in range(num_planned_path):
        # In each planned path, the sequencee consists of multiple interactions and navigations.
        num_sub_seq_path = len(path_data_list[p_idx])

        prev_interaction_end_human_pose = None
        prev_navigation_end_human_pose = None
        prev_interaction_motion = None
        prev_navigation_motion = None
        prev_interaction_finger_motion = None
        prev_navigation_finger_motion = None

        if prev_interaction_end_human_pose_last is not None:
            prev_interaction_end_human_pose = (
                prev_interaction_end_human_pose_last.clone()
            )
            prev_interaction_motion = prev_interaction_motion_last.clone()
            prev_interaction_finger_motion = prev_interaction_finger_motion_last.clone()
            print("Using last interaction pose as the initial pose.")

        mesh_save_folders = []
        initial_obj_paths = []
        video_paths = []

        params_save_paths = []
        contact_start_frames = []
        contact_end_frames = []
        for o_idx in range(num_sub_seq_path):
            finger_all_res_list = None

            if (o_idx + 1) % 2 == 0:
                use_navigation_model = False
            else:
                use_navigation_model = True

            # Only use the first path for canonicalization!
            # Use distance heuristics to determine the waypoints at frame 30, 60, 90.
            # Need to consider waypoints in navigation is human root, in interaction is object com.
            if use_navigation_model and prev_interaction_end_human_pose is not None:
                start_waypoint = prev_interaction_end_human_pose[0, :, :2]  # 1 X 2
            else:
                start_waypoint = None

            planned_obj_path = load_planned_path_as_waypoints(
                path_data_list[p_idx][o_idx],
                use_canonicalization=False,
                load_for_nav=use_navigation_model,
                start_waypoint=start_waypoint,
            )

            vis_tag = (
                str(coarse_milestone)
                + "_long_seq_w_waypoints"
                + "_pidx_"
                + str(p_idx)
                + "_oidx_"
                + str(o_idx)
            )

            if interaction_trainer.use_guidance_in_denoising:
                vis_tag = vis_tag + "_interaction_guidance"

            if not use_navigation_model:
                curr_object_name = object_names_list[p_idx][o_idx]
                no_fc = decide_no_force_closure_from_objects(curr_object_name)

                # Interaction motion generation.
                (
                    all_res_list,
                    finger_all_res_list,
                    coarse_all_res_list,
                    fine_all_res_list,
                    prev_interaction_motion,
                    prev_interaction_end_human_pose,
                    prev_interaction_finger_motion,
                    contact_start_frame,
                    contact_end_frame,
                ) = generate_interaction_motion(
                    coarse_interaction_trainer=interaction_trainer,
                    fine_interaction_trainer=fine_interaction_trainer,
                    navigation_trainer=navigation_trainer,
                    finger_trainer=finger_trainer,
                    curr_object_name=curr_object_name,
                    action_name=action_names_list[p_idx][o_idx],
                    text=text_list[p_idx][o_idx],
                    all_object_data_dict=all_object_data_dict,
                    ref_data_dict=ref_data_dict,
                    planned_obj_path=planned_obj_path,
                    path_data=path_data_list[p_idx][o_idx],
                    obj_initial_rot_mat=obj_initial_rot_mat_list[p_idx][o_idx],
                    obj_end_rot_mat=obj_end_rot_mat_list[p_idx][o_idx],
                    overlap_frame_num=overlap_frame_num,
                    prev_navigation_motion=prev_navigation_motion,
                    prev_navigation_end_human_pose=prev_navigation_end_human_pose,
                    prev_navigation_finger_motion=prev_navigation_finger_motion,
                    rest_human_offsets=rest_human_offsets,
                    trans2joint=trans2joint,
                    table_height=table_height_list[p_idx][o_idx],
                    rest_left_hand_local_rot_6d=rest_left_hand_local_rot_6d,
                    rest_right_hand_local_rot_6d=rest_right_hand_local_rot_6d,
                    add_finger_motion=opt.add_finger_motion,
                    add_interaction_root_xy_ori=opt.add_interaction_root_xy_ori,
                    add_interaction_feet_contact=opt.add_interaction_feet_contact,
                    fix_feet_floor_penetration=fix_feet_floor_penetration,
                    fix_feet_sliding=fix_feet_sliding,
                    smooth_whole_traj=smooth_whole_traj,
                    no_force_closure=no_fc,
                )

                # Save the initial and end object mesh for visualization.
                initial_end_meshes = save_initial_end_object_meshes(
                    path_data=path_data_list[p_idx],
                    obj_initial_rot_mats=obj_initial_rot_mat_list[p_idx],
                    obj_end_rot_mats=obj_end_rot_mat_list[p_idx],
                    object_names=object_names_list[p_idx],
                    num_sub_seq_path=num_sub_seq_path,
                    p_idx=p_idx,
                    vis_wdir=opt.vis_wdir,
                    interaction_trainer=interaction_trainer,
                )
                initial_obj_paths.extend(initial_end_meshes)

                # Save the interaction motion mesh for visualization.
                (
                    dest_mesh_vis_folder,
                    params_path,
                ) = save_interaction_motion_meshes(
                    interaction_trainer=interaction_trainer,
                    all_res_list=all_res_list,
                    ref_obj_rot_mat=all_object_data_dict[curr_object_name][
                        "reference_obj_rot_mat"
                    ],
                    ref_data_dict=ref_data_dict,
                    step=coarse_milestone,
                    planned_waypoints_pos=planned_obj_path,
                    curr_object_name=curr_object_name,
                    vis_tag=vis_tag,
                    dest_mesh_vis_folder=dest_mesh_vis_folder,
                    finger_all_res_list=finger_all_res_list,
                )
                mesh_save_folders.append(dest_mesh_vis_folder)
                params_save_paths.append(params_path)
                contact_start_frames.append(contact_start_frame)
                contact_end_frames.append(contact_end_frame)
            else:
                # Navigation motion generation.
                (
                    all_res_list,
                    finger_all_res_list,
                    prev_navigation_motion,
                    prev_navigation_end_human_pose,
                    prev_navigation_finger_motion,
                    whole_cond_mask,
                ) = generate_navigation_motion(
                    interaction_trainer=fine_interaction_trainer,
                    navigation_trainer=navigation_trainer,
                    ref_data_dict=ref_data_dict,
                    prev_interaction_motion=prev_interaction_motion,
                    prev_interaction_end_human_pose=prev_interaction_end_human_pose,
                    trans2joint=trans2joint,
                    rest_human_offsets=rest_human_offsets,
                    planned_obj_path=planned_obj_path,
                    text_list=text_list,
                    p_idx=p_idx,
                    o_idx=o_idx,
                    overlap_frame_num_navi=overlap_frame_num_navi,
                    rest_left_hand_local_rot_6d=rest_left_hand_local_rot_6d,
                    rest_right_hand_local_rot_6d=rest_right_hand_local_rot_6d,
                    prev_interaction_finger_motion=prev_interaction_finger_motion,
                    add_finger_motion=opt.add_finger_motion,
                    fix_feet_floor_penetration=fix_feet_floor_penetration,
                    fix_feet_sliding=fix_feet_sliding,
                )

                # Save the navigation motion mesh for visualization.
                vis_tag = "navigation_pidx_{}_oidx_{}_epoch_{}".format(
                    p_idx, o_idx, navi_milestone
                )
                _, _, _, _, _, dest_mesh_vis_folder, params_path = (
                    navigation_trainer.gen_vis_res_human_only(
                        all_res_list,
                        planned_obj_path,
                        whole_cond_mask,
                        ref_data_dict,
                        vis_tag=vis_tag,
                        vis_wo_scene=True,
                        cano_quat=None,
                        finger_all_res_list=finger_all_res_list,
                    )
                )
                mesh_save_folders.append(dest_mesh_vis_folder)
                params_save_paths.append(params_path)

            # Render the current interaction and navigation motion videos.
            if VISUALIZE and o_idx % 2 == 1:
                use_guidance_str = "1" if opt.use_guidance_in_denoising else "0"
                interaction_epoch = str(coarse_milestone)
                video_save_dir_name = os.path.join("visualizer_results", opt.vis_wdir)
                video_path = render_motion_clip(
                    mesh_save_folders=mesh_save_folders,
                    initial_end_obj_mesh_paths=initial_obj_paths,
                    p_idx=p_idx,
                    video_paths=video_paths,
                    use_guidance_str=use_guidance_str,
                    interaction_checkpoint_epoch=interaction_epoch,
                    video_save_dir_name=video_save_dir_name,
                )

                mesh_save_folders = []
                initial_obj_paths = []
                video_paths.append(video_path)

        # Merge the motion clips.
        if VISUALIZE:
            final_video_path = os.path.join(
                "{}".format(video_save_dir_name),
                "{}_{}".format(p_idx, use_guidance_str),
                "output.mp4",
            )
            merge_motion_clips(
                final_video_path=final_video_path,
                video_paths=video_paths,
            )


def run_sample(
    opt,
    device: torch.device,
    coarse_milestone: str = "10",
    fine_milestone: str = "10",
    navi_milestone: str = "10",
) -> None:
    """
    Samples long sequences for multi-object scenarios using various trainers.

    This function initializes and configures trainers for navigation, coarse
    interaction, fine interaction, and optionally finger motion. It also
    prepares the output directory for visualization results and invokes the
    navigation trainer to generate conditional samples.

    Args:
        opt (object): An object containing configuration options.
        device (torch.device): The device (CPU or GPU) to be used for computations.
        coarse_milestone (str, optional): The milestone version for the coarse
                                          interaction trainer. Defaults to "10".
        fine_milestone (str, optional): The milestone version for the fine
                                        interaction trainer. Defaults to "10".
        navi_milestone (str, optional): The milestone version for the navigation
                                        trainer. Defaults to "10".
    """
    if not opt.vis_wdir:
        raise ValueError("Please specify the vis_wdir for visualization.")

    # Create results folder if it doesn't exist.
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./results/initial_obj_vis"):
        os.makedirs("./results/initial_obj_vis")

    cond_sample_res_w_long_planned_path_for_multi_objects(
        opt,
        device,
        coarse_milestone=coarse_milestone,
        fine_milestone=fine_milestone,
        navi_milestone=navi_milestone,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    run_sample(opt, device)
