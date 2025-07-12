"""
Modified from https://github.com/PKU-EPIC/DexGraspNet
"""

import os
import sys

sys.path.append(".")
sys.path.append("..")

# os.chdir(os.getcwd())
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pytorch3d.transforms as transforms
import torch
from tqdm import tqdm

from grasp_generation.utils.energy import cal_energy
from grasp_generation.utils.hand_model import HandModel
from grasp_generation.utils.initializations import (
    initialize_convex_hull,
)
from grasp_generation.utils.logger import Logger
from grasp_generation.utils.object_model import ObjectModel
from grasp_generation.utils.optimizer import Annealing
from grasp_generation.utils.save import save_hand_meshes


def run_grasp(
    justtest: bool = False,
    lefthand: bool = False,
    seed: int = 1,
    gpu: str = "0",
    object_code_list: List[str] = ["monitor"],
    name: str = "exp_32",
    n_contact: int = 4,
    batch_size: int = 32,
    n_iter: int = 6000,
    switch_possibility: float = 0.5,
    mu: float = 0.98,
    step_size: float = 0.005,
    stepsize_period: int = 50,
    starting_temperature: int = 18,
    annealing_period: int = 30,
    temperature_decay: float = 0.95,
    w_dis: float = 100.0,
    w_pen: float = 100.0,
    w_prior: float = 0.5,
    w_spen: float = 10.0,
    jitter_strength: float = 0.0,
    distance_lower: float = 0.1,
    distance_upper: float = 0.1,
    theta_lower: float = 0,
    theta_upper: float = 0,
    thres_fc: float = 0.3,
    thres_dis: float = 0.005,
    thres_pen: float = 0.001,
    wrist_init_path: Optional[str] = None,
    wrist_init_pose: Optional[Dict] = None,
    no_fc: bool = False,
    in_parallel: bool = False,
    return_multiple_grasp: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on", device)

    if in_parallel:
        if wrist_init_path != "":
            wrist_pos = []
            wrist_rot = []
            for obj in object_code_list:
                hand = "left_hand" if lefthand else "right_hand"
                path = os.path.join(wrist_init_path, "{}_{}.pkl".format(obj, hand))
                wrist_init = pickle.load(open(path, "rb"))
                wrist_pos_init = (
                    torch.from_numpy(wrist_init["wrist_pos"]).float().to(device)
                )
                wrist_rot_init = (
                    torch.from_numpy(wrist_init["wrist_rot"]).float().to(device)
                )
                if justtest:
                    wrist_pos_init = wrist_pos_init[:1]
                    wrist_rot_init = wrist_rot_init[:1]
                wrist_pos.append(wrist_pos_init)
                wrist_rot.append(wrist_rot_init)
            wrist_pos = torch.cat(wrist_pos)
            wrist_rot = torch.cat(wrist_rot)
            batch_size = wrist_pos.shape[0] // len(object_code_list)
        else:
            raise ValueError(
                "wrist_init_path must be provided when in_parallel is True"
            )
    else:
        if wrist_init_pose is not None:
            wrist_pos = wrist_init_pose["wrist_pos"].float().to(device)
            wrist_rot = wrist_init_pose["wrist_rot"].float().to(device)
        elif wrist_init_path is not None:
            wrist_init = pickle.load(open(wrist_init_path, "rb"))
            wrist_pos = wrist_init["wrist_pos"].float().to(device)
            wrist_rot = wrist_init["wrist_rot"].float().to(device)
        else:
            raise ValueError(
                "wrist_init_pose or wrist_init_path must be provided when in_parallel is False"
            )
        wrist_pos = wrist_pos.detach().requires_grad_()  # 3
        wrist_rot = wrist_rot.detach().requires_grad_()  # 3 X 3

    args_dict = {
        "justtest": justtest,
        "lefthand": lefthand,
        "seed": seed,
        "gpu": gpu,
        "object_code_list": object_code_list,
        "name": name,
        "n_contact": n_contact,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "switch_possibility": switch_possibility,
        "mu": mu,
        "step_size": step_size,
        "stepsize_period": stepsize_period,
        "starting_temperature": starting_temperature,
        "annealing_period": annealing_period,
        "temperature_decay": temperature_decay,
        "w_dis": w_dis,
        "w_pen": w_pen,
        "w_prior": w_prior,
        "w_spen": w_spen,
        "jitter_strength": jitter_strength,
        "distance_lower": distance_lower,
        "distance_upper": distance_upper,
        "theta_lower": theta_lower,
        "theta_upper": theta_upper,
        "thres_fc": thres_fc,
        "thres_dis": thres_dis,
        "thres_pen": thres_pen,
        "wrist_init_path": wrist_init_path,
        "no_fc": no_fc,
    }
    args = argparse.Namespace(**args_dict)

    total_batch_size = len(args.object_code_list) * args.batch_size

    grasp_generation_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "grasp_generation",
    )
    hand_model = HandModel(
        mano_root=os.path.join(grasp_generation_root, "mano"),
        contact_indices_path=os.path.join(
            grasp_generation_root, "mano", "contact_indices.json"
        ),
        pose_distrib_path=os.path.join(
            grasp_generation_root, "mano", "pose_distrib.pt"
        ),
        device=device,
        batch_size=args.batch_size * len(args.object_code_list),
        left_hand=args.lefthand,
        no_fc=args.no_fc,
    )

    object_model = ObjectModel(
        data_root_path=os.path.join(
            grasp_generation_root,
            "objects",
        ),
        batch_size_each=args.batch_size,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(args.object_code_list, wrist_pos=None)

    # if wrist_init_path != '':
    initialize_convex_hull(
        hand_model,
        object_model,
        args,
        wrist_pos,
        wrist_rot,
        move_away=True,
        no_fc=args.no_fc,
    )
    # else:
    #     initialize_convex_hull_original(hand_model, object_model, args, no_fc=args.no_fc)

    print("total batch size", total_batch_size)
    hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        "switch_possibility": args.switch_possibility,
        "starting_temperature": args.starting_temperature,
        "temperature_decay": args.temperature_decay,
        "annealing_period": args.annealing_period,
        "step_size": args.step_size,
        "stepsize_period": args.stepsize_period,
        "mu": args.mu,
        "device": device,
    }
    optimizer = Annealing(hand_model, **optim_config)

    os.makedirs(
        os.path.join(
            grasp_generation_root,
            "experiments",
            args.name,
            "logs",
        ),
        exist_ok=True,
    )
    logger_config = {
        "thres_fc": args.thres_fc,
        "thres_dis": args.thres_dis,
        "thres_pen": args.thres_pen,
    }
    logger = Logger(
        log_dir=os.path.join(
            grasp_generation_root,
            "experiments",
            args.name,
            "logs",
        ),
        **logger_config,
    )

    # log settings

    with open(
        os.path.join(
            grasp_generation_root,
            "experiments",
            args.name,
            "output.txt",
        ),
        "w",
    ) as f:
        f.write(str(args) + "\n")

    # optimize
    correct_initial_pose = not in_parallel

    weight_dict = dict(
        w_dis=args.w_dis, w_pen=args.w_pen, w_prior=args.w_prior, w_spen=args.w_spen
    )
    energy, E_fc, E_dis, E_pen, E_prior, E_spen = cal_energy(
        hand_model,
        object_model,
        verbose=True,
        no_fc=args.no_fc,
        correct_initial_pose=correct_initial_pose,
        **weight_dict,
    )

    energy.sum().backward(retain_graph=True)
    logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, 0, show=False)

    epoch = 2 if args.justtest else args.n_iter + 1

    for step in tqdm(range(1, epoch), desc="optimizing"):
        s = optimizer.try_step()

        optimizer.zero_grad()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen = (
            cal_energy(
                hand_model,
                object_model,
                verbose=True,
                no_fc=args.no_fc,
                correct_initial_pose=correct_initial_pose,
                **weight_dict,
            )
        )

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_prior[accept] = new_E_prior[accept]
            E_spen[accept] = new_E_spen[accept]

            logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, step, show=False)

    # Save results
    result_path = os.path.join(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "grasp_generation",
            "experiments",
            args.name,
            "results",
        )
    )
    os.makedirs(result_path, exist_ok=True)

    wrist_pos_in_obj, wrist_rot_mat_in_obj, finger_local_rot_6d, obj_to_palm = (
        None,
        None,
        None,
        None,
    )
    successes = []
    data_lists = []
    min_indices = []
    for i in range(len(args.object_code_list)):
        data_list = []
        min_engery = 1e6
        min_idx = 0
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            qpos = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            hand_pose = hand_pose_st[idx].detach().cpu()
            qpos_st = dict(
                trans=hand_pose[:3].tolist(),
                rot=hand_pose[3:6].tolist(),
                thetas=hand_pose[6:].tolist(),
            )
            data_list.append(
                dict(
                    scale=scale,
                    qpos=qpos,
                    contact_point_indices=hand_model.contact_point_indices[idx]
                    .detach()
                    .cpu()
                    .tolist(),
                    qpos_st=qpos_st,
                    energy=energy[idx].item(),
                    E_fc=E_fc[idx].item(),
                    E_dis=E_dis[idx].item(),
                    E_pen=E_pen[idx].item(),
                    E_prior=E_prior[idx].item(),
                    E_spen=E_spen[idx].item(),
                )
            )
            success_fc = E_fc[idx].item() < args.thres_fc
            success_dis = E_dis[idx].item() < args.thres_dis
            success_pen = E_pen[idx].item() < args.thres_pen
            if args.no_fc:
                success_fc = True
            success = success_fc * success_dis * success_pen
            # dis = torch.norm((hand_model.hand_pose[idx, :3] - hand_model.initial_translation[idx]), dim=-1)

            successes.append(success)
            print("idx {},".format(j), success, "energy", energy[idx].item())
            if success:
                if energy[idx].item() < min_engery:
                    min_engery = energy[idx].item()
                    min_idx = idx
        min_indices.append(min_idx)
        data_lists.append(data_list)
        np.save(
            os.path.join(result_path, args.object_code_list[i] + ".npy"),
            data_list,
            allow_pickle=True,
        )

    # Save hand meshes.
    hand_model.set_parameters(hand_model.hand_pose)
    candidate_hand_mesh_paths, obj_mesh_paths, hand_poses = save_hand_meshes(
        hand_model, object_model, result_path, args, successes, data_lists, lefthand
    )

    if return_multiple_grasp:
        return candidate_hand_mesh_paths, obj_mesh_paths, hand_poses
    else:
        min_indices = torch.tensor(min_indices).to(hand_model.hand_pose.device)
        wrist_pos_in_obj = hand_model.hand_pose[min_indices, :3]
        wrist_rot_mat_in_obj = transforms.axis_angle_to_matrix(
            hand_model.hand_pose[min_indices, 3:6]
        )
        finger_local_rot_6d = transforms.matrix_to_rotation_6d(
            transforms.axis_angle_to_matrix(
                hand_model.hand_pose[min_indices, -45:].reshape(-1, 3)
            )
        ).reshape(-1, 90)
        # obj_to_palm = object_model.get_surface_points_near_wrist(min_idx, wrist_pos_in_obj)
        # print("min_idx", min_idx)
        # print("wrist_pos_in_obj", wrist_pos_in_obj)
        # print("wrist_rot_mat_in_obj", wrist_rot_mat_in_obj)
        # print("finger_local_rot_6d", finger_local_rot_6d)
        # print("obj_to_palm", obj_to_palm)

        return wrist_pos_in_obj, wrist_rot_mat_in_obj, finger_local_rot_6d


if __name__ == "__main__":
    # prepare arguments

    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument("--justtest", action="store_true")
    parser.add_argument("--lefthand", action="store_true")
    parser.add_argument("--no_fc", action="store_true")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--gpu", default="2", type=str)
    parser.add_argument(
        "--object_code_list",
        default=[
            "largebox",
            "smallbox",
        ],
        type=str,
        nargs="*",
    )
    parser.add_argument("--name", default="exp_32", type=str)
    parser.add_argument("--n_contact", default=4, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_iter", default=6000, type=int)
    # hyper parameters (** Magic, don't touch! **)
    parser.add_argument("--switch_possibility", default=0.5, type=float)
    parser.add_argument("--mu", default=0.98, type=float)
    parser.add_argument("--step_size", default=0.005, type=float)
    parser.add_argument("--stepsize_period", default=50, type=int)
    parser.add_argument("--starting_temperature", default=18, type=float)
    parser.add_argument("--annealing_period", default=30, type=int)
    parser.add_argument("--temperature_decay", default=0.95, type=float)
    parser.add_argument("--w_dis", default=100.0, type=float)
    parser.add_argument("--w_pen", default=100.0, type=float)
    parser.add_argument("--w_prior", default=0.5, type=float)
    parser.add_argument("--w_spen", default=10.0, type=float)
    # initialization settings
    parser.add_argument("--jitter_strength", default=0.0, type=float)
    parser.add_argument("--distance_lower", default=0.1, type=float)
    parser.add_argument("--distance_upper", default=0.1, type=float)
    parser.add_argument("--theta_lower", default=0, type=float)
    parser.add_argument("--theta_upper", default=0, type=float)
    # energy thresholds
    parser.add_argument("--thres_fc", default=0.3, type=float)
    parser.add_argument("--thres_dis", default=0.005, type=float)
    parser.add_argument("--thres_pen", default=0.001, type=float)

    parser.add_argument("--wrist_init_path", type=str, default="")
    parser.add_argument("--in_parallel", action="store_true")

    args = parser.parse_args()
    run_grasp(
        justtest=args.justtest,
        lefthand=args.lefthand,
        seed=args.seed,
        gpu=args.gpu,
        object_code_list=args.object_code_list,
        name=args.name,
        n_contact=args.n_contact,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        switch_possibility=args.switch_possibility,
        mu=args.mu,
        step_size=args.step_size,
        stepsize_period=args.stepsize_period,
        starting_temperature=args.starting_temperature,
        annealing_period=args.annealing_period,
        temperature_decay=args.temperature_decay,
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_prior=args.w_prior,
        w_spen=args.w_spen,
        jitter_strength=args.jitter_strength,
        distance_lower=args.distance_lower,
        distance_upper=args.distance_upper,
        theta_lower=args.theta_lower,
        theta_upper=args.theta_upper,
        thres_fc=args.thres_fc,
        thres_dis=args.thres_dis,
        thres_pen=args.thres_pen,
        wrist_init_path=args.wrist_init_path,
        no_fc=args.no_fc,
        in_parallel=args.in_parallel,
    )
