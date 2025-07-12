import argparse
import json
import os
import pickle
import random
import subprocess
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pytorch3d.transforms as transforms
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import wandb
import yaml
from ema_pytorch import EMA
from human_body_prior.body_model.body_model import BodyModel
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils import data
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from manip.data.grab_dataset import (
    GrabAmbientSensorDataset,
    GrabBothSensorDataset,
    GrabDataset,
    GrabProximitySensorDataset,
)
from manip.data.hand_foot_dataset import quat_ik_hand, quat_ik_wholebody
from manip.data.omomo_dataset import (
    OMOMOAmbientSensorDataset,
    OMOMOBothSensorDataset,
    OMOMODataset,
    OMOMOProximitySensorDataset,
)
from manip.model.transformer_hand_foot_manip_cond_diffusion_model import (
    CondGaussianDiffusionBothSensorNew,
)

JOINT_NUM = 52
SMPLX_MODEL_PATH = "./data/smpl_all_models/"

VIS_CHECK = False


def get_sub_mesh(vertices, faces, sub_vertices):
    # vertices: T X Nv X 3
    vertex_map = {v: i for i, v in enumerate(sub_vertices)}

    new_vertices = vertices[:, sub_vertices]

    new_faces = []
    for face in faces:
        if all(vertex in vertex_map for vertex in face):
            new_face = [vertex_map[vertex] for vertex in face]
            new_faces.append(new_face)

    return np.array(new_vertices), np.array(new_faces)


def save_verts_faces_to_mesh_file_w_object(
    mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder
):
    # mesh_verts: T X Nv X 3
    # mesh_faces: Nf X 3
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx], faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d" % (idx) + ".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx], faces=obj_faces)
        curr_obj_mesh_path = os.path.join(
            save_mesh_folder, "%05d" % (idx) + "_object.ply"
        )
        obj_mesh.export(curr_obj_mesh_path)


def run_smplx_model_new(root_trans, aa_rot_rep, gender, vtemp_path):
    # root_trans: T X 3
    # aa_rot_rep: T X 32 X 3
    smplx_model_path = SMPLX_MODEL_PATH
    sbj_vtemp = np.array(trimesh.load(vtemp_path).vertices)

    root_trans = root_trans.cpu()
    aa_rot_rep = aa_rot_rep.cpu()

    num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_body = torch.zeros(num_steps, 20, 3)  # T X 20 X 3
        aa_rot_rep = torch.cat((padding_zeros_body, aa_rot_rep), dim=-2)  # T X 52 X 3
    aa_rot_rep = aa_rot_rep.reshape(num_steps, -1)  # T X 156

    with torch.no_grad():
        sbj_m = smplx.create(
            model_path=smplx_model_path,
            model_type="smplx",
            gender=gender,
            #  v_template=sbj_vtemp,
            batch_size=num_steps,
            flat_hand_mean=True,
            use_pca=False,
        )
        output = sbj_m(
            global_orient=aa_rot_rep[:, :3],
            body_pose=aa_rot_rep[:, 3:66],
            left_hand_pose=aa_rot_rep[:, 66 : 66 + 45],
            right_hand_pose=aa_rot_rep[:, 66 + 45 :],
            transl=root_trans,
        )
    return (
        output.joints[:, :55][None],
        output.vertices[None],
        torch.from_numpy(sbj_m.faces.astype(np.int32)),
    )


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=20000,
        results_folder="./results",
        use_wandb=True,
        load_ds=True,
    ):
        super().__init__()

        self.load_ds = load_ds

        self.use_wandb = use_wandb
        if self.use_wandb:
            # Loggers
            wandb.init(
                config=opt,
                project=opt.wandb_pj_name,
                entity=opt.entity,
                name=opt.exp_name,
                dir=opt.save_dir,
            )

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.milestone = opt.milestone
        self.omomo_obj = opt.omomo_obj

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt

        self.window = opt.window

        self.use_window_bps = True

        self.use_object_split = self.opt.use_object_split

        self.use_joints24 = True

        self.use_blender_data = self.opt.use_blender_data

        self.prep_dataloader(window_size=opt.window)

        # self.bm_dict = self.ds.bm_dict

        self.test_on_train = self.opt.test_sample_res_on_train

        self.remove_condition = self.opt.remove_condition

        self.normalize_condition = self.opt.normalize_condition

        self.add_start_human_pose = self.opt.add_start_human_pose

        self.pred_hand_jpos_only_from_obj = self.opt.pred_hand_jpos_only_from_obj

        self.pred_hand_jpos_and_rot_from_obj = self.opt.pred_hand_jpos_and_rot_from_obj

        self.pred_palm_jpos_from_obj = self.opt.pred_palm_jpos_from_obj

        self.add_hand_processing = self.opt.add_hand_processing

        self.for_quant_eval = self.opt.for_quant_eval

        self.use_gt_hand_for_eval = self.opt.use_gt_hand_for_eval

        self.load_hand_vertexs()

        self.parents_handonly = np.load(
            "./data/smpl_all_models/smplx_parents_onlyhand_32.npy",
        )
        self.parents_wholebody = np.load(
            "./data/smpl_all_models/smplx_parents_52.npy",
        )

        self.min_val_loss = 100000

        # self.train_contact_label = opt.train_contact_label
        self.wrist_obj_traj_condition = opt.wrist_obj_traj_condition
        self.ambient_sensor_condition = opt.ambient_sensor_condition
        self.proximity_sensor_condition = opt.proximity_sensor_condition
        self.ref_pose_condition = opt.ref_pose_condition
        self.contact_label_condition = opt.contact_label_condition

    def load_hand_vertexs(self):
        data = pickle.load(
            open(
                os.path.join(
                    "data",
                    "smpl_all_models",
                    "MANO_SMPLX_vertex_ids.pkl",
                ),
                "rb",
            )
        )
        self.lhand_verts = torch.from_numpy(data["left_hand"]).cuda()
        self.rhand_verts = torch.from_numpy(data["right_hand"]).cuda()

    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/grab_data/processed_omomo"
        # Define dataset
        train_dataset = GrabDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = GrabDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(
            data, os.path.join(self.results_folder, "model-" + str(milestone) + ".pt")
        )

    def load(self, milestone):
        data = torch.load(
            os.path.join(self.results_folder, "model-" + str(milestone) + ".pt")
        )

        self.step = data["step"]
        self.model.load_state_dict(data["model"], strict=False)
        self.ema.load_state_dict(data["ema"], strict=False)
        self.scaler.load_state_dict(data["scaler"])

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data).to(data.device)  # BS X T X D
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def train(self):
        init_step = self.step
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = (
                False  # If met nan in loss or gradient, need to skip to next data.
            )
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data = data_dict["motion"].cuda()  # BS X T X (22*3+22*6)

                bs, num_steps, _ = data.shape

                data = self.extract_wrist_finger_data(data)  # BS X T X (32 * 6 + 2 * 3)
                obj_bps_data = data_dict["obj_bps"].cuda()
                obj_com_pos = data_dict["obj_com_pos"].cuda()  # BS X T X 3

                contact_label = data_dict["contact_label"].cuda()  # BS X T X 2
                ori_data_cond = torch.cat(
                    (data[..., :18], obj_com_pos, obj_bps_data, contact_label), dim=-1
                )  # BS X T X (21+2048+200+2)
                data = data_dict["local_rot"][..., -180:].cuda()  # BS X T X (30 * 6)

                cond_mask = None
                if self.ref_pose_condition:
                    cond_mask = torch.ones_like(data).to(
                        data.device
                    )  # BS X T X (30 * 6)
                    cond_mask[..., :90] *= 1.0 - contact_label[..., 0:1]
                    cond_mask[..., 90:] *= 1.0 - contact_label[..., 1:2]

                # Generate padding mask
                actual_seq_len = (
                    data_dict["seq_len"] + 1
                )  # BS, + 1 since we need additional timestep for noise level
                tmp_mask = torch.arange(self.window + 1).expand(
                    data.shape[0], self.window + 1
                ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(data.device)

                with autocast(enabled=self.amp):
                    loss_diffusion = self.model(
                        data, ori_data_cond, cond_mask, padding_mask
                    )

                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print("WARNING: NaN loss. Skipping to next data...")
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [
                        p for p in self.model.parameters() if p.grad is not None
                    ]
                    total_norm = torch.norm(
                        torch.stack(
                            [
                                torch.norm(p.grad.detach(), 2.0).to(data.device)
                                for p in parameters
                            ]
                        ),
                        2.0,
                    )
                    if torch.isnan(total_norm):
                        print("WARNING: NaN gradients. Skipping to next data...")
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                        }
                        wandb.log(log_dict)

                    if idx % 10 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_data = val_data_dict["motion"].cuda()

                    bs, num_steps, _ = val_data.shape

                    val_data = self.extract_wrist_finger_data(
                        val_data
                    )  # BS X T X (32 * 6 + 2 * 3)

                    obj_bps_data = val_data_dict["obj_bps"].cuda()
                    obj_com_pos = val_data_dict["obj_com_pos"].cuda()

                    contact_label = val_data_dict["contact_label"].cuda()  # BS X T X 2
                    ori_data_cond = torch.cat(
                        (val_data[..., :18], obj_com_pos, obj_bps_data, contact_label),
                        dim=-1,
                    )  # BS X T X (21+2048+200+2)
                    val_data = val_data_dict["local_rot"][
                        ..., -180:
                    ].cuda()  # BS X T X (30 * 6)

                    cond_mask = None
                    if self.ref_pose_condition:
                        cond_mask = torch.ones_like(val_data).to(
                            val_data.device
                        )  # BS X T X (30 * 6)
                        cond_mask[..., :90] *= 1.0 - contact_label[..., 0:1]
                        cond_mask[..., 90:] *= 1.0 - contact_label[..., 1:2]

                    # Generate padding mask
                    actual_seq_len = (
                        val_data_dict["seq_len"] + 1
                    )  # BS, + 1 since we need additional timestep for noise level
                    tmp_mask = torch.arange(self.window + 1).expand(
                        val_data.shape[0], self.window + 1
                    ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    # Get validation loss
                    val_loss_diffusion = self.model(
                        val_data, ori_data_cond, cond_mask, padding_mask
                    )
                    val_loss = val_loss_diffusion
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                        }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every

                    bs_for_vis = 1

                    vis_gt = True
                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)
                        if val_loss.item() < self.min_val_loss:
                            self.min_val_loss = val_loss.item()
                            self.save("best")
                            print("Best model {} saved.".format(milestone))

                        all_res_list = self.ema.ema_model.sample(
                            val_data, ori_data_cond, cond_mask, padding_mask
                        )
                        all_res_list = all_res_list[:bs_for_vis]

                        self.gen_vis_res(
                            all_res_list,
                            val_data_dict,
                            self.step,
                            vis_tag="pred_jpos",
                            vis_gt=vis_gt,
                        )

            self.step += 1

        print("training complete")

        if self.use_wandb:
            wandb.run.finish()

    def load_palm_sample_verts(self):
        path = os.path.join(
            "data",
            "smpl_all_models",
            "palm_sample_indices.pkl",
        )
        self.palm_sample_indices_dict = pickle.load(open(path, "rb"))
        self.palm_sample_indices = np.concatenate(
            [
                self.palm_sample_indices_dict["left_hand"],
                self.palm_sample_indices_dict["right_hand"],
            ],
            axis=0,
        )
        self.palm_sample_indices = torch.from_numpy(self.palm_sample_indices)

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [
            os.path.join(self.results_folder, weight) for weight in weights
        ]
        weight_path = max(weights_paths, key=os.path.getctime)

        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")

        self.load(milestone)
        self.ema.ema_model.eval()

        num_sample = 50

        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                val_data = val_data_dict["motion"].cuda()

                val_data = self.extract_wrist_finger_data(
                    val_data
                )  # BS X T X (32 * 6 + 2 * 3)
                obj_bps_data = val_data_dict["obj_bps"].cuda()
                obj_com_pos = val_data_dict["obj_com_pos"].cuda()

                contact_label = val_data_dict["contact_label"].cuda()  # BS X T X 2
                ori_data_cond = torch.cat(
                    (val_data[..., :18], obj_com_pos, obj_bps_data, contact_label),
                    dim=-1,
                )  # BS X T X (21+2048+200+2)
                val_data = val_data_dict["local_rot"][
                    ..., -180:
                ].cuda()  # BS X T X (30 * 6)

                cond_mask = None
                if self.ref_pose_condition:
                    cond_mask = torch.ones_like(val_data).to(
                        val_data.device
                    )  # BS X T X (30 * 6)
                    cond_mask[..., :90] *= 1.0 - contact_label[..., 0:1]
                    cond_mask[..., 90:] *= 1.0 - contact_label[..., 1:2]

                # Generate padding mask
                actual_seq_len = (
                    val_data_dict["seq_len"] + 1
                )  # BS, + 1 since we need additional timestep for noise level
                tmp_mask = torch.arange(self.window + 1).expand(
                    val_data.shape[0], self.window + 1
                ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                max_num = 1

                all_res_list = self.ema.ema_model.sample(
                    val_data,
                    ori_data_cond,
                    cond_mask=cond_mask,
                    padding_mask=padding_mask,
                )

                vis_tag = str(milestone) + "_stage1_sample_" + str(s_idx)

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"

                self.gen_vis_res(
                    all_res_list[:max_num], val_data_dict, milestone, vis_tag=vis_tag
                )

    def extract_wrist_finger_data(self, data_input):
        # J = 52
        # data_input: BS X T X D (J*3+J*6)
        lhand_idx = self.ds.lhand_idx
        rhand_idx = self.ds.rhand_idx
        wrist_pos_input = data_input[
            :, :, lhand_idx * 3 : lhand_idx * 3 + 6
        ]  # BS X T X D (2*3)
        rot_input = data_input[
            :, :, JOINT_NUM * 3 + lhand_idx * 6 :
        ]  # BS X T X D (32*6)
        data_input = torch.cat((wrist_pos_input, rot_input), dim=-1)
        # BS X T X D (2*3+32*6)

        return data_input

    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: 4(2) X 3
        lhand_color = np.asarray([255, 87, 51])  # red
        rhand_color = np.asarray([17, 99, 226])  # blue
        lfoot_color = np.asarray([134, 17, 226])  # purple
        rfoot_color = np.asarray([22, 173, 100])  # green

        color_list = [lhand_color, rhand_color, lfoot_color, rfoot_color]

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])

            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=color_list[idx],
                process=False,
            )

            result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding="ascii")
            output_file = open(
                ball_mesh_path.replace(".ply", "_" + str(idx) + ".ply"), "wb+"
            )
            output_file.write(result)
            output_file.close()

    def export_to_mesh(self, mesh_verts, mesh_faces, mesh_path):
        dest_mesh = trimesh.Trimesh(
            vertices=mesh_verts, faces=mesh_faces, process=False
        )

        result = trimesh.exchange.ply.export_ply(dest_mesh, encoding="ascii")
        output_file = open(mesh_path, "wb+")
        output_file.write(result)
        output_file.close()

    def plot_arr(self, t_vec, pred_val, gt_val, dest_path):
        plt.plot(t_vec, gt_val, color="green", label="gt")
        plt.plot(t_vec, pred_val, color="red", label="pred")
        plt.legend(["gt", "pred"])
        plt.savefig(dest_path)
        plt.clf()

    def plot_foot_jpos(self, pred_foot_jpos, gt_foot_jpos, dest_traj_path):
        # pred_foot_jpos: T X 2 X 3
        # gt_foot_jpos: T X 2 X 3
        num_steps = pred_foot_jpos.shape[0]
        t_vec = np.asarray(list(range(num_steps)))

        pred_foot_jpos = pred_foot_jpos.detach().cpu().numpy()
        gt_foot_jpos = gt_foot_jpos.detach().cpu().numpy()

        dest_lfoot_x_path = dest_traj_path.replace(".png", "_lfoot_x.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 0, 0], gt_foot_jpos[:, 0, 0], dest_lfoot_x_path
        )

        dest_lfoot_y_path = dest_traj_path.replace(".png", "_lfoot_y.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 0, 1], gt_foot_jpos[:, 0, 1], dest_lfoot_y_path
        )

        dest_lfoot_z_path = dest_traj_path.replace(".png", "_lfoot_z.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 0, 2], gt_foot_jpos[:, 0, 2], dest_lfoot_z_path
        )

        dest_rfoot_x_path = dest_traj_path.replace(".png", "_rfoot_x.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 1, 0], gt_foot_jpos[:, 1, 0], dest_rfoot_x_path
        )

        dest_rfoot_y_path = dest_traj_path.replace(".png", "_rfoot_y.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 1, 1], gt_foot_jpos[:, 1, 1], dest_rfoot_y_path
        )

        dest_rfoot_z_path = dest_traj_path.replace(".png", "_rfoot_z.png")
        self.plot_arr(
            t_vec, pred_foot_jpos[:, 1, 2], gt_foot_jpos[:, 1, 2], dest_rfoot_z_path
        )

        # Plot velocity for foor joint
        dest_lfoot_x_vel_path = dest_traj_path.replace(".png", "_lfoot_x_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 0, 0] - pred_foot_jpos[:-1, 0, 0],
            gt_foot_jpos[1:, 0, 0] - gt_foot_jpos[:-1, 0, 0],
            dest_lfoot_x_vel_path,
        )

        dest_lfoot_y_vel_path = dest_traj_path.replace(".png", "_lfoot_y_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 0, 1] - pred_foot_jpos[:-1, 0, 1],
            gt_foot_jpos[1:, 0, 1] - gt_foot_jpos[:-1, 0, 1],
            dest_lfoot_y_vel_path,
        )

        dest_lfoot_z_vel_path = dest_traj_path.replace(".png", "_lfoot_z_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 0, 2] - pred_foot_jpos[:-1, 0, 2],
            gt_foot_jpos[1:, 0, 2] - gt_foot_jpos[:-1, 0, 2],
            dest_lfoot_z_vel_path,
        )

        dest_rfoot_x_vel_path = dest_traj_path.replace(".png", "_rfoot_x_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 1, 0] - pred_foot_jpos[:-1, 1, 0],
            gt_foot_jpos[1:, 1, 0] - gt_foot_jpos[:-1, 1, 0],
            dest_rfoot_x_vel_path,
        )

        dest_rfoot_y_vel_path = dest_traj_path.replace(".png", "_rfoot_y_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 1, 1] - pred_foot_jpos[:-1, 1, 1],
            gt_foot_jpos[1:, 1, 1] - gt_foot_jpos[:-1, 1, 1],
            dest_rfoot_y_vel_path,
        )

        dest_rfoot_z_vel_path = dest_traj_path.replace(".png", "_rfoot_z_vel.png")
        self.plot_arr(
            t_vec[:-1],
            pred_foot_jpos[1:, 1, 2] - pred_foot_jpos[:-1, 1, 2],
            gt_foot_jpos[1:, 1, 2] - gt_foot_jpos[:-1, 1, 2],
            dest_rfoot_z_vel_path,
        )

    def calculate_vertex_normals(self, vertices, faces):
        """
        Calculates the vertex normals for a mesh given its vertices and faces.

        Parameters:
            vertices (numpy.ndarray): A numpy array of shape (num_vertices, 3) containing the vertex positions.
            faces (numpy.ndarray): A numpy array of shape (num_faces, 3) containing the vertex indices of each face.

        Returns:
            numpy.ndarray: A numpy array of shape (num_vertices, 3) containing the vertex normals.
        """
        # Initialize an array to hold the vertex normals
        vertex_normals = np.zeros(vertices.shape, dtype=vertices.dtype)

        # Calculate the face normals
        face_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]

        # Accumulate the face normals for each vertex
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)

        # Normalize the vertex normals
        vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]

        return vertex_normals

    def process_hand_foot_contact_jpos(
        self, hand_foot_jpos, object_mesh_verts, object_mesh_faces, obj_rot
    ):
        # hand_foot_jpos: T X 4(2) X 3
        # object_mesh_verts: T X Nv X 3
        # object_mesh_faces: Nf X 3
        # obj_rot: T X 3 X 3
        all_contact_labels = []
        all_object_c_idx_list = []
        all_dist = []

        obj_rot = torch.from_numpy(obj_rot).to(hand_foot_jpos.device)
        object_mesh_verts = object_mesh_verts.to(hand_foot_jpos.device)

        num_joints = hand_foot_jpos.shape[1]
        num_steps = hand_foot_jpos.shape[0]

        # threshold = 0.05
        if self.use_joints24:
            threshold = 0.03  # Use palm position, should be smaller.
        else:
            threshold = 0.08  # For sidx: 91,

        joint2object_dist = torch.cdist(
            hand_foot_jpos, object_mesh_verts.to(hand_foot_jpos.device)
        )  # T X 4 X Nv

        all_dist, all_object_c_idx_list = joint2object_dist.min(dim=2)  # T X 4
        all_contact_labels = all_dist < threshold  # T X 4

        new_hand_foot_jpos = hand_foot_jpos.clone()  # T X 4(2) X 3

        # For each joint, scan the sequence, if contact is true, then use the corresponding object idx for the
        # rest of subsequence in contact until the distance is above a threshold.
        for j_idx in range(num_joints):
            continue_prev_contact = False
            for t_idx in range(num_steps):
                if continue_prev_contact:
                    relative_rot_mat = torch.matmul(
                        obj_rot[t_idx], reference_obj_rot.inverse()
                    )
                    curr_contact_normal = torch.matmul(
                        relative_rot_mat, contact_normal[:, None]
                    ).squeeze(-1)

                    # Add a random noise to avoid the case when object does not move anymore, predicted full pose sliding.
                    # random_noise = torch.randn(3) * 0.002 - 0.001 # [-0.001, 0.001]
                    # random_noise = random_noise.to(curr_contact_normal.device)

                    new_hand_foot_jpos[t_idx, j_idx] = (
                        object_mesh_verts[t_idx, subseq_contact_v_id]
                        + curr_contact_normal
                    )  # 3

                elif (
                    all_contact_labels[t_idx, j_idx] and not continue_prev_contact
                ):  # The first contact frame
                    subseq_contact_v_id = all_object_c_idx_list[t_idx, j_idx]
                    subseq_contact_pos = object_mesh_verts[
                        t_idx, subseq_contact_v_id
                    ]  # 3

                    contact_normal = (
                        new_hand_foot_jpos[t_idx, j_idx] - subseq_contact_pos
                    )  # Keep using this in the following frames.

                    reference_obj_rot = obj_rot[t_idx]  # 3 X 3

                    continue_prev_contact = True

                    # import pdb
                    # pdb.set_trace()

        return new_hand_foot_jpos

    def gen_vis_res(
        self,
        all_res_list,
        data_dict,
        step,
        vis_gt=False,
        vis_tag=None,
        for_quant_eval=False,
        selected_seq_idx=None,
    ):
        assert selected_seq_idx is None
        # all_res_list: BS X T X (2 * 3 + 32 * 6)
        num_seq = all_res_list.shape[0]

        # NOTE: all of the pos and rot are in global coordinate.
        pred_finger_6d = all_res_list

        normalized_wrist_finger_pos = self.extract_wrist_finger_data(
            data_dict["motion"][:num_seq]
        )
        normalized_gt_wrist_pos, normalized_gt_finger_6d = (
            normalized_wrist_finger_pos[..., :6],
            data_dict["local_rot"][:num_seq][..., -32 * 6 :],
        )

        # process wrist pos
        # pred_wrist_pos = self.ds.de_normalize_jpos_min_max_hand_foot(pred_wrist_pos, hand_only=True) # BS X T X 2 X 3

        gt_wrist_pos = self.ds.de_normalize_jpos_min_max_hand_foot(
            normalized_gt_wrist_pos, hand_only=True
        )  # BS X T X 2 X 3
        gt_wrist_pos = gt_wrist_pos.reshape(num_seq, -1, 2, 3)

        # process finger rot
        pred_finger_6d = pred_finger_6d.reshape(num_seq, -1, 30, 6)  # BS X T X 30 X 6
        pred_finger_mat = transforms.rotation_6d_to_matrix(
            pred_finger_6d
        )  # BS X T X 30 X 3 X 3

        gt_finger_6d = normalized_gt_finger_6d.reshape(
            num_seq, -1, 32, 6
        )  # BS X T X 32 X 6
        gt_finger_mat = transforms.rotation_6d_to_matrix(
            gt_finger_6d
        )  # BS X T X 32 X 3 X 3

        pred_finger_mat = torch.cat(
            (gt_finger_mat[:, :, :2].to(pred_finger_mat.device), pred_finger_mat), dim=2
        )  # BS X T X 32 X 3 X 3

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)  # BS X 3

        seq_len = data_dict["seq_len"].detach().cpu().numpy()  # BS

        gt_wholebody_6d = data_dict["local_rot"][:num_seq].reshape(
            num_seq, -1, JOINT_NUM, 6
        )  # BS X T X 52 X 6
        gt_wholebody_mat = transforms.rotation_6d_to_matrix(
            gt_wholebody_6d
        )  # BS X T X 52 X 3 X 3

        mesh_save_folders = []
        for idx in tqdm(range(num_seq), leave=False, desc="Generating vis results..."):
            # NOTE: here all the wrist and finger rot are in global coordinate, after IK, wrist is global and finger are local
            # so we need to set all body joint rot to zero, so that the wrist rot can be seen as local in FK
            curr_local_rot_mat = pred_finger_mat[idx]  # T X 32 X 3 X 3
            # curr_local_rot_mat = quat_ik_hand(curr_global_rot_mat, self.parents_handonly) # T X 32 X 3 X 3
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )  # T X 32 X 3
            # curr_global_root_jpos = torch.zeros((curr_local_rot_aa_rep.shape[0], 3)) # T X 3

            wholebody_local_rot_mat = gt_wholebody_mat[idx]
            # wholebody_local_rot_mat = quat_ik_wholebody(wholebody_global_rot_mat, self.parents_wholebody) # T X 52 X 3 X 3
            wholebody_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                wholebody_local_rot_mat
            )  # T X 52 X 3

            curr_local_rot_aa_rep_gt = wholebody_local_rot_aa_rep
            curr_global_root_jpos_gt = data_dict["ori_motion"][idx][..., :3]

            curr_local_rot_aa_rep = torch.cat(
                (
                    curr_local_rot_aa_rep_gt[:, :22].to(curr_local_rot_aa_rep.device),
                    curr_local_rot_aa_rep[:, 2:],
                ),
                dim=1,
            )

            if selected_seq_idx is None:
                curr_trans2joint = trans2joint[idx : idx + 1].clone()
            else:
                curr_trans2joint = trans2joint[
                    selected_seq_idx : selected_seq_idx + 1
                ].clone()

            # root_trans = curr_global_root_jpos.to(curr_local_rot_aa_rep.device) + curr_trans2joint.to(curr_local_rot_aa_rep.device) # T X 3
            root_trans_gt = curr_global_root_jpos_gt.to(
                curr_local_rot_aa_rep.device
            ) + curr_trans2joint.to(curr_local_rot_aa_rep.device)  # T X 3

            # Generate global joint position
            bs = 1
            if selected_seq_idx is None:
                gender = data_dict["gender"][idx]
                curr_obj_rot_mat = data_dict["obj_rot_mat"][idx]
                curr_obj_trans = data_dict["obj_trans"][idx]
                curr_obj_scale = data_dict["obj_scale"][idx]
                object_mesh_path = data_dict["object_mesh_path"][idx]
                vtemp_path = data_dict["vtemp_path"][idx]
            else:
                gender = data_dict["gender"][selected_seq_idx]
                curr_obj_rot_mat = data_dict["obj_rot_mat"][selected_seq_idx]
                curr_obj_trans = data_dict["obj_trans"][selected_seq_idx]
                curr_obj_scale = data_dict["obj_scale"][selected_seq_idx]
                object_mesh_path = data_dict["object_mesh_path"][selected_seq_idx]
                vtemp_path = data_dict["vtemp_path"][selected_seq_idx]
            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model_new(
                root_trans_gt, curr_local_rot_aa_rep, gender, vtemp_path
            )
            if vis_gt:
                mesh_jnts_gt, mesh_verts_gt, mesh_faces_gt = run_smplx_model_new(
                    root_trans_gt, curr_local_rot_aa_rep_gt, gender, vtemp_path
                )
            # change hand vertexs to wrist coordinate
            # mesh_verts: [1, 120, 10475, 3] BS X T X Nv X 3
            lhand_idx, rhand_idx = self.ds.lhand_idx, self.ds.rhand_idx
            lhand_pos = gt_wrist_pos[idx : idx + 1, :, 0:1].cpu()  # T X 3
            rhand_pos = gt_wrist_pos[idx : idx + 1, :, 1:2].cpu()  # T X 3
            mesh_verts[:, :, self.lhand_verts.cpu()] = (
                mesh_verts[:, :, self.lhand_verts.cpu()]
                - mesh_jnts[:, :, None, lhand_idx]
                + lhand_pos
            )  # rot is corrent, just need to adjust the pos
            mesh_verts[:, :, self.rhand_verts.cpu()] = (
                mesh_verts[:, :, self.rhand_verts.cpu()]
                - mesh_jnts[:, :, None, rhand_idx]
                + rhand_pos
            )  # rot is corrent, just need to adjust the pos

            # Get object verts
            obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(
                object_mesh_path,
                curr_obj_scale.detach().cpu().numpy(),
                curr_obj_trans.detach().cpu().numpy(),
                curr_obj_rot_mat.detach().cpu().numpy(),
            )

            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(
                    self.vis_folder, "blender_mesh_vis", str(step)
                )
            else:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))

            if True:
                if not os.path.exists(dest_mesh_vis_folder):
                    os.makedirs(dest_mesh_vis_folder)

                # if vis_gt:
                mesh_save_folder_gt = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                if os.path.exists(mesh_save_folder_gt):
                    import shutil

                    shutil.rmtree(mesh_save_folder_gt)
                os.makedirs(mesh_save_folder_gt)

                # else:
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )

                mesh_save_folders.append(mesh_save_folder)

                if selected_seq_idx is None:
                    actual_len = seq_len[idx]
                else:
                    actual_len = seq_len[selected_seq_idx]

                hand_vertice_idx = np.concatenate(
                    (self.lhand_verts.cpu().numpy(), self.rhand_verts.cpu().numpy()),
                    axis=0,
                )
                hand_mesh_verts, hand_mesh_faces = get_sub_mesh(
                    mesh_verts.detach().cpu().numpy()[0][:actual_len],
                    mesh_faces.detach().cpu().numpy(),
                    hand_vertice_idx,
                )
                save_verts_faces_to_mesh_file_w_object(
                    hand_mesh_verts,
                    hand_mesh_faces,
                    obj_mesh_verts.detach().cpu().numpy()[:actual_len],
                    obj_mesh_faces,
                    mesh_save_folder,
                )
                if vis_gt:
                    hand_mesh_verts_gt, hand_mesh_faces_gt = get_sub_mesh(
                        mesh_verts_gt.detach().cpu().numpy()[0][:actual_len],
                        mesh_faces_gt.detach().cpu().numpy(),
                        hand_vertice_idx,
                    )
                    save_verts_faces_to_mesh_file_w_object(
                        hand_mesh_verts_gt,
                        hand_mesh_faces_gt,
                        obj_mesh_verts.detach().cpu().numpy()[:actual_len],
                        obj_mesh_faces,
                        mesh_save_folder_gt,
                    )
        return mesh_save_folders


class OMOMOTrainer(Trainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/omomo_data/processed_omomo"
        # Define dataset
        train_dataset = OMOMODataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = OMOMODataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )  # NOTE: set shuffle to false, better compare
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )

    def extract_wrist_finger_data(self, data_input):
        # data_input: BS X T X D (24*3+22*6)
        lhand_idx = self.ds.lhand_idx
        rhand_idx = self.ds.rhand_idx
        wrist_pos_input = data_input[
            :, :, lhand_idx * 3 : lhand_idx * 3 + 6
        ]  # BS X T X D (2*3)
        rot_input = data_input[
            :, :, 24 * 3 + lhand_idx * 6 : 24 * 3 + lhand_idx * 6 + 12
        ]  # BS X T X D (2*6)
        data_input = torch.cat((wrist_pos_input, rot_input), dim=-1)
        # BS X T X D (2*3+2*6)

        return data_input

    def gen_vis_res(
        self,
        all_res_list,
        data_dict,
        step,
        vis_gt=False,
        vis_tag=None,
        for_quant_eval=False,
        selected_seq_idx=None,
    ):
        # all_res_list: BS X T X (2 * 3 + 32 * 6)
        num_seq = all_res_list.shape[0]

        # NOTE: all of the pos and rot are in global coordinate.
        pred_finger_6d = all_res_list
        pred_finger_6d = pred_finger_6d.reshape(num_seq, -1, 30, 6)  # BS X T X 30 X 6
        pred_finger_mat = transforms.rotation_6d_to_matrix(
            pred_finger_6d
        )  # BS X T X 30 X 3 X 3

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)  # BS X 3

        seq_len = data_dict["seq_len"].detach().cpu().numpy()  # BS

        gt_wholebody_6d = data_dict["motion"][:num_seq][..., 24 * 3 :].reshape(
            num_seq, -1, 22, 6
        )  # BS X T X 22 X 6
        gt_wholebody_mat = transforms.rotation_6d_to_matrix(
            gt_wholebody_6d
        )  # BS X T X 22 X 3 X 3
        gt_wholebody_mat = torch.cat(
            (
                gt_wholebody_mat,
                torch.eye(3)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(
                    gt_wholebody_mat.shape[0], gt_wholebody_mat.shape[1], 30, -1, -1
                )
                .to(gt_wholebody_mat.device),
            ),
            dim=2,
        )  # BS X T X 52 X 3 X 3

        seq_names = data_dict["seq_name"]  # BS

        normalized_wrist_finger_pos = self.extract_wrist_finger_data(
            data_dict["motion"][:num_seq]
        )
        normalized_gt_wrist_pos, gt_wrist_6d = (
            normalized_wrist_finger_pos[..., :6],
            normalized_wrist_finger_pos[..., 6:],
        )

        gt_wrist_pos = self.ds.de_normalize_jpos_min_max_hand_foot(
            normalized_gt_wrist_pos, hand_only=True
        )  # BS X T X 2 X 3
        gt_wrist_pos = gt_wrist_pos.reshape(num_seq, -1, 2, 3)

        mesh_save_folders = []
        for idx in tqdm(range(num_seq)):
            object_name = seq_names[idx].split("_")[1]
            obj_scale = data_dict["obj_scale"][idx].detach().cpu().numpy()
            obj_trans = data_dict["obj_trans"][idx].detach().cpu().numpy()
            obj_rot = data_dict["obj_rot_mat"][idx].detach().cpu().numpy()
            if object_name in ["mop", "vacuum"]:
                obj_bottom_scale = (
                    data_dict["obj_bottom_scale"][idx].detach().cpu().numpy()
                )
                obj_bottom_trans = (
                    data_dict["obj_bottom_trans"][idx].detach().cpu().numpy()
                )
                obj_bottom_rot = (
                    data_dict["obj_bottom_rot_mat"][idx].detach().cpu().numpy()
                )
            else:
                obj_bottom_scale = None
                obj_bottom_trans = None
                obj_bottom_rot = None

            curr_local_rot_mat = pred_finger_mat[idx]  # T X 30 X 3 X 3
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )  # T X 30 X 3

            wholebody_global_rot_mat = gt_wholebody_mat[idx]
            wholebody_local_rot_mat = quat_ik_wholebody(
                wholebody_global_rot_mat, self.parents_wholebody
            )  # T X 52 X 3 X 3
            wholebody_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                wholebody_local_rot_mat
            )  # T X 52 X 3

            curr_local_rot_aa_rep_gt = wholebody_local_rot_aa_rep
            curr_global_root_jpos_gt = data_dict["ori_motion"][idx][..., :3]

            curr_local_rot_aa_rep = torch.cat(
                (
                    curr_local_rot_aa_rep_gt[:, :22].to(curr_local_rot_aa_rep.device),
                    curr_local_rot_aa_rep,
                ),
                dim=1,
            )

            curr_trans2joint = trans2joint[idx : idx + 1].clone()

            # root_trans = curr_global_root_jpos.to(curr_local_rot_aa_rep.device) + curr_trans2joint.to(curr_local_rot_aa_rep.device) # T X 3
            root_trans_gt = curr_global_root_jpos_gt.to(
                curr_local_rot_aa_rep.device
            ) + curr_trans2joint.to(curr_local_rot_aa_rep.device)  # T X 3

            # Generate global joint position
            gender = "male"
            vtemp_path = "./data/grab_data/tools/subject_meshes/male/s2.ply"  # now use this template for vis omomo

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model_new(
                root_trans_gt, curr_local_rot_aa_rep, gender, vtemp_path
            )

            # change hand vertexs to wrist coordinate
            # mesh_verts: [1, 120, 10475, 3] BS X T X Nv X 3
            lhand_idx, rhand_idx = self.ds.lhand_idx, self.ds.rhand_idx
            lhand_pos = gt_wrist_pos[idx : idx + 1, :, 0:1].cpu()  # 1 X T X 1 X 3
            rhand_pos = gt_wrist_pos[idx : idx + 1, :, 1:2].cpu()  # 1 X T X 1 X 3
            mesh_verts[:, :, self.lhand_verts.cpu()] = (
                mesh_verts[:, :, self.lhand_verts.cpu()]
                - mesh_jnts[:, :, None, lhand_idx]
                + lhand_pos
            )  # rot is corrent, just need to adjust the pos
            mesh_verts[:, :, self.rhand_verts.cpu()] = (
                mesh_verts[:, :, self.rhand_verts.cpu()]
                - mesh_jnts[:, :, None, rhand_idx]
                + rhand_pos
            )  # rot is corrent, just need to adjust the pos

            # Get object verts
            obj_mesh_verts, obj_mesh_faces = self.ds.load_object_geometry(
                object_name,
                obj_scale,
                obj_trans,
                obj_rot,
                obj_bottom_scale,
                obj_bottom_trans,
                obj_bottom_rot,
            )
            # blend wrist motion
            # from scipy.spatial.transform import Rotation as R
            # start_frame = 25 # 25
            # end_frame = 96 # 96

            # def calc_wrist():
            #     relative_wrist_pos = torch.tensor([ 0.4716, -0.1163,  0.3127], device=gt_wrist_6d.device, dtype=gt_wrist_6d.dtype)
            #     relative_wrist_ori = transforms.axis_angle_to_matrix(torch.tensor([ 0.4725, -0.7605, -0.0407], device=gt_wrist_6d.device, dtype=gt_wrist_6d.dtype))
            #     relative_wrist_pos = relative_wrist_pos.cpu().numpy()
            #     relative_wrist_ori = relative_wrist_ori.cpu().numpy()
            #     com = obj_mesh_verts[start_frame:end_frame].mean(axis=1)

            #     new_wrist_pos = R.from_matrix(obj_rot[start_frame:end_frame]).apply(relative_wrist_pos) + com.cpu().numpy()
            #     new_wrist_ori = (R.from_matrix(obj_rot[start_frame:end_frame]) * R.from_matrix(relative_wrist_ori)).as_matrix()
            #     return new_wrist_pos, new_wrist_ori
            # new_wrist_pos, new_wrist_ori = calc_wrist() # (end_frame-start_frame) X 3, (end_frame-start_frame) X 3 X 3

            # def move_wrist(mesh_verts, old_wrist_pos, new_wrist_pos, old_wrist_ori, new_wrist_ori):
            #     from scipy.spatial.transform import Rotation as R

            #     verts = mesh_verts.cpu().numpy()
            #     old_wrist_pos_ = old_wrist_pos.cpu().numpy()
            #     old_wrist_ori_ = old_wrist_ori.cpu().numpy()
            #     new_wrist_pos_ = new_wrist_pos[None, :, None]
            #     verts -= old_wrist_pos_
            #     for i in range(verts.shape[-2]):
            #         verts[0, :, i] = R.from_matrix(new_wrist_ori).apply(R.from_matrix(old_wrist_ori_).inv().apply(verts[0, :, i]))
            #     verts += new_wrist_pos_
            #     return torch.from_numpy(verts).to(mesh_verts.device)
            # old_wrist_ori = wholebody_global_rot_mat[start_frame:end_frame, 21]
            # mesh_verts[:, start_frame:end_frame, self.rhand_verts.cpu()] = move_wrist(mesh_verts[:, start_frame:end_frame, self.rhand_verts.cpu()], rhand_pos[:, start_frame:end_frame], new_wrist_pos, old_wrist_ori, new_wrist_ori)

            # from manip.inertialize.spring import decay_spring_damper_exact_cubic

            # def interpolate_wrist(src_frame, wrist_pos, wrist_ori, new_wrist_pos, new_wrist_ori):
            #     assert src_frame > 0
            #     prev_src_frame = src_frame - 1
            #     dt = 1 / 30.0
            #     ##### pos #####
            #     dst_pos = new_wrist_pos[0]
            #     dst_pos_next = new_wrist_pos[1]
            #     dst_vel = (dst_pos_next - dst_pos) / dt

            #     src_pos = wrist_pos[prev_src_frame]
            #     src_pos_next = wrist_pos[src_frame]
            #     src_vel = (src_pos_next - src_pos) / dt

            #     diff_pos = src_pos - dst_pos
            #     diff_vel = src_vel - dst_vel

            #     ##### rot #####
            #     dst_ori = transforms.matrix_to_quaternion(new_wrist_ori[0])
            #     dst_ori_next = transforms.matrix_to_quaternion(new_wrist_ori[1])
            #     dst_ang_vel = transforms.quaternion_to_axis_angle(
            #         transforms.quaternion_multiply(
            #             dst_ori_next,
            #             transforms.quaternion_invert(dst_ori)
            #         )
            #     ) / dt

            #     src_ori = transforms.matrix_to_quaternion(wrist_ori[prev_src_frame])
            #     src_ori_next = transforms.matrix_to_quaternion(wrist_ori[src_frame])
            #     src_ang_vel = transforms.quaternion_to_axis_angle(
            #         transforms.quaternion_multiply(
            #             src_ori_next,
            #             transforms.quaternion_invert(src_ori)
            #         )
            #     ) / dt

            #     diff_ori = transforms.quaternion_to_axis_angle(
            #         transforms.quaternion_multiply(
            #             src_ori,
            #             transforms.quaternion_invert(dst_ori)
            #         )
            #     )
            #     diff_ang_vel = src_ang_vel - dst_ang_vel

            #     ##### interpolate #####
            #     for i in reversed(range(src_frame)):
            #         offset = decay_spring_damper_exact_cubic(-diff_pos, diff_vel, 0.5, (src_frame - 1 - i) * dt)
            #         wrist_pos[i] += offset

            #         offset = decay_spring_damper_exact_cubic(-diff_ori, diff_ang_vel, 0.5, (src_frame - 1 - i) * dt)
            #         wrist_ori[i] = transforms.quaternion_to_matrix(
            #             transforms.quaternion_multiply(
            #                 transforms.axis_angle_to_quaternion(offset),
            #                 transforms.matrix_to_quaternion(wrist_ori[i])
            #             )
            #         )
            #     return wrist_pos, wrist_ori
            # if start_frame > 0:
            #     wrist_pos = rhand_pos[0, :, 0].clone() # T X 3
            #     wrsit_ori = wholebody_global_rot_mat[:, 21].clone() # T X 3 X 3
            #     wrist_pos, wrist_ori = interpolate_wrist(start_frame, wrist_pos, wrsit_ori, torch.from_numpy(new_wrist_pos), torch.from_numpy(new_wrist_ori))
            #     old_wrist_ori = wholebody_global_rot_mat[:start_frame, 21]
            #     mesh_verts[:, :start_frame, self.rhand_verts.cpu()] = move_wrist(mesh_verts[:, :start_frame, self.rhand_verts.cpu()], rhand_pos[:, :start_frame], wrist_pos[:start_frame].cpu().numpy(), old_wrist_ori, wrist_ori[:start_frame])

            # NOTE: generate grasping wrist
            # from scipy.spatial.transform import Rotation as R
            # def print_grasp_pose(frame):
            #     com = obj_mesh_verts[frame].mean(axis=0)
            #     gt_wrist_rot_mat = transforms.rotation_6d_to_matrix(gt_wrist_6d[idx, frame, 6:]).cpu().numpy()
            #     hands_pos_obj_frame = R.from_matrix(obj_rot[frame]).inv().apply(rhand_pos[0, frame, 0] - com)
            #     hands_ori_obj_frame = (R.from_matrix(obj_rot[frame]).inv() * R.from_matrix(gt_wrist_rot_mat)).as_matrix()
            #     print(hands_pos_obj_frame)
            #     print(hands_ori_obj_frame)
            # print_grasp_pose(0)
            # import pdb
            # pdb.set_trace()
            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(
                    self.vis_folder, "blender_mesh_vis", str(step)
                )
            else:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))

            if True:
                if not os.path.exists(dest_mesh_vis_folder):
                    os.makedirs(dest_mesh_vis_folder)

                # if vis_gt:
                mesh_save_folder_gt = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                if os.path.exists(mesh_save_folder_gt):
                    import shutil

                    shutil.rmtree(mesh_save_folder_gt)
                os.makedirs(mesh_save_folder_gt)

                # else:
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )

                mesh_save_folders.append(mesh_save_folder)

                actual_len = seq_len[idx]

                hand_vertice_idx = np.concatenate(
                    (self.lhand_verts.cpu().numpy(), self.rhand_verts.cpu().numpy()),
                    axis=0,
                )
                hand_mesh_verts, hand_mesh_faces = get_sub_mesh(
                    mesh_verts.detach().cpu().numpy()[0][:actual_len],
                    mesh_faces.detach().cpu().numpy(),
                    hand_vertice_idx,
                )
                # print("save mesh to,", mesh_save_folder)
                save_verts_faces_to_mesh_file_w_object(
                    hand_mesh_verts,
                    hand_mesh_faces,
                    obj_mesh_verts.detach().cpu().numpy()[:actual_len],
                    obj_mesh_faces,
                    mesh_save_folder,
                )

        return mesh_save_folders


class OMOMOAmbientSensorTrainer(OMOMOTrainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/omomo_data/processed_omomo"
        # Define dataset
        train_dataset = OMOMOAmbientSensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = OMOMOAmbientSensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )


class OMOMOProximitySensorTrainer(OMOMOTrainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/omomo_data/processed_omomo"
        # Define dataset
        train_dataset = OMOMOProximitySensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = OMOMOProximitySensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )


class OMOMOBothSensorTrainer(OMOMOTrainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/omomo_data/processed_omomo"
        # Define dataset
        train_dataset = OMOMOBothSensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
            load_ds=self.load_ds,
        )
        val_dataset = OMOMOBothSensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
            load_ds=self.load_ds,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        if self.load_ds:
            self.dl = cycle(
                data.DataLoader(
                    self.ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=2,
                )
            )
            self.val_dl = cycle(
                data.DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=2,
                )
            )


class AmbientSensorTrainer(Trainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/grab_data/processed_omomo"
        # Define dataset
        train_dataset = GrabAmbientSensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = GrabAmbientSensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )


class ProximitySensorTrainer(Trainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/grab_data/processed_omomo"
        # Define dataset
        train_dataset = GrabProximitySensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = GrabProximitySensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )


class BothSensorTrainer(Trainer):
    def prep_dataloader(self, window_size):
        self.data_root_folder = "./data/grab_data/processed_omomo"
        # Define dataset
        train_dataset = GrabBothSensorDataset(
            train=True,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )
        val_dataset = GrabBothSensorDataset(
            train=False,
            window=window_size,
            use_window_bps=self.use_window_bps,
            use_object_splits=self.use_object_split,
            use_joints24=self.use_joints24,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=2,
            )
        )
        self.val_dl = cycle(
            data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
            )
        )


def build_trainer(opt, wdir, device):
    # Define model
    repr_dim = 30 * 6  # finger
    loss_type = "l1"

    diffusion_model = CondGaussianDiffusionBothSensorNew(
        opt,
        d_feats=repr_dim,
        d_model=opt.d_model,
        n_dec_layers=opt.n_dec_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        max_timesteps=opt.window + 1,
        out_dim=repr_dim,
        timesteps=1000,
        objective="pred_x0",
        loss_type=loss_type,
        batch_size=opt.batch_size,
        bps_in_dim=2048,
        second_bps_in_dim=200,
    )

    diffusion_model.to(device)

    common_params = {
        "opt": opt,
        "diffusion_model": diffusion_model,
        "train_batch_size": opt.batch_size,  # 32
        "train_lr": opt.learning_rate,  # 1e-4
        "train_num_steps": 400000,  # 700000, total training steps
        "gradient_accumulate_every": 2,  # gradient accumulation steps
        "ema_decay": 0.995,  # exponential moving average decay
        "amp": True,  # turn on mixed precision
        "results_folder": str(wdir),
        "use_wandb": opt.use_wandb,
    }
    if opt.use_omomo:
        trainer = OMOMOBothSensorTrainer(**common_params)
    else:
        trainer = BothSensorTrainer(**common_params)
    return trainer


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model
    trainer = build_trainer(opt, wdir, device)

    trainer.train()

    torch.cuda.empty_cache()


def run_sample(opt, device, run_pipeline=False):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"

    # Define model
    trainer = build_trainer(opt, wdir, device)

    trainer.cond_sample_res()

    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default="runs/train",
        help="output folder for weights and visualizations",
    )
    parser.add_argument("--wandb_pj_name", type=str, default="", help="wandb_proj_name")
    parser.add_argument("--entity", default="zhenkirito123", help="W&B entity")
    parser.add_argument("--exp_name", default="", help="save to project/name")
    parser.add_argument("--device", default="0", help="cuda device")

    parser.add_argument("--window", type=int, default=120, help="horizon")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="generator_learning_rate"
    )

    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint")

    parser.add_argument(
        "--n_dec_layers", type=int, default=4, help="the number of decoder layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=4, help="the number of heads in self-attention"
    )
    parser.add_argument(
        "--d_k", type=int, default=256, help="the dimension of keys in transformer"
    )
    parser.add_argument(
        "--d_v", type=int, default=256, help="the dimension of values in transformer"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="the dimension of intermediate representation in transformer",
    )

    parser.add_argument("--generate_refine_dataset", action="store_true")
    parser.add_argument("--generate_quant_eval", action="store_true")
    parser.add_argument("--generate_vis_eval", action="store_true")
    parser.add_argument("--milestone", type=str, default="best")
    # For testing sampled results
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    # For loss type
    parser.add_argument("--use_l2_loss", action="store_true")

    # For training diffusion model without condition
    parser.add_argument("--remove_condition", action="store_true")

    # For normalizing condition
    parser.add_argument("--normalize_condition", action="store_true")

    # For adding first human pose as condition (shared by current trainer and FullBody trainer)
    parser.add_argument("--add_start_human_pose", action="store_true")

    # FullBody trainer config: For adding hand pose (translation+rotation) as condition
    parser.add_argument("--add_hand_pose", action="store_true")

    # FullBody trainer config: For adding hand pose (translation only) as condition
    parser.add_argument("--add_hand_trans_only", action="store_true")

    # FullBody trainer config: For adding hand and foot trans as condition
    parser.add_argument("--add_hand_foot_trans", action="store_true")

    # For canonicalizing the first pose's facing direction
    parser.add_argument("--cano_init_pose", action="store_true")

    # For predicting hand position only
    parser.add_argument("--pred_hand_jpos_only_from_obj", action="store_true")

    # For predicting hand position only
    parser.add_argument("--pred_palm_jpos_from_obj", action="store_true")

    # For predicting hand position only
    parser.add_argument("--pred_hand_jpos_and_rot_from_obj", action="store_true")

    # For running the whole pipeline.
    parser.add_argument("--run_whole_pipeline", action="store_true")

    parser.add_argument("--add_object_bps", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument("--use_joints24", action="store_true")

    parser.add_argument("--add_palm_jpos_only", action="store_true")

    parser.add_argument("--use_blender_data", action="store_true")

    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--use_omomo", action="store_true")
    parser.add_argument("--omomo_obj", type=str, default="")

    parser.add_argument("--train_ambient_sensor", action="store_true")
    parser.add_argument("--train_proximity_sensor", action="store_true")
    parser.add_argument("--train_both_sensor", action="store_true")

    # parser.add_argument("--train_contact_label", action="store_true")

    parser.add_argument("--wrist_obj_traj_condition", action="store_true")
    parser.add_argument("--ambient_sensor_condition", action="store_true")
    parser.add_argument("--proximity_sensor_condition", action="store_true")
    parser.add_argument("--ref_pose_condition", action="store_true")
    parser.add_argument("--contact_label_condition", action="store_true")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split("/")[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
