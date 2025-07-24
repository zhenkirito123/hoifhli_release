import argparse
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import clip
import numpy as np
import pytorch3d.transforms as transforms
import torch
import torch.nn.functional as F
import trimesh
import wandb
import yaml
from ema_pytorch import EMA
from scipy.signal import medfilt
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm

from manip.data.cano_traj_dataset import (
    CanoObjectTrajDataset,
    get_smpl_parents,
    quat_ik_torch,
)
from manip.model.transformer_object_motion_cond_diffusion import (
    ObjectCondGaussianDiffusion,
)
from manip.utils.model_utils import apply_rotation_to_data
from manip.utils.trainer_utils import (
    canonicalize_first_human_and_waypoints,
    cycle,
    find_contact_frames,
    load_palm_vertex_ids,
    run_smplx_model,
    smooth_res,
)
from manip.vis.blender_vis_mesh_motion import (
    save_verts_faces_to_mesh_file_w_object,
)

torch.manual_seed(1)
random.seed(1)


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
        save_and_sample_every=40000,
        results_folder="./results",
        vis_folder=None,
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
        if vis_folder is not None:
            self.vis_folder = vis_folder

        self.opt = opt

        self.window = opt.window

        self.return_diff_level_res = self.opt.return_diff_level_res

        self.add_contact_label = self.opt.add_contact_label

        self.add_wrist_relative = self.opt.add_wrist_relative
        self.add_object_static = self.opt.add_object_static
        self.add_interaction_root_xy_ori = self.opt.add_interaction_root_xy_ori
        self.add_interaction_feet_contact = self.opt.add_interaction_feet_contact

        self.add_language_condition = True

        self.use_first_frame_bps = self.opt.use_first_frame_bps

        self.use_random_frame_bps = True

        self.use_object_keypoints = True

        self.add_semantic_contact_labels = True

        self.use_object_split = self.opt.use_object_split
        self.data_root_folder = self.opt.data_root_folder
        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict

        self.add_start_end_object_pos = self.opt.add_start_end_object_pos

        self.add_start_end_object_pos_rot = self.opt.add_start_end_object_pos_rot

        self.add_start_end_object_pos_xy = self.opt.add_start_end_object_pos_xy

        self.add_waypoints_xy = True

        self.remove_target_z = self.opt.remove_target_z

        self.pred_human_motion = True

        self.use_random_waypoints = self.opt.use_random_waypoints

        self.input_first_human_pose = True

        self.input_full_human_pose = self.opt.input_full_human_pose

        self.use_guidance_in_denoising = self.opt.use_guidance_in_denoising

        self.use_optimization_in_denoising = self.opt.use_optimization_in_denoising

        self.add_rest_human_skeleton = self.opt.add_rest_human_skeleton

        self.loss_w_feet = (
            self.opt.loss_w_feet if hasattr(self.opt, "loss_w_feet") else False
        )
        self.loss_w_fk = self.opt.loss_w_fk if hasattr(self.opt, "loss_w_fk") else False
        self.loss_w_obj_pts = (
            self.opt.loss_w_obj_pts if hasattr(self.opt, "loss_w_obj_pts") else False
        )
        self.loss_w_obj_pts_in_hand = (
            self.opt.loss_w_obj_pts_in_hand
            if hasattr(self.opt, "loss_w_obj_pts_in_hand")
            else False
        )
        self.loss_w_obj_vel = (
            self.opt.loss_w_obj_vel if hasattr(self.opt, "loss_w_obj_vel") else False
        )

        if self.add_language_condition:
            clip_version = "ViT-B/32"
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.use_long_planned_path = self.opt.use_long_planned_path

        (
            self.hand_vertex_idxs,
            self.left_hand_vertex_idxs,
            self.right_hand_vertex_idxs,
        ) = self.load_hand_vertex_ids()
        self.left_palm_vertex_idxs, self.right_palm_vertex_idxs = load_palm_vertex_ids()

    def load_hand_vertex_ids(self):
        data = pickle.load(
            open(
                "./data/smpl_all_models/MANO_SMPLX_vertex_ids.pkl",
                "rb",
            )
        )

        left_hand_vids = data["left_hand"]
        right_hand_vids = data["right_hand"]

        hand_vids = np.concatenate((left_hand_vids, right_hand_vids), axis=0)

        return hand_vids, left_hand_vids, right_hand_vids

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cuda", jit=False
        )  # Must set jit=False for training
        # clip.model.convert_weights(
        #     clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.clip_model.parameters()).device
        max_text_len = 30  # Specific hardcoding for the current dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate

        return self.clip_model.encode_text(texts).float().detach()  # BS X 512

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = CanoObjectTrajDataset(
            train=True,
            data_root_folder=self.data_root_folder,
            window=window_size,
            use_object_splits=self.use_object_split,
            input_language_condition=self.add_language_condition,
            use_first_frame_bps=self.use_first_frame_bps,
            use_random_frame_bps=self.use_random_frame_bps,
            use_object_keypoints=self.use_object_keypoints,
            load_ds=self.load_ds,
        )
        val_dataset = CanoObjectTrajDataset(
            train=False,
            data_root_folder=self.data_root_folder,
            window=window_size,
            use_object_splits=self.use_object_split,
            input_language_condition=self.add_language_condition,
            use_first_frame_bps=True,
            use_object_keypoints=self.use_object_keypoints,
            load_ds=self.load_ds,
        )

        self.ds = train_dataset
        self.val_ds = val_dataset
        if self.load_ds:
            self.dl = cycle(
                data.DataLoader(
                    self.ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4,
                )
            )
            self.val_dl = cycle(
                data.DataLoader(
                    self.val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=4,
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

    def prep_start_end_condition_mask(self, data, actual_seq_len):
        # data: BS X T X D (3+9)
        # actual_seq_len: BS
        tmp_mask = torch.arange(self.window).expand(data.shape[0], self.window) == (
            actual_seq_len[:, None].repeat(1, self.window) - 1
        )
        # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None]  # BS X T X 1

        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data).to(data.device)  # BS X T X D
        mask = mask * (~tmp_mask)  # Only the actual_seq_len frame is 0

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def prep_start_end_condition_mask_pos_only(self, data, actual_seq_len):
        # data: BS X T X D (3+9)
        # actual_seq_len: BS
        tmp_mask = torch.arange(self.window).expand(data.shape[0], self.window) == (
            actual_seq_len[:, None].repeat(1, self.window) - 1
        )
        # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None]  # BS X T X 1

        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data[:, :, :3]).to(data.device)  # BS X T X 3
        mask = mask * (~tmp_mask)  # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given.
        rotation_mask = torch.ones_like(data[:, :, 3:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1)

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def prep_start_end_condition_mask_pos_xy_only(self, data, actual_seq_len):
        # data: BS X T X D
        # actual_seq_len: BS
        tmp_mask = torch.arange(self.window).expand(data.shape[0], self.window) == (
            actual_seq_len[:, None].repeat(1, self.window) - 1
        )
        # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None]  # BS X T X 1

        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data[:, :, :2]).to(data.device)  # BS X T X 2
        mask = mask * (~tmp_mask)  # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given.
        # Also, add z mask, only the first frane's z is given.
        rotation_mask = torch.ones_like(data[:, :, 2:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1)

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def prep_mimic_A_star_path_condition_mask_pos_xy_only(self, data, actual_seq_len):
        # data: BS X T X D
        # actual_seq_len: BS
        tmp_mask = torch.arange(self.window).expand(data.shape[0], self.window) == (
            actual_seq_len[:, None].repeat(1, self.window) - 1
        )
        # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None]  # BS X T X 1
        tmp_mask = ~tmp_mask

        # Random sample a few timesteps as waypoints steps.
        if self.use_random_waypoints:
            num_waypoints_list = list(range(6))
            curr_num_waypoints = random.sample(num_waypoints_list, 1)[0]
            random_steps = random.sample(
                list(range(10, self.window - 10, 10)), curr_num_waypoints
            )  # a list such as [20, 60]
        else:
            # Use fixed number of waypoints.
            random_steps = [30 - 1, 60 - 1, 90 - 1]
        for selected_t in random_steps:
            if selected_t < self.window - 1:
                bs_selected_t = torch.from_numpy(np.asarray([selected_t]))  # 1
                bs_selected_t = bs_selected_t[None, :].repeat(
                    data.shape[0], self.window
                )  # BS X T

                curr_tmp_mask = torch.arange(self.window).expand(
                    data.shape[0], self.window
                ) == (bs_selected_t)
                # BS X max_timesteps
                curr_tmp_mask = curr_tmp_mask.to(data.device)[:, :, None]  # BS X T X 1

                tmp_mask = (~curr_tmp_mask) * tmp_mask

        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data[:, :, :2]).to(data.device)  # BS X T X 2
        mask = mask * tmp_mask  # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given.
        # Also, add z mask, only the first frane's z is given.
        rotation_mask = torch.ones_like(data[:, :, 2:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1)

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(
            data.device
        )  # BS X D

        return mask

    def train(self):
        init_step = self.step
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()
            # start_time = time.time()

            nan_exists = (
                False  # If met nan in loss or gradient, need to skip to next data.
            )

            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                # print("Load data takes:{0}".format(time.time()-start_time))
                human_data = data_dict["motion"].cuda()  # BS X T X (24*3 + 22*6)
                obj_data = data_dict["obj_motion"].cuda()  # BS X T X (3+9)

                obj_bps_data = (
                    data_dict["input_obj_bps"].cuda().reshape(-1, 1, 1024 * 3)
                )  # BS X 1 X 1024 X 3 -> BS X 1 X (1024*3)
                ori_data_cond = obj_bps_data  # BS X 1 X (1024*3)

                rest_human_offsets = data_dict[
                    "rest_human_offsets"
                ].cuda()  # BS X 24 X 3

                # Generate padding mask
                actual_seq_len = (
                    data_dict["seq_len"] + 1
                )  # BS, + 1 since we need additional timestep for noise level
                tmp_mask = torch.arange(self.window + 1).expand(
                    obj_data.shape[0], self.window + 1
                ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(obj_data.device)

                # Generating mask for object waypoints
                if self.add_start_end_object_pos_rot:
                    cond_mask = self.prep_start_end_condition_mask(
                        obj_data, data_dict["seq_len"]
                    )
                elif self.add_start_end_object_pos:
                    cond_mask = self.prep_start_end_condition_mask_pos_only(
                        obj_data, data_dict["seq_len"]
                    )
                elif self.add_start_end_object_pos_xy:
                    cond_mask = self.prep_start_end_condition_mask_pos_xy_only(
                        obj_data, data_dict["seq_len"]
                    )
                elif self.add_waypoints_xy:
                    if self.remove_target_z:
                        end_pos_cond_mask = (
                            self.prep_start_end_condition_mask_pos_xy_only(
                                obj_data, data_dict["seq_len"]
                            )
                        )
                    else:
                        end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(
                            obj_data, data_dict["seq_len"]
                        )

                    cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                        obj_data, data_dict["seq_len"]
                    )
                    cond_mask = end_pos_cond_mask * cond_mask
                else:
                    cond_mask = None

                # Generating mask for object static condition
                if self.add_object_static:
                    object_static_flag = data_dict[
                        "object_static_flag"
                    ].cuda()  # BS X T X 1
                    cond_mask[..., :3] *= 1.0 - object_static_flag

                    # randomly mask orientation
                    random_tensor = torch.rand_like(
                        object_static_flag[:, :1, :1]
                    )  # BS X 1 X 1
                    random_tensor[random_tensor > 0.7] = 1
                    random_tensor[random_tensor <= 0.3] = 0
                    object_static_flag_random = (
                        object_static_flag * random_tensor
                    )  # BS X T X 1, set some rows to be zero
                    cond_mask[..., 3:12] *= 1.0 - object_static_flag_random

                # Generating mask for human motion
                if self.pred_human_motion:
                    human_cond_mask = torch.ones_like(human_data).to(human_data.device)
                    if self.input_first_human_pose:
                        human_cond_mask[:, 0, :] = 0
                    elif self.input_full_human_pose:
                        human_cond_mask *= 0

                    # Generating mask for human root orientation condition
                    if self.add_interaction_root_xy_ori:
                        root_xy_ori = data_dict["root_traj_xy_ori"].cuda()
                        human_data = torch.cat(
                            (human_data, root_xy_ori), dim=-1
                        )  # BS X T X (24*3 + 22*6 + 6)

                        root_xy_ori_mask = torch.ones_like(root_xy_ori).to(
                            human_data.device
                        )
                        tmp_mask = root_xy_ori_mask[
                            :, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :
                        ]
                        random_tensor = torch.rand_like(tmp_mask)
                        random_tensor[random_tensor > 0.5] = 1
                        random_tensor[random_tensor <= 0.5] = 0
                        root_xy_ori_mask[:, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :] = (
                            random_tensor
                        )
                        human_cond_mask = torch.cat(
                            (human_cond_mask, root_xy_ori_mask), dim=-1
                        )

                    # Generating mask for wrist relative condition
                    if self.add_wrist_relative:
                        wrist_relative = data_dict[
                            "wrist_relative"
                        ].cuda()  # BS X T X 18
                        human_data = torch.cat(
                            (human_data, wrist_relative), dim=-1
                        )  # BS X T X (24*3 + 22*6 + 18)
                        wrist_relative_mask = torch.zeros_like(wrist_relative).to(
                            human_data.device
                        )
                        human_cond_mask = torch.cat(
                            (human_cond_mask, wrist_relative_mask), dim=-1
                        )

                    cond_mask = torch.cat(
                        (cond_mask, human_cond_mask), dim=-1
                    )  # BS X T X (3+9 + 24*3+22*6)

                if self.add_contact_label:
                    contact_labels = data_dict["contact_labels"].cuda()
                else:
                    contact_labels = None

                with autocast(enabled=self.amp):
                    if self.pred_human_motion:
                        if self.use_object_keypoints:
                            if self.add_semantic_contact_labels:
                                contact_data = data_dict[
                                    "contact_labels"
                                ].cuda()  # BS X T X 4
                            else:
                                contact_data = data_dict[
                                    "feet_contact"
                                ].cuda()  # BS X T X 4

                            data = torch.cat(
                                (obj_data, human_data, contact_data), dim=-1
                            )
                            cond_mask = torch.cat(
                                (
                                    cond_mask,
                                    torch.ones_like(contact_data).to(cond_mask.device),
                                ),
                                dim=-1,
                            )
                        else:
                            # data = human_data
                            data = torch.cat((obj_data, human_data), dim=-1)

                        if self.add_interaction_feet_contact:
                            feet_contact = data_dict["feet_contact"].cuda()
                            data = torch.cat((data, feet_contact), dim=-1)
                            cond_mask = torch.cat(
                                (
                                    cond_mask,
                                    torch.ones_like(feet_contact).to(cond_mask.device),
                                ),
                                dim=-1,
                            )

                        if self.add_language_condition:
                            text_anno_data = data_dict["text"]
                            language_input = self.encode_text(
                                text_anno_data
                            )  # BS X 512
                            language_input = language_input.to(data.device)
                            if self.use_object_keypoints:
                                loss_dict = self.model(
                                    data,
                                    ori_data_cond,
                                    cond_mask,
                                    padding_mask,
                                    language_input=language_input,
                                    contact_labels=contact_labels,
                                    rest_human_offsets=rest_human_offsets,
                                    ds=self.ds,
                                    data_dict=data_dict,
                                )
                            else:
                                loss_diffusion, loss_obj, loss_human = self.model(
                                    data,
                                    ori_data_cond,
                                    cond_mask,
                                    padding_mask,
                                    language_input=language_input,
                                    contact_labels=contact_labels,
                                    rest_human_offsets=rest_human_offsets,
                                    ds=self.ds,
                                    data_dict=data_dict,
                                )
                        else:
                            loss_diffusion = self.model(
                                data,
                                ori_data_cond,
                                cond_mask,
                                padding_mask,
                                contact_labels=contact_labels,
                                rest_human_offsets=rest_human_offsets,
                            )
                    else:
                        loss_diffusion = self.model(
                            obj_data, ori_data_cond, cond_mask, padding_mask
                        )

                    if self.use_object_keypoints:
                        loss_diffusion = loss_dict.get("loss", 0)
                        loss_obj = loss_dict.get("loss_object", 0)
                        loss_human = loss_dict.get("loss_human", 0)
                        loss_fk = loss_dict.get("loss_fk", 0)
                        loss_obj_pts = loss_dict.get("loss_obj_pts", 0)
                        loss_obj_pts_in_hand = loss_dict.get("loss_obj_pts_in_hand", 0)
                        loss_feet = loss_dict.get("loss_feet", 0)
                        loss_obj_vel = loss_dict.get("loss_obj_vel", 0)

                        loss = (
                            loss_diffusion
                            + self.loss_w_feet * loss_feet
                            + self.loss_w_fk * loss_fk
                            + self.loss_w_obj_pts * loss_obj_pts
                            + self.loss_w_obj_pts_in_hand * loss_obj_pts_in_hand
                            + self.loss_w_obj_vel * loss_obj_vel
                        )
                    else:
                        loss = loss_diffusion

                    # print("Model forward takes:{0}".format(time.time()-start_time))

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
                                torch.norm(p.grad.detach(), 2.0).to(obj_data.device)
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
                        if self.use_object_keypoints:
                            log_dict = {
                                "Train/Loss/Total Loss": float(loss),
                                "Train/Loss/Diffusion Loss": float(loss_diffusion),
                                "Train/Loss/Object Loss": float(loss_obj),
                                "Train/Loss/Human Loss": float(loss_human),
                                "Train/Loss/Feet Contact Loss": float(loss_feet),
                                "Train/Loss/FK Loss": float(loss_fk),
                                "Train/Loss/Object Pts Loss": float(loss_obj_pts),
                                "Train/Loss/Object Pts in Hand Loss": float(
                                    loss_obj_pts_in_hand
                                ),
                                "Train/Loss/Object Vel Loss": float(loss_obj_vel),
                            }
                        else:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                                "Train/Loss/Object Loss": loss_obj.item(),
                                "Train/Loss/Human Loss": loss_human.item(),
                            }
                        wandb.log(log_dict)

                    if idx % 20 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % float(loss))
                        print("Object Loss: %.4f" % float(loss_obj))
                        print("Human Loss: %.4f" % float(loss_human))
                        if self.use_object_keypoints:
                            print("Feet Contact Loss: %.4f" % float(loss_feet))
                            print("FK Loss: %.4f" % float(loss_fk))
                            print("Object Pts Loss: %.4f" % float(loss_obj_pts))
                            print(
                                "Object Pts in Hand Loss: %.4f"
                                % float(loss_obj_pts_in_hand)
                            )
                            print("Object Vel Loss: %.4f" % float(loss_obj_vel))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            # print("A complete step takes:{0}".format(time.time()-start_time))

            # Validation
            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_human_data = val_data_dict["motion"].cuda()
                    val_obj_data = val_data_dict["obj_motion"].cuda()

                    obj_bps_data = (
                        val_data_dict["input_obj_bps"].cuda().reshape(-1, 1, 1024 * 3)
                    )
                    ori_data_cond = obj_bps_data

                    rest_human_offsets = val_data_dict[
                        "rest_human_offsets"
                    ].cuda()  # BS X 24 X 3

                    if self.add_contact_label:
                        contact_labels = val_data_dict["contact_labels"].cuda()
                    else:
                        contact_labels = None

                    # Generate padding mask
                    actual_seq_len = (
                        val_data_dict["seq_len"] + 1
                    )  # BS, + 1 since we need additional timestep for noise level
                    tmp_mask = torch.arange(self.window + 1).expand(
                        val_obj_data.shape[0], self.window + 1
                    ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_obj_data.device)

                    # Generating mask for object waypoints
                    if self.add_start_end_object_pos_rot:
                        cond_mask = self.prep_start_end_condition_mask(
                            val_obj_data, val_data_dict["seq_len"]
                        )
                    elif self.add_start_end_object_pos:
                        cond_mask = self.prep_start_end_condition_mask_pos_only(
                            val_obj_data, val_data_dict["seq_len"]
                        )
                    elif self.add_start_end_object_pos_xy:
                        cond_mask = self.prep_start_end_condition_mask_pos_xy_only(
                            val_obj_data, val_data_dict["seq_len"]
                        )
                    elif self.add_waypoints_xy:
                        if self.remove_target_z:
                            end_pos_cond_mask = (
                                self.prep_start_end_condition_mask_pos_xy_only(
                                    val_obj_data, val_data_dict["seq_len"]
                                )
                            )
                        else:
                            end_pos_cond_mask = (
                                self.prep_start_end_condition_mask_pos_only(
                                    val_obj_data, val_data_dict["seq_len"]
                                )
                            )
                        cond_mask = (
                            self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                                val_obj_data, val_data_dict["seq_len"]
                            )
                        )
                        cond_mask = end_pos_cond_mask * cond_mask
                    else:
                        cond_mask = None

                    # Generating mask for object static condition
                    if self.add_object_static:
                        object_static_flag = val_data_dict[
                            "object_static_flag"
                        ].cuda()  # BS X T X 1
                        cond_mask[..., :3] *= 1.0 - object_static_flag

                        # randomly mask orientation
                        random_tensor = torch.rand_like(
                            object_static_flag[:, :1, :1]
                        )  # BS X 1 X 1
                        random_tensor[random_tensor > 0.7] = 1
                        random_tensor[random_tensor <= 0.3] = 0
                        object_static_flag_random = (
                            object_static_flag * random_tensor
                        )  # BS X T X 1, set some rows to be zero
                        cond_mask[..., 3:12] *= 1.0 - object_static_flag_random

                    # Generating mask for human motion
                    if self.pred_human_motion:
                        human_cond_mask = torch.ones_like(val_human_data).to(
                            val_human_data.device
                        )
                        if self.input_first_human_pose:
                            human_cond_mask[:, 0, :] = 0
                        elif self.input_full_human_pose:
                            human_cond_mask *= 0

                        # Generating mask for human root orientation condition
                        if self.add_interaction_root_xy_ori:
                            root_xy_ori = val_data_dict["root_traj_xy_ori"].cuda()
                            val_human_data = torch.cat(
                                (val_human_data, root_xy_ori), dim=-1
                            )  # BS X T X (24*3 + 22*6 + 6)
                            root_xy_ori_mask = torch.ones_like(root_xy_ori).to(
                                val_human_data.device
                            )
                            tmp_mask = root_xy_ori_mask[
                                :, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :
                            ]
                            random_tensor = torch.rand_like(tmp_mask)
                            random_tensor[random_tensor > 0.5] = 1
                            random_tensor[random_tensor <= 0.5] = 0
                            root_xy_ori_mask[
                                :, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :
                            ] = random_tensor
                            human_cond_mask = torch.cat(
                                (human_cond_mask, root_xy_ori_mask), dim=-1
                            )

                        # Generating mask for wrist relative condition
                        if self.add_wrist_relative:
                            wrist_relative = val_data_dict[
                                "wrist_relative"
                            ].cuda()  # BS X T X 18
                            val_human_data = torch.cat(
                                (val_human_data, wrist_relative), dim=-1
                            )  # BS X T X (24*3 + 22*6 + 18)
                            wrist_relative_mask = torch.zeros_like(wrist_relative).to(
                                val_human_data.device
                            )
                            human_cond_mask = torch.cat(
                                (human_cond_mask, wrist_relative_mask), dim=-1
                            )
                        cond_mask = torch.cat(
                            (cond_mask, human_cond_mask), dim=-1
                        )  # BS X T X (3+6+24*3+22*6)

                    # Get validation loss
                    if self.pred_human_motion:
                        if self.use_object_keypoints:
                            if self.add_semantic_contact_labels:
                                contact_data = val_data_dict[
                                    "contact_labels"
                                ].cuda()  # BS X T X 4
                            else:
                                contact_data = val_data_dict[
                                    "feet_contact"
                                ].cuda()  # BS X T X 4

                            data = torch.cat(
                                (val_obj_data, val_human_data, contact_data), dim=-1
                            )
                            cond_mask = torch.cat(
                                (
                                    cond_mask,
                                    torch.ones_like(contact_data).to(cond_mask.device),
                                ),
                                dim=-1,
                            )
                        else:
                            data = torch.cat((val_obj_data, val_human_data), dim=-1)

                        if self.add_interaction_feet_contact:
                            feet_contact = val_data_dict["feet_contact"].cuda()
                            data = torch.cat((data, feet_contact), dim=-1)
                            cond_mask = torch.cat(
                                (
                                    cond_mask,
                                    torch.ones_like(feet_contact).to(cond_mask.device),
                                ),
                                dim=-1,
                            )

                        if self.add_language_condition:
                            text_anno_data = val_data_dict["text"]
                            language_input = self.encode_text(
                                text_anno_data
                            )  # BS X 512
                            language_input = language_input.to(data.device)
                            if self.use_object_keypoints:
                                loss_dict = self.model(
                                    data,
                                    ori_data_cond,
                                    cond_mask,
                                    padding_mask,
                                    language_input=language_input,
                                    contact_labels=contact_labels,
                                    rest_human_offsets=rest_human_offsets,
                                    ds=self.val_ds,
                                    data_dict=val_data_dict,
                                )
                            else:
                                val_loss_diffusion, val_loss_obj, val_loss_human = (
                                    self.model(
                                        data,
                                        ori_data_cond,
                                        cond_mask,
                                        padding_mask,
                                        language_input=language_input,
                                        contact_labels=contact_labels,
                                        rest_human_offsets=rest_human_offsets,
                                        ds=self.val_ds,
                                        data_dict=val_data_dict,
                                    )
                                )
                        else:
                            val_loss_diffusion = self.model(
                                data,
                                ori_data_cond,
                                cond_mask,
                                padding_mask,
                                contact_labels=contact_labels,
                                rest_human_offsets=rest_human_offsets,
                            )
                    else:
                        val_loss_diffusion = self.model(
                            val_obj_data, ori_data_cond, cond_mask, padding_mask
                        )

                    if self.use_object_keypoints:
                        val_loss_diffusion = loss_dict.get("loss", 0)
                        val_loss_obj = loss_dict.get("loss_object", 0)
                        val_loss_human = loss_dict.get("loss_human", 0)
                        val_loss_fk = loss_dict.get("loss_fk", 0)
                        val_loss_obj_pts = loss_dict.get("loss_obj_pts", 0)
                        val_loss_obj_pts_in_hand = loss_dict.get(
                            "loss_obj_pts_in_hand", 0
                        )
                        val_loss_feet = loss_dict.get("loss_feet", 0)
                        val_loss_obj_vel = loss_dict.get("loss_obj_vel", 0)

                        val_loss = (
                            val_loss_diffusion
                            + self.loss_w_feet * val_loss_feet
                            + self.loss_w_fk * val_loss_fk
                            + self.loss_w_obj_pts * val_loss_obj_pts
                            + self.loss_w_obj_pts_in_hand * val_loss_obj_pts_in_hand
                            + self.loss_w_obj_vel * val_loss_obj_vel
                        )
                    else:
                        val_loss = val_loss_diffusion

                    if self.use_wandb:
                        if self.use_object_keypoints:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": float(val_loss),
                                "Validation/Loss/Diffusion Loss": float(
                                    val_loss_diffusion
                                ),
                                "Validation/Loss/Object Loss": float(val_loss_obj),
                                "Validation/Loss/Human Loss": float(val_loss_human),
                                "Validation/Loss/Feet Contact Loss": float(
                                    val_loss_feet
                                ),
                                "Validation/Loss/FK Loss": float(val_loss_fk),
                                "Validation/Loss/Object Pts Loss": float(
                                    val_loss_obj_pts
                                ),
                                "Validation/Loss/Object Pts in Hand Loss": float(
                                    val_loss_obj_pts_in_hand
                                ),
                                "Validation/Loss/Object Vel Loss": float(
                                    val_loss_obj_vel
                                ),
                            }
                        else:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                                "Validation/Loss/Object Loss": val_loss_obj.item(),
                                "Validation/Loss/Human Loss": val_loss_human.item(),
                            }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        if self.pred_human_motion:
                            # data = torch.cat((val_obj_data, val_human_data), dim=-1)
                            if self.add_language_condition:
                                all_res_list = self.ema.ema_model.sample(
                                    data,
                                    ori_data_cond,
                                    cond_mask,
                                    padding_mask,
                                    language_input=language_input,
                                    contact_labels=contact_labels,
                                    rest_human_offsets=rest_human_offsets,
                                )
                            else:
                                all_res_list = self.ema.ema_model.sample(
                                    data,
                                    ori_data_cond,
                                    cond_mask,
                                    padding_mask,
                                    contact_labels=contact_labels,
                                    rest_human_offsets=rest_human_offsets,
                                )
                        else:
                            all_res_list = self.ema.ema_model.sample(
                                val_obj_data, ori_data_cond, cond_mask, padding_mask
                            )

                        # Visualization
                        if self.pred_human_motion:
                            for_vis_gt_data = torch.cat(
                                (val_obj_data, val_human_data), dim=-1
                            )
                        else:
                            for_vis_gt_data = val_obj_data.to(all_res_list.device)

                        if self.add_interaction_feet_contact:
                            all_res_list = all_res_list[:, :, :-4]
                            cond_mask = cond_mask[:, :, :-4]
                        if self.use_object_keypoints:
                            all_res_list = all_res_list[:, :, :-4]
                            cond_mask = cond_mask[:, :, :-4]
                        if self.add_interaction_root_xy_ori:
                            all_res_list = all_res_list[:, :, :-6]
                            cond_mask = cond_mask[:, :, :-6]
                        if self.add_wrist_relative:
                            all_res_list = all_res_list[:, :, :-18]
                            cond_mask = cond_mask[:, :, :-18]

                        # self.gen_vis_res(for_vis_gt_data, val_data_dict, self.step, cond_mask, vis_gt=True)
                        # self.gen_vis_res(all_res_list, val_data_dict, self.step, cond_mask)

            self.step += 1

        print("training complete")

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(
        self,
        milestone: str = "10",
        render_results: bool = False,
    ):
        # Load the model.
        print(f"Loaded weight: {milestone}")
        self.load(milestone)
        self.ema.ema_model.eval()

        # Load dataset.
        test_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

        for s_idx, val_data_dict in tqdm(
            enumerate(test_loader), ncols=150, desc="Sampling sequences"
        ):
            print(f"Sampling sequence {s_idx} / {len(test_loader)} ...")

            if self.add_wrist_relative:
                wrist_relative_batch = val_data_dict["wrist_relative"].cuda()
                wrist_relative_mask_batch = torch.zeros_like(
                    wrist_relative_batch
                ).cuda()
            else:
                wrist_relative_batch = None
                wrist_relative_mask_batch = None

            if self.add_object_static:
                object_static_flag_batch = val_data_dict["object_static_flag"].cuda()
            else:
                object_static_flag_batch = None

            # Sample.
            num_samples_per_seq = 1
            all_res_list, pred_feet_contact, all_res_list_gt, *_ = (
                self.call_interaction_model(
                    val_data_dict=val_data_dict,
                    add_root_ori=self.add_interaction_root_xy_ori,
                    add_wrist_relative=self.add_wrist_relative,
                    add_feet_contact=self.add_interaction_feet_contact,
                    add_object_static=self.add_object_static,
                    wrist_relative=wrist_relative_batch,
                    wrist_relative_mask=wrist_relative_mask_batch,
                    object_static_flag=object_static_flag_batch,
                    s_idx=s_idx,
                    render_results=render_results,
                    num_samples_per_seq=num_samples_per_seq,
                )
            )

    def call_interaction_model(
        self,
        val_data_dict: Dict[str, Any],
        add_root_ori: bool = False,
        add_wrist_relative: bool = False,
        add_feet_contact: bool = False,
        add_object_static: bool = False,
        render_results: bool = False,
        wrist_relative: Optional[torch.Tensor] = None,
        wrist_relative_mask: Optional[torch.Tensor] = None,
        object_static_flag: Optional[torch.Tensor] = None,
        s_idx: int = 0,
        num_samples_per_seq: int = 1,
        smooth_whole_traj: bool = False,
        inpaint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Call the interaction model from dataset."""

        if add_object_static and object_static_flag is None:
            raise ValueError(
                "object_static_flag must be provided if add_object_static is True"
            )

        if add_wrist_relative and (
            wrist_relative is None or wrist_relative_mask is None
        ):
            raise ValueError(
                "wrist_relative and wrist_relative_mask must be provided if add_wrist_relative is True"
            )

        seq_name_list = val_data_dict["seq_name"]
        object_name_list = val_data_dict["obj_name"]
        start_frame_idx_list = val_data_dict["s_idx"]
        end_frame_idx_list = val_data_dict["e_idx"]

        val_human_data = val_data_dict["motion"].cuda()
        val_obj_data = val_data_dict["obj_motion"].cuda()

        obj_bps_data = val_data_dict["input_obj_bps"].cuda().reshape(-1, 1, 1024 * 3)
        ori_data_cond = obj_bps_data  # BS X 1 X (1024*3)

        rest_human_offsets = val_data_dict["rest_human_offsets"].cuda()  # BS X 24 X 3

        if "contact_labels" in val_data_dict:
            contact_labels = val_data_dict["contact_labels"].cuda()  # BS X T X 4
        else:
            contact_labels = None

        # Generate padding mask
        actual_seq_len = (
            val_data_dict["seq_len"] + 1
        )  # BS, + 1 since we need additional timestep for noise level
        tmp_mask = torch.arange(self.window + 1).expand(
            val_obj_data.shape[0], self.window + 1
        ) < actual_seq_len[:, None].repeat(1, self.window + 1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_obj_data.device)

        if self.add_waypoints_xy:
            end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(
                val_obj_data, val_data_dict["seq_len"]
            )
            cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                val_obj_data, val_data_dict["seq_len"]
            )
            cond_mask = end_pos_cond_mask * cond_mask
        else:
            cond_mask = None

        if add_object_static:
            cond_mask[..., :12] *= 1.0 - object_static_flag.to(cond_mask.device)

        if self.pred_human_motion:
            human_cond_mask = torch.ones_like(val_human_data).to(val_human_data.device)
            if self.input_first_human_pose:
                human_cond_mask[:, 0, :] = 0
            elif self.input_full_human_pose:
                human_cond_mask *= 0

        # Not used in dataset evaluation, padding with 0.
        if add_root_ori:
            root_xy_ori = torch.zeros_like(val_data_dict["root_traj_xy_ori"]).cuda()
            val_human_data = torch.cat(
                (val_human_data, root_xy_ori), dim=-1
            )  # BS X T X (24*3 + 22*6 + 6)
            root_xy_ori_mask = torch.ones_like(root_xy_ori).to(val_human_data.device)
            human_cond_mask = torch.cat((human_cond_mask, root_xy_ori_mask), dim=-1)

        if add_wrist_relative:
            wrist_relative = wrist_relative.cuda()  # BS X T X 18
            val_human_data = torch.cat(
                (val_human_data, wrist_relative), dim=-1
            )  # BS X T X (24*3 + 22*6 + 18)
            wrist_relative_mask = wrist_relative_mask.to(val_human_data.device)
            human_cond_mask = torch.cat((human_cond_mask, wrist_relative_mask), dim=-1)
        cond_mask = torch.cat(
            (cond_mask, human_cond_mask), dim=-1
        )  # BS X T X (3+6+24*3+22*6)

        # Repeat data for multiple samples, used for calculating metrics.
        val_obj_data = val_obj_data.repeat(num_samples_per_seq, 1, 1)  # BS X T X D
        val_human_data = val_human_data.repeat(num_samples_per_seq, 1, 1)
        cond_mask = cond_mask.repeat(num_samples_per_seq, 1, 1)
        padding_mask = padding_mask.repeat(num_samples_per_seq, 1, 1)
        ori_data_cond = ori_data_cond.repeat(num_samples_per_seq, 1, 1)
        rest_human_offsets = rest_human_offsets.repeat(num_samples_per_seq, 1, 1)

        if self.use_object_keypoints:
            contact_data = torch.zeros(
                val_obj_data.shape[0], val_obj_data.shape[1], 4
            ).to(val_obj_data.device)
            data = torch.cat((val_obj_data, val_human_data, contact_data), dim=-1)
            cond_mask = torch.cat(
                (cond_mask, torch.ones_like(contact_data).to(cond_mask.device)), dim=-1
            )
        else:
            data = torch.cat((val_obj_data, val_human_data), dim=-1)

        if add_feet_contact:
            feet_contact = torch.zeros(
                val_obj_data.shape[0], val_obj_data.shape[1], 4
            ).to(val_obj_data.device)
            data = torch.cat((data, feet_contact), dim=-1)
            cond_mask = torch.cat(
                (cond_mask, torch.ones_like(feet_contact).to(cond_mask.device)), dim=-1
            )

        if self.use_guidance_in_denoising:
            guidance_fn = self.apply_different_guidance_loss
        else:
            guidance_fn = None

        if self.add_language_condition:
            text_anno_data = val_data_dict["text"]
            language_input = self.encode_text(text_anno_data)  # BS X 512
            language_input = language_input.to(data.device)
            language_input = language_input.repeat(num_samples_per_seq, 1)
            all_res_list = self.ema.ema_model.sample(
                data,
                ori_data_cond,
                cond_mask,
                padding_mask,
                language_input=language_input,
                contact_labels=contact_labels,
                rest_human_offsets=rest_human_offsets,
                guidance_fn=guidance_fn,
                data_dict=val_data_dict,
                inpaint=inpaint,
            )
        else:
            all_res_list = self.ema.ema_model.sample(
                data,
                ori_data_cond,
                cond_mask,
                padding_mask,
                contact_labels=contact_labels,
                rest_human_offsets=rest_human_offsets,
                guidance_fn=guidance_fn,
                data_dict=val_data_dict,
                return_diff_level_res=self.return_diff_level_res,
                inpaint=inpaint,
            )

        pred_feet_contact = None
        if add_feet_contact:
            all_res_list = all_res_list[:, :, :-4]
            pred_feet_contact = all_res_list[:, :, -4:].clone()
        if self.use_object_keypoints:
            pred_contact_labels = all_res_list[:, :, -4:-2].clone()
            all_res_list = all_res_list[:, :, :-4]
        else:
            pred_contact_labels = None
        if add_root_ori:
            all_res_list = all_res_list[:, :, :-6]
        if add_wrist_relative:
            all_res_list = all_res_list[:, :, :-18]

        all_res_list_gt = torch.cat((val_obj_data, val_human_data), dim=-1)

        if smooth_whole_traj:
            all_res_list = smooth_res(all_res_list)

        if render_results:
            self.render_interaction_model_results(
                all_res_list,
                cond_mask,
                object_name_list,
                val_data_dict,
                val_obj_data,
                val_human_data,
                s_idx=s_idx,
                # vis_gt=False,
            )

        return all_res_list, pred_feet_contact, all_res_list_gt, pred_contact_labels

    def render_interaction_model_results(
        self,
        all_res_list: torch.Tensor,
        cond_mask: torch.Tensor,
        object_name_list: List[str],
        val_data_dict: Dict,
        val_obj_data: torch.Tensor,
        val_human_data: torch.Tensor,
        vis_gt: bool = True,
        s_idx: int = 0,
        finger_all_res_list: Optional[torch.Tensor] = None,
        dest_folder: Optional[str] = None,
    ):
        milestone = "10"
        for_vis_gt_data = torch.cat((val_obj_data, val_human_data), dim=-1)
        vis_tag = "test_sample_interaction_model_from_dataset_{}".format(s_idx)
        if dest_folder is None:
            dest_out_obj_root_folder = "./results/test_results"
        else:
            dest_out_obj_root_folder = os.path.join("./results", dest_folder)
            if not os.path.exists(dest_out_obj_root_folder):
                os.makedirs(dest_out_obj_root_folder)
        curr_dest_out_mesh_folder = os.path.join(dest_out_obj_root_folder, vis_tag)
        curr_dest_out_vid_path = os.path.join(
            dest_out_obj_root_folder, "sample_" + str(s_idx) + ".mp4"
        )

        *_, motion_results_path = self.gen_vis_res(
            all_res_list,
            val_data_dict,
            milestone,
            cond_mask,
            curr_object_name=object_name_list[0],
            vis_tag=vis_tag,
            dest_out_vid_path=curr_dest_out_vid_path,
            dest_mesh_vis_folder=curr_dest_out_mesh_folder,
            save_obj_only=True,
            vis_gt=False,
            finger_all_res_list=finger_all_res_list,
            save_motion_params=False,
        )

        if vis_gt:
            # NOTE: This has to be put behind no_gt, since we need to call `rmtree`.
            self.gen_vis_res(
                for_vis_gt_data,
                val_data_dict,
                milestone,
                cond_mask,
                curr_object_name=object_name_list[0],
                vis_tag=vis_tag,
                dest_out_vid_path=curr_dest_out_vid_path,
                dest_mesh_vis_folder=curr_dest_out_mesh_folder,
                save_obj_only=True,
                vis_gt=True,
            )

        epoch = milestone
        print(
            "Visualizing interaction results for seq: {0}".format(
                curr_dest_out_mesh_folder
            )
        )
        subprocess.run(
            [
                "python",
                "visualizer/vis/visualize_interaction_results.py",
                "--result-path",
                curr_dest_out_mesh_folder,
                "--s-idx",
                str(s_idx),
                "--interaction-epoch",
                str(epoch),
                "--model-path",
                "../data/smplx/models_smplx_v1_1/models/",
                "--offscreen",
            ]
        )

        return motion_results_path

    def get_only_object_mesh(
        self, all_res_list, data_dict, ds, curr_window_ref_obj_rot_mat=None
    ):
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3]  # N X T X 3

        if self.use_first_frame_bps or self.use_random_frame_bps:
            pred_obj_rel_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 30
            if curr_window_ref_obj_rot_mat is not None:
                pred_obj_rot_mat = ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, curr_window_ref_obj_rot_mat
                )
            else:
                pred_obj_rot_mat = ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, data_dict["reference_obj_rot_mat"]
                )
        else:
            pred_obj_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3

        pred_seq_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        object_mesh_verts_list = []
        for idx in range(num_seq):
            curr_obj_rot_mat = pred_obj_rot_mat[idx]  # T X 3 X 3
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(
                curr_obj_quat
            )  # Potentially avoid some prediction not satisfying rotation matrix requirements.

            object_name = data_dict["obj_name"][0]

            # For generating all the vertices of the object
            obj_rest_verts, obj_mesh_faces = ds.load_rest_pose_object_geometry(
                object_name
            )
            obj_rest_verts = (
                torch.from_numpy(obj_rest_verts).float().to(pred_seq_com_pos.device)
            )

            obj_mesh_verts = ds.load_object_geometry_w_rest_geo(
                curr_obj_rot_mat.cuda(), pred_seq_com_pos[idx], obj_rest_verts
            )  # T X Nv X 3

            # For generating object keypoints
            # num_steps = pred_seq_com_pos[idx].shape[0]
            # rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].cuda()[0] # K X 3
            # pred_seq_obj_kpts = torch.matmul(curr_obj_rot_mat[:, None, :, :].repeat(1, \
            #         rest_pose_obj_kpts.shape[0], 1, 1), \
            #         rest_pose_obj_kpts[None, :, :, None].repeat(num_steps, 1, 1, 1)) + \
            #         pred_seq_com_pos[idx][:, None, :, None] # T X K X 3 X 1

            # pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1) # T X K X 3

            object_mesh_verts_list.append(obj_mesh_verts)
            # object_mesh_verts_list.append(pred_seq_obj_kpts)

        object_mesh_verts_list = torch.stack(object_mesh_verts_list)

        return object_mesh_verts_list, obj_mesh_faces

    def get_object_mesh_from_prediction(
        self, all_res_list, data_dict, ds, curr_window_ref_obj_rot_mat=None
    ):
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3]  # N X T X 3

        if self.use_first_frame_bps or self.use_random_frame_bps:
            pred_obj_rel_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 30
            if curr_window_ref_obj_rot_mat is not None:
                pred_obj_rot_mat = ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, curr_window_ref_obj_rot_mat
                )
            else:
                pred_obj_rot_mat = ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, data_dict["reference_obj_rot_mat"]
                )
                # ??? In some cases, this is not the first window!!! Bug? Ok for single-window generation.
        else:
            pred_obj_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3

        pred_seq_com_pos = ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        num_joints = 24

        normalized_global_jpos = all_res_list[
            :, :, 3 + 9 : 3 + 9 + num_joints * 3
        ].reshape(num_seq, -1, num_joints, 3)
        global_jpos = ds.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3
        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        global_rot_6d = all_res_list[
            :, :, 3 + 9 + num_joints * 3 : 3 + 9 + num_joints * 3 + 22 * 6
        ].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(
            global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)  # N X 3

        if trans2joint.shape[0] != all_res_list.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1)

        human_mesh_verts_list = []
        human_mesh_jnts_list = []
        object_mesh_verts_list = []
        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx]  # T X 22 X 3 X 3
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)  # T X 22 X 3 X 3
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )  # T X 22 X 3

            curr_global_root_jpos = global_root_jpos[idx]  # T X 3

            curr_trans2joint = trans2joint[idx : idx + 1].clone()

            root_trans = curr_global_root_jpos + curr_trans2joint.to(
                curr_global_root_jpos.device
            )  # T X 3

            # Generate global joint position
            bs = 1
            betas = data_dict["betas"][0]
            gender = data_dict["gender"][0]

            curr_obj_rot_mat = pred_obj_rot_mat[idx]  # T X 3 X 3
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(
                curr_obj_quat
            )  # Potentially avoid some prediction not satisfying rotation matrix requirements.

            object_name = data_dict["obj_name"][0]

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].cuda(),
                curr_local_rot_aa_rep[None].cuda(),
                betas.cuda(),
                [gender],
                ds.bm_dict,
                return_joints24=True,
            )

            # For generating all the vertices of the object
            obj_rest_verts, obj_mesh_faces = ds.load_rest_pose_object_geometry(
                object_name
            )
            obj_rest_verts = (
                torch.from_numpy(obj_rest_verts).float().to(pred_seq_com_pos.device)
            )

            obj_mesh_verts = ds.load_object_geometry_w_rest_geo(
                curr_obj_rot_mat.cuda(), pred_seq_com_pos[idx], obj_rest_verts
            )  # T X Nv X 3

            # For generating object keypoints
            # num_steps = pred_seq_com_pos[idx].shape[0]
            # rest_pose_obj_kpts = data_dict['rest_pose_obj_pts'].cuda()[0] # K X 3
            # pred_seq_obj_kpts = torch.matmul(curr_obj_rot_mat[:, None, :, :].repeat(1, \
            #         rest_pose_obj_kpts.shape[0], 1, 1), \
            #         rest_pose_obj_kpts[None, :, :, None].repeat(num_steps, 1, 1, 1)) + \
            #         pred_seq_com_pos[idx][:, None, :, None] # T X K X 3 X 1

            # pred_seq_obj_kpts = pred_seq_obj_kpts.squeeze(-1) # T X K X 3

            human_mesh_verts_list.append(mesh_verts)
            human_mesh_jnts_list.append(mesh_jnts)

            object_mesh_verts_list.append(obj_mesh_verts)
            # object_mesh_verts_list.append(pred_seq_obj_kpts)

        human_mesh_verts_list = torch.stack(human_mesh_verts_list)
        human_mesh_jnts_list = torch.stack(human_mesh_jnts_list)

        object_mesh_verts_list = torch.stack(object_mesh_verts_list)

        return (
            human_mesh_verts_list,
            human_mesh_jnts_list,
            mesh_faces,
            object_mesh_verts_list,
            obj_mesh_faces,
        )

    def apply_matching_condition_guidance(self, pred_clean_x, x_pose_cond, cond_mask):
        # pred_clean_x: BS X T X D
        # x_pose_cond: BS X T X D
        # cond_mask: BS X T X D, 1 represents missing regions.

        # classifier_scale = 1e7

        x_in = pred_clean_x

        loss = F.mse_loss(x_in, x_pose_cond, reduction="none") * (1 - cond_mask)

        loss = loss.sum()

        print("Matching condition MSE loss:{0}".format(loss))

        # return torch.autograd.grad(-loss, x_in)[0] * classifier_scale # Notice! Use minus loss!
        return loss

    def apply_feet_floor_contact_guidance(
        self,
        pred_clean_x,
        rest_human_offsets,
        data_dict,
        contact_labels=None,
        curr_window_ref_obj_rot_mat=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
        use_feet_contact=False,
    ):
        # pred_clean_x: BS X T X D
        # x_pose_cond: BS X T X D
        # cond_mask: BS X T X D, 1 represents missing regions.

        num_seq = pred_clean_x.shape[0]

        # For penalize all the joints after fk for penetration loss.
        # x_in = pred_clean_x[:, :, 12:12+3].requires_grad_(True) # BS X T X 3

        human_verts, human_jnts, human_faces, obj_verts, obj_faces = (
            self.get_object_mesh_from_prediction(
                pred_clean_x,
                data_dict,
                ds=self.val_ds,
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            )
        )
        # BS X 1 X T X Nv X 3, BS X 1 X T X 24 X 3, BS X T X Nv' X 3

        # ori_floor_height = determine_floor_height_and_contacts(human_jnts[0, 0, :, :22, :].cpu().detach().numpy())

        # print("current floor height: {0}".format(ori_floor_height))

        left_toe_idx = 10
        right_toe_idx = 11
        l_toe_height = human_jnts[:, 0, :, left_toe_idx, 2:]  # BS X T X 1
        r_toe_height = human_jnts[:, 0, :, right_toe_idx, 2:]  # BS X T X 1
        support_foot_height = torch.minimum(l_toe_height, r_toe_height)

        loss_feet_floor_contact = F.mse_loss(
            support_foot_height, torch.ones_like(support_foot_height) * 0.02
        )

        # left_not_in_contact = (l_toe_height > r_toe_height + 0.015).float().detach()
        # left_in_contact = 1 - left_not_in_contact
        # right_not_in_contact = (r_toe_height > l_toe_height + 0.015).float().detach()
        # right_in_contact = 1 - right_not_in_contact
        # foot_contact_labels = torch.cat((left_in_contact, right_in_contact), dim=-1)[..., None].detach() # BS X T X 2 X 1

        # foot_sliding_loss = 0
        # start = 0
        # while start < left_in_contact.shape[1]:
        #     end = start
        #     while end < left_in_contact.shape[1] and left_in_contact[0, end] == left_in_contact[0, start]:
        #         end += 1
        #     if left_in_contact[0, start, 0] > 0.95:
        #         offset = human_jnts[0, 0, start:end, left_toe_idx] - human_jnts[0, 0, start:start+1, left_toe_idx]
        #         foot_sliding_loss += F.mse_loss(offset, torch.zeros_like(offset))
        #     start = end
        # start = 0
        # while start < right_in_contact.shape[1]:
        #     end = start
        #     while end < right_in_contact.shape[1] and right_in_contact[0, end] == right_in_contact[0, start]:
        #         end += 1
        #     if right_in_contact[0, start, 0] > 0.95:
        #         offset = human_jnts[0, 0, start:end, right_toe_idx] - human_jnts[0, 0, start:start+1, right_toe_idx]
        #         foot_sliding_loss += F.mse_loss(offset, torch.zeros_like(offset))
        #     start = end

        # foot_sliding_loss *= 0.1

        # import pdb
        # pdb.set_trace()

        loss = num_seq * loss_feet_floor_contact * 10
        if use_feet_contact:
            foot_contact_labels = pred_clean_x[:, :, -4:-2].unsqueeze(
                -1
            )  # BS X T X 2 X 1
            foot_velocity = (
                human_jnts[:, 0, 1:, [left_toe_idx, right_toe_idx]]
                - human_jnts[:, 0, :-1, [left_toe_idx, right_toe_idx]]
            )  # BS X (T-1) X 2 X 3
            foot_sliding_loss = (
                F.mse_loss(
                    foot_velocity * foot_contact_labels[:, :-1],
                    torch.zeros_like(foot_velocity),
                )
                * 300
            )  # BS X (T-1) X 2 X 3
            # print("Fett-Floor contact loss: {0}".format(loss_feet_floor_contact), "Foot sliding loss: {0}".format(foot_sliding_loss))
            loss += foot_sliding_loss

        return loss

    def apply_velocity_guidance_loss(
        self,
        pred_clean_x,
        rest_human_offsets,
        data_dict,
        contact_labels=None,
        curr_window_ref_obj_rot_mat=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
    ):
        # prev_window_cano_rot_mat: BS X 3 X 3
        # prev_window_init_root_trans: BS X 1 X 3
        # pass
        # pred_clean_x = torch.cat((data_dict['obj_motion'], data_dict['motion'], data_dict['feet_contact']), dim=-1).to(pred_clean_x.device)

        num_seq = pred_clean_x.shape[0]

        if not self.use_long_planned_path:
            # SIngle window generation
            pred_clean_x = pred_clean_x[:, : data_dict["seq_len"][0]]

        # For penalize all the joints after fk for penetration loss.
        # x_in = pred_clean_x[:, :, 12:12+3].requires_grad_(True) # BS X T X 3

        human_verts, human_jnts, human_faces, obj_verts, obj_faces = (
            self.get_object_mesh_from_prediction(
                pred_clean_x,
                data_dict,
                ds=self.val_ds,
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            )
        )
        # # BS X 1 X T X Nv X 3, BS X 1 X T X 24 X 3, BS X T X Nv' X 3 ]

        parents = torch.from_numpy(get_smpl_parents()).to(human_jnts.device)  # 24
        human_local_jnts = (
            human_jnts - human_jnts[:, :, :, parents, :]
        )  # BS X 1 X T X 24 X 3
        human_local_jnts[:, :, :, 0, :] = human_jnts[
            :, :, :, 0, :
        ]  # BS X 1 X T X 24 X 3
        human_local_vels = (
            human_local_jnts[:, 0, 1:, 16:, :] - human_local_jnts[:, 0, :-1, 16:, :]
        )  # BS X T X 24 X 3
        human_local_vels = torch.norm(human_local_vels, dim=-1)  # BS X T X 24

        return torch.mean(human_local_vels**2) * 100

    def apply_hand_object_interaction_guidance_loss(
        self,
        pred_clean_x,
        x_pose_cond,
        rest_human_offsets,
        data_dict,
        contact_labels=None,
        curr_window_ref_obj_rot_mat=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
        start_phase=False,
        end_phase=False,
        end_frame_obj_rot_mat=None,
        end_height=None,
        use_feet_contact=False,
    ):
        # prev_window_cano_rot_mat: BS X 3 X 3
        # prev_window_init_root_trans: BS X 1 X 3
        num_seq = pred_clean_x.shape[0]

        if not self.use_long_planned_path:
            # SIngle window generation
            pred_clean_x = pred_clean_x[:, : data_dict["seq_len"][0]]

        human_verts, human_jnts, human_faces, obj_verts, obj_faces = (
            self.get_object_mesh_from_prediction(
                pred_clean_x,
                data_dict,
                ds=self.val_ds,
                curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            )
        )

        if end_frame_obj_rot_mat is not None:
            x_pose_cond[:, -1, 3 : 3 + 9] = end_frame_obj_rot_mat.reshape(-1, 9)
        static_obj_verts, static_obj_faces = self.get_only_object_mesh(
            x_pose_cond,
            data_dict,
            ds=self.val_ds,
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
        )
        # # BS X 1 X T X Nv X 3, BS X 1 X T X 24 X 3, BS X T X Nv' X 3 ]

        # Need to downsample object vertices sometimes.
        num_obj_verts = obj_verts.shape[2]
        if num_obj_verts > 30000:
            downsample_rate = num_obj_verts // 30000 + 1
            obj_verts = obj_verts[:, :, ::downsample_rate, :]
            static_obj_verts = static_obj_verts[:, :, ::downsample_rate, :]

        # 1. Compute penetration loss between hand vertices and object vertices.
        # hand_verts = human_verts.squeeze(1)[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
        pred_normalized_obj_trans = pred_clean_x[:, :, :3]  # N X T X 3

        if self.use_first_frame_bps or self.use_random_frame_bps:
            pred_obj_rel_rot_mat = pred_clean_x[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 30
            if curr_window_ref_obj_rot_mat is not None:
                pred_obj_rot_mat = self.ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, curr_window_ref_obj_rot_mat
                )
            else:
                pred_obj_rot_mat = self.ds.rel_rot_to_seq(
                    pred_obj_rel_rot_mat, data_dict["reference_obj_rot_mat"]
                )  # Bug? Since for the windows except the first one, the reference obj mat is not the originbal one in data?
        else:
            pred_obj_rot_mat = pred_clean_x[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3

        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(
            pred_normalized_obj_trans
        )  # N X T X 3

        # # hand_verts_in_rest_frame = hand_verts - pred_seq_com_pos[:, :, None, :] # N X T X N_hand X 3
        # # hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
        # #                     hand_verts_in_rest_frame.shape[2], 1, 1), \
        # #                     hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # N X T X N_hand X 3

        # human_verts_in_rest_frame = human_verts.squeeze(1) - pred_seq_com_pos[:, :, None, :] # N X T X N_v X 3
        # human_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
        #                     human_verts_in_rest_frame.shape[2], 1, 1), \
        #                     human_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # N X T X N_v X 3

        num_steps = pred_clean_x.shape[1]

        # # # Convert hand vertices to align with rest pose object.
        # # signed_dists = compute_signed_distances(self.object_sdf, self.object_sdf_centroid, \
        # #     self.object_sdf_extents, hand_verts_in_rest_frame.reshape(num_seq*num_steps, -1, 3)) # we always use bs = 1 now!!! T(120) X 1535
        # signed_dists = compute_signed_distances(self.object_sdf, self.object_sdf_centroid, \
        #     self.object_sdf_extents, human_verts_in_rest_frame.reshape(num_seq*num_steps, -1, 3)) # we always use bs = 1 now!!!
        # print("Object sdf min:{0}".format(signed_dists.min()))
        # # # signed_dists: T X N_hand (120 X 1535) fullbody: 120 X 10475

        # # # penetration_tolerance = 0.02
        # loss_penetration = torch.minimum(signed_dists, \
        #             torch.zeros_like(signed_dists)).abs().mean()
        # print("Human-Object Penetration loss:{0}".format(loss_penetration))
        # loss_penetration_sum = torch.minimum(signed_dists, \
        #             torch.zeros_like(signed_dists)).abs().sum()
        # print("Hand-Object Penetration loss:{0}".format(loss_penetration_sum)) # if the sum value is larger than 2, then gradient would be huge!!!

        # For contact loss, temporal consistency loss, both need to know whether this frame is in contact or not.

        # 2. Compute contact loss, minimize the distance between hand vertices and nearest neugbor points on the object mesh.
        l_palm_idx = 22
        r_palm_idx = 23
        left_palm_jpos = human_jnts.squeeze(1)[:, :, l_palm_idx, :]  # BS X T X 3
        right_palm_jpos = human_jnts.squeeze(1)[:, :, r_palm_idx, :]  # BS X T X 3

        contact_points = torch.cat(
            (left_palm_jpos[:, :, None, :], right_palm_jpos[:, :, None, :]), dim=2
        )  # BS X T X 2 X 3
        bs, seq_len, _, _ = contact_points.shape
        # assert bs == 1

        # print("Object # vertices:{0}".format(obj_verts.shape))
        dists = torch.cdist(
            contact_points.reshape(bs * seq_len, 2, 3)[:, :, :],
            obj_verts.reshape(bs * seq_len, -1, 3),
        )  # (BS*T) X 2 X N_object
        dists, _ = torch.min(dists, 2)  # (BS*T) X 2

        # Determine contact labels.
        # determine_contact_threshold = 0.1

        if use_feet_contact:
            pred_contact_semantic = pred_clean_x[:, :, -8:-6]  # BS X T X 2
        else:
            pred_contact_semantic = pred_clean_x[:, :, -4:-2]  # BS X T X 2
        contact_labels = pred_contact_semantic > 0.95
        contact_labels = contact_labels.reshape(bs * seq_len, -1)[
            :, :2
        ].detach()  # (BS*T) X 2
        # NOTE: set two hands to contact with the object at the same time, if they're pretty similar
        if start_phase or end_phase:
            xor = contact_labels[:, 0] ^ contact_labels[:, 1]
            if xor.sum() < 10:  # then we assume they're in contact at the same time
                contact_labels[:, 1] = contact_labels[:, 0]

        zero_target = torch.zeros_like(dists).to(dists.device)
        contact_threshold = 0.02

        # loss_contact = torch.sum(dists*contact_labels) # Open question: how to avoid penalizing non-contact frames?
        loss_contact = F.l1_loss(
            torch.maximum(
                dists * contact_labels[:, :2] - contact_threshold, zero_target
            ),
            zero_target,
        )
        # temp_dists = dists.reshape(bs, seq_len, -1)
        # temp_contact_labels = contact_labels.reshape(bs, seq_len, -1)
        # temp_zero_target = torch.zeros_like(temp_dists).to(dists.device)
        # loss_contact = F.l1_loss(torch.maximum(temp_dists*temp_contact_labels-contact_threshold, temp_zero_target), \
        #         temp_zero_target, reduction='none')[1].mean()

        # print("Palm-Object Contact loss:{0}".format(loss_contact))

        # 3. Compute temporal consistency loss.
        left_palm_to_obj_com = left_palm_jpos - pred_seq_com_pos.detach()  # BS X T X 3
        right_palm_to_obj_com = right_palm_jpos - pred_seq_com_pos.detach()
        relative_left_palm_jpos = torch.matmul(
            pred_obj_rot_mat.detach().transpose(2, 3),
            left_palm_to_obj_com[:, :, :, None],
        ).squeeze(-1)  # BS X T X 3
        relative_right_palm_jpos = torch.matmul(
            pred_obj_rot_mat.detach().transpose(2, 3),
            right_palm_to_obj_com[:, :, :, None],
        ).squeeze(-1)

        contact_labels = contact_labels.reshape(num_seq, num_steps, -1)  # BS X T X 2
        # For debug GT
        # contact_labels = data_dict['contact_labels'][:, :, :2].to(relative_left_palm_jpos.device)

        # Expand dimensions of contact_labels for multiplication
        left_contact_labels_expanded = contact_labels[:, :, 0:1]
        left_contact_mask = (
            left_contact_labels_expanded
            * left_contact_labels_expanded.transpose(-1, -2)
        )

        right_contact_labels_expanded = contact_labels[:, :, 1:2]
        right_contact_mask = (
            right_contact_labels_expanded
            * right_contact_labels_expanded.transpose(-1, -2)
        )  # BS X T X T

        left_norms = torch.norm(relative_left_palm_jpos, dim=-1, keepdim=True)
        left_normalized = relative_left_palm_jpos / left_norms
        left_similarity = torch.matmul(
            left_normalized, left_normalized.transpose(-1, -2)
        )

        right_norms = torch.norm(relative_right_palm_jpos, dim=-1, keepdim=True)
        right_normalized = relative_right_palm_jpos / right_norms
        right_similarity = torch.matmul(
            right_normalized, right_normalized.transpose(-1, -2)
        )  # BS X T X T

        loss_consistency = (
            1
            - torch.mean(left_similarity * left_contact_mask)
            + 1
            - torch.mean(right_similarity * right_contact_mask)
        )  # GT: 0.11
        # print("Palm-Obj Temporal Consistency loss:{0}".format(loss_consistency))

        # 3b. Compute temporal consistency loss, for wrist orientation.
        l_wrist_idx = 20
        r_wrist_idx = 21
        left_wrist_jpos = human_jnts.squeeze(1)[:, :, l_wrist_idx, :]  # BS X T X 3
        right_wrist_jpos = human_jnts.squeeze(1)[:, :, r_wrist_idx, :]  # BS X T X 3
        left_wrist_to_palm = left_palm_jpos - left_wrist_jpos  # BS X T X 3
        right_wrist_to_plam = right_palm_jpos - right_wrist_jpos  # BS X T X 3
        relative_left_wrist_to_palm = torch.matmul(
            pred_obj_rot_mat.detach().transpose(2, 3), left_wrist_to_palm[:, :, :, None]
        ).squeeze(-1)  # BS X T X 3
        relative_right_wrist_to_plam = torch.matmul(
            pred_obj_rot_mat.detach().transpose(2, 3),
            right_wrist_to_plam[:, :, :, None],
        ).squeeze(-1)

        left_norms = torch.norm(relative_left_wrist_to_palm, dim=-1, keepdim=True)
        left_normalized = relative_left_wrist_to_palm / left_norms
        left_similarity = torch.matmul(
            left_normalized, left_normalized.transpose(-1, -2)
        )

        right_norms = torch.norm(relative_right_wrist_to_plam, dim=-1, keepdim=True)
        right_normalized = relative_right_wrist_to_plam / right_norms
        right_similarity = torch.matmul(
            right_normalized, right_normalized.transpose(-1, -2)
        )  # BS X T X T

        loss_ori_consistency = (
            1
            - torch.mean(left_similarity * left_contact_mask)
            + 1
            - torch.mean(right_similarity * right_contact_mask)
        )  # GT: 0.11
        # print("Contact loss:{}, Palm-Obj Temporal Consistency loss:{}, Orientation Consistency loss:{}".format(loss_contact, loss_consistency, loss_ori_consistency))

        # 4. Add floor-object penetration loss
        if end_height is not None:
            loss_floor_object = (
                torch.minimum(
                    obj_verts[:, :, :, -1] - end_height,
                    torch.zeros_like(obj_verts[:, :, :, -1]),
                )
                .abs()
                .mean()
            )
        else:
            loss_floor_object = (
                torch.minimum(
                    obj_verts[:, :, :, -1], torch.zeros_like(obj_verts[:, :, :, -1])
                )
                .abs()
                .mean()
            )
        # print("Floor-Object Penetration loss:{0}".format(loss_floor_object))

        # print("left", torch.sum(left_contact_labels_expanded))
        # print("right", torch.sum(right_contact_labels_expanded))
        # 5. Add object static loss
        if start_phase:
            contact_label_all = torch.all(contact_labels, dim=-1)  # BS X T
            uncontact_obj_verts = obj_verts[contact_label_all == False]
            loss_static_object = F.l1_loss(
                uncontact_obj_verts,
                static_obj_verts[0, 0:1].repeat(uncontact_obj_verts.shape[0], 1, 1),
            )  # assume bs == 1
        elif end_phase:
            contact_label_all = torch.all(contact_labels, dim=-1)  # BS X T
            uncontact_obj_verts = obj_verts[0, -45:]
            loss_static_object = F.l1_loss(
                uncontact_obj_verts,
                static_obj_verts[0, -1:].repeat(uncontact_obj_verts.shape[0], 1, 1),
            )  # assume bs == 1
        # Even GT has loss
        # Human-Object Penetration loss:0.00011554262164281681
        # Palm-Object Contact loss:0.013422583229839802
        # Palm-Obj Temporal Consistency loss:0.10843487083911896

        # loss = loss_penetration
        # loss = loss_penetration + 0.1 * loss_contact
        # loss = loss_consistency * 100000
        # loss = loss_penetration + loss_contact * 10000 + 10000 * loss_consistency

        # loss = loss_penetration * 1000 + loss_contact * 0 + loss_consistency * 0
        # loss = loss_penetration * 10000 * 1./pen_scale

        # For seen objects
        # loss = bs * (loss_penetration + loss_contact + loss_consistency + loss_floor_object * 100)

        loss = bs * (
            loss_contact
            + loss_consistency
            + loss_floor_object * 100
            + loss_ori_consistency
        )
        if start_phase or end_phase:
            # print("Object static loss:{0}".format(loss_static_object))
            loss += bs * loss_static_object * 2
        # For unseen objects
        # loss = loss_penetration * 0 + loss_contact + loss_consistency + loss_floor_object * 100

        # loss = loss_contact + loss_consistency + loss_floor_object * 100

        # loss = bs * (loss_contact + loss_consistency + loss_floor_object * 0)

        # loss = (loss_contact + loss_consistency + loss_floor_object * 0)

        # loss = loss_penetration * 0.1 + loss_contact + loss_consistency + loss_floor_object * 100
        # import pdb
        # pdb.set_trace()

        # Full body penetration, not very stable.
        # loss = loss_penetration * 10000 * 1./pen_scale + loss_contact * 2 + loss_consistency

        return loss

    def apply_different_guidance_loss(
        self,
        noise_level,
        pred_clean_x,
        x_pose_cond,
        cond_mask,
        rest_human_offsets,
        data_dict,
        contact_labels=None,
        curr_window_ref_obj_rot_mat=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
        start_phase=False,
        end_phase=False,
        end_frame_obj_rot_mat=None,
        available_conditions=None,
        available_conditions_wrist_relative=None,
        end_height=None,
        use_feet_contact=False,
    ):
        # Combine all the guidance we need during denoising step.

        # 1. Match the input conditions, especially the start and end object position.
        # loss_match_cond = self.apply_matching_condition_guidance(pred_clean_x, x_pose_cond, cond_mask)

        # 2. Feet (the one that is supporting each frame) and floor should be in contact.
        loss_feet_floor_contact = self.apply_feet_floor_contact_guidance(
            pred_clean_x,
            rest_human_offsets,
            data_dict,
            contact_labels=contact_labels,
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            prev_window_cano_rot_mat=prev_window_cano_rot_mat,
            prev_window_init_root_trans=prev_window_init_root_trans,
            use_feet_contact=use_feet_contact,
        )

        # 3. Hand and object should be in contact, not penetrating, temporal consistent.
        # if available_conditions is None and available_conditions_wrist_relative is None:
        loss_hand_object_interaction = self.apply_hand_object_interaction_guidance_loss(
            pred_clean_x,
            x_pose_cond,
            rest_human_offsets,
            data_dict,
            contact_labels=contact_labels,
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            prev_window_cano_rot_mat=prev_window_cano_rot_mat,
            prev_window_init_root_trans=prev_window_init_root_trans,
            start_phase=start_phase,
            end_phase=end_phase,
            end_frame_obj_rot_mat=end_frame_obj_rot_mat,
            end_height=end_height,
            use_feet_contact=use_feet_contact,
        )

        # 4. avoid big velocity.
        loss_velocity = self.apply_velocity_guidance_loss(
            pred_clean_x,
            rest_human_offsets,
            data_dict,
            contact_labels=contact_labels,
            curr_window_ref_obj_rot_mat=curr_window_ref_obj_rot_mat,
            prev_window_cano_rot_mat=prev_window_cano_rot_mat,
            prev_window_init_root_trans=prev_window_init_root_trans,
        )

        loss = (
            loss_hand_object_interaction * 10000
            + loss_feet_floor_contact * 100000 * 3
            + loss_velocity * 1000000
        )

        return loss

    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: K X 3
        ball_color = np.asarray([22, 173, 100])  # green

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])

            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=ball_color,
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

    def gen_vis_res_long_seq(
        self,
        all_res_list,
        ref_obj_rot_mat,
        ref_data_dict,
        step: Union[str, int],
        planned_waypoints_pos: torch.Tensor,
        curr_object_name: str,
        vis_gt: bool = False,
        vis_tag: Optional[str] = None,
        vis_wo_scene: bool = False,
        cano_quat: Optional[torch.Tensor] = None,
        dest_mesh_vis_folder: Optional[str] = None,
        finger_all_res_list: Optional[torch.Tensor] = None,
    ):
        human_object_results_path = None
        self.save_obj_info = {}

        # Prepare list used for evaluation.
        human_verts_list = []
        human_mesh_faces_list = []
        human_root_pos_list = []
        human_jnts_pos_list = []
        human_jnts_local_rot_aa_list = []

        finger_jnts_local_rot_aa_list = []

        obj_verts_list = []
        obj_mesh_faces_list = []
        obj_pos_list = []
        obj_rot_mat_list = []

        # dest_out_vid_path = None

        # all_res_list: N X T X (3+9)
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3]  # N X T X 3
        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(
            pred_normalized_obj_trans
        )

        if self.use_first_frame_bps or self.use_random_frame_bps:
            reference_obj_rot_mat = ref_obj_rot_mat.repeat(
                1, all_res_list.shape[1], 1, 1
            )  # N X 1 X 3 X 3

            pred_obj_rel_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3
            pred_obj_rot_mat = self.ds.rel_rot_to_seq(
                pred_obj_rel_rot_mat, reference_obj_rot_mat
            )
        else:
            assert True, "Shouldn't be used!!!"
            pred_obj_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3

        num_joints = 24

        if self.pred_human_motion:
            normalized_global_jpos = all_res_list[
                :, :, 3 + 9 : 3 + 9 + num_joints * 3
            ].reshape(num_seq, -1, num_joints, 3)
            global_jpos = self.ds.de_normalize_jpos_min_max(
                normalized_global_jpos.reshape(-1, num_joints, 3)
            )
        else:  # Not used!!!
            assert True, "Shouldn't be used!!!"

        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        human_jnts_global_rot_6d = all_res_list[
            :, :, 3 + 9 + 24 * 3 : 3 + 9 + 24 * 3 + 22 * 6
        ].reshape(num_seq, -1, 22, 6)
        human_jnts_global_rot_mat = transforms.rotation_6d_to_matrix(
            human_jnts_global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = (
            ref_data_dict["trans2joint"].to(all_res_list.device).squeeze(1)
        )  # BS X  3
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1)  # N X 24 X 3

        if finger_all_res_list is not None:
            pred_finger_6d = finger_all_res_list.reshape(
                num_seq, -1, 30, 6
            )  # N X T X 30 X 6
            pred_finger_aa_rep = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pred_finger_6d)
            )  # N X T X 30 X 3

        for idx in range(num_seq):
            curr_global_rot_mat = human_jnts_global_rot_mat[idx]  # T X 22 X 3 X 3
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)  # T X 22 X 3 X 3
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )  # T X 22 X 3
            if finger_all_res_list is not None:
                curr_local_rot_aa_rep = torch.cat(
                    (curr_local_rot_aa_rep, pred_finger_aa_rep[idx]), dim=-2
                )  # T X 52 X 3

            curr_global_root_jpos = global_root_jpos[idx]  # T X 3

            curr_trans2joint = trans2joint[idx : idx + 1].clone()  # 1 X 3

            root_trans = curr_global_root_jpos + curr_trans2joint.to(
                curr_global_root_jpos.device
            )  # T X 3

            # Generate global joint position
            bs = 1
            betas = ref_data_dict["betas"][0]
            gender = ref_data_dict["gender"][0]

            curr_obj_rot_mat = pred_obj_rot_mat[idx]  # T X 3 X 3
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(
                curr_obj_quat
            )  # Potentially avoid some prediction not satisfying rotation matrix requirements.

            object_name = curr_object_name

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].cuda(),
                curr_local_rot_aa_rep[None].cuda(),
                betas.cuda(),
                [gender],
                self.ds.bm_dict,
                return_joints24=True,
            )

            # Get object verts
            obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(
                object_name
            )
            obj_rest_verts = torch.from_numpy(obj_rest_verts)

            obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(
                curr_obj_rot_mat,
                pred_seq_com_pos[idx],
                obj_rest_verts.float().to(pred_seq_com_pos.device),
            )

            self.save_obj_info["obj_rot_mat"] = (
                curr_obj_rot_mat.detach().cpu().numpy()
            )  # T X 3 X 3
            self.save_obj_info["obj_com_pos"] = (
                pred_seq_com_pos[idx].detach().cpu().numpy()
            )  # T X 3

            human_verts_list.append(mesh_verts[0])
            human_mesh_faces_list.append(mesh_faces)
            human_root_pos_list.append(root_trans)
            human_jnts_pos_list.append(mesh_jnts[0])
            human_jnts_local_rot_aa_list.append(curr_local_rot_aa_rep[:, :22])

            if finger_all_res_list is not None:
                finger_jnts_local_rot_aa_list.append(pred_finger_aa_rep[idx])

            obj_verts_list.append(obj_mesh_verts)
            obj_mesh_faces_list.append(obj_mesh_faces)
            obj_pos_list.append(pred_seq_com_pos[idx])
            obj_rot_mat_list.append(curr_obj_rot_mat)

            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(
                        self.vis_folder, "blender_mesh_vis"
                    )
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag)

            if os.path.exists(dest_mesh_vis_folder):
                import shutil

                shutil.rmtree(dest_mesh_vis_folder)
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)
            print("Visualizing results for {}".format(dest_mesh_vis_folder))

            if vis_gt:
                ball_mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "ball_objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt.mp4",
                )
            else:
                ball_mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "ball_objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_sideview_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "sideview_imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )

                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )
                out_sideview_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "sideview_vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )

                if vis_wo_scene:
                    ball_mesh_save_folder = ball_mesh_save_folder + "_vis_no_scene"
                    mesh_save_folder = mesh_save_folder + "_vis_no_scene"
                    out_rendered_img_folder = out_rendered_img_folder + "_vis_no_scene"
                    out_vid_file_path = out_vid_file_path.replace(
                        ".mp4", "_vis_no_scene.mp4"
                    )

            # end_object_mesh = gt_obj_mesh_verts[actual_len-1] # Nv X 3
            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")
            start_mesh_path = os.path.join(ball_mesh_save_folder, "start_object.ply")
            end_mesh_path = os.path.join(ball_mesh_save_folder, "end_object.ply")
            # self.export_to_mesh(start_object_mesh, obj_mesh_faces, start_mesh_path)
            if self.add_waypoints_xy:
                waypoints_list = []

                for t_idx in range(planned_waypoints_pos.shape[0]):
                    selected_waypoint = planned_waypoints_pos[t_idx : t_idx + 1]
                    if t_idx % 4 != 0:
                        selected_waypoint[:, 2] = 0.05
                    waypoints_list.append(selected_waypoint)
                ball_for_vis_data = torch.cat(waypoints_list, dim=0)  # K X 3

                if cano_quat is not None:
                    cano_quat_for_ball = transforms.quaternion_invert(
                        cano_quat[0:1].repeat(ball_for_vis_data.shape[0], 1)
                    )  # K X 4
                    ball_for_vis_data = transforms.quaternion_apply(
                        cano_quat_for_ball.to(ball_for_vis_data.device),
                        ball_for_vis_data,
                    )

                self.create_ball_mesh(ball_for_vis_data.cpu(), ball_mesh_path)

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3
                # obj_mesh_verts: T X Nv' X 3
                # cano_quat: K X 4
                cano_quat_for_human = transforms.quaternion_invert(
                    cano_quat[0:1][None].repeat(
                        mesh_verts.shape[1], mesh_verts.shape[2], 1
                    )
                )  # T X Nv X 4
                cano_quat_for_obj = transforms.quaternion_invert(
                    cano_quat[0:1][None].repeat(
                        obj_mesh_verts.shape[0], obj_mesh_verts.shape[1], 1
                    )
                )  # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(
                    cano_quat_for_human.to(mesh_verts.device), mesh_verts[0]
                )
                obj_mesh_verts = transforms.quaternion_apply(
                    cano_quat_for_obj.to(obj_mesh_verts.device), obj_mesh_verts
                )

                save_verts_faces_to_mesh_file_w_object(
                    mesh_verts.detach().cpu().numpy(),
                    mesh_faces.detach().cpu().numpy(),
                    obj_mesh_verts.detach().cpu().numpy(),
                    obj_mesh_faces,
                    mesh_save_folder,
                )
            else:
                save_verts_faces_to_mesh_file_w_object(
                    mesh_verts.detach().cpu().numpy()[0],
                    mesh_faces.detach().cpu().numpy(),
                    obj_mesh_verts.detach().cpu().numpy(),
                    obj_mesh_faces,
                    mesh_save_folder,
                )

            # Save human and object info, only for the final results.
            if finger_all_res_list is not None:
                human_object_results = {
                    "human_root_pos": human_root_pos_list[-1],
                    "human_jnts_pos": human_jnts_pos_list[-1],
                    "human_jnts_local_rot_aa": human_jnts_local_rot_aa_list[-1],
                    "finger_jnts_local_rot_aa": finger_jnts_local_rot_aa_list[-1],
                    "obj_pos": obj_pos_list[-1],
                    "obj_rot_mat": obj_rot_mat_list[-1],
                    "object_name": curr_object_name,
                }
                human_object_results_path = os.path.join(
                    mesh_save_folder, "human_object_results.pkl"
                )
                with open(human_object_results_path, "wb") as f:
                    pickle.dump(human_object_results, f)
                print(
                    "Saving human and object results to {}".format(
                        human_object_results_path
                    )
                )

            if idx > 1:
                break

        return (
            dest_mesh_vis_folder,
            human_verts_list,
            human_root_pos_list,
            human_jnts_pos_list,
            human_jnts_local_rot_aa_list,
            finger_jnts_local_rot_aa_list,
            obj_verts_list,
            obj_pos_list,
            obj_rot_mat_list,
            human_object_results_path,
        )

    def gen_vis_res(
        self,
        all_res_list,
        data_dict,
        step,
        cond_mask,
        vis_gt=False,
        vis_tag=None,
        planned_end_obj_com=None,
        move_to_planned_path=None,
        planned_waypoints_pos=None,
        vis_long_seq=False,
        overlap_frame_num=1,
        planned_scene_names=None,
        planned_path_floor_height=None,
        vis_wo_scene=False,
        text_anno=None,
        cano_quat=None,
        gen_long_seq=False,
        curr_object_name=None,
        dest_out_vid_path=None,
        dest_mesh_vis_folder=None,
        save_obj_only=False,
        finger_all_res_list: Optional[torch.Tensor] = None,
        save_motion_params: bool = False,
    ):
        """Save mesh files for visualization.

        Only process the first sample in the batch.

        """

        human_object_results_path = None

        # Prepare list used for evaluation.
        human_jnts_list = []
        human_verts_list = []
        obj_verts_list = []
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = []

        human_root_pos_list = []
        human_jnts_pos_list = []
        human_jnts_local_rot_aa_list = []

        finger_jnts_local_rot_aa_list = []

        obj_pos_list = []
        obj_rot_mat_list = []
        # dest_out_vid_path = None

        # all_res_list: N X T X (3+9)
        num_seq = all_res_list.shape[0]

        pred_normalized_obj_trans = all_res_list[:, :, :3]  # N X T X 3
        pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(
            pred_normalized_obj_trans
        )

        if self.use_first_frame_bps or self.use_random_frame_bps:
            reference_obj_rot_mat = data_dict["reference_obj_rot_mat"].repeat(
                1, all_res_list.shape[1], 1, 1
            )  # N X 1 X 3 X 3

            pred_obj_rel_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3
            pred_obj_rot_mat = self.ds.rel_rot_to_seq(
                pred_obj_rel_rot_mat, reference_obj_rot_mat
            )
        else:  # Not used!!!
            pred_obj_rot_mat = all_res_list[:, :, 3 : 3 + 9].reshape(
                num_seq, -1, 3, 3
            )  # N X T X 3 X 3
        num_joints = 24

        if self.pred_human_motion:
            normalized_global_jpos = all_res_list[
                :, :, 3 + 9 : 3 + 9 + num_joints * 3
            ].reshape(num_seq, -1, num_joints, 3)
            global_jpos = self.ds.de_normalize_jpos_min_max(
                normalized_global_jpos.reshape(-1, num_joints, 3)
            )
        else:  # Not used!!!
            global_jpos = data_dict["ori_motion"][:, :, : num_joints * 3]

        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3

        # For putting human into 3D scene
        if move_to_planned_path is not None:
            pred_seq_com_pos = pred_seq_com_pos + move_to_planned_path
            global_jpos = global_jpos + move_to_planned_path[:, :, None, :]

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        global_rot_6d = all_res_list[
            :, :, 3 + 9 + 24 * 3 : 3 + 9 + 24 * 3 + 22 * 6
        ].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(
            global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = (
            data_dict["trans2joint"].to(all_res_list.device).squeeze(1)
        )  # BS X  3
        seq_len = data_dict[
            "seq_len"
        ]  # BS, should only be used during for single window generation.
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1)  # N X 24 X 3
            seq_len = seq_len.repeat(num_seq)  # N
        seq_len = seq_len.detach().cpu().numpy()  # N

        if finger_all_res_list is not None:
            pred_finger_6d = finger_all_res_list.reshape(
                num_seq, -1, 30, 6
            )  # N X T X 30 X 6
            pred_finger_aa_rep = transforms.matrix_to_axis_angle(
                transforms.rotation_6d_to_matrix(pred_finger_6d)
            )  # N X T X 30 X 3

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx]  # T X 22 X 3 X 3
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)  # T X 22 X 3 X 3
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
                curr_local_rot_mat
            )  # T X 22 X 3
            if finger_all_res_list is not None:
                curr_local_rot_aa_rep = torch.cat(
                    (curr_local_rot_aa_rep, pred_finger_aa_rep[idx]), dim=-2
                )  # T X 52 X 3

            curr_global_root_jpos = global_root_jpos[idx]  # T X 3

            curr_trans2joint = trans2joint[idx : idx + 1].clone()  # 1 X 3

            root_trans = curr_global_root_jpos + curr_trans2joint.to(
                curr_global_root_jpos.device
            )  # T X 3

            # Generate global joint position
            bs = 1
            betas = data_dict["betas"][0]
            gender = data_dict["gender"][0]

            curr_gt_obj_rot_mat = data_dict["obj_rot_mat"][0]  # T X 3 X 3
            curr_gt_obj_com_pos = data_dict["obj_com_pos"][0]  # T X 3

            curr_obj_rot_mat = pred_obj_rot_mat[idx]  # T X 3 X 3
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(
                curr_obj_quat
            )  # Potentially avoid some prediction not satisfying rotation matrix requirements.

            if curr_object_name is not None:
                object_name = curr_object_name
            else:
                curr_seq_name = data_dict["seq_name"][0]
                object_name = data_dict["obj_name"][0]

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].cuda(),
                curr_local_rot_aa_rep[None].cuda(),
                betas.cuda(),
                [gender],
                self.ds.bm_dict,
                return_joints24=True,
            )

            # Get object verts
            obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(
                object_name
            )
            obj_rest_verts = torch.from_numpy(obj_rest_verts)

            gt_obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(
                curr_gt_obj_rot_mat, curr_gt_obj_com_pos, obj_rest_verts.float()
            )
            obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(
                curr_obj_rot_mat,
                pred_seq_com_pos[idx],
                obj_rest_verts.float().to(pred_seq_com_pos.device),
            )

            actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0])
            obj_verts_list.append(obj_mesh_verts)
            trans_list.append(root_trans)

            human_mesh_faces_list.append(mesh_faces)
            obj_mesh_faces_list.append(obj_mesh_faces)

            human_root_pos_list.append(root_trans)
            human_jnts_pos_list.append(mesh_jnts[0])
            human_jnts_local_rot_aa_list.append(curr_local_rot_aa_rep[:, :22])

            if finger_all_res_list is not None:
                finger_jnts_local_rot_aa_list.append(pred_finger_aa_rep[idx])

            obj_pos_list.append(pred_seq_com_pos[idx])
            obj_rot_mat_list.append(curr_obj_rot_mat)

            if finger_all_res_list is not None and save_motion_params:
                human_object_results = {
                    "human_root_pos": human_root_pos_list[-1],
                    "human_jnts_pos": human_jnts_pos_list[-1],
                    "human_jnts_local_rot_aa": human_jnts_local_rot_aa_list[-1],
                    "finger_jnts_local_rot_aa": finger_jnts_local_rot_aa_list[-1],
                    "obj_pos": obj_pos_list[-1],
                    "obj_rot_mat": obj_rot_mat_list[-1],
                    "object_name": curr_object_name,
                }
                human_object_results_path = os.path.join(
                    mesh_save_folder, "human_object_results.pkl"
                )
                with open(human_object_results_path, "wb") as f:
                    pickle.dump(human_object_results, f)
                print(
                    "Saving human and object results to {}".format(
                        human_object_results_path
                    )
                )

            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(
                        self.vis_folder, "blender_mesh_vis"
                    )
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag)

            if not vis_gt and os.path.exists(dest_mesh_vis_folder):
                import shutil

                shutil.rmtree(dest_mesh_vis_folder)
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)
            print("Visualizing results for {}".format(dest_mesh_vis_folder))

            if vis_gt:
                ball_mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "ball_objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt.mp4",
                )
            else:
                ball_mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "ball_objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )
                out_sideview_rendered_img_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "sideview_imgs_step_" + str(step) + "_bs_idx_" + str(idx),
                )

                out_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )
                out_sideview_vid_file_path = os.path.join(
                    dest_mesh_vis_folder,
                    "sideview_vid_step_" + str(step) + "_bs_idx_" + str(idx) + ".mp4",
                )

                if vis_wo_scene:
                    ball_mesh_save_folder = ball_mesh_save_folder + "_vis_no_scene"
                    mesh_save_folder = mesh_save_folder + "_vis_no_scene"
                    out_rendered_img_folder = out_rendered_img_folder + "_vis_no_scene"
                    out_vid_file_path = out_vid_file_path.replace(
                        ".mp4", "_vis_no_scene.mp4"
                    )

            if text_anno is not None:
                out_vid_file_path.replace(
                    ".mp4", "_" + text_anno.replace(" ", "_") + ".mp4"
                )

            start_obj_com_pos = data_dict["ori_obj_motion"][0, 0:1, :3]  # 1 X 3
            if planned_end_obj_com is None:
                end_obj_com_pos = data_dict["ori_obj_motion"][
                    0, actual_len - 1 : actual_len, :3
                ]  # 1 X 3
            else:
                end_obj_com_pos = planned_end_obj_com[idx].to(
                    start_obj_com_pos.device
                )  # 1 X 3
            start_object_mesh = gt_obj_mesh_verts[0]  # Nv X 3
            if move_to_planned_path is not None:
                start_object_mesh += move_to_planned_path[idx].to(
                    start_object_mesh.device
                )
            end_object_mesh = gt_obj_mesh_verts[actual_len - 1]  # Nv X 3
            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")
            start_mesh_path = os.path.join(ball_mesh_save_folder, "start_object.ply")
            end_mesh_path = os.path.join(ball_mesh_save_folder, "end_object.ply")
            self.export_to_mesh(start_object_mesh, obj_mesh_faces, start_mesh_path)
            if self.add_waypoints_xy:
                if planned_waypoints_pos is not None:
                    if planned_path_floor_height is None:
                        num_waypoints = planned_waypoints_pos[idx].shape[0]
                        for tmp_idx in range(num_waypoints):
                            if (tmp_idx + 1) % 4 != 0:
                                planned_waypoints_pos[idx, tmp_idx, 2] = 0.05
                    else:
                        planned_waypoints_pos[idx, :, 2] = (
                            planned_path_floor_height + 0.05
                        )

                    if move_to_planned_path is None:
                        ball_for_vis_data = torch.cat(
                            (
                                start_obj_com_pos,
                                planned_waypoints_pos[idx].to(end_obj_com_pos.device),
                                end_obj_com_pos,
                            ),
                            dim=0,
                        )
                    else:
                        ball_for_vis_data = torch.cat(
                            (
                                start_obj_com_pos
                                + move_to_planned_path[idx].to(
                                    start_obj_com_pos.device
                                ),
                                planned_waypoints_pos[idx].to(end_obj_com_pos.device),
                                end_obj_com_pos,
                            ),
                            dim=0,
                        )
                    # ball_for_vis_data: K X 3
                    #
                    if cano_quat is not None:
                        cano_quat_for_ball = transforms.quaternion_invert(
                            cano_quat[0:1].repeat(ball_for_vis_data.shape[0], 1)
                        )  # K X 4
                        ball_for_vis_data = transforms.quaternion_apply(
                            cano_quat_for_ball.to(ball_for_vis_data.device),
                            ball_for_vis_data,
                        )

                    self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)
                else:
                    curr_cond_mask = cond_mask[idx, :, 0]  # T
                    waypoints_list = [start_obj_com_pos]
                    end_obj_com_pos_xy = end_obj_com_pos.clone()
                    # end_obj_com_pos_xy[:, 2] = 0.05
                    waypoints_list.append(end_obj_com_pos_xy)
                    curr_timesteps = curr_cond_mask.shape[0]
                    for t_idx in range(curr_timesteps):
                        if curr_cond_mask[t_idx] == 0 and t_idx != 0:
                            selected_waypoint = data_dict["ori_obj_motion"][
                                idx, t_idx : t_idx + 1, :3
                            ]
                            selected_waypoint[:, 2] = 0.05
                            waypoints_list.append(selected_waypoint)

                    ball_for_vis_data = torch.cat(waypoints_list, dim=0)  # K X 3
                    self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)

            # For faster debug visualization!!
            # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
            # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3
                # obj_mesh_verts: T X Nv' X 3
                # cano_quat: K X 4
                cano_quat_for_human = transforms.quaternion_invert(
                    cano_quat[0:1][None].repeat(
                        mesh_verts.shape[1], mesh_verts.shape[2], 1
                    )
                )  # T X Nv X 4
                cano_quat_for_obj = transforms.quaternion_invert(
                    cano_quat[0:1][None].repeat(
                        obj_mesh_verts.shape[0], obj_mesh_verts.shape[1], 1
                    )
                )  # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(
                    cano_quat_for_human.to(mesh_verts.device), mesh_verts[0]
                )
                obj_mesh_verts = transforms.quaternion_apply(
                    cano_quat_for_obj.to(obj_mesh_verts.device), obj_mesh_verts
                )

                save_verts_faces_to_mesh_file_w_object(
                    mesh_verts.detach().cpu().numpy(),
                    mesh_faces.detach().cpu().numpy(),
                    obj_mesh_verts.detach().cpu().numpy(),
                    obj_mesh_faces,
                    mesh_save_folder,
                )
            else:
                if gen_long_seq:
                    save_verts_faces_to_mesh_file_w_object(
                        mesh_verts.detach().cpu().numpy()[0],
                        mesh_faces.detach().cpu().numpy(),
                        obj_mesh_verts.detach().cpu().numpy(),
                        obj_mesh_faces,
                        mesh_save_folder,
                    )
                else:  # For single window
                    save_verts_faces_to_mesh_file_w_object(
                        mesh_verts.detach().cpu().numpy()[0][: seq_len[idx]],
                        mesh_faces.detach().cpu().numpy(),
                        obj_mesh_verts.detach().cpu().numpy()[: seq_len[idx]],
                        obj_mesh_faces,
                        mesh_save_folder,
                    )

                # TODO: save videos

            if idx > 1:
                break

        return (
            human_verts_list,
            human_jnts_list,
            trans_list,
            global_rot_mat,
            pred_seq_com_pos,
            pred_obj_rot_mat,
            obj_verts_list,
            human_mesh_faces_list,
            obj_mesh_faces_list,
            dest_out_vid_path,
            human_object_results_path,
        )

    def calc_jpos_from_interaction_res(self, all_res_list, ref_data_dict, idx=0):
        num_seq = all_res_list.shape[0]

        num_joints = 24
        normalized_global_jpos = all_res_list[
            :, :, 3 + 9 : 3 + 9 + num_joints * 3
        ].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        global_rot_6d = all_res_list[
            :, :, 3 + 9 + 24 * 3 : 3 + 9 + 24 * 3 + 22 * 6
        ].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(
            global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = (
            ref_data_dict["trans2joint"].to(all_res_list.device).squeeze(1)
        )  # BS X  3
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1)  # N X 24 X 3

        curr_global_rot_mat = global_rot_mat[idx]  # T X 22 X 3 X 3
        curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat)  # T X 22 X 3 X 3
        curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(
            curr_local_rot_mat
        )  # T X 22 X 3

        curr_global_root_jpos = global_root_jpos[idx]  # T X 3

        curr_trans2joint = trans2joint[idx : idx + 1].clone()  # 1 X 3

        root_trans = curr_global_root_jpos + curr_trans2joint.to(
            curr_global_root_jpos.device
        )  # T X 3

        # Generate global joint position
        betas = ref_data_dict["betas"][idx]
        gender = ref_data_dict["gender"][idx]

        # Get human verts
        mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
            root_trans[None].cuda(),
            curr_local_rot_aa_rep[None].cuda(),
            betas.cuda(),
            [gender],
            self.ds.bm_dict,
            return_joints24=True,
        )

        return (
            mesh_jnts[0],
            root_trans,
            curr_local_rot_aa_rep,
        )  # T X 24 X 3, T X 3, T X 22 X 3

    def load_end_frame_height_heuristics(self, action_name, object_name):
        assert action_name == "lift" or action_name == "push" or action_name == "pull"
        if action_name != "lift":
            assert object_name in ["largebox", "smallbox", "woodchair"]
        heuristic_dict = {}

        heuristic_dict["push"] = {}
        heuristic_dict["pull"] = {}
        heuristic_dict["lift"] = {}
        # heuristic_dict['kick'] = {}

        # 1. Floorlamp type
        heuristic_dict["push"]["floorlamp"] = [0.8, 0.9]  # not used!
        heuristic_dict["pull"]["floorlamp"] = [0.8, 0.9]
        heuristic_dict["lift"]["floorlamp"] = [1.1, 1.3]
        # heuristic_dict['kick']['floorlamp'] = [object_static_height_floor['floorlamp'], object_static_height_floor['floorlamp']+0.01]

        heuristic_dict["push"]["floorlamp1"] = [0.8, 0.9]  # not used!
        heuristic_dict["pull"]["floorlamp1"] = [0.8, 0.9]
        heuristic_dict["lift"]["floorlamp1"] = [1.1, 1.3]

        heuristic_dict["push"]["clothesstand"] = [0.4, 0.55]  # not used!
        heuristic_dict["pull"]["clothesstand"] = [0.4, 0.55]
        heuristic_dict["lift"]["clothesstand"] = [0.6, 0.8]
        # heuristic_dict['kick']['clothesstand'] = [object_static_height_floor['clothesstand'], object_static_height_floor['clothesstand']+0.01]

        heuristic_dict["push"]["tripod"] = [0.4, 0.55]  # not used!
        heuristic_dict["pull"]["tripod"] = [0.4, 0.55]
        heuristic_dict["lift"]["tripod"] = [0.6, 0.8]
        # heuristic_dict['kick']['tripod'] = [object_static_height_floor['tripod'], object_static_height_floor['tripod']+0.01]

        # 2. Table type
        heuristic_dict["push"]["largetable"] = [0.175, 0.176]
        heuristic_dict["pull"]["largetable"] = [0.175, 0.176]
        heuristic_dict["lift"]["largetable"] = [0.9, 1.0]
        # heuristic_dict['kick']['largetable'] = [object_static_height_floor['largetable'], object_static_height_floor['largetable']+0.01]

        heuristic_dict["push"]["smalltable"] = [0.26, 0.27]
        heuristic_dict["pull"]["smalltable"] = [0.26, 0.31]
        heuristic_dict["lift"]["smalltable"] = [0.8, 0.9]
        # heuristic_dict['kick']['smalltable'] = [object_static_height_floor['smalltable'], object_static_height_floor['smalltable']+0.01]

        # 3. Chair type
        heuristic_dict["push"]["woodchair"] = [0.44, 0.45]
        heuristic_dict["pull"]["woodchair"] = [0.48, 0.50]
        heuristic_dict["lift"]["woodchair"] = [0.8, 1.0]
        # heuristic_dict['kick']['woodchair'] = [object_static_height_floor['woodchair'], object_static_height_floor['woodchair']+0.01]

        heuristic_dict["push"]["whitechair"] = [0.46, 0.47]
        heuristic_dict["pull"]["whitechair"] = [0.50, 0.52]
        heuristic_dict["lift"]["whitechair"] = [0.8, 1.0]
        # heuristic_dict['kick']['whitechair'] = [object_static_height_floor['whitechair'], object_static_height_floor['whitechair']+0.01]

        # 4. Box type
        heuristic_dict["push"]["smallbox"] = [0.062, 0.068]
        heuristic_dict["pull"]["smallbox"] = [0.062, 0.068]
        heuristic_dict["lift"]["smallbox"] = [0.8, 0.9]
        # heuristic_dict['kick']['smallbox'] = [object_static_height_floor['smallbox'], object_static_height_floor['smallbox']+0.01]

        heuristic_dict["push"]["largebox"] = [0.155, 0.16]
        heuristic_dict["pull"]["largebox"] = [0.155, 0.16]
        heuristic_dict["lift"]["largebox"] = [0.8, 0.9]
        # heuristic_dict['kick']['largebox'] = [object_static_height_floor['largebox'], object_static_height_floor['largebox']+0.01]

        heuristic_dict["push"]["plasticbox"] = [0.08, 0.09]
        heuristic_dict["pull"]["plasticbox"] = [0.13, 0.14]
        heuristic_dict["lift"]["plasticbox"] = [0.7, 0.8]
        # heuristic_dict['kick']['plasticbox'] = [object_static_height_floor['plasticbox'], object_static_height_floor['plasticbox']+0.01]

        heuristic_dict["push"]["suitcase"] = [0.322, 0.324]
        heuristic_dict["pull"]["suitcase"] = [0.322, 0.324]
        heuristic_dict["lift"]["suitcase"] = [0.8, 0.9]
        # heuristic_dict['kick']['suitcase'] = [object_static_height_floor['suitcase'], object_static_height_floor['suitcase']+0.01]

        # 5. Monitor type
        heuristic_dict["push"]["monitor"] = [0.23, 0.25]  # not used!
        heuristic_dict["pull"]["monitor"] = [0.23, 0.28]
        heuristic_dict["lift"]["monitor"] = [0.75, 0.9]
        # heuristic_dict['kick']['monitor'] = [object_static_height_floor['monitor'], object_static_height_floor['monitor']+0.01]

        # 6. Trashcan type
        heuristic_dict["push"]["trashcan"] = [0.15, 0.155]
        heuristic_dict["pull"]["trashcan"] = [0.15, 0.155]
        heuristic_dict["lift"]["trashcan"] = [0.7, 0.8]
        # heuristic_dict['kick']['trashcan'] = [object_static_height_floor['trashcan'], object_static_height_floor['trashcan']+0.01]

        if object_name not in heuristic_dict[action_name]:
            raise ValueError(
                "Object name not in the heuristic dict for action {0}".format(
                    action_name
                )
            )
        return heuristic_dict[action_name][object_name]

    def gen_language_for_long_seq(self, num_windows, text):
        text_list = []
        for w_idx in range(num_windows):
            text_list.append(text)

        text_clip_feats_list = []
        for window_text in text_list:
            language_input = self.encode_text(window_text)  # 1 X 512
            text_clip_feats_list.append(language_input)

        return text_clip_feats_list

    def call_interaction_model_long_seq(
        self,
        object_data_dict: Dict[str, Any],
        planned_obj_path: torch.Tensor,
        text: str,
        obj_initial_rot_mat: torch.Tensor,
        overlap_frame_num: int,
        prev_navigation_end_human_pose: Optional[torch.Tensor],
        rest_human_offsets: torch.Tensor,
        trans2joint: torch.Tensor,
        curr_object_name: str,
        curr_action_name: str,
        table_height: Optional[float],
        obj_end_rot_mat: Optional[torch.Tensor] = None,
        available_conditions: Optional[Dict] = None,
        available_conditions_wrist_relative: Optional[Dict] = None,
        add_root_ori: bool = False,
        add_feet_contact: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample interaction model for long sequence.

        Args:
            available_conditions = {
                "left_in_contact": True/False,
                "right_in_contact": True/False,
                "left_contact_start_frame": int,
                "left_contact_end_frame": int,
                "right_contact_start_frame": int,
                "right_contact_end_frame": int,
                "left_wrist_pos_in_object": left_wrist_pos_in_object, # BS X 1 X 3
                "right_wrist_pos_in_object": right_wrist_pos_in_object, # BS X 1 X 3
                "left_wrist_rot_mat_in_object": left_wrist_rot_mat_in_object, # BS X 1 X 3 X 3
                "right_wrist_rot_mat_in_object": right_wrist_rot_mat_in_object, # BS X 1 X 3 X 3
                "x_start": x_start, # BS X T X D
            }
            available_conditions_wrist_relative = {
                "default_wrist_relative": wrist_relative, # BS X T X 18
            }

        Returns:
            all_res_list: The predicted human motion and object motion. It's composed of the following:
                - object position: BS X T X 3
                - object relative rotation matrix: BS X T X 9
                - human joint position: BS X T X 24*3
                - human joint orientation: BS X T X 22*6
                All of the above are in the world coordinate system, except for the object orientation
                which is in relative to the object's reference orientation.
                The object position and joint position are normalized.
            pred_feet_contact: The predicted feet contact labels. It's a tensor of shape BS X T X 4.
                Could be None if add_feet_contact is False.

        """
        if rest_human_offsets is None:
            raise ValueError(
                "rest_human_offsets should not be None for interaction sequence."
            )

        if trans2joint is None:
            raise ValueError("trans2joint should not be None for interaction sequence.")

        extra_dim = 0
        if self.use_object_keypoints:
            extra_dim += 4
        if add_feet_contact:
            extra_dim += 4

        start_obj_pos_on_planned_path = planned_obj_path[0:1, :]  # 1 X 3

        curr_height_range = self.load_end_frame_height_heuristics(
            curr_action_name, curr_object_name
        )

        seq_obj_com_pos = torch.zeros(
            prev_navigation_end_human_pose.shape[0],
            (planned_obj_path.shape[0] - 1) * 30,
            3,
        ).cuda()
        seq_obj_com_pos[:, 0:1, :] = start_obj_pos_on_planned_path[
            None
        ].clone()  # unnormalized

        if self.add_waypoints_xy:  # Need to consider overlapped frames.
            waypoints_com_pos = planned_obj_path[None][
                :, 1:, :
            ].cuda()  # BS X (K-1) X 3

            num_pts = waypoints_com_pos.shape[1]
            last_num_pts = num_pts - (num_pts // 4) * 4

            window_cnt = 0

            window_t = (self.window - overlap_frame_num) // 30
            window_cnt = (
                1 + ((planned_obj_path.shape[0] - 5) + (window_t - 1)) // window_t
            )
            for tmp_p_idx in range(num_pts):
                t_idx = (tmp_p_idx + 1) * 30 - 1
                seq_obj_com_pos[:, t_idx, :2] = waypoints_com_pos[:, tmp_p_idx, :2]
                if tmp_p_idx == num_pts - 1:
                    seq_obj_com_pos[:, t_idx, 2] = waypoints_com_pos[
                        :, tmp_p_idx, 2
                    ]  # Put down.
                else:
                    seq_obj_com_pos[:, t_idx, 2] = random.uniform(
                        curr_height_range[0], curr_height_range[1]
                    )

        seq_obj_com_pos_normalized = self.val_ds.normalize_obj_pos_min_max(
            seq_obj_com_pos
        )  # BS X T X 3
        val_obj_data = torch.cat(
            (
                seq_obj_com_pos_normalized,
                torch.zeros(seq_obj_com_pos.shape[0], seq_obj_com_pos.shape[1], 9).to(
                    seq_obj_com_pos.device
                ),
            ),
            dim=-1,
        )  # BS X T X (3+9)
        val_obj_data[:, 0:1, 3:] = obj_initial_rot_mat.to(val_obj_data.device).reshape(
            1, 1, 9
        )  # Reaplce the first frame's object rotation.

        if self.add_language_condition:
            if last_num_pts > 0:
                language_input = self.gen_language_for_long_seq(window_cnt + 1, text)
            else:
                language_input = self.gen_language_for_long_seq(window_cnt, text)
        else:
            language_input = None

        # Manually define contact labels, the heuristic is that the start and end frames are not in contact,
        # the middle frames are in contact.
        # contact_labels = interaction_trainer.gen_contact_label_for_long_seq(actual_num_frames) # T
        # contact_labels = contact_labels[None].repeat(seq_obj_com_pos.shape[0], 1).to(seq_obj_com_pos.device) # BS X T
        contact_labels = None

        if self.use_guidance_in_denoising:
            # Load current sequence's object SDF
            # self.object_sdf, self.object_sdf_centroid, \
            # self.object_sdf_extents = \
            #     self.load_object_sdf_data(curr_object_name)

            guidance_fn = self.apply_different_guidance_loss
        else:
            guidance_fn = None

        # Generate padding mask
        actual_seq_len = (
            torch.ones(prev_navigation_end_human_pose.shape[0], self.window + 1)
            * self.window
            + 1
        )
        tmp_mask = (
            torch.arange(self.window + 1).expand(
                prev_navigation_end_human_pose.shape[0], self.window + 1
            )
            < actual_seq_len
        )
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].cuda()  # 1 X 1 X 121

        bs_window_len = torch.zeros(val_obj_data.shape[0])
        bs_window_len[:] = self.window
        if self.add_waypoints_xy:
            end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(
                torch.zeros(
                    val_obj_data.shape[0], self.window, val_obj_data.shape[-1]
                ).to(val_obj_data.device),
                bs_window_len,
            )
            cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                torch.zeros(
                    val_obj_data.shape[0], self.window, val_obj_data.shape[-1]
                ).to(val_obj_data.device),
                bs_window_len,
            )
            cond_mask = end_pos_cond_mask * cond_mask
        else:
            raise NotImplementedError("Not implemented yet.")

        human_cond_mask = torch.ones(cond_mask.shape[0], cond_mask.shape[1], 204).cuda()
        if self.input_first_human_pose:
            human_cond_mask[:, 0, :] = 0
        cond_mask = torch.cat(
            (cond_mask, human_cond_mask), dim=-1
        )  # BS X T X (3+9+24*3+22*6)

        if extra_dim > 0:
            cond_mask = torch.cat(
                (
                    cond_mask,
                    torch.ones(cond_mask.shape[0], cond_mask.shape[1], extra_dim).to(
                        cond_mask.device
                    ),
                ),
                dim=-1,
            )  # BS X T X (3+9+24*3+22*6+4) 4 for 0/1 contacts.

        tmp_val_human_data = torch.zeros(
            prev_navigation_end_human_pose.shape[0], val_obj_data.shape[1], 204
        )
        tmp_val_obj_data = torch.zeros_like(val_obj_data).to(val_obj_data.device)
        if prev_navigation_end_human_pose is not None:  # unnormalized positions.
            #################################################### pre transformation ####################################################
            ######################################### canonicalize prev_navigation_end_human_pose #########################################
            bs, num_steps, _ = prev_navigation_end_human_pose.shape

            cano_prev_human_pose, cano_seq_obj_com_pos, cano_rot_mat = (
                canonicalize_first_human_and_waypoints(
                    first_human_pose=prev_navigation_end_human_pose,
                    seq_waypoints_pos=seq_obj_com_pos,
                    trans2joint=trans2joint,
                    parents=self.ds.parents,
                    trainer=self,
                    is_interaction=True,
                )
            )
            ######################################### canonicalize prev_navigation_end_human_pose #########################################

            ######################################### human data #########################################
            tmp_val_human_data[:, 0:1, :] = cano_prev_human_pose.clone()
            ######################################### human data #########################################

            ######################################### obj data #########################################
            ##### obj pos #####
            tmp_val_obj_data[:, :, :3] = (
                cano_seq_obj_com_pos.clone()
            )  # already normalized
            ##### obj rot #####
            ori_obj_rot_mat = obj_initial_rot_mat  # BS(1) X 3 X 3, in the global space
            cano_obj_rot_mat = torch.matmul(
                cano_rot_mat, ori_obj_rot_mat.to(cano_rot_mat.device)
            )  # BS(1) X 3 X 3 # in the canonicalized space
            end_obj_rot_mat = obj_end_rot_mat  # BS(1) X 3 X 3, in the global space
            cano_end_obj_rot_mat = torch.matmul(
                cano_rot_mat, end_obj_rot_mat.to(cano_rot_mat.device)
            )  # BS(1) X 3 X 3 # in the canonicalized space

            reference_rot_mat = object_data_dict["reference_obj_rot_mat"][
                :, 0, :, :
            ].cuda()  # BS(1) X 3 X 3
            tmp_val_obj_data[:, 0:1, 3:] = self.ds.prep_rel_obj_rot_mat_w_reference_mat(
                cano_obj_rot_mat[None], reference_rot_mat[None]
            ).reshape(-1, 9)  # BS(1) X 3 X 3
            tmp_val_obj_data[:, -1:, 3:] = self.ds.prep_rel_obj_rot_mat_w_reference_mat(
                cano_end_obj_rot_mat[None], reference_rot_mat[None]
            ).reshape(-1, 9)  # BS(1) X 3 X 3
            ##### obj bps #####
            ori_data_cond = (
                object_data_dict["input_obj_bps"].cuda().reshape(-1, 1, 1024 * 3)
            )
            ######################################### obj data #########################################

            if extra_dim > 0:
                data = torch.cat(
                    (
                        tmp_val_obj_data,
                        tmp_val_human_data.to(tmp_val_obj_data.device),
                        torch.zeros(
                            tmp_val_obj_data.shape[0],
                            tmp_val_obj_data.shape[1],
                            extra_dim,
                        ).to(tmp_val_obj_data.device),
                    ),
                    dim=-1,
                )
            else:
                data = torch.cat(
                    (tmp_val_obj_data, tmp_val_human_data.to(tmp_val_obj_data.device)),
                    dim=-1,
                )
            #################################################### pre transformation ####################################################

            #################################################### sample ####################################################
            all_res_list, original_curr_x = (
                self.ema.ema_model.sample_sliding_window_w_canonical(
                    self.ds,
                    object_data_dict["obj_name"],
                    trans2joint,
                    data,
                    ori_data_cond,
                    cond_mask=cond_mask,
                    padding_mask=padding_mask,
                    overlap_frame_num=overlap_frame_num,
                    input_waypoints=True,
                    language_input=language_input,
                    contact_labels=contact_labels,
                    rest_human_offsets=rest_human_offsets,
                    guidance_fn=guidance_fn,
                    opt_fn=None,
                    data_dict=object_data_dict,
                    available_conditions=available_conditions,
                    available_conditions_wrist_relative=available_conditions_wrist_relative,
                    add_root_ori=add_root_ori,
                    add_feet_contact=add_feet_contact,
                    table_height=table_height,
                )
            )
            #################################################### sample ####################################################

            #################################################### post transformation ####################################################
            pred_feet_contact = None
            if add_feet_contact:
                pred_feet_contact = all_res_list[:, :, -4:].clone()  # BS X T X 4
            if extra_dim > 0:
                all_res_list = all_res_list[:, :, :-extra_dim]
            ######################################### De-canonicalize the result back to the original direction #########################################
            (
                converted_obj_com_pos,
                converted_obj_rot_mat,
                converted_human_jpos,
                converted_rot_6d,
            ) = apply_rotation_to_data(
                self.ds,
                trans2joint,
                cano_rot_mat,
                reference_rot_mat[None],
                all_res_list,
            )

            converted_obj_rel_rot_mat = self.ds.prep_rel_obj_rot_mat_w_reference_mat(
                converted_obj_rot_mat,
                reference_rot_mat.reshape(bs, -1, 3, 3).to(
                    converted_obj_rot_mat.device
                ),
            )
            ######################################### De-canonicalize the result back to the original direction #########################################

            ######################################### Move the traj starting point from zero to the original starting point #########################################
            aligned_human_trans = (
                prev_navigation_end_human_pose[:, :, :3]
                - converted_human_jpos[:, 0:1, 0, :]
            )

            converted_human_jpos += aligned_human_trans[
                :, :, None, :
            ]  # BS(1) X T X 24 X 3
            converted_obj_com_pos += aligned_human_trans  # BS(1) X T X 3

            converted_normalized_obj_com_pos = self.ds.normalize_obj_pos_min_max(
                converted_obj_com_pos
            )
            converted_normalized_human_jpos = self.ds.normalize_jpos_min_max(
                converted_human_jpos
            )
            all_res_list = torch.cat(
                (
                    converted_normalized_obj_com_pos.reshape(bs, -1, 3),
                    converted_obj_rel_rot_mat.reshape(bs, -1, 9),
                    converted_normalized_human_jpos.reshape(bs, -1, 24 * 3),
                    converted_rot_6d.reshape(bs, -1, 22 * 6),
                ),
                dim=-1,
            )
            ######################################### Move the traj starting point from zero to the original starting point #########################################
            #################################################### post transformation ####################################################

            return all_res_list, pred_feet_contact
        else:
            raise NotImplementedError("Not implemented yet.")


def build_wrist_relative_conditions(
    object_data_dict: Dict,
    left_wrist_pose: Optional[Dict],
    right_wrist_pose: Optional[Dict],
    trainer: Trainer,
    left_contact: bool,
    right_contact: bool,
    left_begin_frame: int,
    left_end_frame: int,
    right_begin_frame: int,
    right_end_frame: int,
    contact_begin_frame: int,
    contact_end_frame: int,
):
    reference_obj_rot_mat = object_data_dict["reference_obj_rot_mat"][0, 0]  # 3 X 3
    left_wrist_relative_pos = (
        (reference_obj_rot_mat @ left_wrist_pose["wrist_pos"].reshape(3, 1))
        .reshape(3)
        .cuda()
        if left_wrist_pose
        else None
    )
    left_wrist_relative_rot = (
        (reference_obj_rot_mat @ left_wrist_pose["wrist_rot"].reshape(3, 3))
        .reshape(3, 3)
        .cuda()
        if left_wrist_pose
        else None
    )
    right_wrist_relative_pos = (
        (reference_obj_rot_mat @ right_wrist_pose["wrist_pos"].reshape(3, 1))
        .reshape(3)
        .cuda()
        if right_wrist_pose
        else None
    )
    right_wrist_relative_rot = (
        (reference_obj_rot_mat @ right_wrist_pose["wrist_rot"].reshape(3, 3))
        .reshape(3, 3)
        .cuda()
        if right_wrist_pose
        else None
    )

    left_wrist_pos_in_object = (
        left_wrist_relative_pos.reshape(1, 1, 3).cuda()
        if (left_wrist_relative_pos is not None)
        else torch.zeros((1, 1, 3)).cuda()
    )
    left_wrist_rot_6d_in_object = (
        transforms.matrix_to_rotation_6d(left_wrist_relative_rot)
        .reshape(1, 1, 6)
        .cuda()
        if (left_wrist_relative_rot is not None)
        else torch.zeros((1, 1, 6)).cuda()
    )
    right_wrist_pos_in_object = (
        right_wrist_relative_pos.reshape(1, 1, 3).cuda()
        if (right_wrist_relative_pos is not None)
        else torch.zeros((1, 1, 3)).cuda()
    )
    right_wrist_rot_6d_in_object = (
        transforms.matrix_to_rotation_6d(right_wrist_relative_rot)
        .reshape(1, 1, 6)
        .cuda()
        if (right_wrist_relative_rot is not None)
        else torch.zeros((1, 1, 6)).cuda()
    )

    # Calaulate default_wrist_relative, the mask is calculated on the fly.
    default_wrist_relative = (
        torch.cat(
            (
                trainer.ds.normalize_wrist_relative_pos(left_wrist_pos_in_object),
                left_wrist_rot_6d_in_object,
                trainer.ds.normalize_wrist_relative_pos(right_wrist_pos_in_object),
                right_wrist_rot_6d_in_object,
            ),
            dim=-1,
        )
        .repeat(1, 120, 1)
        .cuda()
    )

    available_conditions_wrist_relative = {
        "left_in_contact": left_contact,
        "right_in_contact": right_contact,
        "left_contact_start_frame": left_begin_frame,
        "left_contact_end_frame": left_end_frame,
        "right_contact_start_frame": right_begin_frame,
        "right_contact_end_frame": right_end_frame,
        "left_wrist_pos_in_object": left_wrist_relative_pos,
        "left_wrist_rot_mat_in_object": left_wrist_relative_rot,
        "right_wrist_pos_in_object": right_wrist_relative_pos,
        "right_wrist_rot_mat_in_object": right_wrist_relative_rot,
        "contact_begin_frame": contact_begin_frame,
        "contact_end_frame": contact_end_frame,
        "default_wrist_relative": default_wrist_relative,
    }
    return available_conditions_wrist_relative


def build_interaction_trainer(
    opt,
    device: str,
    vis_wdir: Optional[str],
    results_folder: str,
    repr_dim: int,
    loss_type: str,
    use_feet_contact: bool,
    use_wandb: bool = False,
    load_ds: bool = False,
) -> Trainer:
    interaction_model = ObjectCondGaussianDiffusion(
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
        input_contact_labels=opt.add_contact_label,
        input_first_human_pose=True,
        input_rest_human_skeleton=opt.add_rest_human_skeleton,
        use_object_keypoints=True,
        use_feet_contact=use_feet_contact,
    )
    interaction_model.to(device)

    interaction_trainer = Trainer(
        opt,
        interaction_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=8000000,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=results_folder,
        vis_folder=vis_wdir,
        use_wandb=use_wandb,
        load_ds=load_ds,
    )
    return interaction_trainer


def run_interaction_trainer_after_sampling(
    trainer: Trainer,
    all_res_list: torch.Tensor,
    idx: int,
    object_data_dict: Dict,
    ref_data_dict: Dict,
    curr_object_name: str,
    async_hand: bool = False,
    contact_labels: Optional[torch.Tensor] = None,
):
    # Extract obj position and rotation.
    obj_com_pos = trainer.ds.de_normalize_obj_pos_min_max(
        all_res_list[idx, :, :3].clone()
    )  # T X 3
    obj_rel_rot_mat = all_res_list[idx, :, 3:12].reshape(-1, 3, 3)  # T X 3 X 3
    reference_obj_rot_mat = object_data_dict["reference_obj_rot_mat"].repeat(
        1, obj_rel_rot_mat.shape[0], 1, 1
    )  # 1 X T X 3 X 3
    obj_rot_mat = trainer.ds.rel_rot_to_seq(
        obj_rel_rot_mat[None], reference_obj_rot_mat
    )[0]
    obj_quat = transforms.matrix_to_quaternion(obj_rot_mat)
    obj_rot_mat = transforms.quaternion_to_matrix(obj_quat)

    # Use FK to calculate the joint position, instead of using the predicted joint position.
    human_jnts, human_root_trans, human_local_rot_aa_reps = (
        trainer.calc_jpos_from_interaction_res(
            all_res_list, ref_data_dict, idx
        )  # T X 24 X 3, T X 3, T X 22 X 3
    )
    human_jnts_rot_mat_global = transforms.rotation_6d_to_matrix(
        all_res_list[idx, :, 12 + 24 * 3 : 12 + 24 * 3 + 22 * 6]
        .clone()
        .reshape(-1, 22, 6)
    ).reshape(-1, 22, 3, 3)  # T X 22 X 3 X 3
    left_wrist_pos = human_jnts[:, 20]  # T X 3
    left_wrist_rot_mat = human_jnts_rot_mat_global[:, 20]  # T X 3 X 3
    right_wrist_pos = human_jnts[:, 21]  # T X 3
    right_wrist_rot_mat = human_jnts_rot_mat_global[:, 21]  # T X 3 X 3

    # Find the contact frames.
    (
        left_contact,
        right_contact,
        contact_begin_frame,
        contact_end_frame,
        left_begin_frame,
        left_end_frame,
        right_begin_frame,
        right_end_frame,
        left_wrist_pose,
        right_wrist_pose,
    ) = find_contact_frames(
        interaction_trainer=trainer,
        object_name=curr_object_name,
        obj_com_pos=obj_com_pos,
        obj_rot_mat=obj_rot_mat,
        left_wrist_pos=left_wrist_pos,
        left_wrist_rot_mat=left_wrist_rot_mat,
        right_wrist_pos=right_wrist_pos,
        right_wrist_rot_mat=right_wrist_rot_mat,
        async_hand=async_hand,
        contact_labels=contact_labels,
    )
    if not async_hand:
        left_begin_frame = contact_begin_frame
        right_begin_frame = contact_begin_frame
        left_end_frame = contact_end_frame
        right_end_frame = contact_end_frame

    print(
        "contact begin: {}, contact end: {}".format(
            contact_begin_frame, contact_end_frame
        )
    )
    print("left_contact, ", left_contact)
    print("right_contact, ", right_contact)
    print("left_begin_frame, ", left_begin_frame, "left_end_frame, ", left_end_frame)
    print(
        "right_begin_frame, ", right_begin_frame, "right_end_frame, ", right_end_frame
    )
    print(
        "contact_begin_frame, ",
        contact_begin_frame,
        "contact_end_frame, ",
        contact_end_frame,
    )
    return (
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
    )


def run_interaction_trainer(
    trainer: Trainer,
    object_data_dict: Dict,
    ref_data_dict: Dict,
    planned_obj_path: torch.Tensor,
    text: str,
    obj_initial_rot_mat: torch.Tensor,
    overlap_frame_num: int,
    prev_navigation_end_human_pose: Optional[torch.Tensor],
    rest_human_offsets: torch.Tensor,
    trans2joint: torch.Tensor,
    obj_end_rot_mat: Optional[torch.Tensor],
    curr_object_name: str,
    curr_action_name: str,
    table_height: Optional[float],
    available_conditions_wrist_relative: Optional[Dict] = None,
    add_root_ori: bool = False,
    add_feet_contact: bool = False,
):
    # Sample interaction motion.
    all_res_list, pred_feet_contact, *_ = trainer.call_interaction_model_long_seq(
        object_data_dict=object_data_dict,
        planned_obj_path=planned_obj_path,
        text=text,
        obj_initial_rot_mat=obj_initial_rot_mat,
        overlap_frame_num=overlap_frame_num,
        prev_navigation_end_human_pose=prev_navigation_end_human_pose,
        rest_human_offsets=rest_human_offsets,
        trans2joint=trans2joint,
        curr_object_name=curr_object_name,
        curr_action_name=curr_action_name,
        table_height=table_height,
        obj_end_rot_mat=obj_end_rot_mat,
        available_conditions_wrist_relative=available_conditions_wrist_relative,
        add_root_ori=add_root_ori,
        add_feet_contact=add_feet_contact,
    )
    saved_all_res_list = all_res_list.clone()

    # NOTE: Assume the batch size is 1.
    idx = 0
    (
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
    ) = run_interaction_trainer_after_sampling(
        trainer=trainer,
        all_res_list=all_res_list,
        idx=idx,
        object_data_dict=object_data_dict,
        ref_data_dict=ref_data_dict,
        curr_object_name=curr_object_name,
    )

    return (
        saved_all_res_list,
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
    )


def find_static_frame_at_end(
    human_jnts: torch.Tensor,
    contact_end_frame: int,
):
    v = human_jnts[1:] - human_jnts[:-1]  # (T-1) X 24 X 3
    v_norm = torch.sum(torch.norm(v, dim=-1), dim=-1)
    v_norm_smooth = medfilt(v_norm.detach().cpu().numpy(), kernel_size=9)
    bone = human_jnts[:, 6] - human_jnts[:, 3]
    bone /= torch.norm(bone, dim=1, keepdim=True)
    cut_frame = v_norm_smooth.shape[0] - 1
    for cut_frame in reversed(range(contact_end_frame + 1, v_norm_smooth.shape[0])):
        if bone[cut_frame][2] < 0.9:
            break
        if v_norm_smooth[cut_frame] > 0.12:
            break
    return cut_frame


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model
    repr_dim = 3 + 9  # Object relative translation and relative rotation matrix

    if True:  # opt.pred_human_motion:
        repr_dim += 24 * 3 + 22 * 6

    if True: # opt.use_object_keypoints:
        repr_dim += 4

    if opt.add_interaction_feet_contact:
        repr_dim += 4

    if opt.add_interaction_root_xy_ori:
        repr_dim += 6

    if opt.add_wrist_relative:
        repr_dim += 18

    if opt.use_l2_loss:
        loss_type = "l2"
    else:
        loss_type = "l1"

    diffusion_model = ObjectCondGaussianDiffusion(
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
        input_contact_labels=opt.add_contact_label,
        input_first_human_pose=True,
        input_rest_human_skeleton=opt.add_rest_human_skeleton,
        use_object_keypoints=True,
        use_feet_contact=opt.add_interaction_feet_contact,
        add_object_in_wrist_loss=opt.add_object_in_wrist_loss,
        add_object_vel_loss=opt.add_object_vel_loss,
    )

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=8000000,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()


def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"

    # Define model
    repr_dim = 3 + 9

    if True:  # opt.pred_human_motion:
        repr_dim += 24 * 3 + 22 * 6

    if True: # opt.use_object_keypoints:
        repr_dim += 4

    if opt.add_interaction_feet_contact:
        repr_dim += 4

    if opt.add_interaction_root_xy_ori:
        repr_dim += 6

    if opt.add_wrist_relative:
        repr_dim += 18

    if opt.use_l2_loss:
        loss_type = "l2"
    else:
        loss_type = "l1"

    diffusion_model = ObjectCondGaussianDiffusion(
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
        input_contact_labels=opt.add_contact_label,
        input_first_human_pose=True,
        input_rest_human_skeleton=opt.add_rest_human_skeleton,
        use_object_keypoints=True,
        use_feet_contact=opt.add_interaction_feet_contact,
    )

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=8000000,
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False,
    )

    trainer.cond_sample_res(milestone=opt.milestone, render_results=True)

    torch.cuda.empty_cache()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--wandb_pj_name", type=str, default="", help="project name")
    parser.add_argument("--entity", default="zhenkirito123", help="W&B entity")
    parser.add_argument("--exp_name", default="", help="save to project/name")
    parser.add_argument("--device", default="0", help="cuda device")

    parser.add_argument("--window", type=int, default=120, help="horizon")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="generator_learning_rate"
    )

    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint")

    parser.add_argument("--data_root_folder", type=str, default="", help="checkpoint")

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

    # For testing sampled results
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results
    parser.add_argument("--test_long_seq", action="store_true")

    # For testing sampled results w planned path
    parser.add_argument("--use_planned_path", action="store_true")

    # For testing sampled results w planned path
    parser.add_argument("--use_long_planned_path", action="store_true")

    # For loss type
    parser.add_argument("--use_l2_loss", action="store_true")

    # Train and test on different objects.
    parser.add_argument("--use_object_split", action="store_true")

    # For adding start and end object position (xyz) and rotation (6D rotation).
    parser.add_argument("--add_start_end_object_pos_rot", action="store_true")

    # For adding start and end object position (xyz).
    parser.add_argument("--add_start_end_object_pos", action="store_true")

    # For adding start and end object position at z plane (xy).
    parser.add_argument("--add_start_end_object_pos_xy", action="store_true")

    # For adding waypoints (xy).
    parser.add_argument("--add_waypoints_xy", action="store_true")

    # Random sample waypoints instead of fixed intervals.
    parser.add_argument("--use_random_waypoints", action="store_true")

    # Add language conditions.
    parser.add_argument("--add_contact_label", action="store_true")

    # Input the first human pose, maybe can connect the windows better.
    parser.add_argument("--remove_target_z", action="store_true")

    # Input the first human pose, maybe can connect the windows better.
    parser.add_argument("--use_guidance_in_denoising", action="store_true")

    parser.add_argument("--use_optimization_in_denoising", action="store_true")

    # Add rest offsets for body shape information.
    parser.add_argument("--add_rest_human_skeleton", action="store_true")

    # Add rest offsets for body shape information.
    parser.add_argument("--use_first_frame_bps", action="store_true")

    # Visualize the results from different noise levels.
    parser.add_argument("--return_diff_level_res", action="store_true")

    parser.add_argument("--input_full_human_pose", action="store_true")

    parser.add_argument(
        "--loss_w_feet",
        type=float,
        default=1,
        help="the loss weight for feet contact loss",
    )

    parser.add_argument(
        "--loss_w_fk", type=float, default=1, help="the loss weight for fk loss"
    )

    parser.add_argument(
        "--loss_w_obj_pts",
        type=float,
        default=1,
        help="the loss weight for object sampling points",
    )

    parser.add_argument(
        "--loss_w_obj_pts_in_hand",
        type=float,
        default=0.5,
        help="the loss weight for object sampling points in wrist frame",
    )

    parser.add_argument(
        "--loss_w_obj_vel",
        type=float,
        default=1,
        help="the loss weight for object velocity",
    )

    # Add extra loss.
    parser.add_argument(
        "--add_object_in_wrist_loss",
        action="store_true",
        help="Add object-hand relative loss.",
    )

    parser.add_argument(
        "--add_object_vel_loss", action="store_true", help="Add object velocity loss."
    )

    # Add extra conditions.
    parser.add_argument(
        "--add_wrist_relative",
        action="store_true",
        help="Add wrist relative pose as condition.",
    )

    parser.add_argument(
        "--add_object_static",
        action="store_true",
        help="Add object static pose as condition.",
    )

    parser.add_argument(
        "--add_interaction_root_xy_ori",
        action="store_true",
        help="Add root xy orientation as condition.",
    )

    parser.add_argument(
        "--add_interaction_feet_contact",
        action="store_true",
        help="Add feet contact as condition.",
    )

    parser.add_argument(
        "--milestone",
        type=str,
        default="10",
    )

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
