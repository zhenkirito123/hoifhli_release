import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import pytorch3d.transforms as transforms
import torch
import trimesh
import wandb
import yaml
from ema_pytorch import EMA
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils import data

from argument_parser import parse_opt
from manip.data.humanml3d_dataset import HumanML3DDataset, quat_ik_torch
from manip.model.transformer_navigation_cond_diffusion import (
    NavigationCondGaussianDiffusion,
)
from manip.utils.trainer_utils import (
    canonicalize_first_human_and_waypoints,
    cycle,
    run_smplx_model,
)
from manip.vis.blender_vis_mesh_motion import (
    save_verts_faces_to_mesh_file,
)

torch.manual_seed(0)
random.seed(0)


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

        self.add_language_condition_for_nav = True
        if self.add_language_condition_for_nav:
            clip_version = "ViT-B/32"
            self.clip_model = self.load_and_freeze_clip(clip_version)

        self.data_root_folder = self.opt.data_root_folder
        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict

        self.test_on_train = self.opt.test_on_train

        self.add_waypoints_xy = True
        self.add_root_xy_ori = self.opt.add_root_xy_ori
        self.use_noisy_traj = self.opt.use_noisy_traj

        self.input_first_human_pose_for_nav = True

        self.add_feet_contact = self.opt.add_feet_contact

        self.loss_w_feet = self.opt.loss_w_feet
        self.loss_w_fk = self.opt.loss_w_fk

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = HumanML3DDataset(
            train=True,
            data_root_folder=self.data_root_folder,
            window=window_size,
            load_ds=self.load_ds,
        )
        val_dataset = HumanML3DDataset(
            train=False,
            data_root_folder=self.data_root_folder,
            window=window_size,
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
        max_text_len = 20  # Specific hardcoding for the current dataset
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

    def prep_mimic_A_star_path_condition_mask_pos_xy_only(self, data, actual_seq_len):
        # data: BS X T X D
        # actual_seq_len: BS
        # The start/end frame, frame 29, 59, 89's xy are given.
        tmp_mask = None
        # Use fixed number of waypoints.
        # random_steps = [30-1, 60-1, 90-1, 120-1] # for FPS 30
        random_steps = [30 - 1, 60 - 1, 90 - 1, 120 - 1]  # for FPS 20
        for selected_t in random_steps:
            if selected_t < self.window:
                bs_selected_t = torch.from_numpy(np.asarray([selected_t]))  # 1
                bs_selected_t = bs_selected_t[None, :].repeat(
                    data.shape[0], self.window
                )  # BS X T

                curr_tmp_mask = torch.arange(self.window).expand(
                    data.shape[0], self.window
                ) == (bs_selected_t)
                # BS X max_timesteps
                curr_tmp_mask = curr_tmp_mask.to(data.device)[:, :, None]  # BS X T X 1

                if tmp_mask is None:
                    tmp_mask = ~curr_tmp_mask
                else:
                    tmp_mask = (~curr_tmp_mask) * tmp_mask

        # Missing regions are ones, the condition regions are zeros.
        mask = torch.ones_like(data[:, :, :2]).to(data.device)  # BS X T X 2
        mask = mask * tmp_mask  # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given.
        # Also, add z mask, only the first frane's z is given.
        rotation_mask = torch.ones_like(data[:, :, 2:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1)

        mask[:, 0, :2] = torch.zeros(data.shape[0], 2).to(data.device)  # BS X 2

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

                # Generate padding mask
                actual_seq_len = (
                    data_dict["seq_len"] + 1
                )  # BS, + 1 since we need additional timestep for noise level
                tmp_mask = torch.arange(self.window + 1).expand(
                    human_data.shape[0], self.window + 1
                ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(human_data.device)

                if self.add_waypoints_xy:
                    cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                        human_data, data_dict["seq_len"]
                    )
                else:
                    cond_mask = torch.ones_like(human_data)

                # Condition on the first frame's human pose.
                if self.input_first_human_pose_for_nav:
                    cond_mask[:, 0, :] = 0

                # Condition on the human root's orientation on the xy plane.
                if self.add_root_xy_ori:
                    root_xy_ori = data_dict["root_traj_xy_ori"].cuda()  # BS X T X 6
                    human_data = torch.cat(
                        (human_data, root_xy_ori), dim=-1
                    )  # BS X T X (24*3 + 22*6 + 6)
                    root_xy_ori_mask = torch.ones_like(root_xy_ori).to(
                        human_data.device
                    )
                    root_xy_ori_mask[:, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :] = 0
                    cond_mask = torch.cat((cond_mask, root_xy_ori_mask), dim=-1)

                with autocast(enabled=self.amp):
                    if self.add_feet_contact:
                        contact_data = data_dict["feet_contact"].cuda()  # BS X T X 4
                        data = torch.cat((human_data, contact_data), dim=-1)
                        cond_mask = torch.cat(
                            (
                                cond_mask,
                                torch.ones_like(contact_data).to(cond_mask.device),
                            ),
                            dim=-1,
                        )
                    else:
                        data = human_data

                    if self.add_language_condition_for_nav:
                        text_anno_data = data_dict["text"]
                        language_input = self.encode_text(text_anno_data)  # BS X 512
                        language_input = language_input.to(data.device)
                    else:
                        language_input = None

                    if self.add_feet_contact:
                        loss_diffusion, loss_feet, loss_fk = self.model(
                            data,
                            cond_mask=cond_mask,
                            padding_mask=padding_mask,
                            language_input=language_input,
                            rest_human_offsets=data_dict["rest_human_offsets"],
                            ds=self.ds,
                            use_noisy_traj=self.use_noisy_traj,
                        )
                        loss = (
                            loss_diffusion
                            + self.loss_w_feet * loss_feet
                            + self.loss_w_fk * loss_fk
                        )
                    else:
                        loss_diffusion = self.model(
                            data,
                            cond_mask=cond_mask,
                            padding_mask=padding_mask,
                            language_input=language_input,
                            use_noisy_traj=self.use_noisy_traj,
                        )

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
                                torch.norm(p.grad.detach(), 2.0).to(human_data.device)
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
                        if self.add_feet_contact:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                                "Train/Loss/Feet Contact Loss": loss_feet.item(),
                                "Train/Loss/FK Loss": loss_fk.item(),
                            }
                        else:
                            log_dict = {
                                "Train/Loss/Total Loss": loss.item(),
                                "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            }
                        wandb.log(log_dict)

                    if idx % 20 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))
                        print("Loss diffusion: %.4f" % (loss_diffusion.item()))
                        if self.add_feet_contact:
                            print("Loss feet: %.4f" % (loss_feet.item()))
                            print("Loss FK: %.4f" % (loss_fk.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            # print("A complete step takes:{0}".format(time.time()-start_time))

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_human_data = val_data_dict["motion"].cuda()

                    # Generate padding mask
                    actual_seq_len = (
                        val_data_dict["seq_len"] + 1
                    )  # BS, + 1 since we need additional timestep for noise level
                    tmp_mask = torch.arange(self.window + 1).expand(
                        val_human_data.shape[0], self.window + 1
                    ) < actual_seq_len[:, None].repeat(1, self.window + 1)
                    # BS X max_timesteps
                    padding_mask = tmp_mask[:, None, :].to(val_human_data.device)

                    if self.add_waypoints_xy:
                        cond_mask = (
                            self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                                val_human_data, val_data_dict["seq_len"]
                            )
                        )
                    else:
                        cond_mask = torch.ones_like(val_human_data)

                    # Condition on the first frame's human pose.
                    if self.input_first_human_pose_for_nav:
                        cond_mask[:, 0, :] = 0

                    # Condition on the human root's orientation on the xy plane.
                    if self.add_root_xy_ori:
                        root_xy_ori = val_data_dict[
                            "root_traj_xy_ori"
                        ].cuda()  # BS X T X 6
                        val_human_data = torch.cat(
                            (val_human_data, root_xy_ori), dim=-1
                        )  # BS X T X (24*3 + 22*6 + 6)
                        root_xy_ori_mask = torch.ones_like(root_xy_ori).to(
                            human_data.device
                        )
                        root_xy_ori_mask[:, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :] = 0
                        cond_mask = torch.cat((cond_mask, root_xy_ori_mask), dim=-1)

                    if self.add_feet_contact:
                        contact_data = val_data_dict[
                            "feet_contact"
                        ].cuda()  # BS X T X 4
                        data = torch.cat((val_human_data, contact_data), dim=-1)
                        cond_mask = torch.cat(
                            (
                                cond_mask,
                                torch.ones_like(contact_data).to(cond_mask.device),
                            ),
                            dim=-1,
                        )
                    else:
                        data = val_human_data

                    if self.add_language_condition_for_nav:
                        text_anno_data = val_data_dict["text"]
                        language_input = self.encode_text(text_anno_data)  # BS X 512
                        language_input = language_input.to(data.device)
                    else:
                        language_input = None

                    if self.add_feet_contact:
                        val_loss_diffusion, val_loss_feet, val_loss_fk = self.model(
                            data,
                            cond_mask=cond_mask,
                            padding_mask=padding_mask,
                            language_input=language_input,
                            rest_human_offsets=val_data_dict["rest_human_offsets"],
                            ds=self.val_ds,
                        )

                        val_loss = (
                            val_loss_diffusion
                            + self.loss_w_feet * val_loss_feet
                            + self.loss_w_fk * val_loss_fk
                        )
                    else:
                        val_loss_diffusion = self.model(
                            data,
                            cond_mask=cond_mask,
                            padding_mask=padding_mask,
                            language_input=language_input,
                        )

                        val_loss = val_loss_diffusion

                    if self.use_wandb:
                        if self.add_feet_contact:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                                "Validation/Loss/Feet Contact Loss": val_loss_feet.item(),
                                "Validation/Loss/FK Loss": val_loss_fk.item(),
                            }
                        else:
                            val_log_dict = {
                                "Validation/Loss/Total Loss": val_loss.item(),
                                "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                            }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        # all_res_list = self.ema.ema_model.sample(
                        #     data,
                        #     cond_mask=cond_mask,
                        #     padding_mask=padding_mask,
                        #     language_input=language_input,
                        # )

                        # for_vis_gt_data = val_human_data
                        # if self.add_feet_contact:
                        #     all_res_list = all_res_list[:, :, :-4]
                        #     cond_mask = cond_mask[:, :, :-4]
                        # if self.add_root_xy_ori:
                        #     all_res_list = all_res_list[:, :, :-6]
                        #     cond_mask = cond_mask[:, :, :-6]

                        # self.gen_vis_res(
                        #     for_vis_gt_data,
                        #     val_data_dict,
                        #     self.step,
                        #     cond_mask,
                        #     vis_gt=True,
                        # )
                        # self.gen_vis_res(
                        #     all_res_list,
                        #     val_data_dict,
                        #     self.step,
                        #     cond_mask,
                        # )

            self.step += 1

        print("training complete")

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(
        self,
        milestone: str = "10",
    ):
        print(
            f"Loaded weight: {os.path.join(self.results_folder, 'model-' + str(milestone) + '.pt')}"
        )

        self.load(milestone)
        self.ema.ema_model.eval()

        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False,
            )

        for s_idx, val_data_dict in enumerate(test_loader):
            print("s_idx / total: {0} / {1}".format(s_idx, len(test_loader)))

            # Only test walking
            if "walk" not in val_data_dict["text"][0]:
                continue

            val_human_data = val_data_dict["motion"].cuda()
            rest_human_offsets = val_data_dict[
                "rest_human_offsets"
            ].cuda()  # BS X 24 X 3

            # Generate padding mask
            actual_seq_len = (
                val_data_dict["seq_len"] + 1
            )  # BS, + 1 since we need additional timestep for noise level
            tmp_mask = torch.arange(self.window + 1).expand(
                val_human_data.shape[0], self.window + 1
            ) < actual_seq_len[:, None].repeat(1, self.window + 1)
            # BS X max_timesteps
            padding_mask = tmp_mask[:, None, :].to(val_human_data.device)

            if self.add_waypoints_xy:
                cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                    val_human_data, val_data_dict["seq_len"]
                )
            else:
                cond_mask = None

            if self.input_first_human_pose_for_nav:
                cond_mask[:, 0, :] = 0

            if self.add_root_xy_ori:
                root_xy_ori = val_data_dict["root_traj_xy_ori"].cuda()  # BS X T X 6
                val_human_data = torch.cat(
                    (val_human_data, root_xy_ori), dim=-1
                )  # BS X T X (24*3 + 22*6 + 6)
                root_xy_ori_mask = torch.ones_like(root_xy_ori).to(
                    val_human_data.device
                )
                root_xy_ori_mask[:, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :] = 0
                cond_mask = torch.cat((cond_mask, root_xy_ori_mask), dim=-1)

            guidance_fn = None

            if self.add_language_condition_for_nav:
                text_anno_data = val_data_dict[
                    "text"
                ]  # 'a man walks forward then turns right after a short pause'
                print(s_idx, val_data_dict["text"])
                language_input = self.encode_text(text_anno_data)  # BS X 512
                language_input = language_input.to(val_human_data.device)
            else:
                language_input = None

            if self.add_feet_contact:
                contact_data = val_data_dict["feet_contact"].cuda()  # BS X T X 4
                data = torch.cat((val_human_data, contact_data), dim=-1)
                cond_mask = torch.cat(
                    (cond_mask, torch.ones_like(contact_data).to(cond_mask.device)),
                    dim=-1,
                )
            else:
                data = val_human_data

            num_samples_per_seq = 1
            for sample_idx in range(num_samples_per_seq):
                all_res_list = self.ema.ema_model.sample(
                    data,
                    cond_mask=cond_mask,
                    padding_mask=padding_mask,
                    language_input=language_input,
                    rest_human_offsets=rest_human_offsets,
                    guidance_fn=guidance_fn,
                    data_dict=val_data_dict,
                )

                if self.add_feet_contact:
                    all_res_list = all_res_list[:, :, :-4]
                    cond_mask = cond_mask[:, :, :-4]
                if self.add_root_xy_ori:
                    all_res_list = all_res_list[:, :, :-6]
                    cond_mask = cond_mask[:, :, :-6]
                    val_human_data = val_human_data[:, :, :-6]

                for_vis_gt_data = val_human_data

                vis_tag = (
                    str(milestone)
                    + "_sidx_"
                    + str(s_idx)
                    + "_sample_cnt_"
                    + str(sample_idx)
                )

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"

                dest_mesh_vis_folder, *_ = self.gen_vis_res(
                    all_res_list,
                    val_data_dict,
                    milestone,
                    cond_mask,
                    vis_tag=vis_tag,
                )

                self.gen_vis_res(
                    for_vis_gt_data,
                    val_data_dict,
                    milestone,
                    cond_mask,
                    vis_gt=True,
                    vis_tag=vis_tag,
                )

                print("Visualizing navigation results for seq: {0}".format(s_idx))
                subprocess.run(
                    [
                        "python",
                        "visualizer/vis/visualize_navigation_results.py",
                        "--result-path",
                        dest_mesh_vis_folder,
                        "--s-idx",
                        str(s_idx),
                        "--interaction-epoch",
                        str(milestone),
                        "--model-path",
                        "../data/smplx/models_smplx_v1_1/models/",
                        "--offscreen",
                    ]
                )

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

    def gen_vis_res(
        self,
        all_res_list,
        data_dict,
        step,
        cond_mask,
        vis_gt=False,
        vis_tag=None,
    ):
        # Prepare list used for evaluation.
        human_jnts_list = []
        human_verts_list = []
        trans_list = []
        human_mesh_faces_list = []

        # all_res_list: N X T X (3+9)
        num_seq = all_res_list.shape[0]

        num_joints = 24

        normalized_global_jpos = all_res_list[:, :, : num_joints * 3].reshape(
            num_seq, -1, num_joints, 3
        )
        global_jpos = self.ds.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )

        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3
        global_rot_6d = all_res_list[:, :, -22 * 6 :].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(
            global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)  # N X 3

        seq_len = data_dict["seq_len"].detach().cpu().numpy()  # BS

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
            betas = data_dict["betas"][idx]
            gender = data_dict["gender"][idx]

            curr_seq_name = data_dict["seq_name"][idx]

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].cuda().float(),
                curr_local_rot_aa_rep[None].cuda().float(),
                betas[None].cuda().float(),
                [gender],
                self.ds.bm_dict,
                return_joints24=True,
            )

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0])
            trans_list.append(root_trans)

            human_mesh_faces_list.append(mesh_faces)

            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(
                    self.vis_folder, "blender_mesh_vis" + "_" + str(step)
                )
            else:
                dest_mesh_vis_folder = os.path.join(
                    self.vis_folder, vis_tag + "_" + str(step)
                )

            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                ball_mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "ball_objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
                )
                mesh_save_folder = os.path.join(
                    dest_mesh_vis_folder,
                    "objs_step_" + str(step) + "_bs_idx_" + str(idx) + "_gt",
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

            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")

            if self.add_waypoints_xy:
                curr_cond_mask = cond_mask[idx, :, 0]
                waypoints_list = []
                curr_timesteps = curr_cond_mask.shape[0]
                for t_idx in range(curr_timesteps):
                    if curr_cond_mask[t_idx] == 0:
                        selected_waypoint = data_dict["ori_motion"][
                            idx, t_idx : t_idx + 1, :3
                        ]
                        selected_waypoint[:, 2] = 0.05
                        waypoints_list.append(selected_waypoint)

                ball_for_vis_data = torch.cat(waypoints_list, dim=0)
                self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)

            save_verts_faces_to_mesh_file(
                mesh_verts.detach().cpu().numpy()[0][: seq_len[idx]],
                mesh_faces.detach().cpu().numpy(),
                mesh_save_folder,
            )

            # Only visualize the first sequence in each batch.
            if idx > 0:
                break

        return (
            dest_mesh_vis_folder,
            human_verts_list,
            human_jnts_list,
            trans_list,
            global_rot_mat,
            human_mesh_faces_list,
        )

    def gen_language_for_long_seq(self, num_windows, text):
        text_list = []
        for w_idx in range(num_windows):
            text_list.append(text)

        text_clip_feats_list = []
        for window_text in text_list:
            language_input = self.encode_text(window_text)  # 1 X 512
            text_clip_feats_list.append(language_input)

        return text_clip_feats_list

    def calc_jpos_from_navi_res(self, all_res_list, ref_data_dict):
        num_seq = all_res_list.shape[0]

        num_joints = 24
        normalized_global_jpos = all_res_list[:, :, : num_joints * 3].reshape(
            num_seq, -1, num_joints, 3
        )
        global_jpos = self.ds.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        global_rot_6d = all_res_list[:, :, 24 * 3 : 24 * 3 + 22 * 6].reshape(
            num_seq, -1, 22, 6
        )
        global_rot_mat = transforms.rotation_6d_to_matrix(
            global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = (
            ref_data_dict["trans2joint"].to(all_res_list.device).squeeze(1)
        )  # BS X  3
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1)  # N X 24 X 3

        idx = 0

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
        betas = ref_data_dict["betas"][0]
        gender = ref_data_dict["gender"][0]

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

    def call_navigation_model_long_seq(
        self,
        trans2joint: torch.Tensor,
        rest_human_offsets: torch.Tensor,
        prev_interaction_end_human_pose: torch.Tensor,
        planned_obj_path: torch.Tensor,
        text_list: List[List[str]],
        p_idx: int,
        o_idx: int,
        overlap_frame_num: int,
        step_dis=0.8,
        use_cut_step=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # In navigation model, planned_obj_path is for human root translation xy.
        # prev_interaction_end_human_pose: BS(1) X 1 X D(24*3+24*6), in the coordinate frame that makes the first interaction seqq's first human pose in canonical direction

        start_human_root_pos = prev_interaction_end_human_pose[
            :, 0:1, :3
        ]  # BS X 1 X 3, z shouldn't be used!

        seq_human_root_pos = torch.zeros(
            start_human_root_pos.shape[0], (planned_obj_path.shape[0] - 1) * 30, 3
        ).cuda()
        seq_human_root_pos[:, 0:1, :] = start_human_root_pos.clone()  # unnormalized

        if self.add_waypoints_xy:  # Need to consider overlapped frames.
            planned_start_point = planned_obj_path[0:1, :2]  # 1 X 2
            real_start_point = start_human_root_pos[0, 0, :2]  # 1 X 2
            if (
                torch.norm(planned_start_point.cuda() - real_start_point.cuda()) > 0.1
            ):  # will not happen, will set real_start_point to be planned_start_point.
                seq_human_root_pos = torch.zeros(
                    start_human_root_pos.shape[0], (planned_obj_path.shape[0]) * 30, 3
                ).cuda()
                seq_human_root_pos[:, 0:1, :] = (
                    start_human_root_pos.clone()
                )  # unnormalized
                waypoints_com_pos = planned_obj_path[None][
                    :, :, :
                ].cuda()  # BS X (K) X 3
            else:
                waypoints_com_pos = planned_obj_path[None][
                    :, 1:, :
                ].cuda()  # BS X (K-1) X 3

            num_pts = waypoints_com_pos.shape[1]

            window_cnt = 0
            for tmp_p_idx in range(num_pts):
                t_idx = (tmp_p_idx + 1) * 30 - 1
                seq_human_root_pos[:, t_idx, :2] = waypoints_com_pos[:, tmp_p_idx, :2]

                window_cnt += 1

        actual_num_frames = window_cnt * self.window
        seq_human_root_pos = seq_human_root_pos[:, :actual_num_frames]

        if self.add_language_condition_for_nav:
            text_clip_feats_list = self.gen_language_for_long_seq(
                window_cnt, text_list[p_idx][o_idx]
            )

        # Generate padding mask
        actual_seq_len = torch.ones(1, self.window + 1) * self.window + 1
        tmp_mask = (
            torch.arange(self.window + 1).expand(1, self.window + 1) < actual_seq_len
        )  # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].cuda()  # 1 X 1 X 121

        bs_window_len = torch.zeros(prev_interaction_end_human_pose.shape[0])
        bs_window_len[:] = self.window
        if self.add_waypoints_xy:
            cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(
                torch.zeros(
                    prev_interaction_end_human_pose.shape[0],
                    self.window,
                    prev_interaction_end_human_pose.shape[-1],
                ).to(prev_interaction_end_human_pose.device),
                bs_window_len,
            )
        else:
            cond_mask = None

        if self.input_first_human_pose_for_nav:
            cond_mask[:, 0, :] = 0

        guidance_fn = None

        if self.add_language_condition_for_nav:
            language_input = text_clip_feats_list
        else:
            language_input = None

        # Canonicalize the first human pose and corresponding waypoints.
        cano_prev_human_pose, cano_seq_human_root_pos, cano_rot_mat = (
            canonicalize_first_human_and_waypoints(
                first_human_pose=prev_interaction_end_human_pose,
                seq_waypoints_pos=seq_human_root_pos,
                trans2joint=trans2joint,
                parents=self.ds.parents,
                trainer=self,
                is_interaction=False,
            )
        )

        data = torch.zeros(
            prev_interaction_end_human_pose.shape[0],
            seq_human_root_pos.shape[1],
            prev_interaction_end_human_pose.shape[-1],
        )  # BS X T X D
        data[:, :, :2] = cano_seq_human_root_pos[:, :, :2]  # BS X T X 2
        data[:, 0:1, :] = cano_prev_human_pose  # BS X 1 X D

        cano_seq_human_root_pos[:, 0:1, :2] = cano_prev_human_pose[:, :, :2].clone()

        if self.add_feet_contact:
            data = torch.cat(
                (data, torch.zeros(data.shape[0], data.shape[1], 4).to(data.device)),
                dim=-1,
            )
            cond_mask = torch.cat(
                (
                    cond_mask,
                    torch.ones(cond_mask.shape[0], cond_mask.shape[1], 4).to(
                        cond_mask.device
                    ),
                ),
                dim=-1,
            )

        all_res_list, whole_cond_mask = (
            self.ema.ema_model.sample_sliding_window_w_canonical(
                self.ds,
                trans2joint,
                data.cuda(),
                cond_mask=cond_mask,
                padding_mask=padding_mask,
                overlap_frame_num=overlap_frame_num,
                input_waypoints=True,
                language_input=language_input,
                rest_human_offsets=rest_human_offsets,
                guidance_fn=guidance_fn,
                opt_fn=None,
                data_dict=None,  # NOTE: This is only used when using guidance_fn.
                add_root_ori=self.add_root_xy_ori,
                step_dis=step_dis,
                use_cut_step=use_cut_step,
            )
        )

        ##################### post transformation #####################
        pred_feet_contact = None
        if self.add_feet_contact:
            pred_feet_contact = all_res_list[:, :, -4:]  # BS X T X 4
            all_res_list = all_res_list[:, :, :-4]

        # De-canonicalize the result back to the original direction.
        unnormalized_human_jpos, human_rot6d_res = (
            self.ema.ema_model.apply_rotation_to_data_human_only(
                self.ds, trans2joint, cano_rot_mat, all_res_list
            )
        )
        # BS X T X 24 X 3, BS X T X 22 X 6, convert the canonicalized direction to original one

        # After de-canonicalize, needs to move the traj starting point from zero to the original starting point.
        align_to_prev_subseq = (
            start_human_root_pos - unnormalized_human_jpos[:, 0:1, 0, :]
        )  # BS X 1 X 3
        align_to_prev_subseq[:, :, 2] = 0
        unnormalized_human_jpos += align_to_prev_subseq

        normalized_human_jpos = self.ds.normalize_jpos_min_max(unnormalized_human_jpos)

        all_res_list[:, :, : 24 * 3] = normalized_human_jpos.reshape(
            all_res_list.shape[0], all_res_list.shape[1], 24 * 3
        )
        all_res_list[:, :, 24 * 3 :] = human_rot6d_res.reshape(
            all_res_list.shape[0], all_res_list.shape[1], 22 * 6
        )

        return all_res_list, whole_cond_mask, seq_human_root_pos, pred_feet_contact

    def gen_vis_res_human_only(
        self,
        all_res_list,
        planned_waypoints_pos,
        cond_mask,
        data_dict,
        vis_tag=None,
        finger_all_res_list=None,
        planned_end_obj_com=None,
        move_to_planned_path=None,
        overlap_frame_num=10,
        planned_scene_names=None,
        planned_path_floor_height=None,
        vis_wo_scene=False,
        text_anno=None,
        cano_quat=None,
    ):
        # Prepare list used for saving params.
        human_root_pos_list = []
        human_jnts_pos_list = []
        human_jnts_local_rot_aa_list = []

        finger_jnts_local_rot_aa_list = []

        # Prepare list used for evaluation.
        human_jnts_list = []
        human_verts_list = []
        trans_list = []
        human_mesh_faces_list = []

        # all_res_list: N X T X (3+9)
        num_seq = all_res_list.shape[0]

        num_joints = 24

        normalized_global_jpos = all_res_list[:, :, : num_joints * 3].reshape(
            num_seq, -1, num_joints, 3
        )
        global_jpos = self.ds.de_normalize_jpos_min_max(
            normalized_global_jpos.reshape(-1, num_joints, 3)
        )

        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3)  # N X T X 22 X 3
        if move_to_planned_path is not None:
            global_jpos = global_jpos + move_to_planned_path[:, :, None, :]

        global_root_jpos = global_jpos[:, :, 0, :].clone()  # N X T X 3

        human_jnts_global_rot_6d = all_res_list[:, :, -22 * 6 :].reshape(
            num_seq, -1, 22, 6
        )
        human_jnts_global_rot_mat = transforms.rotation_6d_to_matrix(
            human_jnts_global_rot_6d
        )  # N X T X 22 X 3 X 3

        trans2joint = data_dict["trans2joint"].to(all_res_list.device)  # N X 3

        dest_out_vid_path = None

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

            curr_trans2joint = trans2joint[idx : idx + 1].clone()

            root_trans = curr_global_root_jpos + curr_trans2joint.to(
                curr_global_root_jpos.device
            )  # T X 3

            # Generate global joint position
            bs = 1
            betas = data_dict["betas"][idx]
            gender = data_dict["gender"][idx]

            # curr_seq_name = data_dict['seq_name'][idx]

            # Get human verts
            mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(
                root_trans[None].cuda().float(),
                curr_local_rot_aa_rep[None].cuda().float(),
                betas.cuda().float(),
                [gender],
                self.ds.bm_dict,
                return_joints24=True,
            )

            # actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0])
            trans_list.append(root_trans)

            human_mesh_faces_list.append(mesh_faces)

            human_root_pos_list.append(root_trans)
            human_jnts_pos_list.append(mesh_jnts[0])
            human_jnts_local_rot_aa_list.append(curr_local_rot_aa_rep[:, :22])

            if finger_all_res_list is not None:
                finger_jnts_local_rot_aa_list.append(pred_finger_aa_rep[idx])

            if vis_tag is None:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, "navigation_vis")
            else:
                dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag)
            if os.path.exists(dest_mesh_vis_folder):
                import shutil

                shutil.rmtree(dest_mesh_vis_folder)
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            ball_mesh_save_folder = os.path.join(
                dest_mesh_vis_folder, "ball_objs" + "_bs_idx_" + str(idx)
            )
            mesh_save_folder = os.path.join(
                dest_mesh_vis_folder, "objs" + "_bs_idx_" + str(idx)
            )
            out_rendered_img_folder = os.path.join(
                dest_mesh_vis_folder, "imgs" + "_bs_idx_" + str(idx)
            )

            out_vid_file_path = os.path.join(
                dest_mesh_vis_folder, "vid" + "_bs_idx_" + str(idx) + ".mp4"
            )

            if vis_wo_scene:
                ball_mesh_save_folder = ball_mesh_save_folder + "_vis_no_scene"
                mesh_save_folder = mesh_save_folder + "_vis_no_scene"
                out_rendered_img_folder = out_rendered_img_folder + "_vis_no_scene"
                out_vid_file_path = out_vid_file_path.replace(
                    ".mp4", "_vis_no_scene.mp4"
                )

            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")

            if self.add_waypoints_xy:
                waypoints_list = []
                for t_idx in range(planned_waypoints_pos.shape[0] - 3):
                    selected_waypoint = planned_waypoints_pos[t_idx : t_idx + 1]
                    selected_waypoint[:, 2] = 0.05
                    waypoints_list.append(selected_waypoint)

                ball_for_vis_data = torch.cat(waypoints_list, dim=0)  # K X 3

                if cano_quat is not None:
                    cano_quat_inv = transforms.quaternion_invert(cano_quat[0:1])
                    cano_quat_for_ball = cano_quat_inv.repeat(
                        ball_for_vis_data.shape[0], 1
                    )  # K X 4
                    ball_for_vis_data = transforms.quaternion_apply(
                        cano_quat_for_ball.to(ball_for_vis_data.device),
                        ball_for_vis_data,
                    )

                self.create_ball_mesh(
                    ball_for_vis_data.detach().cpu().numpy(), ball_mesh_path
                )

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3
                # obj_mesh_verts: T X Nv' X 3
                # cano_quat: K X 4
                cano_quat_inv = transforms.quaternion_invert(cano_quat[0:1])
                cano_quat_for_human = cano_quat_inv[None].repeat(
                    mesh_verts.shape[1], mesh_verts.shape[2], 1
                )  # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(
                    cano_quat_for_human.to(mesh_verts.device), mesh_verts[0]
                )

                save_verts_faces_to_mesh_file(
                    mesh_verts.detach().cpu().numpy(),
                    mesh_faces.detach().cpu().numpy(),
                    mesh_save_folder,
                )
            else:
                save_verts_faces_to_mesh_file(
                    mesh_verts.detach().cpu().numpy()[0],
                    mesh_faces.detach().cpu().numpy(),
                    mesh_save_folder,
                )

            # Save human info.
            results = {
                "human_root_pos": human_root_pos_list[-1],
                "human_jnts_pos": human_jnts_pos_list[-1],
                "human_jnts_local_rot_aa": human_jnts_local_rot_aa_list[-1],
                "finger_jnts_local_rot_aa": finger_jnts_local_rot_aa_list[-1],
            }
            results_path = os.path.join(mesh_save_folder, "navigation_results.pkl")
            with open(results_path, "wb") as f:
                pickle.dump(results, f)
            print("Saving navigation results to {}".format(results_path))

            if idx > 0:
                break

        return (
            human_verts_list,
            human_jnts_list,
            trans_list,
            human_jnts_global_rot_mat,
            human_mesh_faces_list,
            dest_mesh_vis_folder,
            results_path,
        )


def calculate_navi_representation_dim(opt):
    # Human motion representation
    repr_dim = 24 * 3 + 22 * 6

    # Feet floor contact label
    if opt.add_feet_contact:
        repr_dim += 4

    # Human root orientation in XY plane
    if opt.add_root_xy_ori:
        repr_dim += 6

    return repr_dim


def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

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

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,  # 32
        train_lr=opt.learning_rate,  # 1e-4
        train_num_steps=8000000,  # 700000, total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        results_folder=str(wdir),
    )
    trainer.train()

    torch.cuda.empty_cache()


def run_sample(opt, device):
    # Prepare Directories
    if not opt.vis_wdir:
        raise ValueError("Please specify the vis_wdir for visualization.")
    vis_wdir = "./results/navigation/{}".format(opt.vis_wdir)

    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"

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

    trainer = Trainer(
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
        load_ds=True,
    )

    trainer.cond_sample_res()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split("/")[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
