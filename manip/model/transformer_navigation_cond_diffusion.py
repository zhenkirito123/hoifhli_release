import math
import os
import pickle
from inspect import isfunction

import matplotlib.pyplot as plt
import pytorch3d.transforms as transforms
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from scipy.signal import medfilt
from torch import nn
from tqdm.auto import tqdm

from manip.data.humanml3d_dataset import (
    normalize,
    quat_between,
    quat_fk_torch,
    quat_ik_torch,
)
from manip.inertialize.inert import apply_inertialize
from manip.lafan1.utils import quat_slerp, rotate_at_frame_w_obj_global
from manip.model.transformer_module import Decoder
from manip.utils.model_utils import wxyz_to_xyzw, xyzw_to_wxyz


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# def wxyz_to_xyzw(input_quat):
#     # input_quat: 1 X w X 22 X 4
#     w = input_quat[:, :, :, 0:1]
#     x = input_quat[:, :, :, 1:2]
#     y = input_quat[:, :, :, 2:3]
#     z = input_quat[:, :, :, 3:4]

#     out_quat = torch.cat((x, y, z, w), dim=-1) # 1 X w X 22 X 4

#     return out_quat

# def xyzw_to_wxyz(input_quat):
#     # input_quat: 1 X w X 22 X 4
#     x = input_quat[:, :, :, 0:1]
#     y = input_quat[:, :, :, 1:2]
#     z = input_quat[:, :, :, 2:3]
#     w = input_quat[:, :, :, 3:4]

#     out_quat = torch.cat((w, x, y, z), dim=-1) # 1 X w X 22 X 4

#     return out_quat


def interpolate_transition_human_only(
    prev_jpos, prev_rot_6d, window_jpos, window_rot_6d
):
    # prev_jpos: 1 X overlap_num X 24 X 3
    # prev_rot_6d: 1 X overlap_num X 22 X 6
    # window_jpos: 1 X w X 24 X 3
    # window_rot_6d: 1 X w X 22 X 6
    num_overlap_frames = prev_jpos.shape[1]

    fade_out = torch.linspace(1, 0, num_overlap_frames)[None, :, None].to(
        prev_jpos.device
    )  # 1 X overlap_num X 1
    fade_in = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(
        prev_jpos.device
    )

    window_jpos[:, :num_overlap_frames, :, :] = (
        fade_out[:, :, None, :] * prev_jpos
        + fade_in[:, :, None, :] * window_jpos[:, :num_overlap_frames, :, :]
    )

    # stitch joint angles with slerp
    slerp_weight = torch.linspace(0, 1, num_overlap_frames)[None, :, None].to(
        prev_rot_6d.device
    )  # 1 X overlap_num X 1

    prev_rot_mat = transforms.rotation_6d_to_matrix(prev_rot_6d)
    prev_q = transforms.matrix_to_quaternion(prev_rot_mat)
    window_rot_mat = transforms.rotation_6d_to_matrix(window_rot_6d)
    window_q = transforms.matrix_to_quaternion(window_rot_mat)  # 1 X w X 22 X 4

    human_q_left = prev_q.clone()  # 1 X overlap_num X 22 X 4
    human_q_right = window_q[:, :num_overlap_frames, :, :]  # 1 X overlap_num X 22 X 4

    human_q_left = wxyz_to_xyzw(human_q_left)
    human_q_right = wxyz_to_xyzw(human_q_right)

    # quat_slerp needs xyzw
    slerped_human_q = quat_slerp(
        human_q_left, human_q_right, slerp_weight
    )  # 1 X overlap_num X 22 X 4

    slerped_human_q = xyzw_to_wxyz(slerped_human_q)

    new_human_q = torch.cat(
        (slerped_human_q, window_q[:, num_overlap_frames:, :, :]), dim=1
    )  # 1 X w X 22 X 4

    new_human_rot_mat = transforms.quaternion_to_matrix(new_human_q)
    new_human_rot_6d = transforms.matrix_to_rotation_6d(new_human_rot_mat)

    return window_jpos, new_human_rot_6d


def visualize_array(array, dest_fig_path):
    """
    Visualizes a 2D array using a heatmap.

    Parameters:
    - array (2D list or numpy array): The data to visualize.

    """

    plt.figure(figsize=(10, 6))  # adjust the size to your preference
    plt.imshow(
        array, cmap="viridis", interpolation="nearest"
    )  # 'viridis' is a good perceptually uniform colormap. Change it if needed.
    plt.colorbar(label="Value")
    plt.title("Gradient Array Visualization")
    plt.savefig(dest_fig_path)

    # Example usage:
    # dummy_data = np.random.rand(120, 200) * 100  # This creates a 120x200 array of random values between 0 and 100
    # visualize_array(dummy_data)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()

        self.d_feats = d_feats
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k
        self.d_v = d_v
        self.max_timesteps = max_timesteps

        # Input: BS X D X T
        # Output: BS X T X D'
        self.motion_transformer = Decoder(
            d_feats=d_input_feats,
            d_model=self.d_model,
            n_layers=self.n_dec_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            max_timesteps=self.max_timesteps,
            use_full_attention=True,
        )

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model),
        )

    def forward(
        self, src, noise_t, condition, language_embedding=None, padding_mask=None
    ):
        # src: BS X T X D
        # noise_t: int

        src = torch.cat((src, condition), dim=-1)

        noise_t_embed = self.time_mlp(noise_t)  # BS X d_model
        if language_embedding is not None:
            noise_t_embed += language_embedding  # BS X d_model
        noise_t_embed = noise_t_embed[:, None, :]  # BS X 1 X d_model

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            padding_mask = (
                torch.ones(bs, 1, num_steps).to(src.device).bool()
            )  # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) + 1  # timesteps
        pos_vec = (
            pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)
        )  # BS X 1 X timesteps

        data_input = src.transpose(1, 2)  # BS X D X T
        feat_pred, _ = self.motion_transformer(
            data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed
        )
        # ("feat_pred: {0}".format(feat_pred.isnan().max()))
        output = self.linear_out(feat_pred[:, 1:])  # BS X T X D

        return output  # predicted noise, the same size as the input


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        # input: BS
        idx = input.to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


class NavigationCondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps=1000,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        input_first_human_pose=False,
        input_rest_human_skeleton=False,
        add_feet_contact=False,
    ):
        super().__init__()

        self.add_feet_contact = add_feet_contact

        self.input_rest_human_skeleton = input_rest_human_skeleton
        if input_rest_human_skeleton:
            self.body_shape_encoder = nn.Sequential(
                nn.Linear(in_features=24 * 3, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=128),
            )

        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
        )

        self.input_first_human_pose = input_first_human_pose

        d_input_feats = 2 * d_feats

        if self.input_rest_human_skeleton:
            d_input_feats += 128

        self.denoise_fn = TransformerDiffusionModel(
            d_input_feats=d_input_feats,
            d_feats=d_feats,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            n_dec_layers=n_dec_layers,
            max_timesteps=max_timesteps,
        )
        # Input condition and noisy motion, noise level t, predict gt motion

        self.objective = objective

        self.seq_len = max_timesteps - 1
        self.out_dim = out_dim

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        print(
            alphas_cumprod[0],
            torch.sqrt(alphas_cumprod)[0],
            torch.sqrt(1.0 - alphas_cumprod)[0],
        )
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting

        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x,
        t,
        x_cond,
        language_embedding=None,
        padding_mask=None,
        clip_denoised=True,
    ):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output = self.denoise_fn(
            x,
            t,
            x_cond,
            language_embedding=language_embedding,
            padding_mask=padding_mask,
        )

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # Not ready to use!
    def p_mean_variance_reconstruction_guidance(
        self,
        x,
        t,
        x_cond,
        x_start,
        guidance_fn,
        opt_fn=None,
        language_embedding=None,
        padding_mask=None,
        cond_mask=None,
        rest_human_offsets=None,
        data_dict=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
        clip_denoised=True,
        contact_labels=None,
    ):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            model_output = self.denoise_fn(
                x,
                t,
                x_cond,
                language_embedding=language_embedding,
                padding_mask=padding_mask,
            )

            if self.objective == "pred_noise":
                x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
            elif self.objective == "pred_x0":
                x_start = model_output
            else:
                raise ValueError(f"unknown objective {self.objective}")

            classifier_scale = 1e3

            loss = guidance_fn(x_start, rest_human_offsets, data_dict)

            gradient = (
                torch.autograd.grad(-loss, x)[0] * classifier_scale
            )  # BS(1) X 120 X 216
            print("gradient mean:{0}".format(gradient.mean()))

            # Peturb predicted clean x
            tmp_posterior_variance = extract(self.posterior_variance, t, x_start.shape)
            # print("variance max:{0}".format(tmp_posterior_variance.max()))
            x_start = x_start + tmp_posterior_variance * gradient.float()

        if opt_fn is not None and t[0] < 10:
            # x_start = opt_fn(x_start, x_start_gt) # For using GT l2 loss debug
            x_start = opt_fn(x_start.detach(), rest_human_offsets, data_dict)

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # Not ready to use!!!
    def p_sample_guided_reconstruction_guidance(
        self,
        x,
        t,
        x_cond,
        x_start,
        guidance_fn,
        language_embedding=None,
        opt_fn=None,
        clip_denoised=True,
        rest_human_offsets=None,
        data_dict=None,
        cond_mask=None,
        prev_window_cano_rot_mat=None,
        prev_window_init_root_trans=None,
        contact_labels=None,
    ):
        b, *_, device = *x.shape, x.device

        # x_start = self.q_sample(x_start, t) # Add noise to the target, for debugging.
        model_mean, _, model_log_variance = (
            self.p_mean_variance_reconstruction_guidance(
                x=x,
                t=t,
                x_cond=x_cond,
                x_start=x_start,
                guidance_fn=guidance_fn,
                language_embedding=language_embedding,
                opt_fn=opt_fn,
                clip_denoised=clip_denoised,
                cond_mask=cond_mask,
                rest_human_offsets=rest_human_offsets,
                data_dict=data_dict,
                prev_window_cano_rot_mat=prev_window_cano_rot_mat,
                prev_window_init_root_trans=prev_window_init_root_trans,
                contact_labels=contact_labels,
            )
        )

        new_mean = model_mean

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        sampled_x = new_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return sampled_x

    # Not ready to use!!!
    def p_sample_guided(
        self,
        x,
        t,
        x_cond,
        x_start,
        guidance_fn,
        reconstruction_guidance=False,
        opt_fn=None,
        clip_denoised=True,
        rest_human_offsets=None,
        data_dict=None,
        cond_mask=None,
    ):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, x_cond=x_cond, clip_denoised=clip_denoised
        )

        model_variance = torch.exp(model_log_variance)

        if guidance_fn is not None:
            # x_start = self.q_sample(x_start, t) # Add noise to the target, for debugging.
            # gradient = guidance_fn(x, x_start) # For using GT loss debug
            # gradient = guidance_fn(x, rest_human_offsets=rest_human_offsets, data_dict=data_dict) # For contact constrints guidance.
            gradient = guidance_fn(x, x_cond[:, :, -x.shape[-1] :], cond_mask)
        else:
            gradient = None

        new_mean = model_mean + model_variance * gradient.float()
        # new_mean = model_mean + gradient.float()
        print("model variance:{0}".format(model_variance))
        print("gradient max:{0}".format(gradient.max()))

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        sampled_x = new_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # For debug
        # model_variance = torch.exp(model_log_variance)
        # x_start = self.q_sample(x_start, t) # Add noise to the target, for debugging.
        # gradient = guidance_fn(x, x_start)
        # sampled_x = sampled_x + model_variance * gradient.float()

        return sampled_x

    # Not ready to use!!!
    def p_sample_loop_guided(
        self,
        shape,
        x_start,
        x_cond,
        guidance_fn=None,
        opt_fn=None,
        rest_human_offsets=None,
        data_dict=None,
        cond_mask=None,
    ):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # x = self.p_sample_guided(x, torch.full((b,), i, device=device, dtype=torch.long), \
            #             x_cond, x_start, guidance_fn)
            if i > 0 and i < 10:
                # x = self.p_sample_guided(x, torch.full((b,), i, device=device, dtype=torch.long), \
                #             x_cond, x_start, guidance_fn, reconstruction_guidance=False, \
                #             rest_human_offsets=rest_human_offsets, data_dict=data_dict, cond_mask=cond_mask)
                x = self.p_sample_guided_reconstruction_guidance(
                    x,
                    torch.full((b,), i, device=device, dtype=torch.long),
                    x_cond,
                    x_start,
                    guidance_fn,
                    opt_fn=opt_fn,
                    rest_human_offsets=rest_human_offsets,
                    data_dict=data_dict,
                    cond_mask=cond_mask,
                )

                known_pose_conditions = x_cond[:, :, -x.shape[-1] :]
                # known_pose_conditions = self.q_sample(known_pose_conditions, \
                #                 torch.full((b,), i-1, device=device, dtype=torch.long))

                # if i < 5:
                #     print("known_pose_conditions:{0}".format(known_pose_conditions[0, -1, :3]))
                #     print("x:{0}".format(x[0, -1, :3]))
                # import pdb
                # pdb.set_trace()

                x = (1 - cond_mask) * known_pose_conditions + cond_mask * x
            else:
                x = self.p_sample(
                    x, torch.full((b,), i, device=device, dtype=torch.long), x_cond
                )

        # import pdb
        # pdb.set_trace()

        if opt_fn is not None:
            # x_start = opt_fn(x_start, x_start_gt) # For using GT l2 loss debug
            x = opt_fn(x.detach(), rest_human_offsets, data_dict)

        return x  # BS X T X D

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        x_cond,
        language_embedding=None,
        padding_mask=None,
        clip_denoised=True,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            x_cond=x_cond,
            language_embedding=language_embedding,
            padding_mask=padding_mask,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        x_cond,
        language_embedding=None,
        padding_mask=None,
        cond_mask=None,
        ori_pose_cond=None,
        return_diff_level_res=False,
    ):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        if return_diff_level_res:
            x_all_list = []

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # Debug, replace object with GT.
            # known_object_conditions = x_cond[:, :, -x.shape[-1]:-x.shape[-1]+3+9]

            # known_object_conditions = ori_pose_cond[:, :, :3+9]
            # known_object_conditions = self.q_sample(known_object_conditions, torch.full((b,), i, device=device, dtype=torch.long))
            # x[:, :, :3+9] = known_object_conditions

            x = self.p_sample(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                x_cond,
                language_embedding=language_embedding,
                padding_mask=padding_mask,
            )

            # if i % 50 == 0 and return_diff_level_res:
            #     x_all_list.append(x)

            if i < 50 and return_diff_level_res and i % 5 == 0:
                x_all_list.append(x)

            # if i > 0:
            #     known_object_conditions = ori_pose_cond[:, :, :3+9]
            #     known_object_conditions = self.q_sample(known_object_conditions, torch.full((b,), i-1, device=device, dtype=torch.long))
            #     x[:, :, :3+9] = known_object_conditions

            # import pdb
            # pdb.set_trace()

            # if self.inpainting_conditions:
            # x: 1 X 120 X 216, x_cond: 1 X 120 X 475, cond_mask: 1 X 120 X 216
            #     # Use a paper's method to apply conditions.

            #     # prev_conditions = torch.cat((cano_prev_sample_res, torch.zeros(b, \
            #     #                 self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)
            #     # x_w_conditions = self.q_sample(prev_conditions, torch.full((b,), i-1, device=device, dtype=torch.long))
            #     # prev_condition_mask = torch.ones(b, cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)
            #     # prev_condition_mask = torch.cat((prev_condition_mask, \
            #     #         torch.zeros(b, self.seq_len-cano_prev_sample_res.shape[1], cano_prev_sample_res.shape[-1]).to(cano_prev_sample_res.device)), dim=1)

            #     curr_x = prev_condition_mask * x_w_conditions + (1 - prev_condition_mask) * curr_x

            # if i > 0:
            #     known_pose_conditions = x_cond[:, :, -x.shape[-1]:]
            #     # known_pose_conditions = self.q_sample(known_pose_conditions, torch.full((b,), i-1, device=device, dtype=torch.long))

            #     # if i < 5:
            #     #     print("known_pose_conditions:{0}".format(known_pose_conditions[0, -1, :3]))
            #     #     print("x:{0}".format(x[0, -1, :3]))
            #         # import pdb
            #         # pdb.set_trace()

            #     x = (1 - cond_mask) * known_pose_conditions + cond_mask * x

            # Debug, replace object motion with GT motion.

        # x[:, :, :3+9] = known_object_conditions

        if return_diff_level_res:
            x_all_list = torch.stack(x_all_list, dim=1)  # BS X K X T X D

            # import pdb
            # pdb.set_trace()

            return x_all_list

        return x  # BS X T X D

    def apply_rotation_to_data(self, ds, trans2joint, cano_rot_mat, curr_x):
        # cano_rot_mat:BS X 3 X 3, convert from the coodinate frame which canonicalize the first frame of a sequence to
        # the frame that canonicalize the first frame of a window.
        # trans2joint: BS X 3
        # new_obj_rot_mat: BS X 10(overlapped -length) X 3 X 3
        # curr_x: BS X window_size X D
        # This function is to convert window data to sequence data.
        bs, timesteps, _ = curr_x.shape

        pred_human_normalized_jpos = curr_x[:, :, : 24 * 3]  # BS X window_size X (24*3)
        pred_human_jpos = ds.de_normalize_jpos_min_max(
            pred_human_normalized_jpos.reshape(bs, timesteps, 24, 3)
        )  # BS X window_size X 24 X 3
        pred_human_rot_6d = curr_x[:, :, 24 * 3 :]  # BS X window_size X (22*6)

        pred_human_rot_mat = transforms.rotation_6d_to_matrix(
            pred_human_rot_6d.reshape(bs, timesteps, 22, 6)
        )  # BS X T X 22 X 3 X 3

        converted_human_jpos = torch.matmul(
            cano_rot_mat[:, None, None, :, :]
            .repeat(1, timesteps, 24, 1, 1)
            .transpose(3, 4),
            pred_human_jpos[:, :, :, :, None],
        ).squeeze(-1)  # BS X T X 24 X 3
        converted_rot_mat = torch.matmul(
            cano_rot_mat[:, None, None, :, :]
            .repeat(1, timesteps, 22, 1, 1)
            .transpose(3, 4),
            pred_human_rot_mat,
        )  # BS X T X 22 X 3 X 3

        converted_rot_6d = transforms.matrix_to_rotation_6d(converted_rot_mat)

        # converted_obj_com_pos = ds.normalize_obj_pos_min_max(converted_obj_com_pos)
        # converted_human_jpos = ds.normalize_jpos_min_max(converted_human_jpos)
        # converted_curr_x = torch.cat((converted_obj_com_pos, converted_obj_rot_mat, \
        #         converted_human_jpos, converted_rot_6d), dim=-1)

        return converted_human_jpos, converted_rot_6d

    # @torch.no_grad()
    def p_sample_loop_sliding_window_w_canonical(
        self,
        ds,
        trans2joint,
        x_start,
        cond_mask,
        padding_mask,
        overlap_frame_num=1,
        input_waypoints=False,
        language_input=None,
        rest_human_offsets=None,
        data_dict=None,
        guidance_fn=None,
        opt_fn=None,
        add_root_ori=False,
        step_dis=0.8,
        use_cut_step=True,
    ):
        """
        NOTE: This function includes several tricks to ensure smooth walking motion generation.
        """

        # object_names: BS
        # obj_scales: BS X T
        # trans2joint: BS X 3
        # first_frame_obj_com2trans: BS X 1 X 3
        # x_start: BS X T X D(24*3+22*6) (T can be larger than the window_size), without normalization.
        # ori_x_cond: BS X 1 X (3+1024*3), the first frame's object BPS + com position.
        # cond_mask: BS X window_size X D
        # padding_mask: BS X T
        # contact_labels: BS X T
        def quat_between_x_axis(forward):
            assert forward.shape[0] == 1
            x_axis = torch.zeros(forward.shape[0], 3).to(forward.device)  # BS X 3
            x_axis[:, 0] = 1.0

            if abs(forward[0, 1]) < 1e-3:
                if forward[0, 0] > 0:
                    yrot = torch.Tensor([1.0, 0.0, 0.0, 0.0]).to(forward.device)
                else:
                    yrot = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(forward.device)
            else:
                yrot = normalize(
                    quat_between(x_axis, forward)
                )  # 4-dim, from current direction to canonicalized direction
            return yrot

        def cal_root_traj_xy_ori_from_root(motion):
            # forward: BS X 208
            num_joints = 24
            root_rot = motion[:, 2 * num_joints * 3 : 2 * num_joints * 3 + 6]  # BS X 6
            root_rot = transforms.rotation_6d_to_matrix(root_rot)  # BS X 3 X 3

            z_axis = torch.zeros(root_rot.shape[0], 3).to(root_rot.device)  # BS X 3
            z_axis[:, 2] = 1.0
            z_axis = z_axis.reshape(root_rot.shape[0], 3, 1)  # BS X 3 X 1

            rotated_z_axis = torch.matmul(root_rot.float(), z_axis.float()).reshape(
                root_rot.shape[0], 3
            )  # BS X 3
            rotated_z_axis[:, 2] = 0.0  # T X 3

            forward = normalize(rotated_z_axis)  # BS X 3
            yrot = quat_between_x_axis(forward)  # BS X 4

            yrot = transforms.matrix_to_rotation_6d(
                transforms.quaternion_to_matrix(yrot)
            )  # BS X 6
            return yrot.detach()

        def cal_root_traj_xy_ori(forward):
            # forward: BS X 3
            forward[:, 2] = 0.0

            forward = normalize(forward)  # T X 3
            yrot = quat_between_x_axis(forward)

            yrot = transforms.matrix_to_rotation_6d(
                transforms.quaternion_to_matrix(yrot)
            )  # T X 6
            return yrot.detach()

        shape = x_start.shape
        if add_root_ori:
            shape = (shape[0], shape[1], shape[2] + 6)

            root_xy_ori_mask = torch.ones(cond_mask.shape[0], cond_mask.shape[1], 6).to(
                cond_mask.device
            )
            root_xy_ori_mask[:, [0, 30 - 1, 60 - 1, 90 - 1, 120 - 1], :] = 0
            cond_mask = torch.cat(
                (cond_mask[..., :204], root_xy_ori_mask, cond_mask[..., 204:]), dim=-1
            )

        device = self.betas.device

        b = shape[0]
        assert b == 1

        x_all = torch.randn(shape, device=device)

        whole_sample_res = None  # BS X T X D (24*3+22*6)
        whole_cond_mask = None

        # whole_sample_res = x_start[:, 0:1, :].repeat(1, self.seq_len, 1) # BS X 10 X D

        x_start_denormalized = ds.de_normalize_jpos_min_max(
            x_start[:, :, : 24 * 3].reshape(b, -1, 24, 3)
        )

        num_steps = shape[1]
        stride = self.seq_len - overlap_frame_num
        window_idx = 0
        for t_idx in range(0, num_steps, stride):
            if t_idx + self.seq_len > num_steps:
                break
            cut_step = False
            if t_idx == 0:
                curr_x = x_all[:, t_idx : t_idx + self.seq_len]  # Random noise.
                curr_x_start = x_start[
                    :, t_idx : t_idx + self.seq_len
                ].clone()  # BS X window_szie X D (3+9+24*3+22*6)

                if add_root_ori:
                    curr_x_start_denormalized = ds.de_normalize_jpos_min_max(
                        curr_x_start[:, :, : 24 * 3].reshape(b, -1, 24, 3)
                    )
                    curr_root_ori = (
                        torch.zeros(curr_x_start.shape[0], curr_x_start.shape[1], 6)
                        .float()
                        .to(curr_x_start.device)
                    )  # BS X T X 3
                    curr_root_ori[:, 0] = cal_root_traj_xy_ori_from_root(
                        curr_x_start[:, 0]
                    )
                    curr_root_ori[:, 30 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 30 - 1, 0]
                        - curr_x_start_denormalized[:, 0, 0]
                    )
                    curr_root_ori[:, 60 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 60 - 1, 0]
                        - curr_x_start_denormalized[:, 30 - 1, 0]
                    )
                    curr_root_ori[:, 90 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 90 - 1, 0]
                        - curr_x_start_denormalized[:, 60 - 1, 0]
                    )
                    curr_root_ori[:, 120 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 120 - 1, 0]
                        - curr_x_start_denormalized[:, 90 - 1, 0]
                    )
                    if t_idx + self.seq_len == num_steps:
                        curr_root_ori[:, 30 - 1] = curr_root_ori[:, 60 - 1]

                    cur_step = (
                        curr_x_start_denormalized[0, 30 - 1, 0]
                        - curr_x_start_denormalized[0, 0, 0]
                    )
                    cur_step[..., 2] = 0
                    cur_dis = torch.norm(cur_step).item()

                    curr_x_start = torch.cat(
                        (
                            curr_x_start[..., :204],
                            curr_root_ori,
                            curr_x_start[..., 204:],
                        ),
                        dim=-1,
                    )

                if language_input is not None:
                    language_embedding = self.clip_encoder(
                        language_input[window_idx]
                    )  # BS X d_model
                else:
                    language_embedding = None

                x_pose_cond = curr_x_start * (
                    1.0 - cond_mask
                )  # Remove noise, overall better than adding random noise.
                curr_x_cond = x_pose_cond  # BS X T X (24*3+22*6)

                for i in tqdm(
                    reversed(range(0, self.num_timesteps)),
                    desc="sampling loop time step",
                    total=self.num_timesteps,
                ):
                    # padding mask is not used now!
                    curr_x = self.p_sample(
                        curr_x,
                        torch.full((b,), i, device=device, dtype=torch.long),
                        curr_x_cond,
                        language_embedding=language_embedding,
                    )

                whole_sample_res = (
                    curr_x.clone()
                )  # BS X window_size X D (3+9+24*3+22*6)
                if add_root_ori:
                    whole_sample_res = torch.cat(
                        (whole_sample_res[..., :204], whole_sample_res[..., 210:]),
                        dim=-1,
                    )

                whole_cond_mask = cond_mask.clone()

                whole_sample_res = whole_sample_res[
                    :, :-overlap_frame_num, :
                ]  # BS X 10 X D
                whole_cond_mask = whole_cond_mask[:, :-overlap_frame_num, :]

                window_idx += 1
            else:
                curr_x = x_all[:, t_idx : t_idx + self.seq_len]  # Random noise.

                prev_sample_res = whole_sample_res[:, -1:, :]  # BS X 10 X D

                curr_x_start_init = x_start[
                    :, t_idx : t_idx + self.seq_len
                ].clone()  # BS X window_szie X D (24*3+22*6)

                if (
                    curr_x.shape[1] < self.seq_len
                ):  # The last window with a smaller size. Better to not use this code.
                    break

                # Canonicalize the first human pose in the current window and make the root trans to (0,0).
                global_human_normalized_jpos = prev_sample_res[:, :, : 24 * 3].reshape(
                    b, -1, 24, 3
                )  # BS X 10 X J(24) X 3
                global_human_jpos = ds.de_normalize_jpos_min_max(
                    global_human_normalized_jpos
                )  # BS X 10 X J X 3

                global_human_6d = prev_sample_res[
                    :, :, 24 * 3 : 24 * 3 + 22 * 6
                ].reshape(b, -1, 22, 6)  # BS X 10 X 22 X 6
                global_human_rot_mat = transforms.rotation_6d_to_matrix(global_human_6d)
                global_human_q = transforms.matrix_to_quaternion(
                    global_human_rot_mat
                )  # BS X 10 X 22 X 4

                def use_T_start_pose():
                    curr_seq_local_jpos = rest_human_offsets.clone()  # 1 X 24 X 3
                    curr_seq_local_jpos[0, 0, :3] = global_human_jpos[
                        0, 0, 0, :3
                    ].clone()  # Set the root joint to the same position.
                    local_joint_rot_mat = pickle.load(
                        open(
                            os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                "../../data/local_joint_rot_mat.pkl",
                            ),
                            "rb",
                        )
                    )[0:1]  # 1 X 22 X 3 X 3
                    local_joint_rot_mat[0, 0] = (
                        global_human_rot_mat[0, 0, 0]
                        .to(local_joint_rot_mat.device)
                        .clone()
                    )  # Set the root joint to the same orientation.
                    global_quat, global_pos = quat_fk_torch(
                        local_joint_rot_mat.cuda(), curr_seq_local_jpos.cuda()
                    )
                    return global_pos[0], global_quat[0]

                def generate_curr_start_frame():
                    global_human_jpos_clone = global_human_jpos.clone()
                    global_human_q_clone = global_human_q.clone()
                    global_human_jpos_clone[0, 0], global_human_q_clone[0, 0] = (
                        use_T_start_pose()
                    )
                    global_human_jpos_clone[0, 0, :11] = global_human_jpos[
                        0, 0, :11
                    ].clone()  # NOTE: Keep the hand position.
                    global_human_q_clone[0, 0, :11] = global_human_q[
                        0, 0, :11
                    ].clone()  # Keep the hand orientation.

                    new_glob_jpos, new_glob_q, _, _ = rotate_at_frame_w_obj_global(
                        global_human_jpos_clone[:, :, 0, :].data.cpu().numpy(),
                        global_human_q_clone[:, :, 0, :].data.cpu().numpy(),
                        ds.parents,
                        n_past=1,
                        floor_z=True,
                        global_q=global_human_q_clone.data.cpu().numpy(),
                        global_x=global_human_jpos_clone.data.cpu().numpy(),
                        use_global=True,
                    )
                    # BS X T X J X 3, BS X T X J X 4, BS X T X 3, BS X T X 4
                    new_glob_jpos = (
                        torch.from_numpy(new_glob_jpos)
                        .float()
                        .to(prev_sample_res.device)
                    )
                    new_glob_q = (
                        torch.from_numpy(new_glob_q).float().to(prev_sample_res.device)
                    )

                    global_human_root_jpos = new_glob_jpos[
                        :, :, 0, :
                    ].clone()  # BS X T X 3
                    global_human_root_trans = global_human_root_jpos + trans2joint[
                        :, None, :
                    ].to(global_human_root_jpos.device)  # BS X T X 3

                    move_to_zero_trans = (
                        global_human_root_trans[
                            :,
                            0:1,
                        ].clone()
                    )  # Move the first frame's root joint x, y to 0,  BS X 1 X 3
                    move_to_zero_trans[:, :, 2] = 0  # BS X 1 X 3

                    global_human_root_trans -= move_to_zero_trans
                    global_human_root_jpos -= move_to_zero_trans
                    new_glob_jpos -= move_to_zero_trans[:, :, None, :]
                    new_glob_rot_mat = transforms.quaternion_to_matrix(
                        new_glob_q
                    )  # BS X T X J X 3 X 3
                    new_glob_rot_6d = transforms.matrix_to_rotation_6d(
                        new_glob_rot_mat
                    )  # BS X T X J X 6

                    # Add original object position information to current window. (in a new canonicalized frame)
                    # This is only for given the target single frame as condition.

                    # Prepare canonicalized results of previous overlapped frames.
                    curr_normalized_global_jpos = ds.normalize_jpos_min_max(
                        new_glob_jpos
                    )  # BS X T X J X 3

                    if self.add_feet_contact:
                        cano_prev_sample_res = torch.cat(
                            (
                                curr_normalized_global_jpos.reshape(b, -1, 24 * 3),
                                new_glob_rot_6d.reshape(b, -1, 22 * 6),
                                prev_sample_res[:, :, -4:],
                            ),
                            dim=-1,
                        )
                    else:
                        cano_prev_sample_res = torch.cat(
                            (
                                curr_normalized_global_jpos.reshape(b, -1, 24 * 3),
                                new_glob_rot_6d.reshape(b, -1, 22 * 6),
                            ),
                            dim=-1,
                        )

                    curr_start_frame = cano_prev_sample_res[
                        :, 0:1
                    ]  # BS X 1 X D(24*3+22*6)

                    return curr_start_frame

                new_glob_jpos, new_glob_q, _, _ = rotate_at_frame_w_obj_global(
                    global_human_jpos[:, :, 0, :].data.cpu().numpy(),
                    global_human_q[:, :, 0, :].data.cpu().numpy(),
                    ds.parents,
                    n_past=1,
                    floor_z=True,
                    global_q=global_human_q.data.cpu().numpy(),
                    global_x=global_human_jpos.data.cpu().numpy(),
                    use_global=True,
                )
                # BS X T X J X 3, BS X T X J X 4, BS X T X 3, BS X T X 4
                new_glob_jpos = (
                    torch.from_numpy(new_glob_jpos).float().to(prev_sample_res.device)
                )
                new_glob_q = (
                    torch.from_numpy(new_glob_q).float().to(prev_sample_res.device)
                )

                global_human_root_jpos = new_glob_jpos[:, :, 0, :].clone()  # BS X T X 3
                global_human_root_trans = global_human_root_jpos + trans2joint[
                    :, None, :
                ].to(global_human_root_jpos.device)  # BS X T X 3

                move_to_zero_trans = global_human_root_trans[
                    :,
                    0:1,
                ].clone()  # Move the first frame's root joint x, y to 0,  BS X 1 X 3
                move_to_zero_trans[:, :, 2] = 0  # BS X 1 X 3

                global_human_root_trans -= move_to_zero_trans
                global_human_root_jpos -= move_to_zero_trans
                new_glob_jpos -= move_to_zero_trans[:, :, None, :]
                new_glob_rot_mat = transforms.quaternion_to_matrix(
                    new_glob_q
                )  # BS X T X J X 3 X 3
                new_glob_rot_6d = transforms.matrix_to_rotation_6d(
                    new_glob_rot_mat
                )  # BS X T X J X 6

                # The matrix that convert the orientation to canonicalized direction.
                cano_rot_mat = torch.matmul(
                    new_glob_rot_mat[:, 0, 0, :, :],
                    global_human_rot_mat[:, 0, 0, :, :].transpose(1, 2),
                )  # BS X 3 X 3

                # Add original object position information to current window. (in a new canonicalized frame)
                # This is only for given the target single frame as condition.
                curr_end_frame_init = (
                    curr_x_start_init.clone()
                )  # BS X W X D (24*3+22*6)
                curr_end_root_pos = ds.de_normalize_specific_jpos_min_max(
                    curr_end_frame_init[:, :, :3], 0
                )  # BS X W X 3
                curr_end_root_pos = torch.matmul(
                    cano_rot_mat[:, None, :, :].repeat(
                        1, curr_end_root_pos.shape[1], 1, 1
                    ),
                    curr_end_root_pos[:, :, :, None],
                )  # BS X W X 3 X 1
                curr_end_root_pos = curr_end_root_pos.squeeze(-1)  # BS X W X 3
                curr_end_root_pos -= move_to_zero_trans  # BS X W X 3

                # Assign target frame's object position infoirmation.
                curr_end_frame = torch.zeros_like(curr_end_frame_init)  # BS X W X D

                # Assign previous window's object information as condition to generate for current window.
                curr_end_frame[:, :, :3] = ds.normalize_specific_jpos_min_max(
                    curr_end_root_pos, 0
                )

                # Prepare canonicalized results of previous overlapped frames.
                curr_normalized_global_jpos = ds.normalize_jpos_min_max(
                    new_glob_jpos
                )  # BS X T X J X 3

                if self.add_feet_contact:
                    cano_prev_sample_res = torch.cat(
                        (
                            curr_normalized_global_jpos.reshape(b, -1, 24 * 3),
                            new_glob_rot_6d.reshape(b, -1, 22 * 6),
                            prev_sample_res[:, :, -4:],
                        ),
                        dim=-1,
                    )
                else:
                    cano_prev_sample_res = torch.cat(
                        (
                            curr_normalized_global_jpos.reshape(b, -1, 24 * 3),
                            new_glob_rot_6d.reshape(b, -1, 22 * 6),
                        ),
                        dim=-1,
                    )

                # curr_start_frame = cano_prev_sample_res[:, 0:1]  # BS X 1 X D(24*3+22*6)
                # NOTE: use self-defined start pose
                curr_start_frame = generate_curr_start_frame()

                curr_x_start = torch.cat(
                    (curr_start_frame, curr_end_frame[:, 1:, :]), dim=1
                )
                curr_x_start_denormalized = ds.de_normalize_jpos_min_max(
                    curr_x_start[:, :, : 24 * 3].reshape(b, -1, 24, 3)
                )
                if (
                    use_cut_step and t_idx + self.seq_len != num_steps and t_idx != 0
                ):  # exclude the first and last window
                    cut_step = True

                    cur_step = (
                        curr_x_start_denormalized[0, 30 - 1, 0]
                        - curr_x_start_denormalized[0, 0, 0]
                    )
                    cur_step[..., 2] = 0
                    cur_dis = torch.norm(cur_step).item()

                    def calc_dis(dir1, dir2):
                        dir1[2] = 0
                        dir2[2] = 0
                        dot_product = torch.dot(dir1, dir2)
                        norm_v1 = torch.norm(dir1)
                        norm_v2 = torch.norm(dir2)
                        cos_theta = dot_product / (norm_v1 * norm_v2)
                        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
                        dis = 0.2 + 0.7 * (0.5 * (1 + math.cos(theta)))  # 0.2 ~ 0.9
                        return dis

                    last_dir = (
                        x_start_denormalized[0, t_idx - 1, 0]
                        - x_start_denormalized[0, max(t_idx - 30 - 1, 0), 0]
                    )
                    cur_dir = (
                        x_start_denormalized[0, t_idx + 30 - 1, 0]
                        - x_start_denormalized[0, t_idx - 1, 0]
                    )
                    first_step_dis = calc_dis(last_dir, cur_dir)
                    # print("first_step_dis:{0}".format(first_step_dis))

                    curr_x_start_denormalized[0, 30 - 1, 0] = (
                        curr_x_start_denormalized[0, 0, 0]
                        + cur_step / cur_dis * first_step_dis
                    )
                    curr_x_start_denormalized[0, 60 - 1, 0] = (
                        curr_x_start_denormalized[0, 30 - 1, 0]
                        + cur_step / cur_dis * step_dis
                    )
                    curr_x_start_denormalized[0, 90 - 1, 0] = (
                        curr_x_start_denormalized[0, 60 - 1, 0]
                        + cur_step / cur_dis * step_dis
                    )
                    curr_x_start_denormalized[0, 120 - 1, 0] = (
                        curr_x_start_denormalized[0, 90 - 1, 0]
                        + cur_step / cur_dis * step_dis
                    )
                    new_x_start = ds.normalize_jpos_min_max(
                        curr_x_start_denormalized
                    )  # 1 X T X 24 X 3
                    curr_x_start[0, 1:, :3] = new_x_start[0, 1:, 0]

                if add_root_ori:
                    curr_root_ori = (
                        torch.zeros(curr_x_start.shape[0], curr_x_start.shape[1], 6)
                        .float()
                        .to(curr_x_start.device)
                    )  # BS X T X 3
                    curr_root_ori[:, 0] = cal_root_traj_xy_ori_from_root(
                        curr_x_start[:, 0]
                    )
                    curr_root_ori[:, 30 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 30 - 1, 0]
                        - curr_x_start_denormalized[:, 0, 0]
                    )
                    curr_root_ori[:, 60 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 60 - 1, 0]
                        - curr_x_start_denormalized[:, 30 - 1, 0]
                    )
                    curr_root_ori[:, 90 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 90 - 1, 0]
                        - curr_x_start_denormalized[:, 60 - 1, 0]
                    )
                    curr_root_ori[:, 120 - 1] = cal_root_traj_xy_ori(
                        curr_x_start_denormalized[:, 120 - 1, 0]
                        - curr_x_start_denormalized[:, 90 - 1, 0]
                    )
                    if t_idx + self.seq_len == num_steps:
                        # Use the same orientation as the next frame to avoid abrupt changes due to small distance differences.
                        curr_root_ori[:, 30 - 1] = curr_root_ori[:, 60 - 1]

                    curr_x_start = torch.cat(
                        (
                            curr_x_start[..., :204],
                            curr_root_ori,
                            curr_x_start[..., 204:],
                        ),
                        dim=-1,
                    )

                x_pose_cond = curr_x_start * (
                    1.0 - cond_mask
                )  # Remove noise, overall better than adding random noise.
                curr_x_cond = x_pose_cond  # BS X T X (24*3+22*6)

                if language_input is not None:
                    language_embedding = self.clip_encoder(
                        language_input[window_idx]
                    )  # BS X d_model
                else:
                    language_embedding = None

                for i in tqdm(
                    reversed(range(0, self.num_timesteps)),
                    desc="sampling loop time step",
                    total=self.num_timesteps,
                ):
                    # Apply previous window prediction as additional condition, direcly replacement.

                    curr_x = self.p_sample(
                        curr_x,
                        torch.full((b,), i, device=device, dtype=torch.long),
                        curr_x_cond,
                        language_embedding=language_embedding,
                        padding_mask=padding_mask,
                    )

                curr_x_human_jpos = curr_x[:, :, : 24 * 3].reshape(b, -1, 24, 3)
                curr_x_human_rot_6d = curr_x[:, :, 24 * 3 : 24 * 3 + 22 * 6].reshape(
                    b, -1, 22, 6
                )

                curr_x = torch.cat(
                    (
                        curr_x_human_jpos.reshape(b, -1, 24 * 3),
                        curr_x_human_rot_6d.reshape(b, -1, 22 * 6),
                        curr_x[:, :, -4:],
                    ),
                    dim=-1,
                )

                # Convert the results of this window to be at the canonical frame of the first frame in this whole sequence.
                converted_human_jpos, converted_rot_6d = self.apply_rotation_to_data(
                    ds, trans2joint, cano_rot_mat, curr_x[:, :, :-4]
                )
                # 1 X window X 24 X 3 (unnormalized), 1 X window X 22 X 6

                aligned_human_trans = (
                    global_human_jpos[:, 0:1, 0, :] - converted_human_jpos[:, 0:1, 0, :]
                )
                converted_human_jpos += aligned_human_trans[:, :, None, :]

                # need to first denormalize the human jpos
                (
                    converted_human_jpos,
                    converted_rot_6d,
                    new_prev_jpos,
                    new_prev_rot_6d,
                ) = apply_inertialize(
                    prev_jpos=ds.de_normalize_jpos_min_max(
                        whole_sample_res[:, :, : 24 * 3].reshape(b, -1, 24, 3)
                    ),
                    prev_rot_6d=whole_sample_res[
                        :, :, 24 * 3 : 24 * 3 + 22 * 6
                    ].reshape(b, -1, 22, 6),
                    window_jpos=converted_human_jpos,
                    window_rot_6d=converted_rot_6d,
                )

                converted_normalized_human_jpos = ds.normalize_jpos_min_max(
                    converted_human_jpos
                )
                converted_curr_x = torch.cat(
                    (
                        converted_normalized_human_jpos.reshape(b, self.seq_len, -1),
                        converted_rot_6d.reshape(b, self.seq_len, -1),
                        curr_x[:, :, -4:],
                    ),
                    dim=-1,
                )

                new_prev = torch.cat(
                    (
                        ds.normalize_jpos_min_max(new_prev_jpos).reshape(b, -1, 24 * 3),
                        new_prev_rot_6d.reshape(b, -1, 22 * 6),
                        whole_sample_res[:, :, -4:],
                    ),
                    dim=-1,
                )
                cut_frame = self.seq_len
                if cut_step:
                    generated_denormalized = ds.de_normalize_jpos_min_max(
                        converted_curr_x[:, :, : 24 * 3].reshape(b, -1, 24, 3)
                    )
                    generated_step = (
                        generated_denormalized[0, :, 0]
                        - generated_denormalized[0, 0, 0]
                    )
                    generated_step[..., 2] = 0
                    generated_dis = torch.norm(generated_step, dim=-1)

                    is_turn = first_step_dis < 0.7  # assume max is 0.9

                    for i in range(1, self.seq_len):
                        if abs(generated_dis[i] - cur_dis) < 0.03:
                            cut_frame = i
                            break
                    if is_turn:
                        pred_feet_contact = curr_x[:, :, -4:]
                        left_feet_contact = medfilt(
                            pred_feet_contact[0, :, 0].cpu().numpy(), kernel_size=3
                        )  # T
                        right_feet_contact = medfilt(
                            pred_feet_contact[0, :, 1].cpu().numpy(), kernel_size=3
                        )  # T
                        left_feet_contact = left_feet_contact > 0.95
                        right_feet_contact = right_feet_contact > 0.95

                        begin = max(1, cut_frame - 30)
                        end = min(self.seq_len, cut_frame + 30)
                        min_dis = 1e5
                        for i in range(begin, end):
                            if (left_feet_contact[i] == right_feet_contact[i]) or (
                                left_feet_contact[i] != left_feet_contact[i + 1]
                            ):
                                tmp_dis = abs(generated_dis[i] - cur_dis)
                                if tmp_dis < min_dis:
                                    min_dis = tmp_dis
                                    cut_frame = i
                    # print("cut step", cut_frame)

                    assert cut_frame > 0

                if not cut_step:
                    whole_sample_res = torch.cat(
                        (new_prev, converted_curr_x[:, :-overlap_frame_num]), dim=1
                    )
                    whole_cond_mask = torch.cat(
                        (whole_cond_mask, cond_mask[:, :-overlap_frame_num]), dim=1
                    )  # BS X T' X D
                else:
                    whole_sample_res = torch.cat(
                        (new_prev, converted_curr_x[:, :cut_frame]), dim=1
                    )
                    whole_cond_mask = torch.cat(
                        (whole_cond_mask, cond_mask[:, :cut_frame]), dim=1
                    )  # BS X T' X D
                    whole_cond_mask[:, -1, :2] = 0

                window_idx += 1

        whole_cond_mask = whole_cond_mask
        whole_sample_res = whole_sample_res
        return whole_sample_res, whole_cond_mask  # BS X T X D (24*3+22*6)

    def apply_rotation_to_data_human_only(self, ds, trans2joint, cano_rot_mat, curr_x):
        # cano_rot_mat:BS X 3 X 3, convert from the coodinate frame which canonicalize the first frame of a sequence to
        # the frame that canonicalize the first frame of a window.
        # trans2joint: BS X 3
        # curr_x: BS X window_size X D
        # This function is to convert window data to sequence data.
        bs, timesteps, _ = curr_x.shape

        pred_human_normalized_jpos = curr_x[:, :, : 24 * 3]  # BS X window_size X (24*3)
        pred_human_jpos = ds.de_normalize_jpos_min_max(
            pred_human_normalized_jpos.reshape(bs, timesteps, 24, 3)
        )  # BS X window_size X 24 X 3
        pred_human_rot_6d = curr_x[:, :, 24 * 3 :]  # BS X window_size X (22*6)

        pred_human_rot_mat = transforms.rotation_6d_to_matrix(
            pred_human_rot_6d.reshape(bs, timesteps, 22, 6)
        )  # BS X T X 22 X 3 X 3

        converted_human_jpos = torch.matmul(
            cano_rot_mat[:, None, None, :, :]
            .repeat(1, timesteps, 24, 1, 1)
            .transpose(3, 4),
            pred_human_jpos[:, :, :, :, None],
        ).squeeze(-1)  # BS X T X 24 X 3
        converted_rot_mat = torch.matmul(
            cano_rot_mat[:, None, None, :, :]
            .repeat(1, timesteps, 22, 1, 1)
            .transpose(3, 4),
            pred_human_rot_mat,
        )  # BS X T X 22 X 3 X 3

        converted_rot_6d = transforms.matrix_to_rotation_6d(converted_rot_mat)

        return converted_human_jpos, converted_rot_6d

    # @torch.no_grad()
    def sample(
        self,
        x_start,
        cond_mask=None,
        padding_mask=None,
        language_input=None,
        rest_human_offsets=None,
        data_dict=None,
        guidance_fn=None,
        opt_fn=None,
        return_diff_level_res=False,
    ):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.clip_encoder.eval()
        self.denoise_fn.eval()

        if cond_mask is not None:
            x_pose_cond = x_start * (
                1.0 - cond_mask
            )  # Remove noise, overall better than adding random noise.
            # x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)

            x_cond = x_pose_cond

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None

        if guidance_fn is not None:
            sample_res = self.p_sample_loop_guided(
                x_start.shape,
                x_start,
                x_cond,
                guidance_fn,
                opt_fn=opt_fn,
                rest_human_offsets=rest_human_offsets,
                data_dict=data_dict,
                cond_mask=cond_mask,
            )
            # BS X T X D
        else:
            sample_res = self.p_sample_loop(
                x_start.shape,
                x_cond,
                language_embedding=language_embedding,
                padding_mask=padding_mask,
                cond_mask=cond_mask,
                ori_pose_cond=x_start,
                return_diff_level_res=return_diff_level_res,
            )
            # BS X T X D

        self.denoise_fn.train()
        self.clip_encoder.train()

        return sample_res

    # @torch.no_grad()
    def sample_sliding_window_w_canonical(
        self,
        ds,
        trans2joint,
        x_start,
        cond_mask=None,
        padding_mask=None,
        overlap_frame_num=1,
        input_waypoints=False,
        rest_human_offsets=None,
        data_dict=None,
        language_input=None,
        guidance_fn=None,
        opt_fn=None,
        add_root_ori=False,
        step_dis=0.8,
        use_cut_step=True,
    ):
        # object_names: BS
        # object_scales: BS X T
        # trans2joint: BS X 3
        # first_frame_obj_com2tran: BS X 1 X 3
        # x_start: BS X T X D(3+9+24*3+22*6) (T can be larger than the window_size)
        # ori_x_cond: BS X 1 X (3+1024*3)
        # cond_mask: BS X window_size X D
        # padding_mask: BS X T
        self.denoise_fn.eval()
        self.clip_encoder.eval()

        sample_res, whole_cond_mask = self.p_sample_loop_sliding_window_w_canonical(
            ds,
            trans2joint,
            x_start,
            cond_mask=cond_mask,
            padding_mask=padding_mask,
            overlap_frame_num=overlap_frame_num,
            input_waypoints=input_waypoints,
            language_input=language_input,
            rest_human_offsets=rest_human_offsets,
            data_dict=data_dict,
            guidance_fn=guidance_fn,
            opt_fn=opt_fn,
            add_root_ori=add_root_ori,
            step_dis=step_dis,
            use_cut_step=use_cut_step,
        )

        # BS X T X D
        self.denoise_fn.train()
        self.clip_encoder.train()

        return sample_res, whole_cond_mask

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(
        self,
        x_start,
        x_cond,
        t,
        language_embedding=None,
        noise=None,
        padding_mask=None,
        rest_human_offsets=None,
        ds=None,
    ):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(
            x_start=x_start, t=t, noise=noise
        )  # noisy motion in noise level t.

        model_out = self.denoise_fn(
            x,
            t,
            x_cond,
            language_embedding=language_embedding,
            padding_mask=padding_mask,
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if padding_mask is not None:
            loss = (
                self.loss_fn(model_out, target, reduction="none")
                * padding_mask[:, 0, 1:][:, :, None]
            )
        else:
            loss = self.loss_fn(model_out, target, reduction="none")  # BS X T X D

        loss = reduce(loss, "b ... -> b (...)", "mean")  # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        if self.add_feet_contact:
            foot_idx = [7, 8, 10, 11]

            bs, num_steps, _ = model_out.shape

            model_contact = model_out[:, :, -4:]

            gt_global_jpos = target[:, :, : 24 * 3].reshape(bs, num_steps, 24, 3)
            gt_global_jpos = ds.de_normalize_jpos_min_max(
                gt_global_jpos
            )  # BS X T X 24 X 3

            global_jpos = model_out[:, :, : 24 * 3].reshape(bs, num_steps, 24, 3)
            global_jpos = ds.de_normalize_jpos_min_max(global_jpos)  # BS X T X 24 X 3

            static_idx = model_contact > 0.95  # BS x T x 4

            # FK to get joint positions. rest_human_offsets: BS X 24 X 3
            curr_seq_local_jpos = (
                rest_human_offsets[:, None].repeat(1, num_steps, 1, 1).cuda()
            )  # BS X T X 24 X 3
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(
                bs * num_steps, 24, 3
            )  # (BS*T) X 24 X 3
            curr_seq_local_jpos[:, 0, :] = global_jpos.reshape(bs * num_steps, 24, 3)[
                :, 0, :
            ]  # (BS*T) X 3

            global_joint_rot_6d = model_out[:, :, 24 * 3 : 24 * 3 + 22 * 6].reshape(
                bs, num_steps, 22, 6
            )  # BS X T X 22 X 6
            global_joint_rot_mat = transforms.rotation_6d_to_matrix(
                global_joint_rot_6d
            )  # BS X T X 22 X 3 X 3
            local_joint_rot_mat = quat_ik_torch(
                global_joint_rot_mat.reshape(-1, 22, 3, 3)
            )  # (BS*T) X 22 X 3 X 3
            _, human_jnts = quat_fk_torch(local_joint_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts.reshape(bs, num_steps, 24, 3)  # BS X T X 24 X 3

            # Add fk loss
            fk_loss = (
                self.loss_fn(human_jnts, gt_global_jpos, reduction="none")
                * padding_mask[:, 0, 1:][:, :, None, None]
            )  # BS X T X 24 X 3
            fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")

            fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

            # Add foot contact loss
            # model_feet = gt_global_jpos[:, :, foot_idx]  # foot positions (BS, T, 4, 3), GT debug
            model_feet = human_jnts[:, :, foot_idx]  # foot positions (BS, T, 4, 3)
            model_foot_v = torch.zeros_like(model_feet)
            model_foot_v[:, :-1] = (
                model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
            )  # (N, S-1, 4, 3)
            model_foot_v[~static_idx] = 0

            foot_loss = (
                self.loss_fn(
                    model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
                )
                * padding_mask[:, 0, 1:][:, :, None, None]
            )
            foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

            foot_loss = foot_loss * extract(self.p2_loss_weight, t, foot_loss.shape)
            # print("foot loss gt:{0}".format(foot_loss.mean()))

            return loss.mean(), foot_loss.mean(), fk_loss.mean()

        return loss.mean()

    def forward(
        self,
        x_start,
        cond_mask=None,
        padding_mask=None,
        language_input=None,
        rest_human_offsets=None,
        ds=None,
        use_noisy_traj=False,
    ):
        # x_start: BS X T X D, we predict object motion
        # (relative rotation matrix 9-dim with respect to the first frame, absolute translation 3-dim)
        # ori_x_cond: BS X 1 X D' (com pos + BPS representation), we only use the first frame.
        # language_embedding: BS X D(512)
        # contact_labels: BS X T
        # rest_human_offsets: BS X 24 X 3
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        if cond_mask is not None:
            # x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)

            # Not adding noise, for set9, 10 object motion.
            if use_noisy_traj:
                noise = torch.randn_like(x_start).to(x_start.device) * 0.1
                noise[:, 0] = 0.0  # Set the first frame to be 0.
                noise[:, -1] = 0.0  # Set the last frame to be 0.
                noise[:, :, 2:] = 0.0  # only add noise to the xy
                x_pose_cond = (x_start + noise) * (1.0 - cond_mask)
            else:
                x_pose_cond = x_start * (1.0 - cond_mask)

            x_cond = x_pose_cond

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)  # BS X d_model
        else:
            language_embedding = None

        if self.add_feet_contact:
            curr_loss, curr_feet_contact_loss, curr_fk_loss = self.p_losses(
                x_start,
                x_cond,
                t,
                language_embedding=language_embedding,
                padding_mask=padding_mask,
                rest_human_offsets=rest_human_offsets,
                ds=ds,
            )
            return curr_loss, curr_feet_contact_loss, curr_fk_loss
        else:
            curr_loss = self.p_losses(
                x_start,
                x_cond,
                t,
                language_embedding=language_embedding,
                padding_mask=padding_mask,
            )
            return curr_loss
