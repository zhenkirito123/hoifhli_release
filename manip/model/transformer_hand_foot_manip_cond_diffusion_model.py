import math
import os
from inspect import isfunction

import pytorch3d.transforms as transforms
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from tqdm.auto import tqdm

from manip.model.transformer_module import Decoder

# class EmbedAction(nn.Module):
#     def __init__(self, num_actions, latent_dim):
#         super().__init__()
#         self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

#     def forward(self, input):
#         # input: BS
#         idx = input.to(torch.long)  # an index array must be long
#         output = self.action_embedding[idx]
#         return output


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

    def forward(self, src, noise_t, condition, padding_mask=None):
        # src: BS X T X D
        # noise_t: int

        src = torch.cat((src, condition), dim=-1)

        noise_t_embed = self.time_mlp(noise_t)  # BS X d_model
        noise_t_embed = noise_t_embed[:, None, :]  # BS X 1 X d_model

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking
            padding_mask = (
                torch.ones(bs, 1, num_steps).to(src.device).bool()
            )  # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) + 1  # timesteps
        pos_vec = (
            pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)
        )  # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach()  # BS X D X T
        feat_pred, _ = self.motion_transformer(
            data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed
        )

        output = self.linear_out(feat_pred[:, 1:])  # BS X T X D

        return output  # predicted noise, the same size as the input


class CondGaussianDiffusion(nn.Module):
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
        batch_size=None,
        bps_in_dim=1024 * 3,
        bps_hidden_dim=512,
        bps_out_dim=256,
        enable_dropout=False,
    ):
        super().__init__()

        obj_feats_dim = bps_out_dim
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_in_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=obj_feats_dim),
        )
        # Input: (BS*T) X 3 X N
        # Output: (BS*T) X d X N, (BS*T) X d
        # self.object_encoder = Pointnet()

        self.object_wrist_feats_dim = 21  # 3 + 2 * 3 + 2 * 6

        if opt.add_start_human_pose:
            d_input_feats = 2 * d_feats + 3 + obj_feats_dim
            # d_input_feats = d_feats+obj_feats_dim # For debug
        else:
            d_input_feats = d_feats + self.object_wrist_feats_dim + obj_feats_dim

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

    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output = self.denoise_fn(x, t, x_cond, padding_mask)

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

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, padding_mask=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            x_cond=x_cond,
            padding_mask=padding_mask,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, shape, x_start, x_cond, padding_mask=None, ref_pose=None, ref_mask=None
    ):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
            leave=False,
        ):
            # NOTE: inpainting
            if ref_mask is not None and ref_pose is not None:
                x = x * (1.0 - ref_mask) + ref_pose * ref_mask

            x = self.p_sample(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                x_cond,
                padding_mask=padding_mask,
            )

        return x  # BS X T X D

    # @torch.no_grad()
    def sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        self.bps_encoder.eval()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        ori_x_cond[:, :, :bps_start] = 0  # NOTE: enable this for sensor only
        x_cond = torch.cat(
            (
                ori_x_cond[:, :, :bps_start],
                self.bps_encoder(ori_x_cond[:, :, bps_start:]),
            ),
            dim=-1,
        )  # BS X T X (21+256)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        sample_res = self.p_sample_loop(x_start.shape, x_start, x_cond, padding_mask)
        # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()

        return sample_res

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

    def p_losses(self, x_start, x_cond, t, noise=None, padding_mask=None):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T
        b, timesteps, d_input = x_start.shape  # BS X T X D(3*4/3*2)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(
            x_start=x_start, t=t, noise=noise
        )  # noisy motion in noise level t.

        model_out = self.denoise_fn(x, t, x_cond, padding_mask)

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

        return loss.mean()

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        ori_x_cond[:, :, :bps_start] = 0  # NOTE: enable this for sensor only
        x_cond = torch.cat(
            (
                ori_x_cond[:, :, :bps_start],
                self.bps_encoder(ori_x_cond[:, :, bps_start:]),
            ),
            dim=-1,
        )  # BS X T X (21+256)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss


class CondGaussianDiffusionBothSensor(CondGaussianDiffusion):
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
        batch_size=None,
        bps_in_dim=1024 * 3,
        bps_hidden_dim=512,
        bps_out_dim=256,
        enable_dropout=False,
        second_bps_in_dim=200,
        second_bps_hidden_dim=512,
        second_bps_out_dim=256,
    ):
        super(CondGaussianDiffusion, self).__init__()

        obj_feats_dim = bps_out_dim + second_bps_out_dim
        self.bps_in_dim = bps_in_dim
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_in_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=bps_out_dim),
        )
        self.second_bps_in_dim = second_bps_in_dim
        self.second_bps_encoder = nn.Sequential(
            nn.Linear(
                in_features=second_bps_in_dim, out_features=second_bps_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=second_bps_hidden_dim, out_features=second_bps_out_dim
            ),
        )
        # Input: (BS*T) X 3 X N
        # Output: (BS*T) X d X N, (BS*T) X d
        # self.object_encoder = Pointnet()

        self.object_wrist_feats_dim = 21  # 3 + 2 * 3 + 2 * 6

        if opt.add_start_human_pose:
            d_input_feats = 2 * d_feats + 3 + obj_feats_dim
            # d_input_feats = d_feats+obj_feats_dim # For debug
        else:
            d_input_feats = d_feats + self.object_wrist_feats_dim + obj_feats_dim

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

    def sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.second_bps_encoder.eval()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        ori_x_cond[:, :, :bps_start] = 0  # NOTE: enable this for sensor only
        x_cond = torch.cat(
            (
                ori_x_cond[:, :, :bps_start],
                self.bps_encoder(
                    ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
                ),
                self.second_bps_encoder(
                    ori_x_cond[:, :, bps_start + self.bps_in_dim :]
                ),
            ),
            dim=-1,
        )  # BS X T X (21+256)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        sample_res = self.p_sample_loop(x_start.shape, x_start, x_cond, padding_mask)
        # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.second_bps_encoder.train()

        return sample_res

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        ori_x_cond[:, :, :bps_start] = 0  # NOTE: enable this for sensor only
        x_cond = torch.cat(
            (
                ori_x_cond[:, :, :bps_start],
                self.bps_encoder(
                    ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
                ),
                self.second_bps_encoder(
                    ori_x_cond[:, :, bps_start + self.bps_in_dim :]
                ),
            ),
            dim=-1,
        )  # BS X T X (21+256)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss


class CondGaussianDiffusionBothSensorNew(CondGaussianDiffusion):
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
        batch_size=None,
        bps_in_dim=1024 * 3,
        bps_hidden_dim=512,
        bps_out_dim=256,
        enable_dropout=False,
        second_bps_in_dim=200,
        second_bps_hidden_dim=512,
        second_bps_out_dim=256,
        contact_label_out_dim=256,
        object_wrist_feats_dim=21,
    ):
        super(CondGaussianDiffusion, self).__init__()

        obj_feats_dim = bps_out_dim + second_bps_out_dim
        self.bps_in_dim = bps_in_dim
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_in_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=bps_out_dim),
        )
        self.second_bps_in_dim = second_bps_in_dim
        self.second_bps_encoder = nn.Sequential(
            nn.Linear(
                in_features=second_bps_in_dim, out_features=second_bps_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=second_bps_hidden_dim, out_features=second_bps_out_dim
            ),
        )
        # Input: (BS*T) X 3 X N
        # Output: (BS*T) X d X N, (BS*T) X d
        # self.object_encoder = Pointnet()

        self.object_wrist_feats_dim = object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6

        # if opt.add_start_human_pose:
        #     d_input_feats = 2*d_feats+3+obj_feats_dim
        #     # d_input_feats = d_feats+obj_feats_dim # For debug
        # else:
        #     d_input_feats = d_feats + self.object_wrist_feats_dim + obj_feats_dim

        d_input_feats = d_feats
        self.wrist_obj_traj_condition = opt.wrist_obj_traj_condition
        self.ambient_sensor_condition = opt.ambient_sensor_condition
        self.proximity_sensor_condition = opt.proximity_sensor_condition
        self.ref_pose_condition = opt.ref_pose_condition
        self.contact_label_condition = opt.contact_label_condition
        if opt.wrist_obj_traj_condition:
            d_input_feats += self.object_wrist_feats_dim

        if opt.ambient_sensor_condition:
            d_input_feats += bps_out_dim

        if opt.proximity_sensor_condition:
            d_input_feats += second_bps_out_dim

        if opt.ref_pose_condition:
            d_input_feats += d_feats

        if opt.contact_label_condition:
            d_input_feats += contact_label_out_dim

        print(f"d_input_feats: {d_input_feats}")

        self.contact_embedding = nn.Linear(2, contact_label_out_dim)

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

    def sample(
        self,
        x_start,
        ori_x_cond,
        cond_mask=None,
        padding_mask=None,
        ref_pose=None,
        ref_mask=None,
    ):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.second_bps_encoder.eval()
        self.contact_embedding.eval()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6

        x_cond = []
        if self.wrist_obj_traj_condition:
            wrist_obj_traj = ori_x_cond[:, :, :bps_start]
            x_cond.append(wrist_obj_traj)
        if self.ambient_sensor_condition:
            ambient_sensor = self.bps_encoder(
                ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
            )
            x_cond.append(ambient_sensor)
        if self.proximity_sensor_condition:
            proximity_sensor = self.second_bps_encoder(
                ori_x_cond[
                    :,
                    :,
                    bps_start + self.bps_in_dim : bps_start
                    + self.bps_in_dim
                    + self.second_bps_in_dim,
                ]
            )
            x_cond.append(proximity_sensor)
        if self.ref_pose_condition and cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask)
            x_cond.append(x_pose_cond)
        if self.contact_label_condition:
            contact_label = self.contact_embedding(
                ori_x_cond[:, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :]
            )
            x_cond.append(contact_label)
        x_cond = torch.cat(x_cond, dim=-1)

        sample_res = self.p_sample_loop(x_start.shape, x_start, x_cond, padding_mask)
        # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.second_bps_encoder.train()
        self.contact_embedding.train()

        return sample_res

    def forward(
        self, x_start, ori_x_cond, cond_mask=None, padding_mask=None, target_pose=None
    ):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6

        x_cond = []
        if self.wrist_obj_traj_condition:
            wrist_obj_traj = ori_x_cond[:, :, :bps_start]
            x_cond.append(wrist_obj_traj)
        if self.ambient_sensor_condition:
            ambient_sensor = self.bps_encoder(
                ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
            )
            x_cond.append(ambient_sensor)
        if self.proximity_sensor_condition:
            proximity_sensor = self.second_bps_encoder(
                ori_x_cond[
                    :,
                    :,
                    bps_start + self.bps_in_dim : bps_start
                    + self.bps_in_dim
                    + self.second_bps_in_dim,
                ]
            )
            x_cond.append(proximity_sensor)
        if self.ref_pose_condition and cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask)
            x_cond.append(x_pose_cond)
        if self.contact_label_condition:
            contact_label = self.contact_embedding(
                ori_x_cond[:, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :]
            )
            x_cond.append(contact_label)
        x_cond = torch.cat(x_cond, dim=-1)

        curr_loss = self.p_losses(
            x_start, x_cond, t, padding_mask=padding_mask, target_pose=target_pose
        )

        return curr_loss

    def p_losses(
        self, x_start, x_cond, t, noise=None, padding_mask=None, target_pose=None
    ):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T
        # target_pose: BS X T X D ()
        b, timesteps, d_input = x_start.shape  # BS X T X D(3*4/3*2)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(
            x_start=x_start, t=t, noise=noise
        )  # noisy motion in noise level t.

        model_out = self.denoise_fn(x, t, x_cond, padding_mask)

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

        return loss.mean()


class CondGaussianDiffusionBothSensorContact(CondGaussianDiffusionBothSensor):
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
        batch_size=None,
        bps_in_dim=1024 * 3,
        bps_hidden_dim=512,
        bps_out_dim=256,
        enable_dropout=False,
        second_bps_in_dim=200,
        second_bps_hidden_dim=512,
        second_bps_out_dim=256,
    ):
        super(CondGaussianDiffusion, self).__init__()

        obj_feats_dim = bps_out_dim + second_bps_out_dim
        self.bps_in_dim = bps_in_dim
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_in_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=bps_out_dim),
        )
        self.second_bps_in_dim = second_bps_in_dim
        self.second_bps_encoder = nn.Sequential(
            nn.Linear(
                in_features=second_bps_in_dim, out_features=second_bps_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=second_bps_hidden_dim, out_features=second_bps_out_dim
            ),
        )
        # Input: (BS*T) X 3 X N
        # Output: (BS*T) X d X N, (BS*T) X d
        # self.object_encoder = Pointnet()

        self.object_wrist_feats_dim = 21  # 3 + 2 * 3 + 2 * 6

        if opt.add_start_human_pose:
            d_input_feats = 2 * d_feats + 3 + obj_feats_dim
            # d_input_feats = d_feats+obj_feats_dim # For debug
        else:
            d_input_feats = d_feats + 2 + obj_feats_dim

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

    def sample(
        self,
        x_start,
        ori_x_cond,
        cond_mask=None,
        padding_mask=None,
        ref_pose=None,
        ref_mask=None,
    ):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.second_bps_encoder.eval()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        x_cond = torch.cat(
            (
                self.bps_encoder(
                    ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
                ),
                self.second_bps_encoder(
                    ori_x_cond[
                        :,
                        :,
                        bps_start + self.bps_in_dim : bps_start
                        + self.bps_in_dim
                        + self.second_bps_in_dim,
                    ]
                ),
                ori_x_cond[
                    :, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :
                ],
            ),
            dim=-1,
        )  # BS X T X (2 * 256 + 2)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        sample_res = self.p_sample_loop(
            x_start.shape,
            x_start,
            x_cond,
            padding_mask,
            ref_pose=ref_pose,
            ref_mask=ref_mask,
        )
        # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.second_bps_encoder.train()

        return sample_res

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        x_cond = torch.cat(
            (
                self.bps_encoder(
                    ori_x_cond[:, :, bps_start : bps_start + self.bps_in_dim]
                ),
                self.second_bps_encoder(
                    ori_x_cond[
                        :,
                        :,
                        bps_start + self.bps_in_dim : bps_start
                        + self.bps_in_dim
                        + self.second_bps_in_dim,
                    ]
                ),
                ori_x_cond[
                    :, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :
                ],
            ),
            dim=-1,
        )  # BS X T X (2 * 256 + 2)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss


class CondGaussianDiffusionProximitySensorContact(
    CondGaussianDiffusionBothSensorContact
):
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
        batch_size=None,
        bps_in_dim=1024 * 3,
        bps_hidden_dim=512,
        bps_out_dim=256,
        enable_dropout=False,
        second_bps_in_dim=200,
        second_bps_hidden_dim=512,
        second_bps_out_dim=256,
    ):
        super(CondGaussianDiffusion, self).__init__()

        obj_feats_dim = second_bps_out_dim
        self.bps_in_dim = bps_in_dim
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_in_dim, out_features=bps_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=bps_hidden_dim, out_features=bps_out_dim),
        )
        self.second_bps_in_dim = second_bps_in_dim
        self.second_bps_encoder = nn.Sequential(
            nn.Linear(
                in_features=second_bps_in_dim, out_features=second_bps_hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=second_bps_hidden_dim, out_features=second_bps_out_dim
            ),
        )
        # Input: (BS*T) X 3 X N
        # Output: (BS*T) X d X N, (BS*T) X d
        # self.object_encoder = Pointnet()

        self.object_wrist_feats_dim = 21  # 3 + 2 * 3 + 2 * 6

        if opt.add_start_human_pose:
            d_input_feats = 2 * d_feats + 3 + obj_feats_dim
            # d_input_feats = d_feats+obj_feats_dim # For debug
        else:
            d_input_feats = d_feats + 2 + obj_feats_dim

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

    def sample(
        self,
        x_start,
        ori_x_cond,
        cond_mask=None,
        padding_mask=None,
        ref_pose=None,
        ref_mask=None,
    ):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        self.bps_encoder.eval()
        self.second_bps_encoder.eval()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        x_cond = torch.cat(
            (
                # self.bps_encoder(ori_x_cond[:, :, bps_start:bps_start+self.bps_in_dim]),
                self.second_bps_encoder(
                    ori_x_cond[
                        :,
                        :,
                        bps_start + self.bps_in_dim : bps_start
                        + self.bps_in_dim
                        + self.second_bps_in_dim,
                    ]
                ),
                ori_x_cond[
                    :, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :
                ],
            ),
            dim=-1,
        )  # BS X T X (256 + 2)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)

        sample_res = self.p_sample_loop(
            x_start.shape,
            x_start,
            x_cond,
            padding_mask,
            ref_pose=ref_pose,
            ref_mask=ref_mask,
        )
        # BS X T X D

        self.denoise_fn.train()
        self.bps_encoder.train()
        self.second_bps_encoder.train()

        return sample_res

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors.
        bps_start = self.object_wrist_feats_dim  # 3 + 2 * 3 + 2 * 6
        x_cond = torch.cat(
            (
                # self.bps_encoder(ori_x_cond[:, :, bps_start:bps_start+self.bps_in_dim]),
                self.second_bps_encoder(
                    ori_x_cond[
                        :,
                        :,
                        bps_start + self.bps_in_dim : bps_start
                        + self.bps_in_dim
                        + self.second_bps_in_dim,
                    ]
                ),
                ori_x_cond[
                    :, :, bps_start + self.bps_in_dim + self.second_bps_in_dim :
                ],
            ),
            dim=-1,
        )  # BS X T X (256 + 2)

        if cond_mask is not None:
            x_pose_cond = x_start * (1.0 - cond_mask) + cond_mask * torch.randn_like(
                x_start
            ).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss
