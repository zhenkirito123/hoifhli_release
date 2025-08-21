import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")

parser.add_argument("--resume", action="store_true", default=False)

parser.add_argument("--motion", nargs="+", type=str, default=None)
parser.add_argument("--render_to", type=str, default=None)
parser.add_argument("--blender", type=str, default=None)



parser.add_argument("--test1", action="store_true", default=False)
parser.add_argument("--test2", action="store_true", default=False)
parser.add_argument("--terminate", type=float, default=-1)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--actor_lr", type=float, default=5e-6)
parser.add_argument("--replay", action="store_true", default=False)

settings = parser.parse_args()

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
    )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None and torch.cuda.is_available and "cuda" in str(device):
        torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
    
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

TRAINING_PARAMS = dict(
    horizon = 8,
    num_envs = 512,
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = settings.actor_lr,
    critic_lr = 1e-4,
    gamma = settings.gamma,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000,
    save_interval = None,
    log_interval = 50,
    terminate_reward = settings.terminate,
    control_mode="position"
)

def test(env, model):
    model.eval()
    env.eval()
    env.reset()
    blender, dones = [], np.array(False)
    while not env.request_quit:
        obs, info = env.reset_done()
        if settings.blender:
            if dones.item():
                if len(blender) > 100:
                    import pickle
                    with open(settings.blender, "wb") as f:
                        pickle.dump(blender, f)
                    print("Export poses of", len(blender), "frames to:", settings.blender)
                    exit()
                blender.clear()
            blender.append(env.pose())
        seq_len = info["ob_seq_lens"]
        actions = model.act(obs, seq_len-1)
        _, _, dones, _ = env.step(actions)

        # if env.simulation_step == 378:
        #     state_dict = torch.load("ckpt_seq17b0_new4/ckpt", map_location=torch.device(settings.device))
        #     model.load_state_dict(state_dict["model"])
        # if env.simulation_step == 700:
        #     env.viewer_pause = True


def train(env, model, ckpt_dir, training_params, ckpt=None):
    if ckpt_dir is not None:
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None

    optimizer = torch.optim.Adam([
        {"params": model.actor.parameters(), "lr": training_params.actor_lr},
        {"params": model.critic.parameters(), "lr": training_params.critic_lr}
    ])
    ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
    disc_optimizer = {name: torch.optim.Adam(disc.parameters(), training_params.disc_lr) for name, disc in model.discriminators.items()}

    buffer = dict(
        s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[],
        ob_seq_len=[]
    )
    multi_critics = env.reward_weights is not None and env.reward_weights.size(-1) > 1
    if multi_critics:
        buffer["reward_weights"] = []
    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"] = []

    buffer_disc = {
        name: dict(fake=[], real=[]) for name in env.discriminators.keys()
    }
    real_losses, fake_losses = {n:[] for n in buffer_disc.keys()}, {n:[] for n in buffer_disc.keys()}
    

    BATCH_SIZE = training_params.batch_size
    HORIZON = training_params.horizon
    GAMMA = training_params.gamma
    GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
    OPT_EPOCHS = training_params.opt_epochs
    LOG_INTERVAL = training_params.log_interval

    epoch = 0

    if ckpt is not None and os.path.exists(settings.ckpt):
        if os.path.isdir(settings.ckpt):
            ckpt = os.path.join(settings.ckpt, "ckpt")
        else:
            ckpt = settings.ckpt
            settings.ckpt = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=device)
            model.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optimizer"])
            for n, opt in disc_optimizer.items():
                opt.load_state_dict(state_dict["disc_optimizer"][n])
            if "rng_state" in state_dict:
                set_rng_state(torch.load(ckpt, map_location="cpu")["rng_state"], device)
            epoch = state_dict["epoch"]

    model.eval()
    env.train()
    env.reset()
    tic = time.time()
    while not env.request_quit:
        with torch.no_grad():
            obs, info = env.reset_done()
            seq_len = info["ob_seq_lens"]
            reward_weights = info["reward_weights"]
            actions, values, log_probs = model.act(obs, seq_len-1, stochastic=True)
            obs_, rews, dones, info = env.step(actions)
            log_probs = log_probs.sum(-1, keepdim=True)
            not_done = (~dones).unsqueeze_(-1)
            terminate = info["terminate"]
            
            if env.discriminators:
                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]

            values_ = model.evaluate(obs_, seq_len)

        buffer["s"].append(obs)
        buffer["a"].append(actions)
        buffer["v"].append(values)
        buffer["lp"].append(log_probs)
        buffer["v_"].append(values_)
        buffer["not_done"].append(not_done)
        buffer["terminate"].append(terminate)
        buffer["ob_seq_len"].append(seq_len)
        if has_goal_reward:
            buffer["r"].append(rews)
        if multi_critics:
            buffer["reward_weights"].append(reward_weights)
        if env.discriminators:
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])

        if len(buffer["s"]) == HORIZON:
            disc_data = []
            ob_seq_lens = torch.cat(buffer["ob_seq_len"])
            ob_seq_end_frames = ob_seq_lens - 1
            if env.discriminators:
                with torch.no_grad():
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"])
                        real_ = torch.cat(data["real"])
                        end_frame = ob_seq_lens # N

                        length = torch.arange(fake.size(1), 
                            dtype=end_frame.dtype, device=end_frame.device
                        ).unsqueeze_(0)         # 1 x L
                        mask = length <= end_frame.unsqueeze(1)     # N x L
                        mask_ = length >= fake.size(1)-1 - end_frame.unsqueeze(1)

                        real = torch.zeros_like(real_)
                        real[mask] = real_[mask_]
                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])
                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)
                        disc_data.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    if len(ref) != n_samples:
                        n_samples = len(ref)
                        idx = torch.randperm(n_samples)
                    for batch in range(n_samples//BATCH_SIZE):
                        sample = idx[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)
                    
                        loss_r = torch.nn.functional.relu(1-score_r).mean()
                        loss_f = torch.nn.functional.relu(1+score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(r.size(0), dtype=r.dtype, device=r.device)
                            alpha = alpha.view(-1, *([1]*(r.ndim-1)))
                            interp = alpha*r+(1-alpha)*f
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        grad = torch.autograd.grad(
                            score_interp, interp, torch.ones_like(score_interp),
                            retain_graph=True, create_graph=True, only_inputs=True
                        )[0]
                        gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()
                        l = loss_f + loss_r + 10*gp
                        l.backward()
                        opt.step()
                        opt.zero_grad()

                        real_loss.append(score_r.mean().item())
                        fake_loss.append(score_f.mean().item())


            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                if multi_critics:
                    reward_weights = torch.cat(buffer["reward_weights"])
                    rewards = torch.empty_like(reward_weights)
                else:
                    reward_weights = None
                    rewards = None
                for name, disc, _, ob, seq_end_frame in disc_data:
                    r = (disc(ob, seq_end_frame, normalize=False).clamp_(-1, 1)
                            .mean(-1, keepdim=True))
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                if has_goal_reward:
                    rewards_task = torch.cat(buffer["r"])
                    if rewards is None:
                        rewards = rewards_task
                    else:
                        rewards[:, -rewards_task.size(-1):] = rewards_task
                else:
                    rewards_task = None
                rewards[terminate] = training_params.terminate_reward

                values = torch.cat(buffer["v"])
                values_ = torch.cat(buffer["v_"])
                if model.value_normalizer is not None:
                    values = model.value_normalizer(values, unnorm=True)
                    values_ = model.value_normalizer(values_, unnorm=True)
                values_[terminate] = 0
                rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                values = values.view(HORIZON, -1, values.size(-1))
                values_ = values_.view(HORIZON, -1, values_.size(-1))

                not_done = buffer["not_done"]
                advantages = (rewards - values).add_(values_, alpha=GAMMA)
                for t in reversed(range(HORIZON-1)):
                    advantages[t].add_(advantages[t+1]*not_done[t], alpha=GAMMA_LAMBDA)

                advantages = advantages.view(-1, advantages.size(-1))
                returns = advantages + values.view(-1, advantages.size(-1))

                log_probs = torch.cat(buffer["lp"])
                actions = torch.cat(buffer["a"])
                states = torch.cat(buffer["s"])
                ob_seq_lens = torch.cat(buffer["ob_seq_len"])
                ob_seq_end_frames = ob_seq_lens - 1

                sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                advantages = (advantages - mu) / (sigma + 1e-8) # (HORIZON x N_ENVS) x N_DISC
                
                length = torch.arange(env.ob_horizon, 
                    dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                states_raw = model.observe(states, norm=False)[0]
                model.ob_normalizer.update(states_raw[mask])
                if model.value_normalizer is not None:
                    model.value_normalizer.update(returns)
                    returns = model.value_normalizer(returns)
                if multi_critics:
                    advantages = advantages.mul_(reward_weights)

            n_samples = advantages.size(0)
            epoch += 1
            model.train()
            policy_loss, value_loss = [], []
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE *(batch+1)]
                    s = states[sample]
                    a = actions[sample]
                    lp = log_probs[sample]
                    adv = advantages[sample]
                    v_t = returns[sample]
                    end_frame = ob_seq_end_frames[sample]

                    pi_, v_ = model(s, end_frame)
                    lp_ = pi_.log_prob(a).sum(-1, keepdim=True)

                    ratio = torch.exp(lp_ - lp)
                    clipped_ratio = torch.clamp(ratio, 1.0-0.2, 1.0+0.2)
                    pg_loss = -torch.min(adv*ratio, adv*clipped_ratio).sum(-1).mean()
                    vf_loss = (v_ - v_t).square().mean()
                    #if torch.isnan(pg_loss).item():
                    #    print(torch.where(torch.isnan(adv)))
                    #    print(torch.where(torch.isnan(ratio)))
                    #    print(torch.max(lp_), torch.min(lp_))
                    #    print(torch.max(lp), torch.min(lp))
                    #    print(torch.where(torch.isinf(lp_)))
                    #    print(torch.where(torch.isinf(lp)))
                    #    print(a[torch.where(torch.isinf(lp_))[0][0]])

                    loss = pg_loss + 0.5*vf_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    policy_loss.append(pg_loss.item())
                    value_loss.append(vf_loss.item())
            model.eval()
            for v in buffer.values(): v.clear()
            for buf in buffer_disc.values():
                for v in buf.values(): v.clear()
                
            if epoch % LOG_INTERVAL == 1:
                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
                if multi_critics:
                    rewards = rewards.view(*reward_weights.shape)
                    r = rewards.mean(0).cpu().tolist()
                    # reward_tot = (rewards * reward_weights).sum(-1, keepdims=True).mean(0).item()
                else:
                    r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                terminate = terminate.sum().cpu().item()
                if not settings.silent:
                    print("Epoch: {}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f}/{} -- {:.4f}s".format(
                        epoch, policy_loss, value_loss, "/".join(list(map("{:.4f}".format, r))), lifetime, terminate, time.time()-tic
                    ))
                if logger is not None:
                    logger.add_scalar("train/lifetime", lifetime, epoch)
                    logger.add_scalar("train/terminate", terminate, epoch)
                    logger.add_scalar("train/reward", np.mean(r), epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, epoch)
                    logger.add_scalar("train/loss_value", value_loss, epoch)
                    for name, r_loss in real_losses.items():
                        if r_loss: logger.add_scalar("score_real/{}".format(name), sum(r_loss)/len(r_loss), epoch)
                    for name, f_loss in fake_losses.items():
                        if f_loss: logger.add_scalar("score_fake/{}".format(name), sum(f_loss)/len(f_loss), epoch)
                    if rewards_task is not None and rewards_task.size(-1) < rewards.size(-1):
                        rewards_task = rewards_task.mean(0).cpu().tolist()
                        for i in range(len(rewards_task)):
                            logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], epoch)
                for v in real_losses.values(): v.clear()
                for v in fake_losses.values(): v.clear()
            
            if ckpt_dir is not None:
                state = None
                if epoch % 50 == 0:
                    state = dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        disc_optimizer={n: disc_opt.state_dict() for n, disc_opt in disc_optimizer.items()},
                        rng_state=get_rng_state(device),
                        epoch=epoch
                    )
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                if epoch % training_params.save_interval == 0:
                    if state is None:
                        state = dict(
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            disc_optimizer={n: disc_opt.state_dict() for n, disc_opt in disc_optimizer.items()},
                            rng_state=get_rng_state(device),
                            epoch=epoch
                        )
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
                if epoch >= training_params.max_epochs: exit()
            tic = time.time()

if __name__ == "__main__":
    if os.path.splitext(settings.config)[-1] in [".pkl", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(
            motion_file = settings.config
        )
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print(TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())
    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid
    if settings.replay:
        env_cls = env.ICCGANHumanoidDemo
        settings.test = True
    print(env_cls, config.env_params)

    if settings.test:
        num_envs = 1
    else:
        num_envs = training_params.num_envs
        if settings.ckpt and not settings.resume:
            if os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt")):
                raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(settings.ckpt))
            import shutil, sys
            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            shutil.copy("main.py", settings.ckpt)
            shutil.copy("env.py", settings.ckpt)
            shutil.copy("utils.py", settings.ckpt)
            shutil.copy("models.py", settings.ckpt)
            shutil.copy("ref_motion.py", settings.ckpt)
            with open(os.path.join(settings.ckpt, "command_{}.txt".format(time.time())), "w") as f:
                f.write(" ".join(sys.argv))
    if settings.motion is not None:
        config.env_params["motion_file"] = settings.motion

    env = env_cls(num_envs,
        discriminators=discriminators,
        compute_device=settings.device, 
        render_to=settings.render_to,
        test1=settings.test1, test2=settings.test2,
        **config.env_params
    )
    if settings.test:
        env.episode_length = 500000
    value_dim = len(env.discriminators)+env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                print("Load model from {}".format(ckpt))
                state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
                model.load_state_dict(state_dict["model"])
        env.render()
        test(env, model)
    else:
        train(env, model, settings.ckpt, training_params, settings.ckpt if settings.resume else None)
