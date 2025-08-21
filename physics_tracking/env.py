from typing import Callable, Optional, Union, List, Dict, Any
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized, quat2axang, expmap2quat

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: Optional[bool]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,

        render_to: Optional[str] = None,

        **kwargs
    ):
        self.viewer = None
        self.render_to = render_to

        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if type(self.character_model) == str:
            self.character_model = [self.character_model]

        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors, self.actuated_dofs = self.create_envs(n_envs)
        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        self.root_tensor.fill_(0)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.joint_tensor.fill_(0)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_tensor))
        self.root_updated_actors, self.dof_updated_actors = [], []
        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.num_position_iterations = 4
        p.physx.num_velocity_iterations = 0
        p.physx.contact_offset = 0.01
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 1000.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

    def ground_height(self, p, env_ids=None):
        return None

    def create_envs(self, n: int, start_height: float=0.89,  actuate_all_dofs: bool=True, asset_options: Dict[str, Any]=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3

        actor_assets = []
        controllable_dofs = []
        for character_model in self.character_model:
            asset_opt = gymapi.AssetOptions()
            asset_opt.angular_damping = 0.01
            asset_opt.max_angular_velocity = 100.0
            asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
            for k, v in asset_options.items():
                setattr(asset_opt, k, v)

            # when loading mesh in mjcf, the convex hull settings does not work
            asset_opt.vhacd_enabled = True
            asset_opt.vhacd_params.max_convex_hulls = 32
            asset_opt.vhacd_params.max_num_vertices_per_ch = 64
            asset_opt.vhacd_params.resolution = 300000
            if "jiaman" not in character_model and "box" in character_model:
                asset_opt.angular_damping = 2
                #asset_opt.linear_damping = 1

            asset = self.gym.load_asset(self.sim,
                os.path.abspath(os.path.dirname(character_model)),
                os.path.basename(character_model),
                asset_opt)
            actor_assets.append(asset)
            if actuate_all_dofs:
                controllable_dofs.append([i for i in range(self.gym.get_asset_dof_count(asset))])
            else:
                actuators = []
                for i in range(self.gym.get_asset_actuator_count(asset)):
                    name = self.gym.get_asset_actuator_joint_name(asset, i)
                    actuators.append(self.gym.find_asset_dof_index(asset, name))
                    if actuators[-1] == -1:
                        raise ValueError("Failed to find joint with name {}".format(name))
                controllable_dofs.append(sorted(actuators) if len(actuators) else [])

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(start_height))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        total_rigids = sum([self.gym.get_asset_rigid_body_count(asset) for asset in actor_assets])
        total_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in actor_assets])

        actuated_dofs = []
        for env_id in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            self.gym.begin_aggregate(env, total_rigids, total_shapes, True)
            for aid, (asset, dofs) in enumerate(zip(actor_assets, controllable_dofs)):
                actor = self.gym.create_actor(env, asset, start_pose, "actor{}_{}".format(env_id, aid), env_id, -1, 0)

                dof_prop = self.gym.get_asset_dof_properties(asset)
                for k in range(len(dof_prop)):
                    if k in dofs:
                        dof_prop[k]["driveMode"] = control_mode
                    else:
                        dof_prop[k]["driveMode"] = gymapi.DOF_MODE_NONE
                        dof_prop[k]["stiffness"] = 0
                        dof_prop[k]["damping"] = 0
                self.gym.set_actor_dof_properties(env, actor, dof_prop)
                if env_id == n-1:
                    actors.append(actor)
                    actuated_dofs.append(dofs)
            self.gym.end_aggregate(env)
            envs.append(env)
        return envs, actors, actuated_dofs

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)
        self.gym.clear_lines(self.viewer)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        if self.actuated_dofs.size(-1) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        actuated_dof = []
        dof_cnts = 0
        action_lower, action_upper = [], []
        action_scale = []
        for i, dofs in zip(range(self.gym.get_actor_count(self.envs[0])), self.actuated_dofs):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in dofs])
                action_upper.extend([dof_prop["effort"][j] for j in dofs])
                action_scale.extend([1]*len(dofs))
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_scale.extend([2]*len(dofs))
            for j in dofs:
                actuated_dof.append(dof_cnts+j)
            dof_cnts += len(dof_prop)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        self.actuated_dofs = torch.tensor(actuated_dof, dtype=torch.int64, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dofs] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

        self.i = 0

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)
        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dofs] = ref_joint_tensor
        self.root_updated_actors.append(self.actor_ids[env_ids].flatten())
        self.dof_updated_actors.append(self.actor_ids_having_dofs[env_ids].flatten())
        self.lifetime[env_ids] = 0

    def do_simulation(self):
        # root tensor inside isaacgym would be overwritten
        # when set_actor_root_state_tensor is called multiple times before doing simulation
        if self.root_updated_actors:
            actor_ids = torch.unique(torch.cat(self.root_updated_actors))
            if actor_ids.numel() == self.actor_ids.numel():
                self.gym.set_actor_root_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor)
                )
            else:
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.root_updated_actors.clear()
        if self.dof_updated_actors:
            actor_ids = torch.unique(torch.cat(self.dof_updated_actors))
            if actor_ids.numel() == self.actor_ids_having_dofs.numel():
                self.gym.set_dof_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor)
                )
            else:
                self.gym.set_dof_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.dof_updated_actors.clear()
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            # print("BEFORE", self.root_tensor[0])
            self.do_simulation()
            self.refresh_tensors()
            # print("AFTER", self.root_tensor[0])
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

            # if self.render_to:
            #     self.gym.write_viewer_image_to_file(self.viewer, "{}/frame{:04d}.png".format(self.render_to, self.i))
            #     self.i += 1
                # if self.i == 1800:
                #     exit()

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        # if self.render_to and overtime.item():
        #     exit()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self):
        if self.episode_length is not None:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = os.path.join("assets", "humanoid.xml")
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None    # All links
    PARENT_LINK = None  # root link


    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)
        self.key_links = parse_kwarg(kwargs, "key_links", self.KEY_LINKS)
        self.parent_link = parse_kwarg(kwargs, "parent_link", self.PARENT_LINK)
        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)

        if contactable_links is None:
            self.contactable_links = None
        else:
            # contact = np.zeros((n_envs, n_links), dtype=bool)
            contact = np.full((n_envs, n_links), 0.2)
            if type(contactable_links) != dict:
                contactable_links = {link: -10000 for link in contactable_links}
            for link, h in contactable_links.items():
                lids = []
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid >= 0:
                        contact[:, lid] = h
                        lids.append(lid)
                if not lids: print("[Warning] Unrecognized contactable link {}".format(link))
            self.contactable_links = torch.tensor(contact, dtype=torch.float32).to(self.contact_force_tensor.device)

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight)) 

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            if config.key_links is None:
                key_links = None
            else:
                key_links = []
                for link in config.key_links:
                    for actor in self.actors:
                        lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                        if lid != -1:
                            key_links.append(lid)
                            break
                    assert lid != -1, "Unrecognized key link {}".format(link)
                key_links = sorted(key_links)
            if config.parent_link is None:
                parent_link = None
            else:
                for j in self.actors:
                    parent_link = self.gym.find_actor_rigid_body_handle(self.envs[0], j, config.parent_link)
                    if parent_link != -1: break
                assert parent_link != -1, "Unrecognized parent link {}".format(parent_link)
            assert key_links is None or all(lid >= 0 for lid in key_links)
            assert parent_link is None or parent_link >= 0
            config.parent_link = parent_link
            config.key_links = key_links
            
            if config.motion_file is None:
                config.motion_file = motion_file
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            config.id = i
            config.name = id
            self.discriminators[id] = config
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.zeros((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        if self.reward_weights is None:
            self.reward_weights = torch.ones((n_envs, 1), dtype=torch.float32, device=self.device)
        elif self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
            
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.goal_dim = self.GOAL_DIM
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
            self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}

        self.ref_motion, self.root_links = self.build_motion_lib(motion_file)
        self.sampling_workers = []
        self.real_samples = []

    def build_motion_lib(self, motion_file):
        ref_motion = ReferenceMotion(motion_file=motion_file, character_model=self.character_model, device=self.device)
        root_links = [i for i, p in enumerate(ref_motion.skeleton.parents) if p == -1]
        return ref_motion, root_links
    
    def __del__(self):
        if hasattr(self, "sampling_workers"):
            for p in self.sampling_workers:
                p.terminate()
            for p in self.sampling_workers:
                p.join()
        super().__del__()

    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)

        contacted = torch.any(self.char_contact_force_tensor.abs() > 1., dim=-1)      # N x n_links

        ground_height = self.ground_height(self.char_root_tensor[:, :3])
        if ground_height is not None:
            low_threshold = (ground_height+self.contactable_links).unsqueeze_(1)
        else:
            low_threshold = self.contactable_links                        # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < low_threshold    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        ground_height = self.ground_height(ref_link_tensor[:, self.root_links[0]], env_ids)
        if ground_height is not None:
            ref_link_tensor[:, :, 2] += ground_height.unsqueeze_(1)

        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = sum([self.gym.get_actor_dof_count(self.envs[0], actor) for actor in self.actors])
        n_links = sum([self.gym.get_actor_rigid_body_count(self.envs[0], actor) for actor in self.actors])
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]
    
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), n_links*13),
            dtype=self.root_tensor.dtype, device=self.device)

        if self.key_links is None:
            self.key_links = None
        else:
            key_links = []
            for link in self.key_links:
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid != -1:
                        key_links.append(lid)
                        break
                assert lid != -1, "Unrecognized key link {}".format(link)
            self.key_links = key_links
        if self.parent_link is None:
            self.parent_link = None
        else:
            for actor in self.actors:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, self.parent_link)
                if lid != -1:
                    parent_link = lid
                    break
            assert lid != -1, "Unrecognized parent link {}".format(self.parent_link)
            self.parent_link = parent_link
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.char_link_tensor[env_ids].view(n_envs, -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            ground_height = self.ground_height(self.state_hist[-1, :, :3])
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)
        else:
            ground_height = self.ground_height(self.state_hist[-1, env_ids, :3], env_ids)
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)

    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
            return res
        else:
            # real
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_iccgan(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        if not self.real_samples:
            if not self.sampling_workers:
                self.disc_ref_motion = {}
                import torch.multiprocessing as mp
                mp.set_start_method("spawn")
                manager = mp.Manager()
                seed = np.random.get_state()[1][0]
                for n, config in self.discriminators.items():
                    q = manager.Queue(maxsize=1)
                    self.disc_ref_motion[n] = q
                    key_links = None if config.key_links is None else config.key_links
                    if key_links is None:  # all links are key links and observable
                        parent_link_index = config.parent_link
                        key_links_index = None
                    elif config.parent_link is None: # parent link is the root, ensure it appears as the first in the key link list
                        parent_link_index = None
                        if 0 in key_links:
                            key_links = [0] + [_ for _ in key_links if _ != 0] # root link is the first key links
                            key_links_index = None # all links in the key link list are key links for observation
                        else:
                            key_links = [0] + key_links # the root link in the key link list but not for observation
                            key_links_index = list(range(1, len(key_links)+1))
                    else:
                        if config.parent_link in key_links:
                            key_links_index = None
                        else:
                            key_links_index = list(range(1, len(key_links)+1))
                            key_links = [config.parent_link] + key_links
                        parent_link_index = key_links.index(config.parent_link)
                    p = mp.Process(target=self.__class__.ref_motion_sample, args=(q,
                        seed+1+config.id, self.step_time, len(self.envs), config.ob_horizon, key_links_index, parent_link_index, config.local_pos, config.replay_speed,
                        dict(motion_file=config.motion_file, character_model=self.character_model,
                            key_links=key_links, device=self.device
                        )
                    ))
                    p.start()
                    self.sampling_workers.append(p)

            self.real_samples = [{n: None for n in self.disc_ref_motion.keys()} for _ in range(128)]
            for n, q in self.disc_ref_motion.items():
                for i, v in enumerate(q.get()):
                    self.real_samples[i][n] = v.to(self.device)
        return self.real_samples.pop()

    @staticmethod
    def ref_motion_sample(queue, seed, step_time, n_inst, ob_horizon, key_links, parent_link, local_pos, replay_speed, kwargs):
        np.random.seed(seed)
        torch.set_num_threads(1)
        lib = ReferenceMotion(**kwargs)
        if replay_speed is not None:
            replay_speed = eval(replay_speed)
        while True:
            obs = []
            for _ in range(128):
                if replay_speed is None:
                    dt = step_time
                else:
                    dt = step_time * replay_speed(n_inst)
                motion_ids, motion_times0 = lib.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
                link_tensor = lib.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = link_tensor.view(ob_horizon, n_inst, -1)
                ob = observe_iccgan(samples, None, key_links, parent_link, include_velocity=False, local_pos=local_pos)
                obs.append(ob.cpu())
            queue.put(obs)


@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: Optional[torch.Tensor]=None,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_velocity: bool=True, local_pos: Optional[bool]=None, ground_height:Optional[torch.Tensor]=None
):
    # state_hist: L x N x (1+N_links) x 13

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist.view(n_hist, n_inst, -1, 13)
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        if local_pos is True:
            origin = root_tensor[:,:, :3]          # L x N x 3
            orient = root_tensor[:,:,3:7]          # L x N x 4
        else:
            origin = root_tensor[-1,:, :3]          # N x 3
            orient = root_tensor[-1,:,3:7]          # N x 4

        heading = heading_zup(orient)               # (L x) N
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1                    # (L x) N x 3
        orient_inv = axang2quat(up_dir, -heading)   # (L x) N x 4
        orient_inv = orient_inv.view(-1, n_inst, 1, 4)   # L x N x 1 x 4 or 1 x N x 1 x 4

        origin = origin.clone()
        if ground_height is None:
            origin[..., UP_AXIS] = 0                # (L x) N x 3
        else:
            origin[..., UP_AXIS] = ground_height    # (L x) N x 3
        origin.unsqueeze_(-2)                       # (L x) N x 1 x 3
    else:
        if local_pos is True or local_pos is None:
            origin = link_tensor[:,:, parent_link, :3]  # L x N x 3
            orient = link_tensor[:,:, parent_link,3:7]  # L x N x 4
        else:
            origin = link_tensor[-1,:, parent_link, :3]  # N x 3
            orient = link_tensor[-1,:, parent_link,3:7]  # N x 4
        orient_inv = quatconj(orient)               # L x N x 4
        orient_inv = orient.view(-1, n_inst, 1, 4)  # L x N x 1 x 4 or 1 x N x 1 x 4
        origin = origin.unsqueeze(-2)               # (L x) N x 1 x 3

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4

    if include_velocity:
        if key_links is None:
            link_lin_vel, link_ang_vel = link_tensor[...,7:10], link_tensor[...,10:13]
        else:
            link_lin_vel, link_ang_vel = link_tensor[:,:,key_links,7:10], link_tensor[:,:,key_links,10:13]
        ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)         # L x N x n_links x 3
        ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)         # L x N x n_links x 3
        ob = torch.cat((ob_link_pos, ob_link_orient,
            ob_link_lin_vel, ob_link_ang_vel), -1)                      # L x N x n_links x 13
    else:
        ob = torch.cat((ob_link_pos, ob_link_orient), -1)               # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7 or 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7 or 13)
    if seq_len is None: return ob1

    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2


class TrackingHumanoid(ICCGANHumanoid):
    OB_HORIZON = 1
    CAMERA_POS= 0, 4.5, 2.0
    GOAL_REWARD_WEIGHT = 1

    def __init__(self, *args, **kwargs):
        self.test1 = kwargs["test1"] if "test1" in kwargs else False
        self.test2 = kwargs["test2"] if "test2" in kwargs else False

        self.random_init = parse_kwarg(kwargs, "random_init", False)
        self.loop_motion = parse_kwarg(kwargs, "loop_motion", False)
        kwargs["ob_horizon"] = 1

        self.key_link_weights_orient = kwargs["key_link_weights_orient"]
        self.key_link_weights_pos = kwargs["key_link_weights_pos"]
        self.key_link_weights_pos_related = kwargs["key_link_weights_pos_related"]
        self.key_link_weights_acc_penalty = kwargs["key_link_weights_acc_penalty"]
        
        assert all([_ >= 0 for _ in self.key_link_weights_orient.values()])
        assert all([_ >= 0 for _ in self.key_link_weights_pos.values()])
        assert all([_ >= 0 for _ in self.key_link_weights_pos_related.values()])
        assert all([_ >= 0 for _ in self.key_link_weights_acc_penalty.values()])

        self.sampling_importance = parse_kwarg(kwargs, "sampling_importance", None)

        super().__init__(*args, **kwargs)

        self.link_state_hist = self.state_hist.view(2, len(self.envs), -1, 13)

    def setup_sim_params(self):
        return super().setup_sim_params(dict(
            contact_offset = 0.0001,
            #bounce_threshold_velocity = 10000,
            #max_depenetration_velocity = 0.1
        ))

    def create_tensors(self):
        super().create_tensors()
        self.tracking_motion_ids = np.zeros((len(self.envs)), dtype=int)
        self.tracking_motion_times = np.zeros((len(self.envs)), dtype=float)
        self.tracking_motion_nframes = torch.zeros((len(self.envs)), dtype=torch.long, device=self.device)

        # the first actor is the humanoid character and the others are the objects
        n_actors = self.gym.get_actor_count(self.envs[0])
        self.n_char_links = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        n_objects = n_actors - 1
        self.multi_objects = n_objects > 2
        self.has_objects = n_objects > 1

        links, n_links = dict(), 0
        for i in range(n_actors):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            for n, id in rigid_body.items():
                handle = id + n_links
                if n in links:
                    links[n].append(handle)
                else:
                    links[n] = [handle]
            n_links += len(rigid_body)

        def convert_weights(weights):
            ww = np.zeros(n_links)
            nw = 0
            for n, w in weights.items():
                if n == "object":
                    nw = w
                else:
                    ww[links[n]] = w
            if self.has_objects:
                ww[self.n_char_links] = nw
                ww /= np.sum(ww)
                ww[self.n_char_links+1:]= ww[self.n_char_links]
            else:
                ww /= np.sum(ww)
            ww = torch.tensor(np.nan_to_num(ww), dtype=torch.float)
            return ww
        self.key_link_weights_orient = convert_weights(self.key_link_weights_orient).to(self.device)
        self.key_link_weights_pos = convert_weights(self.key_link_weights_pos).to(self.device)
        self.key_link_weights_pos_related = convert_weights(self.key_link_weights_pos_related)[:self.n_char_links].to(self.device) # remove objects
        self.activated_obj_mask = torch.ones((len(self.envs), n_links), dtype=torch.bool, device=self.device)

        key_acc_links = sum([links[k] for k, v in self.key_link_weights_acc_penalty.items()], [])
        key_link_weights_acc_penalty = sum([[v]*len(links[k]) for k, v in self.key_link_weights_acc_penalty.items()], [])    
        self.key_acc_links = sorted(key_acc_links)
        key_link_weights_acc_penalty = np.array([v for _, v in sorted(zip(key_acc_links, key_link_weights_acc_penalty))], dtype=float)
        key_link_weights_acc_penalty /= np.sum(key_link_weights_acc_penalty)
        self.key_link_weights_acc_penalty = torch.tensor(key_link_weights_acc_penalty, dtype=torch.float, device=self.device)
        
        self.left_finger_links = sorted(sum([v for n, v in links.items() if "L_" in n and ("Thumb" in n or "Index" in n or "Middle" in n or "Ring" in n or "Pinky" in n)], []))
        self.left_wrist_links = sorted(sum([v for n, v in links.items() if "L_" in n and "Wrist" in n], []))
        self.right_finger_links = sorted(sum([v for n, v in links.items() if "R_" in n and ("Thumb" in n or "Index" in n or "Middle" in n or "Ring" in n or "Pinky" in n)], []))
        self.right_wrist_links = sorted(sum([v for n, v in links.items() if "R_" in n and "Wrist" in n], []))
        self.n_envs_range = np.arange(len(self.envs))


        if self.sampling_importance is None:
            p = 1./(1+n_objects*4)
            self.sampling_importance = [p] + [4*p]*n_objects
        else:
            assert len(self.sampling_importance) == n_actors, "{} vs {}".format(len(self.sampling_importance), n_actors)
        t = sum(self.sampling_importance)
        dist = [int(np.round(len(self.envs)*p/t)) for p in self.sampling_importance]
        dist[0] += len(self.envs)-sum(dist)
        dist = np.cumsum(dist)
        self.sampling_phase = np.full((len(self.envs),), n_objects, dtype=int)
        print(dist, n_actors)
        for i, n in enumerate(reversed(dist[:-1]), 1):
            print(n, n_objects - i)
            self.sampling_phase[:n] = n_objects - i
        print(np.unique(self.sampling_phase, return_counts=True))

        # dummy reference motion for observe function calling during __init__
        class DummyRefMotion():
            def __init__(self, n_links, n_joints, device):
                self.n_links = n_links
                self.n_joints = n_joints
                self.device = device
            def state(self, _, __, with_joint_tensor=False):
                link_state = torch.empty((_.shape[0], self.n_links, 13), device=self.device)
                if with_joint_tensor:
                    return link_state, torch.empty((_.shape[0], self.n_joints, 2), device=self.device)
                return link_state, torch.full((_.shape[0],), -1, dtype=torch.int, device=self.device)
        self.ref_motion = DummyRefMotion(self.link_tensor.size(1), self.joint_tensor.size(1), self.device)

    def init_state(self, env_ids):
        env_ids = env_ids.cpu().numpy()
        motion_ids, motion_times = self.ref_motion.sample(self.sampling_phase[env_ids])
        if not self.training or not self.random_init:
            motion_times[:] = 0 
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)

        self.tracking_motion_times[env_ids] = motion_times
        self.tracking_motion_ids[env_ids] = motion_ids
        tracking_motion_nframes = torch.tensor(
            (self.ref_motion.motion_length[motion_ids]-motion_times)//self.step_time,
            dtype=torch.long, device=self.device)
        if self.training and self.random_init and self.episode_length is not None:
            tracking_motion_nframes.clip_(max=self.episode_length)
        self.tracking_motion_nframes[env_ids] = tracking_motion_nframes
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor
    
    def _observe(self, env_ids):
        if env_ids is None:
            link_tensor = self.link_tensor
            n_envs = link_tensor.size(0)
            motion_ids = self.tracking_motion_ids
            motion_times = self.tracking_motion_times
            target_link_tensor, activated_obj = self.ref_motion.state(motion_ids, motion_times, with_joint_tensor=False)
            self.tracking_motion_times += self.step_time
            self.target_link_tensor = target_link_tensor

            if self.multi_objects:
                self.activated_obj_mask[:, self.n_char_links:] = False
                self.activated_obj_mask[(self.n_envs_range, activated_obj)] = True
                self.target_obj = activated_obj 
                activated_obj_mask = self.activated_obj_mask.unsqueeze(-1)
        else:
            link_tensor = self.link_tensor[env_ids]
            n_envs = link_tensor.size(0)
            env_ids_ = env_ids.cpu().numpy()
            motion_ids = self.tracking_motion_ids[env_ids_]
            motion_times = self.tracking_motion_times[env_ids_]
            target_link_tensor, activated_obj = self.ref_motion.state(motion_ids, motion_times, with_joint_tensor=False)
            self.tracking_motion_times[env_ids_] += self.step_time
            self.target_link_tensor[env_ids] = target_link_tensor

            if self.multi_objects:
                self.activated_obj_mask[env_ids, self.n_char_links:] = False
                self.activated_obj_mask[(env_ids, activated_obj)] = True
                self.target_obj[env_ids_] = activated_obj 
                activated_obj_mask = self.activated_obj_mask[env_ids].unsqueeze(-1)

        orient = link_tensor[..., :1, 3:7]
        origin = link_tensor[:, :1, :3].clone()
        origin[..., 2] = 0
        heading = heading_zup(orient)
        up_dir = torch.zeros_like(origin)
        up_dir[..., 2] = 1
        orient_inv = axang2quat(up_dir, -heading)

        p = rotatepoint(orient_inv, link_tensor[..., :3] - origin)
        q = quatmultiply(orient_inv, link_tensor[..., 3:7])
        v1 = rotatepoint(orient_inv, link_tensor[..., 7:10])
        v2 = rotatepoint(orient_inv, link_tensor[..., 10:13])
        p_ = rotatepoint(orient_inv, target_link_tensor[..., :3] - origin)
        q_ = quatmultiply(orient_inv, target_link_tensor[..., 3:7])
        if self.multi_objects:
            p_.mul_(activated_obj_mask)
            q_.mul_(activated_obj_mask)
        return torch.cat((p.view(n_envs, -1), q.view(n_envs, -1), v1.view(n_envs, -1), v2.view(n_envs, -1), p_.view(n_envs, -1), q_.view(n_envs, -1)), -1)

    def reward(self):
        target_orient = self.target_link_tensor[:,:,3:7]
        target_pos = self.target_link_tensor[:,:,:3]
        link_orient = self.link_tensor[:,:,3:7]
        link_pos = self.link_tensor[:,:,:3]

        _, e_a = quat2axang(quatmultiply(link_orient, quatconj(target_orient)))
        e_p = (link_pos - target_pos).square_().sum(-1)

        eo = (e_a.square_() * self.key_link_weights_orient*self.activated_obj_mask).sum(-1)
        ep = (e_p * self.key_link_weights_pos*self.activated_obj_mask).sum(-1)

        dp = link_pos[:, self.left_finger_links] - link_pos[:, self.left_wrist_links]
        dp = rotatepoint(quatconj(link_orient[:, self.left_wrist_links]), dp)
        dp_ = target_pos[:, self.left_finger_links] - target_pos[:, self.left_wrist_links]
        dp_ = rotatepoint(quatconj(target_orient[:, self.left_wrist_links]), dp_)
        ewl = (dp - dp_).square_().sum(-1).mean(-1)
        dp = link_pos[:, self.right_finger_links] - link_pos[:, self.right_wrist_links]
        dp = rotatepoint(quatconj(link_orient[:, self.right_wrist_links]), dp)
        dp_ = target_pos[:, self.right_finger_links] - target_pos[:, self.right_wrist_links]
        dp_ = rotatepoint(quatconj(target_orient[:, self.right_wrist_links]), dp_)
        ewr = (dp - dp_).square_().sum(-1).mean(-1)
        e_hand = ewl.add_(ewr).mul_(0.5)

        if self.has_objects:
            if self.multi_objects:
                obj = (self.n_envs_range, self.target_obj)
            else:
                obj = (self.n_envs_range, -1)
            dp = link_pos[:, :self.n_char_links] - link_pos[obj].unsqueeze(1)
            dp = rotatepoint(quatconj(link_orient[obj].unsqueeze(1)), dp)
            dp_ = target_pos[:, :self.n_char_links] - target_pos[obj].unsqueeze(1)
            dp_ = rotatepoint(quatconj(target_orient[obj].unsqueeze(1)), dp_)
            eop = ((dp - dp_).square_().sum(-1) * self.key_link_weights_pos_related).sum(-1)

            # this metric may be invalid for some extreme cases where the left and right hands are quite far from each other (e.g. 1.5m)
            dist_threshold_close, dist_threshold_far = 0.5, 1
            dist2obj = (dp_.square_().sum(-1) * self.key_link_weights_pos_related).sum(-1)
            self.close_to_obj = dist2obj < dist_threshold_close

            frac = ((dist2obj-dist_threshold_close)/(dist_threshold_far-dist_threshold_close)).clip_(min=0, max=1)
            e_hand.mul_(frac).add_(eop.mul_(1-frac))
        e_hand.sqrt_()
        # if self.test2:
        #     rew = 0.4*torch.exp(-5*eo) + 0.4*torch.exp(-10*ep) + 0.2*torch.exp(-5*eop)
        # else:
        r_eo  = torch.exp(-15*eo)
        r_ep  = torch.exp(-15*ep)
        r_eh = torch.exp(-5*e_hand)
        rew = 0.4*r_eo + 0.4*r_ep + 0.2*r_eh

        self.e_hand = e_hand
        if self.simulation_step and not self.test1:
            v1 = link_pos[:, self.key_acc_links] - self.link_state_hist[-1, :, self.key_acc_links, :3]
            v2 = self.link_state_hist[-1, :, self.key_acc_links, :3] - self.link_state_hist[-2, :, self.key_acc_links, :3]
            a = (v1-v2).square_().sum(-1).mul_(self.key_link_weights_acc_penalty).sum(-1)
            r_acc = torch.exp(a.mul_(-self.fps*self.fps).mul_(self.lifetime>2))
            rew += 0.05*r_acc

        self.r_eo = r_eo
        self.r_ep = r_ep

        if not self.training:
            print(self.lifetime.item(), rew.item(), r_eo.item(), r_ep.item(), r_eh.item(), r_acc.item(), "Pose Error:", e_p.sqrt_().mean().item())
        # print(self.link_pos[0, self.n_char_links, 2].item(), self.link_pos[0, self.n_char_links+1, 2].item())
        return rew.unsqueeze_(-1)

    def overtime_check(self):
        if self.loop_motion:
            return super().overtime_check()
        return self.lifetime > self.tracking_motion_nframes
    
    def termination_check(self):
        term = super().termination_check()
        if self.has_objects:
            ep = (self.link_tensor[:, self.n_char_links:, :3]-self.target_link_tensor[:, self.n_char_links:, :3]).square_().sum(-1)
            term_obj = torch.any(ep > 0.5, -1).logical_or_((self.e_hand > 0.5).logical_and_(self.close_to_obj)).logical_or_(self.r_eo < 0.001).logical_or_(self.r_ep < 0.1)
            #print(term, term_obj)
            return term.logical_or_(term_obj)
        else:
            return term
    
    def pose(self, env_id=0):
        # the first actor is the humanoid actor
        pose = {"base": self.root_tensor[env_id, 0].cpu().tolist()[:7]}
        env = self.envs[env_id]
        dof = self.gym.get_actor_dof_dict(env, 0)
        joint_pos = self.joint_pos[env_id]
        obj = {}
        n_links = self.gym.get_actor_rigid_body_count(env, 0)
        for i in range(1, self.gym.get_actor_count(env)):
            for n, idx in self.gym.get_actor_rigid_body_dict(env, i).items():
                obj[n] = idx + n_links
            n_links += self.gym.get_actor_rigid_body_count(env, i)
        for n in dof.keys():
            if "_x" not in n: continue
            n = n[:-2]
            idx = min([dof[n+"_x"], dof[n+"_y"], dof[n+"_z"]])
            q = expmap2quat(joint_pos[idx:idx+3])
            pose[n] = q.cpu().tolist()
        return dict(pose=pose, obj={n: self.link_tensor[env_id, i].cpu().tolist()[:7] for n, i in obj.items()})


class ICCGANHumanoidDemo(ICCGANHumanoid):
    OB_HORIZON = 1
    CAMERA_POS= 0, 4.5, 2.0

    def __init__(self, *args, **kwargs):
        self.controllable = True
        if not self.controllable:
            if len(args) > 4:
                args = [_ for _ in args]
                args[4] = "free"
            else:
                kwargs["control_mode"] = "free"
        self.blender = []
        super().__init__(*args, **kwargs)

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=bool, device=self.device)
    
    def create_envs(self, n):
        return super().create_envs(n,
            asset_options=dict(
                fix_base_link = True,
                disable_gravity = True
        ))

    def reset(self):
        super().reset()
        self.motion_ids = torch.zeros(len(self.envs), dtype=torch.int32, device=self.device)
        self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device) - self.step_time

    def apply_actions(self, actions):
        env_ids = torch.arange(len(self.envs), dtype=torch.int64)
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        # remove velocity
        ref_root_tensor[..., 7:12] = 0
        ref_joint_tensor[..., 1] = 0
        ref_link_tensor[..., 7:12] = 0

        self.root_tensor[env_ids] = ref_root_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor
        self.link_tensor[env_ids] = ref_link_tensor

        self.gym.set_actor_root_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor)
        )
        self.gym.set_dof_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor.view(self.joint_tensor.size(0), -1))
        )

    def init_state(self, env_ids):
        self.motion_times[env_ids] += self.step_time
        self.motion_times[env_ids].clip_(min=0)
        motion_ids = self.motion_ids[env_ids].cpu().numpy()
        motion_times = self.motion_times[env_ids].cpu().numpy()
        motion_times[motion_times > self.ref_motion.motion_length[motion_ids]] = 0
        self.motion_times = torch.from_numpy(motion_times)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor


    def pose(self, env_id=0):
        # the first actor is the humanoid actor
        pose = {"base": self.root_tensor[env_id, 0].cpu().tolist()[:7]}
        env = self.envs[env_id]
        dof = self.gym.get_actor_dof_dict(env, 0)
        joint_pos = self.joint_pos[env_id]
        obj = {}
        n_links = self.gym.get_actor_rigid_body_count(env, 0)
        for i in range(1, self.gym.get_actor_count(env)):
            for n, idx in self.gym.get_actor_rigid_body_dict(env, i).items():
                obj[n] = idx + n_links
            n_links += self.gym.get_actor_rigid_body_count(env, i)
        for n in dof.keys():
            if "_x" not in n: continue
            n = n[:-2]
            idx = min([dof[n+"_x"], dof[n+"_y"], dof[n+"_z"]])
            q = expmap2quat(joint_pos[idx:idx+3])
            pose[n] = q.cpu().tolist()
        return dict(pose=pose, obj={n: self.link_tensor[env_id, i].cpu().tolist()[:7] for n, i in obj.items()})
