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
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.02.12

# modify from human_body_prior
from typing import Dict, List

from scipy.ndimage import gaussian_filter1d

can_display = True

try:
    from body_visualizer.mesh.psbody_mesh_cube import points_to_cubes
    from body_visualizer.mesh.psbody_mesh_sphere import points_to_spheres
    from body_visualizer.tools.mesh_tools import rotateXYZ
    from body_visualizer.tools.vis_tools import colors
    from psbody.mesh import Mesh, MeshViewers

except Exception as e:
    print(e)
    print("psbody.mesh based visualization could not be started. skipping ...")
    can_display = False

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from colour import Color
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import flatten_list, log2file
from torch import nn


def visualize(points, bm_f, mvs, kpts_colors, verbosity=2, logger=None):
    from human_body_prior.tools.omni_tools import log2file

    if logger is None:
        logger = log2file()

    def view(opt_objs, body_v, virtual_markers, opt_it):
        if verbosity <= 0:
            return
        opt_objs_cpu = {k: c2c(v) for k, v in opt_objs.items()}

        total_loss = np.sum([np.sum(v) for k, v in opt_objs_cpu.items()])
        message = "it {} -- [total loss = {:.2e}] - {}".format(
            opt_it,
            total_loss,
            " | ".join(
                ["%s = %2.2e" % (k, np.sum(v)) for k, v in opt_objs_cpu.items()]
            ),
        )
        logger(message)
        if verbosity > 1 and can_display:
            bs = body_v.shape[0]
            np.random.seed(100)
            frame_ids = (
                list(range(bs))
                if bs <= len(mvs)
                else np.random.choice(bs, size=len(mvs), replace=False).tolist()
            )
            if bs > len(mvs):
                message += " -- [frame_ids: {}]".format(frame_ids)
            for dispId, fId in enumerate(
                frame_ids
            ):  # check for the number of frames in mvs and show a randomly picked number of frames in body if there is more to show than row*cols available
                new_body_v = rotateXYZ(body_v[fId], [-90, 0, 0])

                orig_mrk_mesh = points_to_spheres(
                    rotateXYZ(c2c(points[fId]), [-90, 0, 0]),
                    radius=0.01,
                    point_color=kpts_colors,
                )
                virtual_markers_mesh = points_to_cubes(
                    rotateXYZ(virtual_markers[fId], [-90, 0, 0]),
                    radius=0.01,
                    point_color=kpts_colors,
                )
                new_body_mesh = Mesh(new_body_v, bm_f, vc=colors["grey"])

                # linev = rotateXYZ(np.hstack((c2c(points[fId]), virtual_markers[fId])).reshape((-1, 3)), [-90,0,0])
                # linee = np.arange(len(linev)).reshape((-1, 2))
                # ll = Lines(v=linev, e=linee)
                # ll.vc = (ll.v * 0. + 1) * np.array([0.00, 0.00, 1.00])
                # mvs[dispId].set_dynamic_lines([ll])

                # orig_mrk_mesh = points_to_spheres(data_pc, radius=0.01, vc=colors['blue'])
                mvs[dispId].set_dynamic_meshes([orig_mrk_mesh, virtual_markers_mesh])
                mvs[dispId].set_static_meshes([new_body_mesh])

            mvs[0].set_titlebar(message)
            # if out_dir is not None: mv.save_snapshot(os.path.join(out_dir, '%05d_it_%.5d.png' %(frame_id, opt_it)))

    return view


class AdamInClosure:
    def __init__(self, var_lr, max_iter=100, tolerance_change=1e-5):
        self.optimizer = torch.optim.Adam(var_lr)
        self.max_iter = max_iter
        self.tolerance_change = tolerance_change

    def step(self, closure):
        prev_loss = None
        for it in range(self.max_iter):
            loss = closure()
            self.optimizer.step()
            if prev_loss is None:
                prev_loss = loss
                continue
            if torch.isnan(loss):
                # breakpoint()
                break
            # if abs(loss - prev_loss) < self.tolerance_change:
            #     print('abs(loss - prev_loss) <  self.tolerance_change')
            #     break

    def zero_grad(self):
        self.optimizer.zero_grad()


def ik_fit(
    optimizer,
    source_kpts_model,
    static_vars,
    extra_params={},
    on_step=None,
    gstep=0,
    loss_mask=None,
    vars_mask=None,
    apply_smooth=True,
    initial_pose_body=None,
):
    data_loss = extra_params.get(
        "data_loss",
        torch.nn.SmoothL1Loss(reduction="mean"),
    )

    def fit(weights, free_vars):
        fit.gstep += 1
        optimizer.zero_grad()

        # free_vars['pose_body'] = vp_model.decode(free_vars['poZ_body'])['pose_body'].contiguous().view(-1, 63)
        # nonan_mask = torch.isnan(free_vars['poZ_body']).sum(-1) == 0

        opt_objs = {}

        if "pose_upper" in free_vars:
            free_vars["pose_body"] = torch.cat(
                [
                    free_vars["pose_lower"][:, :2],
                    free_vars["pose_upper"][:, 0:1],
                    free_vars["pose_lower"][:, 2:4],
                    free_vars["pose_upper"][:, 1:2],
                    free_vars["pose_lower"][:, 4:6],
                    free_vars["pose_upper"][:, 2:3],
                    free_vars["pose_lower"][:, 6:9],
                    free_vars["pose_upper"][:, 3:],
                    free_vars["pose_lower"][:, 9:],
                    free_vars["pose_arm"][:],
                ],
                dim=1,
            ).reshape(free_vars["pose_lower"].shape[0], -1)
        res = source_kpts_model(free_vars)

        if loss_mask is None:
            opt_objs["data"] = data_loss(res["source_kpts"], static_vars["target_kpts"])
        else:
            opt_objs["data"] = (
                data_loss(
                    res["source_kpts"] * loss_mask,
                    static_vars["target_kpts"] * loss_mask,
                )
                * 0.003
            )

        # add temporal smoothness
        temporal_loss = (
            torch.nn.MSELoss(reduction="mean")(
                res["source_kpts"][1:], res["source_kpts"][:-1]
            )
            * 0.0001
        )

        opt_objs["data"] += temporal_loss

        pose = free_vars["pose_body"].detach().cpu().numpy()
        pose = np.apply_along_axis(lambda x: gaussian_filter1d(x, sigma=2), 0, pose)
        free_vars["pose_body"] = torch.tensor(
            pose,
            device=free_vars["pose_body"].device,
            dtype=free_vars["pose_body"].dtype,
            requires_grad=True,
        )

        opt_objs = {
            k: opt_objs[k] * v for k, v in weights.items() if k in opt_objs.keys()
        }
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        # breakpoint()

        loss_total.backward()

        if on_step is not None:
            on_step(opt_objs, c2c(res["body"].v), c2c(res["source_kpts"]), fit.gstep)

        fit.free_vars = {
            k: v for k, v in free_vars.items()
        }  # if k in IK_Engine.fields_to_optimize}
        # fit.nonan_mask = nonan_mask
        fit.final_loss = loss_total

        return loss_total

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    # fit.nonan_mask = None
    return fit


class IK_Engine(nn.Module):
    def __init__(
        self,
        #  vposer_expr_dir: str,
        data_loss,
        optimizer_args: dict = {"type": "ADAM"},
        stepwise_weights: List[Dict] = [{"data": 10.0, "poZ_body": 0.01, "betas": 0.5}],
        display_rc: tuple = (2, 1),
        verbosity: int = 1,
        num_betas: int = 16,
        logger=None,
        right_wrist=False,
        left_wrist=False,
        feet=False,
    ):
        """

        :param vposer_expr_dir: The vposer directory that holds the settings and model snapshot
        :param data_loss: should be a pytorch callable (source, target) that returns the accumulated loss
        :param optimizer_args: arguments for optimizers
        :param stepwise_weights: list of dictionaries. each list element defines weights for one full step of optimization
                                 if a weight value is left out, its respective object item will be removed as well. imagine optimizing without data term!
        :param display_rc: number of row and columns in case verbosity > 1
        :param verbosity: 0: silent, 1: text, 2: text/visual. running 2 over ssh would need extra work
        :param logger: an instance of human_body_prior.tools.omni_tools.log2file
        """

        super(IK_Engine, self).__init__()

        assert isinstance(stepwise_weights, list), ValueError(
            "stepwise_weights should be a list of dictionaries."
        )
        assert np.all(["data" in l for l in stepwise_weights]), ValueError(
            "The term data should be available in every weight of anealed optimization step: {}".format(
                stepwise_weights
            )
        )

        self.data_loss = (
            torch.nn.SmoothL1Loss(reduction="mean") if data_loss is None else data_loss
        )
        self.num_betas = num_betas
        self.stepwise_weights = stepwise_weights
        self.verbosity = verbosity
        self.optimizer_args = optimizer_args

        self.logger = log2file() if logger is None else logger

        if verbosity > 1 and can_display:
            mvs = MeshViewers(display_rc, keepalive=True)
            self.mvs = flatten_list(mvs)
            self.mvs[0].set_background_color(colors["white"])
        else:
            self.mvs = None

        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.feet = feet

        # self.vp_model, _ = load_model(vposer_expr_dir,
        #                               model_code=VPoser,
        #                               remove_words_in_model_weights='vp_model.',
        #                               disable_grad=True)

    def forward(self, source_kpts, target_kpts, initial_body_params={}):
        """
        source_kpts is a function that given body parameters computes source key points that should match target key points
        Try to reconstruct the bps signature by optimizing the body_poZ
        """
        # if self.rt_ps.verbosity > 0: self.logger('Processing {} frames'.format(points.shape[0]))
        bs = target_kpts.shape[0]

        loss_mask = None
        vars_mask = None
        loss_mask = torch.ones_like(target_kpts)
        vars_mask = torch.zeros((21, 3))
        if self.right_wrist and self.left_wrist:
            loss_mask[:, [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]] = 0.0
            vars_mask[[11, 12, 13, 14, 15, 16, 17, 18]] = 1
        elif self.right_wrist:
            loss_mask[:, [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]] = 0.0
            vars_mask[[11, 13, 14, 16, 18]] = 1
        elif self.left_wrist:
            loss_mask[:, [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21]] = 0.0
            vars_mask[[11, 12, 14, 15, 17]] = 1
        else:
            loss_mask = None
            vars_mask = None
        if self.feet:
            if loss_mask is None:
                loss_mask = torch.ones_like(target_kpts)
            loss_mask[:, [1, 2, 4, 5]] = 0.0
        if vars_mask is not None:
            vars_mask = vars_mask.to(target_kpts.device).reshape(1, 63)

        on_step = visualize(
            target_kpts,
            kpts_colors=source_kpts.kpts_colors,
            bm_f=source_kpts.bm_f,
            mvs=self.mvs,
            verbosity=self.verbosity,
            logger=self.logger,
        )

        comp_device = target_kpts.device
        # comp_device = self.vp_model.named_parameters().__next__()[1].device
        if "pose_body" not in initial_body_params:
            initial_body_params["pose_body"] = torch.zeros(
                [bs, 63], device=comp_device, dtype=torch.float, requires_grad=False
            )
        if "trans" not in initial_body_params:
            initial_body_params["trans"] = torch.zeros(
                [bs, 3], device=comp_device, dtype=torch.float, requires_grad=False
            )
        if "betas" not in initial_body_params:
            initial_body_params["betas"] = torch.zeros(
                [bs, self.num_betas],
                device=comp_device,
                dtype=torch.float,
                requires_grad=False,
            )
        if "root_orient" not in initial_body_params:
            initial_body_params["root_orient"] = torch.zeros(
                [bs, 3], device=comp_device, dtype=torch.float, requires_grad=False
            )

        # initial_body_params['poZ_body'] = self.vp_model.encode(initial_body_params['pose_body']).mean

        # free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) for k, v in initial_body_params.items() if
        #              k in ['betas', 'trans', 'poZ_body', 'root_orient', 'pose_body']}
        free_vars = {
            "betas": torch.nn.Parameter(
                initial_body_params["betas"].detach(), requires_grad=False
            ),
            "trans": torch.nn.Parameter(
                initial_body_params["trans"].detach(), requires_grad=False
            ),
            "root_orient": torch.nn.Parameter(
                initial_body_params["root_orient"].detach(), requires_grad=False
            ),
            "pose_body": torch.nn.Parameter(
                initial_body_params["pose_body"].detach(), requires_grad=True
            ),
        }
        static_vars = {
            "target_kpts": target_kpts,
            # 'trans': initial_body_params['trans'].detach(),
            # 'betas': initial_body_params['betas'].detach(),
            # 'poZ_body': initial_body_params['poZ_body'].detach()
        }

        if self.optimizer_args["type"].upper() == "LBFGS":
            optimizer = torch.optim.LBFGS(
                list(free_vars.values()),
                lr=self.optimizer_args.get("lr", 1),
                max_iter=self.optimizer_args.get("max_iter", 100),
                tolerance_change=self.optimizer_args.get("tolerance_change", 1e-5),
                max_eval=self.optimizer_args.get("max_eval", None),
                history_size=self.optimizer_args.get("history_size", 100),
                line_search_fn="strong_wolfe",
            )

        elif self.optimizer_args["type"].upper() == "ADAM":
            pose_body = initial_body_params["pose_body"].detach().reshape(-1, 21, 3)
            if self.left_wrist or self.right_wrist:
                if not self.feet:
                    free_vars = {
                        "betas": torch.nn.Parameter(
                            initial_body_params["betas"].detach(), requires_grad=False
                        ),
                        "trans": torch.nn.Parameter(
                            initial_body_params["trans"].detach(), requires_grad=False
                        ),
                        "root_orient": torch.nn.Parameter(
                            initial_body_params["root_orient"].detach(),
                            requires_grad=False,
                        ),
                        "pose_arm": torch.nn.Parameter(
                            pose_body[:, [15, 16, 17, 18, 19, 20]], requires_grad=True
                        ),
                        "pose_upper": torch.nn.Parameter(
                            pose_body[:, [2, 5, 8, 12, 13]], requires_grad=True
                        ),
                        "pose_lower": torch.nn.Parameter(
                            pose_body[:, [0, 1, 3, 4, 6, 7, 9, 10, 11, 14]],
                            requires_grad=False,
                        ),
                    }
                else:
                    free_vars = {
                        "betas": torch.nn.Parameter(
                            initial_body_params["betas"].detach(), requires_grad=False
                        ),
                        "trans": torch.nn.Parameter(
                            initial_body_params["trans"].detach(), requires_grad=False
                        ),
                        "root_orient": torch.nn.Parameter(
                            initial_body_params["root_orient"].detach(),
                            requires_grad=False,
                        ),
                        "pose_arm": torch.nn.Parameter(
                            pose_body[:, [15, 16, 17, 18, 19, 20]], requires_grad=True
                        ),
                        "pose_upper": torch.nn.Parameter(
                            pose_body[:, [2, 5, 8, 12, 13]], requires_grad=True
                        ),
                        "pose_lower": torch.nn.Parameter(
                            pose_body[:, [0, 1, 3, 4, 6, 7, 9, 10, 11, 14]],
                            requires_grad=True,
                        ),
                    }
            else:
                assert self.feet
                free_vars = {
                    "betas": torch.nn.Parameter(
                        initial_body_params["betas"].detach(), requires_grad=False
                    ),
                    "trans": torch.nn.Parameter(
                        initial_body_params["trans"].detach(), requires_grad=False
                    ),
                    "root_orient": torch.nn.Parameter(
                        initial_body_params["root_orient"].detach(), requires_grad=False
                    ),
                    "pose_arm": torch.nn.Parameter(
                        pose_body[:, [15, 16, 17, 18, 19, 20]], requires_grad=False
                    ),
                    "pose_upper": torch.nn.Parameter(
                        pose_body[:, [2, 5, 8, 12, 13]], requires_grad=False
                    ),
                    "pose_lower": torch.nn.Parameter(
                        pose_body[:, [0, 1, 3, 4, 6, 7, 9, 10, 11, 14]],
                        requires_grad=True,
                    ),
                }

            if "pose_body" in free_vars:
                optimizer = AdamInClosure(
                    list(free_vars.values()),
                    lr=self.optimizer_args.get("lr", 1e-3),
                    max_iter=self.optimizer_args.get("max_iter", 100),
                    tolerance_change=self.optimizer_args.get("tolerance_change", 1e-5),
                )
            else:
                if not self.feet:
                    var_lr = [
                        {"params": free_vars["pose_arm"], "lr": 0.01},
                        {"params": free_vars["pose_upper"], "lr": 0.001},
                    ]
                else:
                    var_lr = [
                        {"params": free_vars["pose_arm"], "lr": 0.01},
                        {"params": free_vars["pose_upper"], "lr": 0.001},
                        {"params": free_vars["pose_lower"], "lr": 0.003},
                    ]
                optimizer = AdamInClosure(
                    var_lr,
                    max_iter=self.optimizer_args.get("max_iter", 100),
                    tolerance_change=self.optimizer_args.get("tolerance_change", 1e-5),
                )
        else:
            raise ValueError("optimizer_type not recognized.")

        if "pose_upper" in free_vars:
            initial_pose_body = torch.cat(
                [
                    free_vars["pose_lower"][:, :2],
                    free_vars["pose_upper"][:, 0:1],
                    free_vars["pose_lower"][:, 2:4],
                    free_vars["pose_upper"][:, 1:2],
                    free_vars["pose_lower"][:, 4:6],
                    free_vars["pose_upper"][:, 2:3],
                    free_vars["pose_lower"][:, 6:9],
                    free_vars["pose_upper"][:, 3:],
                    free_vars["pose_lower"][:, 9:],
                    free_vars["pose_arm"][:],
                ],
                dim=1,
            ).reshape(free_vars["pose_lower"].shape[0], -1)
        else:
            initial_pose_body = free_vars["pose_body"]

        gstep = 0
        closure = ik_fit(
            optimizer,
            source_kpts_model=source_kpts,
            static_vars=static_vars,
            #  vp_model=self.vp_model,
            extra_params={"data_loss": self.data_loss},
            on_step=on_step,
            gstep=gstep,
            loss_mask=loss_mask,
            vars_mask=vars_mask,
            initial_pose_body=initial_pose_body,
        )
        # try:
        for wts in self.stepwise_weights:
            optimizer.step(lambda: closure(wts, free_vars))
            free_vars = closure.free_vars
        # except:
        #
        #     pass

        # if closure.final_loss is None or torch.isnan(closure.final_loss) or torch.any(torch.isnan(free_vars['trans'])):
        #     if self.verbosity > 0:
        #         self.logger('NaN observed in the optimization results. you might want to restart the refinment procedure.')
        #     breakpoint()
        #     return None
        free_vars = closure.free_vars
        if "pose_upper" in free_vars:
            free_vars["pose_body"] = torch.cat(
                [
                    free_vars["pose_lower"][:, :2],
                    free_vars["pose_upper"][:, 0:1],
                    free_vars["pose_lower"][:, 2:4],
                    free_vars["pose_upper"][:, 1:2],
                    free_vars["pose_lower"][:, 4:6],
                    free_vars["pose_upper"][:, 2:3],
                    free_vars["pose_lower"][:, 6:9],
                    free_vars["pose_upper"][:, 3:],
                    free_vars["pose_lower"][:, 9:],
                    free_vars["pose_arm"][:],
                ],
                dim=1,
            ).reshape(free_vars["pose_lower"].shape[0], -1)
        res = source_kpts(free_vars)

        root_orient = free_vars["root_orient"].detach().clone()
        pose_body = free_vars["pose_body"].detach().clone()
        source_kpts = res["source_kpts"].detach().clone()

        return root_orient, pose_body, source_kpts


class SourceKeyPoints(nn.Module):
    def __init__(
        self,
        bm: Union[str, BodyModel],
        n_joints: int = 22,
        kpts_colors: Union[np.ndarray, None] = None,
    ):
        super(SourceKeyPoints, self).__init__()

        self.bm = (
            BodyModel(bm, persistant_buffer=False, num_betas=16)
            if isinstance(bm, str)
            else bm
        )
        self.bm_f = []  # self.bm.f
        self.n_joints = n_joints
        self.kpts_colors = (
            np.array([Color("grey").rgb for _ in range(n_joints)])
            if kpts_colors == None
            else kpts_colors
        )

    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {"source_kpts": new_body.Jtr[:, : self.n_joints], "body": new_body}
