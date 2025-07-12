import pytorch3d.transforms as transforms
import torch

from manip.inertialize.spring import (
    decay_spring_damper_exact_cubic,
)
from manip.lafan1.utils import quat_slerp
from manip.utils.model_utils import wxyz_to_xyzw, xyzw_to_wxyz


def quat_abs(q):
    q[q[..., 0] < 0] *= -1
    return q


def apply_inertialize(
    prev_jpos=None,
    prev_rot_6d=None,
    window_jpos=None,
    window_rot_6d=None,
    ratio=0.5,
    prev_blend_time=0.2,
    window_blend_time=0.2,
    zero_velocity=False,
):
    """
    prev_jpos: BS X T X 24 X 3 / BS X T X 3
    prev_rot_6d: BS X T X 22 X 6 / BS X T X 6
    window_jpos: BS X T X 24 X 3 / BS X T X 3
    window_rot_6d: BS X T X 22 X 6 / BS X T X 6
    ratio: 0 - all prev, 1 - all window
    """

    dt = 1 / 30.0
    ##### pos #####
    if prev_jpos is not None and window_jpos is not None:
        dst_pos = window_jpos[:, 0]
        dst_pos_next = window_jpos[:, 1]
        dst_vel = (dst_pos_next - dst_pos) / dt  # BS X 24 X 3

        src_pos = prev_jpos[:, -1]
        src_pos_prev = prev_jpos[:, -2]
        src_vel = (src_pos - src_pos_prev) / dt  # BS X 24 X 3

        diff_pos = src_pos - dst_pos
        diff_vel = src_vel - dst_vel

        if zero_velocity:
            diff_vel[:] = 0.0

        ##### interpolate #####
        new_jpos = window_jpos.clone()
        new_prev_jpos = prev_jpos.clone()
        for i in range(new_jpos.shape[1]):
            offset = decay_spring_damper_exact_cubic(
                ratio * diff_pos, ratio * diff_vel, window_blend_time, i * dt
            )
            new_jpos[:, i] += offset
        for i in range(prev_jpos.shape[1]):
            offset = decay_spring_damper_exact_cubic(
                (1 - ratio) * -diff_pos,
                (1 - ratio) * diff_vel,
                prev_blend_time,
                (prev_jpos.shape[1] - 1 - i) * dt,
            )
            new_prev_jpos[:, i] += offset
    else:
        new_jpos = None
        new_prev_jpos = None

    ##### rot #####
    if prev_rot_6d is not None and window_rot_6d is not None:
        dst_ori = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(window_rot_6d[:, 0])
        )
        dst_ori_next = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(window_rot_6d[:, 1])
        )
        dst_ang_vel = (
            transforms.quaternion_to_axis_angle(
                quat_abs(
                    transforms.quaternion_multiply(
                        dst_ori_next, transforms.quaternion_invert(dst_ori)
                    )
                )
            )
            / dt
        )

        src_ori = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(prev_rot_6d[:, -1])
        )
        src_ori_prev = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(prev_rot_6d[:, -2])
        )
        src_ang_vel = (
            transforms.quaternion_to_axis_angle(
                quat_abs(
                    transforms.quaternion_multiply(
                        src_ori, transforms.quaternion_invert(src_ori_prev)
                    )
                )
            )
            / dt
        )

        diff_ori = transforms.quaternion_to_axis_angle(
            quat_abs(
                transforms.quaternion_multiply(
                    src_ori, transforms.quaternion_invert(dst_ori)
                )
            )
        )
        diff_ang_vel = src_ang_vel - dst_ang_vel

        if zero_velocity:
            diff_ang_vel[:] = 0.0

        new_rot_6d = window_rot_6d.clone()
        new_prev_rot_6d = prev_rot_6d.clone()

        ##### interpolate #####
        for i in range(new_rot_6d.shape[1]):
            offset = decay_spring_damper_exact_cubic(
                ratio * diff_ori, ratio * diff_ang_vel, window_blend_time, i * dt
            )
            new_rot_6d[:, i] = transforms.matrix_to_rotation_6d(
                transforms.quaternion_to_matrix(
                    transforms.quaternion_multiply(
                        transforms.axis_angle_to_quaternion(offset),
                        transforms.matrix_to_quaternion(
                            transforms.rotation_6d_to_matrix(new_rot_6d[:, i])
                        ),
                    )
                )
            )
        for i in range(new_prev_rot_6d.shape[1]):
            offset = decay_spring_damper_exact_cubic(
                (1 - ratio) * -diff_ori,
                (1 - ratio) * diff_ang_vel,
                prev_blend_time,
                (new_prev_rot_6d.shape[1] - 1 - i) * dt,
            )
            new_prev_rot_6d[:, i] = transforms.matrix_to_rotation_6d(
                transforms.quaternion_to_matrix(
                    transforms.quaternion_multiply(
                        transforms.axis_angle_to_quaternion(offset),
                        transforms.matrix_to_quaternion(
                            transforms.rotation_6d_to_matrix(new_prev_rot_6d[:, i])
                        ),
                    )
                )
            )
    else:
        new_rot_6d = None
        new_prev_rot_6d = None

    return new_jpos, new_rot_6d, new_prev_jpos, new_prev_rot_6d


def apply_linear_offset(
    original_jpos=None,
    original_rot_6d=None,
    new_target_jpos=None,
    new_target_rot_6d=None,
    reversed=False,
):
    """
    original_jpos: BS X T X 3 / BS X T X 24 X 3
    original_rot_6d: BS X T X 6 / BS X T X 22 X 6
    new_target_jpos: BS X 3 / BS X 24 X 3
    new_target_rot_6d: BS X 6 / BS X 22 X 6
    """
    new_jpos, new_rot_6d = None, None
    if reversed:
        if original_jpos is not None:
            original_jpos = torch.flip(original_jpos, dims=[1])
        if original_rot_6d is not None:
            original_rot_6d = torch.flip(original_rot_6d, dims=[1])

    ##### pos #####
    if original_jpos is not None:
        T = original_jpos.shape[1]
        new_jpos = original_jpos.clone()
        pos_offset = new_target_jpos - original_jpos[:, -1]
        for i in range(T):
            new_jpos[:, i] += (i + 1) / T * pos_offset

    ##### rot #####
    if original_rot_6d is not None:
        T = original_rot_6d.shape[1]
        new_target_quat = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(new_target_rot_6d)
        )
        original_quat = transforms.matrix_to_quaternion(
            transforms.rotation_6d_to_matrix(original_rot_6d)
        )
        new_quat = original_quat.clone()
        quat_offset = transforms.quaternion_multiply(
            new_target_quat, transforms.quaternion_invert(original_quat[:, -1])
        )

        # quat_slerp needs xyzw
        quat_offset = wxyz_to_xyzw(quat_offset)
        zero_quat = torch.zeros_like(quat_offset)
        zero_quat[..., -1] = 1
        for i in range(T):
            cur_offset = quat_slerp(zero_quat, quat_offset, (i + 1) / T)
            cur_offset = xyzw_to_wxyz(cur_offset)
            new_quat[:, i] = transforms.quaternion_multiply(
                cur_offset, original_quat[:, i]
            )
        new_rot_6d = transforms.matrix_to_rotation_6d(
            transforms.quaternion_to_matrix(new_quat)
        )

    if reversed:
        if original_jpos is not None:
            new_jpos = torch.flip(new_jpos, dims=[1])
        if original_rot_6d is not None:
            new_rot_6d = torch.flip(new_rot_6d, dims=[1])

    return new_jpos, new_rot_6d
