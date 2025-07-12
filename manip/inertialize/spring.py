"""
This file is inspired by https://theorangeduck.com/page/spring-roll-call
"""

import math

import numpy as np
import torch

# Constants
LN2f = 0.69314718
PIf = 3.14159265


# Helper functions
def square(x):
    return x * x


def neg_exp(x):
    return math.exp(-x)


def lerp(a, b, t):
    return a + t * (b - a)


def clamp(value, min, max):
    return torch.clamp(value, min, max)


def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * LN2f) / (halflife + eps)


def damping_to_halflife(damping, eps=1e-5):
    return (4.0 * LN2f) / (damping + eps)


def frequency_to_stiffness(frequency):
    return square(2.0 * PIf * frequency)


def stiffness_to_frequency(stiffness):
    return torch.sqrt(stiffness) / (2.0 * PIf)


def decay_spring_damper_exact(x, v, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j1 = v + x * y
    eydt = neg_exp(y * dt)

    x_new = eydt * (x + j1 * dt)
    return x_new


def decay_spring_damper_exact_cubic(x, v, blendtime, dt, eps=1e-8):
    t = np.clip(dt / (blendtime + eps), 0.0, 1.0)

    d = x
    c = v * blendtime
    b = -3 * d - 2 * c
    a = 2 * d + c

    return a * t * t * t + b * t * t + c * t + d


def inertialize_transition(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x_new = (src_x + off_x) - dst_x
    off_v_new = (src_v + off_v) - dst_v
    return off_x_new, off_v_new


def inertialize_update(out_x, out_v, off_x, off_v, in_x, in_v, halflife, dt):
    off_x_new, off_v_new = decay_spring_damper_exact(off_x, off_v, halflife, dt)
    out_x_new = in_x + off_x_new
    out_v_new = in_v + off_v_new
    return out_x_new, out_v_new
