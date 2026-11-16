"""Residual RL: learn corrections to swing-up; retain exact local stabilization."""
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import MuscleHumanoid
from .controller import SwingUpTeacher


class ResidualHumanoid(gym.Wrapper):
    def __init__(self, duration=20., residual_scale=.25):
        super().__init__(MuscleHumanoid(duration))
        self.controller = SwingUpTeacher()
        self.residual_scale = residual_scale
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            (self.env.observation_space.shape[0]+3,), dtype=np.float32)

    def observe(self):
        return np.r_[self.env.observe(), self.controller(self.env)].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        return self.observe(), info

    def step(self, action):
        residual = np.asarray(action, dtype=float)
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise ValueError('Expected three finite residual actions')
        base = self.controller(self.env)
        # Once captured, let the calibrated local controller hold equilibrium.
        scale = 0. if self.env.balance_mode else self.residual_scale
        applied = np.clip(base+scale*np.clip(residual, -1, 1), -1, 1)
        nominal_height = .1+.8*math.cos(.35)
        applied[2] = min(applied[2], max(0., (nominal_height-self.env.target_height)/(.5*self.env.dt)))
        _, reward, done, timeout, info = self.env.step(applied)
        info.update(base_action=base.tolist(), residual_action=residual.tolist(),
                    applied_action=applied.tolist(), residual_scale=scale,
                    task_reward=reward)
        # Constant scaling preserves the task objective while keeping critic
        # targets manageable at gamma=.998 (otherwise values approach 4000).
        return self.observe(), reward*.01, done, timeout, info
