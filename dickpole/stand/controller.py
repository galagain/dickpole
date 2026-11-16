"""Physical baseline and demonstrator used by the residual RL controller.

The five energy-controller coefficients were searched on real MuJoCo rollouts.
The LQR gain is a finite-difference linearization of the full muscular plant.
Only pelvis target velocities are output. No position/velocity teleportation.
The fixed baseline remains part of inference; PPO supplies ascent corrections.
"""
from pathlib import Path
import math
import numpy as np

REFERENCE = Path(__file__).resolve().parents[2] / 'models' / 'stand' / 'controller.npz'


class SwingUpTeacher:
    def __init__(self, reference=REFERENCE):
        with np.load(reference, allow_pickle=False) as bundle:
            self.parameters = bundle['parameters'].copy()
            self.gain = bundle['K'].copy()
            self.reference = bundle['reference'].copy()

    def __call__(self, env):
        angle, speed = env.pole_state()
        nominal_height = .1+.8*math.cos(.35)
        if env.balance_mode:
            state = np.r_[env.data.qpos, env.data.qvel, env.data.act,
                          env.target_x, env.target_pitch]
            delta = state-self.reference
            delta[0] -= env.support_origin
            joint = env.qids['pole_0_joint']
            delta[joint] = (delta[joint]+math.pi) % (2*math.pi)-math.pi
            return np.r_[np.clip(-self.gain @ delta, -1, 1),
                         np.clip(10*(nominal_height-env.target_height), -1, 1)]
        p = self.parameters
        f, g = math.sin(angle), speed*math.cos(angle)
        action = [p[0]*f+p[1]*g-4*env.target_x-.5*env.data.qvel[0],
                  p[2]*f+p[3]*g-2*env.target_pitch-.1*env.data.qvel[2],
                  -p[4]*(math.cos(angle)+1)+8*(.80-env.target_height)]
        if env.data.time < .2:
            action[0] += .2
            action[1] += .4
        # Pump below the calibrated standing height; keep that height at capture.
        action[2] = min(action[2], max(0., (nominal_height-env.target_height)/(.5*env.dt)))
        return np.clip(action, -1, 1)
