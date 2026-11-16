"""Episodic policy search and local feedback for the common 3D humanoid."""
import math
from pathlib import Path

import numpy as np

LOW = np.array([.6, .1, 0., 0., 0., .78, .15, .5, .7])
HIGH = np.array([3., 16., 12., 2., 8., .84, .65, 6., 1.8])
INITIAL = np.array([1.0107036890281857,7.702599891239644,4.06099045259859,0.,
                    3.3797835556105875,.8084445999868857,.46586545216449815,
                    1.3408019836673444,1.2542328443542947])
PARAMETERS = ['energy_target', 'pumping_gain', 'position_damping', 'velocity_damping',
              'height_pumping', 'crouch_height', 'capture_angle', 'capture_speed', 'feedback_gain']


class Stabilizer:
    def __init__(self, folder):
        with np.load(Path(folder)/'stabilizer.npz', allow_pickle=False) as bundle:
            for name in ('reference', 'K', 'qids', 'dofs', 'actids'):
                setattr(self, name, bundle[name].copy())
            self.knots = bundle['knots'].copy() if 'knots' in bundle else np.array([0.])
        if self.K.ndim == 1:
            self.K = self.K[None,:]
            self.reference = self.reference[None,:]

    def action(self, env, gain):
        state = np.r_[env.data.qpos[self.qids], env.data.qvel[self.dofs],
                      env.data.act[self.actids], env.target_y, env.target_height]
        if env.lqr_anchor is None:
            env.lqr_anchor = float(env.target_y)
        anchor = np.clip(env.lqr_anchor,self.knots[0],self.knots[-1])
        reference = np.array([np.interp(anchor,self.knots,self.reference[:,i])
                              for i in range(self.reference.shape[1])])
        K = np.array([np.interp(anchor,self.knots,self.K[:,i])
                      for i in range(self.K.shape[1])])
        delta = state-reference
        index = int(np.flatnonzero(self.qids == env.qids['pole_0_joint'])[0])
        delta[index] = (delta[index]+math.pi) % (2*math.pi)-math.pi
        lateral = -gain*float(K @ delta)
        return np.clip([lateral, 20*(.86-env.target_height)], -1, 1)


def balance_action(env, parameters, stabilizer):
    a, b, c, d, h, height, capture, slow, gain = parameters
    angle, speed = env.pole_state()
    error = (angle-math.pi+math.pi) % (2*math.pi)-math.pi
    if abs(error) < capture and abs(speed) < slow and not env.pole_supported():
        env.control_phase = 1
    if abs(error) > .7:
        env.control_phase = 0
        env.lqr_anchor = None
    if env.control_phase:
        return stabilizer.action(env, gain)
    energy = (1-math.cos(angle))/2 + .21*speed*speed/(6*9.81)
    lateral = (-.5*math.sin(angle)-b*(a-energy)*speed*math.cos(angle)
               -c*env.target_y-d*env.data.qvel[env.dofs['root_y']])
    if env.data.time < .2:
        lateral -= .3
    vertical = h*(1-math.cos(angle))+8*(height-env.target_height)
    vertical = min(vertical, max(0., (.8515-env.target_height)/(.5*env.dt)))
    return np.clip([lateral, vertical], -1, 1)
