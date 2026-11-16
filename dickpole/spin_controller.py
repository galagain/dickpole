"""Phase-feedback controller family optimized by the common CEM trainer."""
import math

import numpy as np


# lateral radius, vertical radius, Hz, locked phase, feedback, height, z phase, speed cap
LOW = np.array([.005, .005, .6, -math.pi, 0., .79, -math.pi, 8.])
HIGH = np.array([.08, .06, 2.1, math.pi, 1., .85, math.pi, 18.])


def oscillator_action(env, parameters):
    ay, az, frequency, offset, feedback, center, zphase, cap = parameters
    rotor = env.direction*float(env.data.qpos[env.qids['pole_0_joint']])
    speed = env.direction*float(env.data.qvel[env.dofs['pole_0_joint']])
    if env.data.time < 4.:
        offset, feedback, zphase = math.pi, 0., math.pi
    else:
        scale = np.clip((cap-speed)/3., 0., 1.)
        ay, az = ay*scale, az*scale
    phase = ((1-feedback)*2*math.pi*frequency*env.data.time
             + feedback*rotor+offset)
    goal = np.array([env.direction*ay*math.cos(phase), center+az*math.sin(phase+zphase)])
    targets = np.array([env.target_y, env.target_height])
    return np.clip((goal-targets)/(.5*env.dt), -1, 1)

