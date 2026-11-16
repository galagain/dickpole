"""One controller architecture and one parameter space for both reward functions."""
import numpy as np

from . import balance_controller as energy
from . import spin_controller as oscillator
from .balance_controller import Stabilizer

LOW = np.r_[oscillator.LOW, energy.LOW, 0., 0., -2.]
HIGH = np.r_[oscillator.HIGH, energy.HIGH, 1., 1., 2.]
PARAMETERS = [
    'lateral_radius','vertical_radius','frequency','locked_phase','phase_feedback',
    'height_center','vertical_phase','speed_threshold',*energy.PARAMETERS,
    'oscillator_fraction','capture_enabled','roll_pumping',
]
ARCHITECTURE = 'shared_hybrid_20_v3'


def action(env, parameters, stabilizer):
    """No task/reward input: identical states and weights give identical actions.

    Both policies may use pumping, circular phase feedback and upright feedback.
    The learned parameters decide how these common primitives are combined.
    """
    p = np.asarray(parameters)
    a = energy.balance_action(env,p[8:17],stabilizer,allow_capture=p[18]>=.5)
    if env.control_phase:
        return a
    angle,speed=env.pole_state()
    a[4]=np.clip(p[19]*(-2.13*np.sin(angle)-2.51*speed*np.cos(angle))
                 -2*env.target_roll-.1*env.data.qvel[env.dofs['root_roll']],-1,1)
    fraction = float(p[17])
    if fraction == 0.:
        return a
    b = np.r_[oscillator.oscillator_action(env,p[:8]),0.,0.,0.]
    return np.clip((1-fraction)*a+fraction*b,-1,1)

def effective_key(parameters):
    """Collapse inactive genes so a library contains distinct controllers."""
    p=np.asarray(parameters).copy()
    p[18]=float(p[18]>=.5)
    if p[17]==0.:
        p[:8]=0.
    elif p[17]==1.:
        p[8:14]=0.
        p[19]=0.
    if not p[18]:
        p[14:17]=0.
        if p[17]==1.:
            p[8:17]=0.
            p[19]=0.
    return tuple(np.round(p,10))
