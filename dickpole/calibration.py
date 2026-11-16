"""Offline local identification; evaluation always starts hanging down at rest."""
import copy
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from .humanoid import HumanoidEnv, SIDE_JOINTS


def _identify(target_y=0., height=.86):
    env = HumanoidEnv(task='balance')
    env.reset(seed=7)
    env.target_y = target_y
    env.target_height = height
    rotor = env.qids['pole_0_joint']
    # This upright reference is used ONLY to identify the local linear model.
    # It is never used as a training/evaluation initial condition.
    for _ in range(250):
        env.data.qpos[rotor] = math.pi-env.data.qpos[env.qids['root_roll']]
        env.data.qvel[env.dofs['pole_0_joint']] = -env.data.qvel[env.dofs['root_roll']]
        mujoco.mj_forward(env.model, env.data)
        env.step([0., 0.])
    env.data.qpos[rotor] = math.pi-env.data.qpos[env.qids['root_roll']]
    env.data.qvel[env.dofs['pole_0_joint']] = -env.data.qvel[env.dofs['root_roll']]
    mujoco.mj_forward(env.model, env.data)
    qi = np.array([i for i in range(env.model.nq) if i != env.qids['root_y']])
    vi, ai = np.arange(env.model.nv), np.arange(env.model.na)

    def state():
        return np.r_[env.data.qpos[qi], env.data.qvel[vi], env.data.act[ai],
                     env.target_y, env.target_height]

    reference, baseline = state(), copy.copy(env.data)

    def advance(s, action):
        mujoco.mj_copyData(env.data, env.model, baseline)
        env.data.qpos[qi] = s[:len(qi)]
        env.data.qvel[vi] = s[len(qi):len(qi)+len(vi)]
        env.data.act[ai] = s[-(len(ai)+2):-2]
        env.target_y, env.target_height = float(s[-2]), float(s[-1])
        mujoco.mj_forward(env.model, env.data)
        env.step(action)
        return state()

    n, eps = len(reference), 1e-5
    A, B = np.zeros((n, n)), np.zeros((n, 2))
    for j in range(n):
        delta = np.eye(1, n, j).ravel()*eps
        A[:, j] = (advance(reference+delta, [0,0])-advance(reference-delta, [0,0]))/(2*eps)
    for j in range(2):
        delta = np.eye(1, 2, j).ravel()*eps
        B[:, j] = (advance(reference, delta)-advance(reference, -delta))/(2*eps)
    Q = np.eye(n)*.01
    for name, weight in [('pole_0_joint',1000), ('root_roll',10),
                         ('root_pitch',10), ('pole_splay',100)]:
        j = int(np.flatnonzero(qi == env.qids[name])[0])
        Q[j,j] = weight
    Q[len(qi)+env.dofs['pole_0_joint'],len(qi)+env.dofs['pole_0_joint']] = 50
    R = np.eye(2)*100
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R+B.T@P@B, B.T@P@A)[0]
    # Keep lateral feedback; the reflex independently stabilizes the sagittal
    # plane. This avoids reacting to its small stance oscillations with y moves.
    mask = np.zeros(n, bool)
    for name in ['root_roll', *SIDE_JOINTS, 'pole_0_joint']:
        mask[np.flatnonzero(qi == env.qids[name])[0]] = True
        mask[len(qi)+env.dofs[name]] = True
    mask[len(qi)+env.dofs['root_y']] = True
    mask[len(qi)+len(vi)+14:len(qi)+len(vi)+24] = True
    mask[-2] = True
    K[~mask] = 0.
    env.close()
    return reference, K, qi, vi, ai


def calibrate(folder):
    knots = np.array([-.075,-.05,-.025,0.,.025,.05,.075])
    references,gains = [],[]
    for target in knots:
        reference,K,qi,vi,ai = _identify(float(target))
        references.append(reference);gains.append(K)
        print(f'Identified local balance at pelvis offset {target:.3f} m',flush=True)
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'stabilizer.npz',reference=references,K=gains,
                        qids=qi,dofs=vi,actids=ai,knots=knots)
    return folder/'stabilizer.npz'
