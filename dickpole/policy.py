"""Task-specific learned policies on one versioned 3D physical model."""
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .humanoid import HumanoidEnv, humanoid_model_xml
from .controller import Stabilizer, action, ARCHITECTURE, PARAMETERS

SCHEMA = 2


def model_hash():
    return hashlib.sha256(humanoid_model_xml().encode()).hexdigest()


def write_manifest(folder, task):
    stabilizer_sha256=hashlib.sha256(Path(folder,'stabilizer.npz').read_bytes()).hexdigest()
    metadata = dict(schema=SCHEMA, task=task, physical_model_sha256=model_hash(),
                    action_names=['lateral_pelvis_velocity','vertical_pelvis_velocity','fore_aft_pelvis_velocity','pelvis_tilt_velocity','pelvis_roll_velocity'],
                    episode_seconds=20, target_rotation_rps=2.,
                    controller_architecture=ARCHITECTURE, observation_size=94,
                    parameter_names=PARAMETERS,stabilizer_sha256=stabilizer_sha256,
                    reward_spec='shared_rewards_v3')
    Path(folder,'manifest.json').write_text(json.dumps(metadata, indent=2))


def succeeded(info, task, fell):
    return bool(not fell and (info['best_hold'] >= 5. if task == 'balance' else
                             info['best_consecutive_turns'] >= 3.
                             and info['mean_free_rps'] >= 1.5))


class TaskPolicy:
    def __init__(self, folder, task=None):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        self.trained_task = self.manifest['task']
        self.task = task or self.trained_task
        if self.manifest['schema'] != SCHEMA or self.manifest['physical_model_sha256'] != model_hash():
            raise ValueError('Checkpoint uses a different physical model; retrain it for this environment')
        with np.load(self.folder/'policy.npz', allow_pickle=False) as bundle:
            self.parameters = bundle['parameters'].copy()
            self.direction = int(bundle['direction'])
        if self.manifest.get('controller_architecture') != ARCHITECTURE:
            raise ValueError('Checkpoint uses a different controller architecture')
        expected = len(PARAMETERS)
        if self.parameters.ndim != 2 or self.parameters.shape[1] != expected or not np.isfinite(self.parameters).all():
            raise ValueError('Invalid policy parameters')
        actual=hashlib.sha256((self.folder/'stabilizer.npz').read_bytes()).hexdigest()
        if self.manifest.get('stabilizer_sha256',actual)!=actual:
            raise ValueError('Checkpoint feedback weights do not match its manifest')
        self.stabilizer = Stabilizer(self.folder)
        self.selected = 0

    def predict(self, env):
        p = self.parameters[self.selected]
        return action(env,p,self.stabilizer)

    def select(self, live, horizon=20.):
        """Plan in simulator copies; the live integration state stays untouched."""
        clone = HumanoidEnv(task=live.task, duration=live.duration, direction=live.direction)
        clone.model = copy.copy(live.model)
        clone.data = mujoco.MjData(clone.model)
        fields = ('steps','target_x','target_pitch','target_height','target_y','target_roll',
                  'support_origin','balance_mode','last_action','last_pitch_action',
                  'last_height_action','spin_action','signed_angle','streak_angle',
                  'best_streak_angle','previous_rotor_angle','stall_time','contact_time',
                  'hold','best_hold','upright_time','control_phase',
                  'free_spin_angle','fast_time','peak_rps','lqr_anchor')
        best, winner, attempts = -float('inf'), 0, []
        for i in range(len(self.parameters)):
            mujoco.mj_copyData(clone.data, clone.model, live.data)
            for field in fields:
                setattr(clone, field, copy.copy(getattr(live, field)))
            self.selected = i
            total, done = 0., False
            info = clone.info()
            for _ in range(round(horizon/clone.dt)):
                _, reward, done, timeout, info = clone.step(self.predict(clone))
                total += reward
                if done or timeout:
                    break
            success = succeeded(info, clone.task, done)
            metric = info['best_hold'] if clone.task == 'balance' else info['mean_free_rps']
            score = total
            attempts.append(dict(profile=i, success=success, metric=metric, fell=done))
            if score > best:
                best, winner = score, i

        self.selected = winner
        clone.close()
        return dict(profile=winner, predicted_success=bool(attempts[winner]['success']), attempts=attempts)


def evaluate(policy, episodes=10, seed=3000, plan=True):
    records = []
    for i in range(episodes):
        env = HumanoidEnv(task=policy.task, direction=policy.direction)
        env.reset(seed=seed+i)
        selection = policy.select(env) if plan else None
        total = 0.
        while True:
            _, reward, done, timeout, info = env.step(policy.predict(env))
            total += reward
            if done or timeout:
                break
        record = dict(seed=seed+i, task=policy.task, trained_task=policy.trained_task, **info, reward=total, fell=done,
                      success=succeeded(info, policy.task, done), plan=selection,
                      physical_model_sha256=model_hash())
        records.append(record)
        print(f'{policy.task.upper()} seed={seed+i} success={record["success"]} '
              f'hold={info["best_hold"]:.2f}s free_rps={info["mean_free_rps"]:.2f}', flush=True)
        env.close()
    return records
