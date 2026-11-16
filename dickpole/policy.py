"""Task-specific learned policies on one versioned 3D physical model."""
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .humanoid import HumanoidEnv, humanoid_model_xml
from .balance_controller import Stabilizer, balance_action
from .spin_controller import oscillator_action

SCHEMA = 1


def model_hash():
    return hashlib.sha256(humanoid_model_xml().encode()).hexdigest()


def write_manifest(folder, task):
    metadata = dict(schema=SCHEMA, task=task, physical_model_sha256=model_hash(),
                    action_names=['lateral_pelvis_velocity','vertical_pelvis_velocity'],
                    episode_seconds=20, target_rotation_rps=2.)
    Path(folder,'manifest.json').write_text(json.dumps(metadata, indent=2))


def succeeded(info, task, fell):
    return bool(not fell and (info['best_hold'] >= 5. if task == 'balance' else
                             info['best_consecutive_turns'] >= 3.
                             and info['mean_free_rps'] >= 1.5))


class TaskPolicy:
    def __init__(self, folder, task=None):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        self.task = self.manifest['task']
        if self.manifest['schema'] != SCHEMA or self.manifest['physical_model_sha256'] != model_hash():
            raise ValueError('Checkpoint uses a different physical model; retrain it for this environment')
        if task is not None and self.task != task:
            raise ValueError(f'Expected {task} checkpoint, found {self.task}')
        with np.load(self.folder/'policy.npz', allow_pickle=False) as bundle:
            self.parameters = bundle['parameters'].copy()
            self.direction = int(bundle['direction'])
        expected = 9 if self.task == 'balance' else 8
        if self.parameters.ndim != 2 or self.parameters.shape[1] != expected or not np.isfinite(self.parameters).all():
            raise ValueError('Invalid policy parameters')
        self.stabilizer = Stabilizer(self.folder) if self.task == 'balance' else None
        self.selected = 0

    def predict(self, env):
        if env.task != self.task:
            raise ValueError('Policy task and environment goal differ')
        p = self.parameters[self.selected]
        return balance_action(env, p, self.stabilizer) if self.task == 'balance' else oscillator_action(env, p)

    def select(self, live, horizon=20.):
        """Plan in simulator copies; the live integration state stays untouched."""
        clone = HumanoidEnv(task=self.task, duration=live.duration, direction=live.direction)
        clone.model = copy.copy(live.model)
        clone.data = mujoco.MjData(clone.model)
        fields = ('steps','target_x','target_pitch','target_height','target_y','target_roll',
                  'support_origin','balance_mode','last_action','last_pitch_action',
                  'last_height_action','spin_action','signed_angle','streak_angle',
                  'best_streak_angle','previous_rotor_angle','stall_time','contact_time',
                  'hold','best_hold','upright_time','control_phase',
                  'free_spin_angle','fast_time','peak_rps','lqr_anchor')
        best, winner, attempts = (-1., -float('inf'), -float('inf')), 0, []
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
            success = succeeded(info, self.task, done)
            metric = info['best_hold'] if self.task == 'balance' else info['mean_free_rps']
            score = (float(success), metric-10*done, total)
            attempts.append(dict(profile=i, success=success, metric=metric, fell=done))
            if score > best:
                best, winner = score, i
            if success:
                break
        self.selected = winner
        clone.close()
        return dict(profile=winner, predicted_success=bool(best[0]), attempts=attempts)


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
        record = dict(seed=seed+i, **info, reward=total, fell=done,
                      success=succeeded(info, policy.task, done), plan=selection,
                      physical_model_sha256=model_hash())
        records.append(record)
        print(f'{policy.task.upper()} seed={seed+i} success={record["success"]} '
              f'hold={info["best_hold"]:.2f}s free_rps={info["mean_free_rps"]:.2f}', flush=True)
        env.close()
    return records
