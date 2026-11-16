"""Model-based selection among learned residual PPO policies, then real execution.

Planning runs on independent MuJoCo copies. The live environment is never
advanced or reset by the selector. This is model-based control, not pure PPO.
"""
import copy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from .rl_env import ResidualHumanoid


def build_library(source, search, folder, count=32):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    history = json.loads(Path(search).read_text())
    history.sort(key=lambda row: tuple(row['score']), reverse=True)
    unique, seen = [], set()
    for row in history:
        key = tuple(np.round(row['parameters'], 9))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    chosen = []
    covered = set()
    # Start with validation coverage, then keep distinct policy heads.
    for _ in range(min(8, len(unique))):
        def quality(row):
            wins = {r['seed'] for r in row['episodes'] if r['success']}
            return len(wins-covered), tuple(row['score'])
        selected = max(unique, key=quality)
        chosen.append(selected)
        covered |= {r['seed'] for r in selected['episodes'] if r['success']}
        unique.remove(selected)
    while unique and len(chosen) < count-2:
        def distance(row):
            scale = np.array([.4]*3+[.08]*3)
            return min(np.linalg.norm((np.array(row['parameters'])-np.array(c['parameters']))/scale)
                       for c in chosen)
        selected = max(unique, key=distance)
        chosen.append(selected)
        unique.remove(selected)
    profiles = [np.r_[np.exp(r['parameters'][:3]), r['parameters'][3:]] for r in chosen]
    # Original PPO and zero-residual baseline add complementary trajectories.
    profiles.insert(1, np.array([1., 1., 1., 0., 0., 0.]))
    profiles.insert(2, np.zeros(6))
    np.savez_compressed(folder/'heads.npz', profiles=np.array(profiles))
    shutil.copyfile(source, folder/'policy.zip')
    (folder/'library.json').write_text(json.dumps(dict(
        source=str(source), search=str(search), profiles=len(profiles),
        method='PPO residual policies + CEM head search + MuJoCo lookahead',
        planning_uses_known_simulator=True, validation_seeds=list(range(1000, 1005))), indent=2))


class PolicyLibrary:
    def __init__(self, folder=Path(__file__).resolve().parent.parent / 'models' / 'pretrained'):
        folder = Path(folder)
        self.model = PPO.load(folder/'policy.zip', device='cpu')
        self.weight = self.model.policy.action_net.weight.detach().clone()
        self.bias = self.model.policy.action_net.bias.detach().clone()
        with np.load(folder/'heads.npz', allow_pickle=False) as bundle:
            self.profiles = bundle['profiles'].copy()
        self.selected = None

    def apply(self, index):
        parameters = self.profiles[index]
        layer = self.model.policy.action_net
        gain = torch.as_tensor(parameters[:3], dtype=layer.weight.dtype)
        bias = torch.as_tensor(parameters[3:], dtype=layer.bias.dtype)
        with torch.no_grad():
            layer.weight.copy_(self.weight*gain[:, None])
            layer.bias.copy_(self.bias*gain+bias)
        self.selected = index

    def predict(self, observation):
        return self.model.predict(observation, deterministic=True)[0]

    def select(self, live, horizon=20., max_candidates=None):
        """Roll out candidates in clones of the *current* state, never live."""
        if horizon <= 0 or (max_candidates is not None and max_candidates < 1):
            raise ValueError('Positive planning horizon and candidate count required')
        sim = live.unwrapped
        clone = ResidualHumanoid(duration=sim.duration)
        other = clone.unwrapped
        other.model = copy.copy(sim.model)
        other.data = mujoco.MjData(other.model)
        fields = ('steps', 'hold', 'best_hold', 'upright_time', 'target_x',
                  'target_pitch', 'target_height', 'last_action', 'last_pitch_action',
                  'last_height_action', 'support_origin', 'balance_mode')
        best = (-1., -1., -float('inf'))
        best_index = 0
        attempts = []
        start = time.monotonic()
        limit = min(len(self.profiles), max_candidates or len(self.profiles))
        for index in range(limit):
            mujoco.mj_copyData(other.data, other.model, sim.data)
            for field in fields:
                setattr(other, field, getattr(sim, field))
            self.apply(index)
            obs = clone.observe()
            total = 0.
            done = False
            info = other.info()
            for _ in range(max(1, round(horizon/sim.dt))):
                obs, reward, done, truncated, info = clone.step(self.predict(obs))
                total += info['task_reward']
                if done or truncated:
                    break
            success = not done and info['best_hold'] >= 5.
            score = (float(success), float(info['best_hold']), total)
            attempts.append(dict(profile=index, success=success, hold=info['best_hold'], fell=done))
            if score > best:
                best, best_index = score, index
            if success:
                break
        self.apply(best_index)
        clone.close()
        return dict(profile=best_index, predicted_success=bool(best[0]),
                    planning_seconds=time.monotonic()-start, attempts=attempts)


def run_evaluation(library, episodes, seed, duration):
    records = []
    for index in range(episodes):
        env = ResidualHumanoid(duration=duration)
        obs, _ = env.reset(seed=seed+index)
        plan = library.select(env)
        # This rollout starts from the original untouched, down-at-rest state.
        total = 0.
        while True:
            obs, reward, done, truncated, info = env.step(library.predict(obs))
            total += info['task_reward']
            if done or truncated:
                break
        success = not done and info['best_hold'] >= 5.
        record = dict(seed=seed+index, **info, fell=done, success=success,
                      return_value=total, plan=plan)
        records.append(record)
        print(f"seed={seed+index} success={success} hold={info['best_hold']:.2f}s "
              f"profile={plan['profile']} trials={len(plan['attempts'])}", flush=True)
        env.close()
    return records


def play(library, seed):
    import mujoco.viewer
    env = ResidualHumanoid(duration=40.)
    sim = env.unwrapped
    obs, _ = env.reset(seed=seed)
    print('Préparation du contrôleur...', flush=True)
    plan = library.select(env)
    print(f"Prêt en {plan['planning_seconds']:.1f} s", flush=True)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.azimuth = 110
        viewer.cam.elevation = -10
        viewer.cam.distance = 3.
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            start = time.monotonic()
            obs, _, done, truncated, info = env.step(library.predict(obs))
            viewer.cam.lookat[:] = [sim.data.body('pelvis').xpos[0], 0, .85]
            viewer.sync()
            if done or truncated:
                print(f"Maintien : {info['best_hold']:.1f} s — {'chute' if done else 'épisode terminé'}", flush=True)
                seed += 1
                obs, _ = env.reset(seed=seed)
                plan = library.select(env)
                print(f"Prêt en {plan['planning_seconds']:.1f} s", flush=True)
            time.sleep(max(0, sim.dt-(time.monotonic()-start)))

