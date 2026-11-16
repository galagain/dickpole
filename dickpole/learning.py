"""Common episodic CEM trainer for balance and fast rotation."""
import json
from pathlib import Path
import shutil

import numpy as np

from .humanoid import HumanoidEnv, humanoid_model_xml
from .balance_controller import Stabilizer, balance_action
from .spin_controller import oscillator_action


def rollout(env, parameters, task, seed, stabilizer=None):
    env.reset(seed=seed)
    total = 0.
    previous = 0.
    while True:
        action = (balance_action(env, parameters, stabilizer) if task == 'balance'
                  else oscillator_action(env, parameters))
        _, reward, done, timeout, info = env.step(action)
        record = info['best_hold'] if task == 'balance' else info['best_consecutive_turns']
        total += reward+8*(record-previous)
        previous = record
        if done or timeout:
            break
    from .policy import succeeded
    success = succeeded(info, task, done)
    return dict(seed=seed, **info, fell=bool(done), reward=float(total), success=bool(success))


def fitness(result, task):
    if task == 'balance':
        return (float(result['success']), result['best_hold'],
                result['reward']-20*result['fell'], result['upright_time'])
    return (float(result['success']), result['best_consecutive_turns']-5*result['fell'],
            result['signed_turns'], result['reward'])


def train(task, folder, steps=262144, population=16, seed=7, resume=None, rounds=None):
    from . import balance_controller as balance
    from . import spin_controller as spin
    family = balance if task == 'balance' else spin
    low, high = family.LOW, family.HIGH
    initial = (balance.INITIAL.copy() if task == 'balance'
               else np.array([.06, .045, 1.2, 1., 1., .83, 0., 12.]))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if resume:
        from .policy import TaskPolicy
        TaskPolicy(resume, task)
        with np.load(Path(resume)/'policy.npz') as f:
            initial = f['parameters'][0].copy()
    if task == 'balance':
        if resume:
            source = Path(resume)/'stabilizer.npz'
            if source.resolve() != (folder/'stabilizer.npz').resolve():
                shutil.copyfile(source,folder/'stabilizer.npz')
        elif not (folder/'stabilizer.npz').exists():
            from .calibration import calibrate
            calibrate(folder)
    stabilizer = Stabilizer(folder) if task == 'balance' else None
    env = HumanoidEnv(task=task)
    rng = np.random.default_rng(seed)
    center = np.clip(2*(initial-low)/(high-low)-1, -1, 1)
    spread = np.full(len(center), .3 if resume else .65)
    incumbent = center.copy()
    best = (-float('inf'),)*4
    history, archive = [], []
    consumed = iteration = 0
    (folder/'humanoid.xml').write_text(humanoid_model_xml())
    while consumed < steps and (rounds is None or iteration < rounds):
        candidates = np.clip(rng.normal(center, spread, (population, len(center))), -1, 1)
        candidates[0] = incumbent
        ranked = []
        for normalized in candidates:
            if consumed >= steps and ranked:
                break
            p = low+(normalized+1)*.5*(high-low)
            result = rollout(env, p, task, seed, stabilizer)
            consumed += env.steps
            score = fitness(result, task)
            row = dict(steps=consumed, round=iteration, parameters=p.tolist(),
                       score=score, episode=result)
            history.append(row)
            archive.append(row)
            ranked.append((score, normalized.copy()))
            if score > best:
                best, incumbent = score, normalized.copy()
                print(f'{task.upper()} steps={consumed} best={best[:3]}', flush=True)
            archive.sort(key=lambda x: tuple(x['score']), reverse=True)
            profiles, seen = [], set()
            for row in archive:
                key = tuple(np.round(row['parameters'], 8))
                if key not in seen:
                    profiles.append(row['parameters'])
                    seen.add(key)
                if len(profiles) >= 32:
                    break
            np.savez_compressed(folder/'policy.npz', parameters=profiles, direction=1)
            (folder/'progress.json').write_text(json.dumps(history, indent=2))
        ranked.sort(key=lambda x: x[0], reverse=True)
        successful = [p for score,p in ranked if score[0]>0]
        elite = np.array(successful or [p for _,p in ranked[:max(2,len(ranked)//3)]])
        center = .5*center+.5*elite.mean(axis=0)
        spread = np.maximum(.6*spread+.4*elite.std(axis=0), .05)
        iteration += 1
        print(f'{task.upper()} round={iteration} steps={consumed}', flush=True)
    from .policy import write_manifest
    write_manifest(folder, task)
    (folder/'training.json').write_text(json.dumps(dict(
        task=task, algorithm='CEM episodic policy search', steps=consumed,
        seed=seed, population=population, rounds=iteration,
        parameter_names=(balance.PARAMETERS if task == 'balance' else
                         ['lateral_radius','vertical_radius','frequency','locked_phase',
                          'phase_feedback','height_center','vertical_phase','speed_cap'])), indent=2))
    env.close()
    return folder
