"""Identical return-maximizing CEM trainer for both rewards and the same policy family."""
import json
from pathlib import Path
import shutil

import numpy as np

from .humanoid import HumanoidEnv, humanoid_model_xml
from .controller import Stabilizer, action, LOW, HIGH, PARAMETERS, effective_key

ROOT = Path(__file__).resolve().parents[1]


def rollout(env, parameters, seed, stabilizer):
    env.reset(seed=seed)
    total = 0.
    while True:
        _,reward,done,timeout,info = env.step(action(env,parameters,stabilizer))
        total += reward
        if done or timeout:
            break
    from .policy import succeeded
    return dict(seed=seed,**info,fell=done,reward=float(total),
                success=succeeded(info,env.task,done))


def train(task, folder, steps=262144, population=16, seed=7, resume=None, rounds=None):
    from .policy import TaskPolicy,write_manifest,model_hash
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if resume:
        previous = TaskPolicy(resume,task)
        profiles = previous.parameters.copy()
        source = Path(resume)/'stabilizer.npz'
    else:
        with np.load(ROOT/'models/initial_profiles.npz',allow_pickle=False) as bundle:
            profiles = bundle['parameters'].copy()
        source = ROOT/'models/stabilizer.npz'
        metadata = source.with_suffix('.json')
        if not source.exists() or not metadata.exists() or json.loads(metadata.read_text()).get('physical_model_sha256')!=model_hash() or json.loads(metadata.read_text()).get('action_size')!=5:
            from .calibration import calibrate
            calibrate(folder)
            source = folder/'stabilizer.npz'
    if source.resolve() != (folder/'stabilizer.npz').resolve():
        shutil.copyfile(source,folder/'stabilizer.npz')
    stabilizer = Stabilizer(folder)
    env = HumanoidEnv(task=task)
    rng = np.random.default_rng(seed)
    history,archive = [],[]
    consumed = iteration = 0
    best_return = -float('inf')
    incumbent = profiles[0].copy()
    center = np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
    spread = np.full(len(center),.3 if resume else .45)
    (folder/'humanoid.xml').write_text(humanoid_model_xml())

    def evaluate(parameters):
        nonlocal consumed,best_return,incumbent
        result = rollout(env,parameters,seed,stabilizer)
        consumed += env.steps
        row = dict(steps=consumed,round=iteration,parameters=parameters.tolist(),episode=result)
        history.append(row);archive.append(row)
        if result['reward']>best_return:
            best_return = result['reward'];incumbent=parameters.copy()
            print(f'{task.upper()} steps={consumed} return={best_return:.2f} '
                  f'hold={result["best_hold"]:.2f}s rps={result["mean_free_rps"]:.2f} '
                  f'fell={result["fell"]}',flush=True)
        archive.sort(key=lambda x:x['episode']['reward'],reverse=True)
        saved,seen=[],set()
        for record in archive:
            key=effective_key(record['parameters'])
            if key not in seen:
                saved.append(record['parameters']);seen.add(key)
            if len(saved)>=16:break
        np.savez_compressed(folder/'policy.npz',parameters=saved,direction=1)
        (folder/'progress.json').write_text(json.dumps(history,indent=2))
        return result['reward']

    # Exactly the same initialization pool is screened for both objectives.
    for parameters in profiles:
        if consumed>=steps:break
        evaluate(parameters)
    center = np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
    while consumed<steps and (rounds is None or iteration<rounds):
        candidates=np.clip(rng.normal(center,spread,(population,len(center))),-1,1)
        candidates[0]=np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
        ranked=[]
        for normalized in candidates:
            if consumed>=steps:break
            parameters=LOW+(normalized+1)*.5*(HIGH-LOW)
            score=evaluate(parameters)
            ranked.append((score,normalized.copy()))
        ranked.sort(key=lambda x:x[0],reverse=True)
        elite=np.array([p for _,p in ranked[:max(2,len(ranked)//4)]])
        center=.4*center+.6*elite.mean(axis=0)
        spread=np.maximum(.65*spread+.35*elite.std(axis=0),.02)
        iteration+=1
        print(f'{task.upper()} round={iteration} steps={consumed}',flush=True)
    write_manifest(folder,task)
    (folder/'training.json').write_text(json.dumps(dict(
        task=task,algorithm='shared CEM, undiscounted environment return',steps=consumed,
        seed=seed,population=population,rounds=iteration,parameter_names=PARAMETERS,
        physical_model_sha256=model_hash()),indent=2))
    env.close()
    return folder
