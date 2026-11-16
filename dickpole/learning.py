"""Multi-seed CEM with validation, mode exploration and resumable optimizer state."""
import json
from pathlib import Path
import shutil

import numpy as np

from .humanoid import HumanoidEnv, humanoid_model_xml
from .controller import Stabilizer, action, LOW, HIGH, PARAMETERS, effective_key

ROOT=Path(__file__).resolve().parents[1]


def rollout(env,parameters,seed,stabilizer):
    env.reset(seed=seed);total=0.
    while True:
        _,reward,done,timeout,info=env.step(action(env,parameters,stabilizer))
        total+=reward
        if done or timeout:break
    from .policy import succeeded
    return dict(seed=seed,**info,fell=done,reward=float(total),
                success=succeeded(info,env.task,done))


def active_indices(parameters):
    """Mutate effective genes locally; global proposals still explore all modes."""
    p=np.asarray(parameters)
    active=set(range(len(p)))
    active.discard(18)
    active.discard(17)
    if p[17]==0.:
        active-=set(range(8))
    elif p[17]==1.:
        active-=set(range(8,14));active.discard(19)
    if p[18]<.5:active-=set(range(14,17))
    return sorted(active)


def propose(rng,center,spread,incumbent,population):
    candidates=rng.normal(center,spread,(population,len(center)))
    reference=2*(incumbent-LOW)/(HIGH-LOW)-1
    active=active_indices(incumbent)
    # Half the population refines the current phenotype without inactive noise.
    for row in range(population//2):
        samples=candidates[row,active].copy()
        candidates[row]=reference
        candidates[row,active]=samples
    # Periodic phases wrap rather than accumulating at an arbitrary boundary.
    for j in (3,6):candidates[:,j]=(candidates[:,j]+1)%2-1
    candidates=np.clip(candidates,-1,1)
    candidates[0]=reference
    if population>2:
        candidates[-1,18]=-reference[18]  # explicitly explore the other capture mode
        candidates[-2,17]=rng.choice([-1.,1.])
    return candidates


def robust_score(records):
    rewards=np.asarray([r['reward'] for r in records])
    return float(rewards.mean()-.1*rewards.std())


def train(task,folder,steps=262144,population=16,seed=7,resume=None,rounds=None,episodes=3):
    from .policy import TaskPolicy,write_manifest,model_hash
    if episodes<1:raise ValueError('episodes per candidate must be positive')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if resume:
        if json.loads((Path(resume)/'manifest.json').read_text()).get('backend')=='ppo':
            raise ValueError('Spin training requires a compatible CEM checkpoint')
        previous=TaskPolicy(resume,task)
        profiles=previous.parameters.copy();source=Path(resume)/'stabilizer.npz'
    else:
        with np.load(ROOT/'models/spin/initial_profiles.npz',allow_pickle=False) as bundle:
            profiles=bundle['parameters'].copy()
        source=ROOT/'models/spin/stabilizer.npz';metadata=source.with_suffix('.json')
        if not source.exists() or not metadata.exists() or json.loads(metadata.read_text()).get('physical_model_sha256')!=model_hash() or json.loads(metadata.read_text()).get('action_size')!=5:
            from .calibration import calibrate
            calibrate(folder);source=folder/'stabilizer.npz'
    if source.resolve()!=(folder/'stabilizer.npz').resolve():
        shutil.copyfile(source,folder/'stabilizer.npz')
    stabilizer=Stabilizer(folder);env=HumanoidEnv(task=task);rng=np.random.default_rng(seed)
    training_seeds=list(range(seed,seed+episodes))
    validation_start=1000
    while set(range(validation_start,validation_start+4)) & set(training_seeds):
        validation_start+=4
    validation_seeds=list(range(validation_start,validation_start+4))
    # Initialization cannot consume the entire budget before evolution starts.
    count=min(len(profiles),max(2,population//2))
    indices=np.unique(np.linspace(0,len(profiles)-1,count,dtype=int))
    initial=profiles[indices]
    history=[];archive={}
    consumed=train_steps=validation_steps=iteration=0
    incumbent=profiles[0].copy();best=-float('inf')
    center=np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
    spread=np.full(len(center),.3 if resume else .45)
    resumed_optimizer=False
    if resume and (Path(resume)/'search_state.json').exists():
        state=json.loads((Path(resume)/'search_state.json').read_text())
        if state['training_seeds']==training_seeds:
            with np.load(Path(resume)/'search_state.npz',allow_pickle=False) as saved:
                center=saved['center'].copy();spread=saved['spread'].copy()
                incumbent=saved['incumbent'].copy()
            rng.bit_generator.state=state['rng_state']
            initial=np.vstack([incumbent,initial])
            resumed_optimizer=True
    (folder/'humanoid.xml').write_text(humanoid_model_xml())

    def evaluate(parameters):
        nonlocal consumed,train_steps,best,incumbent
        key=effective_key(parameters)
        if key in archive:return archive[key]['score']
        records=[]
        for episode_seed in training_seeds:
            records.append(rollout(env,parameters,episode_seed,stabilizer))
            consumed+=env.steps;train_steps+=env.steps
        score=robust_score(records)
        row=dict(steps=consumed,round=iteration,parameters=parameters.tolist(),score=score,episodes=records)
        history.append(row);archive[key]=row
        if score>best:
            best=score;incumbent=parameters.copy()
            print(f'CEM {task} steps={consumed} robust_return={best:.2f} '
                  f'hold={np.mean([r["best_hold"] for r in records]):.2f}s '
                  f'rps={np.mean([r["mean_free_rps"] for r in records]):.2f}',flush=True)
        return score

    def checkpoint_search():
        # Keep a usable incumbent even if a long search is interrupted.
        top=sorted(archive.values(),key=lambda row:row['score'],reverse=True)[:16]
        np.savez_compressed(folder/'policy.npz',parameters=[r['parameters'] for r in top],direction=1)
        write_manifest(folder,task)
        (folder/'progress.json').write_text(json.dumps(history,indent=2))
        np.savez_compressed(folder/'search_state.npz',center=center,spread=spread,incumbent=incumbent)
        (folder/'search_state.json').write_text(json.dumps(dict(rng_state=rng.bit_generator.state,
            training_seeds=training_seeds,validation_seeds=validation_seeds),indent=2))

    for p in initial:
        if consumed>=steps:break
        evaluate(p)
    if not resumed_optimizer:
        center=np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
    checkpoint_search()
    stagnation=0
    while consumed<steps and (rounds is None or iteration<rounds):
        previous_best=best
        candidates=propose(rng,center,spread,incumbent,population)
        ranked=[]
        for normalized in candidates:
            if consumed>=steps:break
            p=LOW+(normalized+1)*.5*(HIGH-LOW)
            ranked.append((evaluate(p),normalized.copy()))
        ranked.sort(key=lambda x:x[0],reverse=True)
        elite=np.asarray([p for _,p in ranked[:max(2,len(ranked)//4)]])
        center=.4*center+.6*elite.mean(axis=0)
        spread=np.maximum(.65*spread+.35*elite.std(axis=0),.025)
        stagnation=stagnation+1 if best<=previous_best else 0
        if stagnation>=3:
            center=np.clip(2*(incumbent-LOW)/(HIGH-LOW)-1,-1,1)
            spread=np.maximum(spread,.3);stagnation=0
        iteration+=1
        checkpoint_search()
        print(f'CEM {task} round={iteration} steps={consumed}',flush=True)
    ranked=sorted(archive.values(),key=lambda r:r['score'],reverse=True)
    # Validation chooses the deployed first profile, using distinct frozen seeds.
    finalists=[np.asarray(row['parameters']) for row in ranked[:4]]
    if not any(effective_key(profiles[0])==effective_key(p) for p in finalists):
        finalists.append(profiles[0])
    champion=finalists[0];champion_score=-float('inf')
    validation=[]
    for p in finalists:
        records=[]
        for validation_seed in validation_seeds:
            records.append(rollout(env,p,validation_seed,stabilizer))
            consumed+=env.steps;validation_steps+=env.steps
        score=robust_score(records)
        validation.append(dict(parameters=p.tolist(),score=score,episodes=records))
        if score>champion_score:champion_score=score;champion=p.copy()
    saved=[champion.tolist()];seen={effective_key(champion)}
    for row in ranked:
        key=effective_key(row['parameters'])
        if key not in seen:saved.append(row['parameters']);seen.add(key)
        if len(saved)>=16:break
    np.savez_compressed(folder/'policy.npz',parameters=saved,direction=1)
    write_manifest(folder,task)
    (folder/'progress.json').write_text(json.dumps(history,indent=2))
    (folder/'selection.json').write_text(json.dumps(validation,indent=2))
    np.savez_compressed(folder/'search_state.npz',center=center,spread=spread,incumbent=incumbent)
    (folder/'search_state.json').write_text(json.dumps(dict(rng_state=rng.bit_generator.state,
        training_seeds=training_seeds,validation_seeds=validation_seeds),indent=2))
    (folder/'training.json').write_text(json.dumps(dict(task=task,algorithm='multi-seed robust CEM',
        steps=consumed,training_steps=train_steps,validation_steps=validation_steps,
        training_seeds=training_seeds,validation_seeds=validation_seeds,population=population,
        rounds=iteration,resumed_optimizer=resumed_optimizer,parameter_names=PARAMETERS,physical_model_sha256=model_hash()),indent=2))
    env.close();return folder
