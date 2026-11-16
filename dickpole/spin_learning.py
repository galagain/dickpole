"""Reward-driven CEM search for a phase-feedback helicopter controller."""
import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .spin import SpinHumanoid, spin_model_xml

# lateral radius, vertical radius, Hz, locked phase, feedback, height, z phase, speed cap
LOW = np.array([.005, .005, .5, -math.pi, 0., .79, -math.pi, 8.])
HIGH = np.array([.08, .06, 1.55, math.pi, 1., .85, math.pi, 18.])


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


def run_episode(env, parameters, seed):
    env.reset(seed=seed)
    total = 0.
    previous_best = 0.
    while True:
        _, reward, done, timeout, info = env.step(oscillator_action(env, parameters))
        # Record-progress shaping cannot be farmed by repeated partial swings.
        total += reward+8*(info['best_consecutive_turns']-previous_best)
        previous_best = info['best_consecutive_turns']
        if done or timeout:
            break
    return dict(seed=seed, **info, fell=done, reward=total,
                success=not done and info['best_consecutive_turns'] >= 3.)


class SpinPolicy:
    def __init__(self, folder):
        with np.load(Path(folder)/'policy.npz', allow_pickle=False) as data:
            self.parameters = data['parameters'].copy()
            self.direction = int(data['direction'])
        self.selected = 0

    def predict(self, env):
        return oscillator_action(env, self.parameters[self.selected])

    def select(self, live, horizon=20.):
        """Choose a learned oscillator in independent copies of the live world."""
        clone = SpinHumanoid(duration=live.duration, direction=live.direction)
        clone.model = copy.copy(live.model)
        clone.data = mujoco.MjData(clone.model)
        fields = ('steps', 'target_x', 'target_pitch', 'target_height', 'target_y',
                  'target_roll', 'support_origin', 'balance_mode', 'last_action',
                  'last_pitch_action', 'last_height_action', 'spin_action',
                  'signed_angle', 'streak_angle', 'best_streak_angle',
                  'previous_rotor_angle', 'stall_time', 'contact_time',
                  'hold', 'best_hold', 'upright_time')
        best = (-1., -float('inf'), -float('inf'))
        winner = 0
        attempts = []
        for i, parameters in enumerate(self.parameters):
            mujoco.mj_copyData(clone.data, clone.model, live.data)
            for field in fields:
                setattr(clone, field, copy.copy(getattr(live, field)))
            total = 0.
            done = False
            info = clone.info()
            for _ in range(round(horizon/clone.dt)):
                _, reward, done, timeout, info = clone.step(oscillator_action(clone, parameters))
                total += reward
                if done or timeout:
                    break
            success = not done and info['best_consecutive_turns'] >= 3.
            score = (float(success), info['best_consecutive_turns'], total)
            attempts.append(dict(profile=i, success=success, turns=info['best_consecutive_turns']))
            if score > best:
                best, winner = score, i
            if success:
                break
        self.selected = winner
        clone.close()
        return dict(profile=winner, predicted_success=bool(best[0]), attempts=attempts)


def train_spin(folder, steps=262144, population=12, seed=7, resume=None,
               direction=1, max_rounds=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = SpinHumanoid(direction=direction)
    center = np.array([.06, .045, 1., 1., 1., .83, 0., 12.])
    if resume:
        center = SpinPolicy(resume).parameters[0]
    center = np.clip(2*(center-LOW)/(HIGH-LOW)-1, -1, 1)
    deviation = np.full(len(center), .25 if resume else .65)
    history = []
    archive = []
    consumed = iteration = 0
    best_score = (-1., -float('inf'), -float('inf'), -float('inf'))
    best_normalized = center.copy()
    (folder/'humanoid.xml').write_text(spin_model_xml())
    while consumed < steps and (max_rounds is None or iteration < max_rounds):
        candidates = np.clip(rng.normal(center, deviation, (population, len(center))), -1, 1)
        candidates[0] = best_normalized
        ranked = []
        for normalized in candidates:
            if consumed >= steps and ranked:
                break
            parameters = LOW+(normalized+1)*.5*(HIGH-LOW)
            result = run_episode(env, parameters, seed)
            consumed += env.steps
            score = (float(result['success']),
                     result['best_consecutive_turns']-(5. if result['fell'] else 0.),
                     result['signed_turns'], result['reward'])
            ranked.append((score, normalized.copy()))
            record = dict(steps=consumed, round=iteration, parameters=parameters.tolist(),
                          score=score, episode=result)
            history.append(record)
            archive.append(record)
            if score > best_score:
                best_score, best_normalized = score, normalized.copy()
                print(f'SPIN steps={consumed} free_turns={result["best_consecutive_turns"]:.2f} '
                      f'net_turns={result["signed_turns"]:.2f} success={result["success"]}', flush=True)
            archive.sort(key=lambda item: tuple(item['score']), reverse=True)
            profiles, seen = [], set()
            for item in archive:
                key = tuple(np.round(item['parameters'], 8))
                if key not in seen:
                    seen.add(key)
                    profiles.append(item['parameters'])
                if len(profiles) == 24:
                    break
            np.savez_compressed(folder/'policy.npz', parameters=np.array(profiles), direction=direction)
            (folder/'progress.json').write_text(json.dumps(history, indent=2))
        ranked.sort(key=lambda item: item[0], reverse=True)
        elite = np.array([p for _, p in ranked[:max(2, len(ranked)//3)]])
        center = .5*center+.5*elite.mean(axis=0)
        deviation = np.maximum(.6*deviation+.4*elite.std(axis=0), .05)
        iteration += 1
        print(f'SPIN round={iteration} steps={consumed} best={best_score[:2]}', flush=True)
    (folder/'training.json').write_text(json.dumps(dict(
        task='spin', algorithm='CEM phase-feedback policy search', seed=seed,
        steps=consumed, direction=direction, population=population,
        success='three consecutive free rotations and no body fall',
        parameter_names=['lateral_radius', 'vertical_radius', 'frequency', 'locked_phase',
                         'phase_feedback', 'height_center', 'vertical_phase', 'speed_cap']), indent=2))
    env.close()
    return folder


def play_spin(policy, seed=3000):
    import time
    import mujoco.viewer
    env = SpinHumanoid(direction=policy.direction)
    env.reset(seed=seed)
    print('Selecting a learned controller...', flush=True)
    policy.select(env)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 10, -10, 3.
        while viewer.is_running():
            start = time.monotonic()
            _, _, done, timeout, info = env.step(policy.predict(env))
            viewer.cam.lookat[:] = [*env.data.body('pelvis').xpos[:2], .85]
            viewer.sync()
            if done or timeout:
                print(f'Consecutive turns: {info["best_consecutive_turns"]:.2f}; fell: {done}', flush=True)
                seed += 1
                env.reset(seed=seed)
                policy.select(env)
            time.sleep(max(0., env.dt-(time.monotonic()-start)))
    env.close()


def dispatch(args, parser):
    if args.command == 'env':
        from .viewer import manual
        manual(args.seed, task='spin')
    elif args.command == 'train':
        if args.steps < 1 or args.population < 4 or (args.rounds is not None and args.rounds < 1):
            parser.error('Positive steps/rounds and population >= 4 required')
        train_spin(args.output, args.steps, args.population, args.seed,
                   args.pretrained, max_rounds=args.rounds)
    elif args.command == 'play':
        play_spin(SpinPolicy(args.model), args.seed)
    elif args.command == 'evaluate':
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        results = evaluate_spin(SpinPolicy(args.model), args.episodes, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f'Successes: {sum(r["success"] for r in results)}/{len(results)}')
    else:
        from .video import render_presentation
        render_presentation(args.model, args.output, args.seed, task='spin')


def evaluate_spin(policy, episodes=10, seed=3000, duration=20., plan=True):
    results = []
    for i in range(episodes):
        env = SpinHumanoid(duration=duration, direction=policy.direction)
        env.reset(seed=seed+i)
        selection = policy.select(env, min(20., duration)) if plan else None
        while True:
            _, _, done, timeout, info = env.step(policy.predict(env))
            if done or timeout:
                break
        result = dict(seed=seed+i, **info, fell=done,
                      success=not done and info['best_consecutive_turns'] >= 3., plan=selection)
        results.append(result)
        print(f'SPIN seed={seed+i} turns={info["signed_turns"]:.2f} '
              f'consecutive={info["best_consecutive_turns"]:.2f} success={result["success"]}', flush=True)
        env.close()
    return results
