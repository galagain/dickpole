"""Residual PPO training and physical evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .environment import MuscleHumanoid, model_xml
from .rl_env import ResidualHumanoid
from .controller import SwingUpTeacher


class TaskEvaluation(BaseCallback):
    """Select by real task success/hold first, reward only breaks ties."""
    def __init__(self, folder, frequency=4096):
        super().__init__()
        self.folder = folder
        self.frequency = frequency
        self.best = (-1., -1., -float('inf'))
        self.history = []

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        self._evaluate()
        return True

    def _on_training_start(self):
        self._evaluate()

    def _evaluate(self):
        results = evaluate(self.model, episodes=5, seed=1000)
        score = (float(np.mean([r['success'] for r in results])),
                 float(np.mean([r['best_hold'] for r in results])),
                 float(np.mean([r['return'] for r in results])))
        self.history.append(dict(steps=self.num_timesteps, score=score, episodes=results))
        (self.folder/'progress.json').write_text(json.dumps(self.history, indent=2))
        self.model.save(str(self.folder/'last_model'))
        if score > self.best:
            self.best = score
            self.model.save(str(self.folder/'best_model'))
        print(f'EVAL steps={self.num_timesteps} success={score[0]:.0%} '
              f'hold={score[1]:.2f}s return={score[2]:.2f}', flush=True)


def evaluate(model, episodes=5, seed=2000, duration=20.):
    records = []
    for i in range(episodes):
        env = MuscleHumanoid(duration) if isinstance(model, SwingUpTeacher) else ResidualHumanoid(duration)
        sim = env.unwrapped
        obs, _ = env.reset(seed=seed+i)
        total = 0.
        max_height = -1.
        while True:
            action = model(sim) if isinstance(model, SwingUpTeacher) else model.predict(obs, deterministic=True)[0]
            obs, reward, done, timeout, info = env.step(action)
            total += info.get('task_reward', reward)
            max_height = max(max_height, float(np.cos(sim.pole_state()[0])))
            if done or timeout:
                records.append({'seed': seed+i, **info, 'return': total,
                                'fell': done, 'max_vertical_cosine': max_height,
                                'success': not done and info['best_hold'] >= 5.})
                break
    return records


def train(model_path, steps=262144, resume=None, seed=7, learning_rate=3e-5, episodes=5):
    """Train residual PPO while preserving the best physically evaluated policy."""
    args = SimpleNamespace(model=str(model_path), steps=steps, resume=resume,
                           seed=seed, learning_rate=learning_rate,
                           episodes=episodes)
    torch.set_num_threads(1)
    folder = Path(model_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    (folder/'humanoid.xml').write_text(model_xml())
    (folder/'training_config.json').write_text(json.dumps({
        'seed': args.seed, 'requested_steps': args.steps, 'resume': args.resume,
        'physics_dt': .002, 'control_dt': MuscleHumanoid.dt,
        'muscle_actuators': 14, 'low_level_controller': 'fixed_postural_reflex',
        'high_level_action': ['pelvis_target_velocity', 'pelvis_tilt_rate', 'pelvis_vertical_velocity'],
        'pelvis_tilt_limit_rad': .30,
        'pelvis_tilt_target_rate_rad_s': 2.4,
        'pelvis_height_target_rate_m_s': .5,
        'algorithm': 'Residual PPO + energy swing-up baseline + local LQR',
        'residual_scale': .25,
        'rl_reward_scale': .01,
        'learning_rate': args.learning_rate,
        'exploration': 'gSDE, resample every 8 decisions',
        'success': 'no fall and at least five continuous seconds upright',
    }, indent=2))
    env = DummyVecEnv([lambda: Monitor(ResidualHumanoid()) for _ in range(4)])
    model = PPO.load(args.resume, env=env, device='cpu', learning_rate=args.learning_rate) if args.resume else PPO(
        'MlpPolicy', env, seed=args.seed, device='cpu', verbose=1,
        n_steps=512, batch_size=256, learning_rate=args.learning_rate, gamma=.998,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-2.), ent_coef=0., target_kl=.025,
        use_sde=True, sde_sample_freq=8)
    if not args.resume:
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    callback = TaskEvaluation(folder)
    try:
        model.learn(args.steps, callback=callback,
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(str(folder/'last_model'))
    if not Path(args.model).exists():
        model.save(args.model)
    results = evaluate(PPO.load(args.model), args.episodes)
    (folder/'evaluation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    env.close()
    return Path(model_path)
