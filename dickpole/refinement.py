"""Refine a PPO actor with cross-entropy policy search on separate validation seeds."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO

from .training import evaluate


def refine(model_path, out, rounds=6, population=12, episodes=5, seed=19):
    """Search complementary PPO output heads for the final policy library."""
    if rounds < 1 or population < 4 or episodes < 1:
        raise ValueError('Positive rounds/episodes and population >= 4 required')
    args = SimpleNamespace(model=str(model_path), out=str(out), rounds=rounds,
                           population=population, episodes=episodes, seed=seed)
    torch.set_num_threads(1)
    model = PPO.load(args.model, device='cpu')
    layer = model.policy.action_net
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(6)
    deviation = np.array([.4]*3+[.08]*3)
    best_score = (-1., -1., -float('inf'))
    best_parameters = center.copy()
    history = []

    def apply(parameters):
        gain = torch.as_tensor(np.exp(parameters[:3]), dtype=layer.weight.dtype)
        bias = torch.as_tensor(parameters[3:], dtype=layer.bias.dtype)
        with torch.no_grad():
            layer.weight.copy_(original_weight*gain[:, None])
            layer.bias.copy_(original_bias*gain+bias)

    for iteration in range(args.rounds):
        population = rng.normal(center, deviation, (args.population, 6))
        population[:, :3] = np.clip(population[:, :3], -1.5, 1.)
        population[:, 3:] = np.clip(population[:, 3:], -.2, .2)
        population[0] = best_parameters
        ranked = []
        for parameters in population:
            apply(parameters)
            # 2000+ seeds are held out until the final evaluation.
            results = evaluate(model, args.episodes, seed=1000)
            score = (float(np.mean([r['success'] for r in results])),
                     float(np.mean([r['best_hold'] for r in results])),
                     float(np.mean([r['return'] for r in results])))
            ranked.append((score, parameters.copy()))
            history.append(dict(round=iteration, score=score, parameters=parameters.tolist(), episodes=results))
            if score > best_score:
                best_score, best_parameters = score, parameters.copy()
                model.save(str(folder/'best_model'))
                print(f'NEW BEST validation success={score[0]:.0%} hold={score[1]:.2f}s', flush=True)
            (folder/'search.json').write_text(json.dumps(history, indent=2))
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        elite = np.array([p for _, p in ranked[:max(3, args.population//3)]])
        center = .5*center+.5*elite.mean(axis=0)
        deviation = np.maximum(.6*deviation+.4*elite.std(axis=0), [.05]*3+[.01]*3)
        print(f'ROUND {iteration+1}/{args.rounds} best={best_score}', flush=True)
    best = PPO.load(folder/'best_model.zip', device='cpu')
    best.policy.optimizer.state.clear()
    best.save(str(folder/'best_model'))
    results = evaluate(best, episodes=10, seed=2000, duration=40.)
    (folder/'heldout_40s.json').write_text(json.dumps(results, indent=2))
    (folder/'refinement.json').write_text(json.dumps(dict(
        source=args.model, method='CEM on PPO final-layer gains/biases',
        validation_seed=1000, validation_episodes=args.episodes,
        best_parameters=best_parameters.tolist(), best_score=best_score,
        seed=args.seed, rounds=args.rounds, population=args.population), indent=2))
    print(json.dumps(dict(heldout_successes=sum(r['success'] for r in results),
                         episodes=len(results), holds=[r['best_hold'] for r in results])))

    return folder / "search.json"
