"""Single entry point for the final DickPole simulation."""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PRETRAINED = ROOT/'models'/'pretrained'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    env = commands.add_parser('env', help='Open the manually controlled environment')
    env.add_argument('--seed', type=int, default=3000)
    play = commands.add_parser('play', help='Run the pretrained final controller')
    play.add_argument('--model', type=Path, default=PRETRAINED)
    play.add_argument('--seed', type=int, default=3000)
    train = commands.add_parser('train', help='Train PPO, refine and build a final model')
    train.add_argument('--steps', type=int, default=262144)
    train.add_argument('--output', type=Path, default=ROOT/'outputs'/'training')
    train.add_argument('--pretrained', nargs='?', const=PRETRAINED, type=Path)
    train.add_argument('--learning-rate', type=float, default=3e-5)
    train.add_argument('--rounds', type=int, default=6)
    train.add_argument('--population', type=int, default=12)
    train.add_argument('--seed', type=int, default=7)
    evaluate = commands.add_parser('evaluate', help='Evaluate real rollouts from below')
    evaluate.add_argument('--model', type=Path, default=PRETRAINED)
    evaluate.add_argument('--episodes', type=int, default=20)
    evaluate.add_argument('--seed', type=int, default=4000)
    evaluate.add_argument('--output', type=Path, default=ROOT/'outputs'/'evaluation.json')
    video = commands.add_parser('video', help='Render 10 seconds of the environment (requires FFmpeg)')
    video.add_argument('--model', type=Path, default=PRETRAINED)
    video.add_argument('--output', type=Path, default=ROOT/'media'/'presentation.mp4')
    video.add_argument('--seed', type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'env':
        from dickpole.viewer import manual
        manual(args.seed)
    elif args.command == 'play':
        from dickpole.policy import PolicyLibrary, play
        play(PolicyLibrary(args.model), args.seed)
    elif args.command == 'train':
        if args.steps < 1 or args.rounds < 1 or args.population < 4 or args.learning_rate <= 0:
            parser.error('Positive steps/rounds/learning rate and population >= 4 required')
        from dickpole.policy import PolicyLibrary, build_library
        from dickpole.training import train
        from dickpole.refinement import refine
        args.output.mkdir(parents=True, exist_ok=True)
        resume = None
        if args.pretrained:
            if args.pretrained.is_file():
                resume = str(args.pretrained)
            else:
                library = PolicyLibrary(args.pretrained)
                library.apply(0)
                library.model.policy.optimizer.state.clear()
                resume = str(args.output/'initial_policy.zip')
                library.model.save(resume)
        print('1/3 — Entraînement PPO', flush=True)
        checkpoint = train(args.output/'ppo'/'best_model.zip', args.steps, resume,
                           args.seed, args.learning_rate)
        print('2/3 — Raffinement des politiques', flush=True)
        search = refine(checkpoint, args.output/'search', args.rounds, args.population,
                        seed=args.seed+12)
        print('3/3 — Construction du modèle final', flush=True)
        build_library(checkpoint, search, args.output)
        print(f'Modèle prêt : {args.output}', flush=True)
    elif args.command == 'evaluate':
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        from dickpole.policy import PolicyLibrary, run_evaluation
        records = run_evaluation(PolicyLibrary(args.model), args.episodes, args.seed, 40.)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2))
        print(f"Réussites : {sum(r['success'] for r in records)}/{len(records)}")
    else:
        from dickpole.video import render_presentation
        render_presentation(args.model, args.output, args.seed)


if __name__ == '__main__':
    main()
