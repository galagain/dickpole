"""Run the original stand controller or the current 3D spin environment."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [('env','Manually move the pelvis'),('play','Watch a trained policy'),
                            ('train','Learn a task-specific controller'),('evaluate','Evaluate real episodes'),
                            ('video','Render ten seconds without overlays')]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--task', choices=['stand','balance','spin'], default='balance')
        command.add_argument('--seed', type=int, default=None)
        if name in ('play','evaluate','video'):
            command.add_argument('--model', type=Path)
            command.add_argument('--no-plan', action='store_true', help='Use profile zero without simulator lookahead')
        if name in ('train','evaluate','video'):
            command.add_argument('--output', type=Path)
        if name == 'train':
            command.add_argument('--steps', type=int)
            command.add_argument('--learning-rate',type=float)
            command.add_argument('--population', type=int, default=16)
            command.add_argument('--episodes-per-candidate', type=int, default=3)
            command.add_argument('--rounds', type=int)
            command.add_argument('--pretrained', nargs='?', const=Path('__included__'), type=Path)
        if name == 'evaluate':
            command.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = (7 if args.command=='train' else 3000) if args.task in ('stand','balance') else (3000 if args.command=='evaluate' else 7)
    if args.task in ('stand','balance'):
        from dickpole.stand.cli import run
        run(args,parser)
        return
    included = ROOT/'models'/args.task
    if hasattr(args, 'model'):
        args.model = args.model or included
    if args.command == 'env':
        from dickpole.viewer import manual
        manual(args.seed, args.task)
    elif args.command == 'train':
        if args.steps is None:
            args.steps=262144
        if args.episodes_per_candidate < 1 or args.steps < 1 or args.population < 4 or (args.rounds is not None and args.rounds < 1):
            parser.error('Positive steps/rounds/episodes and population >= 4 required')
        resume = included if args.pretrained == Path('__included__') else args.pretrained
        output = args.output or ROOT/'outputs'/args.task
        from dickpole.learning import train
        train(args.task, output, args.steps, args.population, args.seed, resume, args.rounds, args.episodes_per_candidate)
        print(f'Checkpoint ready: {output}')
    else:
        from dickpole.policy import load_policy
        policy = load_policy(args.model, args.task)
        if args.command == 'play':
            from dickpole.viewer import play
            play(policy, args.seed, plan=not args.no_plan)
        elif args.command == 'evaluate':
            if args.episodes < 1:
                parser.error('--episodes must be positive')
            from dickpole.policy import evaluate
            records = evaluate(policy, args.episodes, args.seed, plan=not args.no_plan)
            output = args.output or ROOT/'outputs'/f'{args.task}_evaluation.json'
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(records, indent=2))
            print(f'Success: {sum(r["success"] for r in records)}/{len(records)}')
        else:
            from dickpole.video import render_presentation
            render_presentation(policy, args.output or ROOT/'media'/f'{args.task}.mp4',
                                args.seed, plan=not args.no_plan)


if __name__ == '__main__':
    main()
