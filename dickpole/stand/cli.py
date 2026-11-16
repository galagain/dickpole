"""Route the restored stand environment and its original trained PPO/LQR library."""
from pathlib import Path
import json
import torch

ROOT=Path(__file__).resolve().parents[2]
PRETRAINED=ROOT/'models/stand/pretrained'


def run(args,parser):
    if args.command=='train':
        args.steps=262144 if args.steps is None else args.steps
        args.learning_rate=3e-5 if args.learning_rate is None else args.learning_rate
        args.rounds=6 if args.rounds is None else args.rounds
        args.output=args.output or ROOT/'outputs/stand'
        if args.pretrained==Path('__included__'):args.pretrained=PRETRAINED
    elif args.command in ('play','evaluate','video'):
        args.model=args.model or PRETRAINED
        if args.command=='evaluate':args.output=args.output or ROOT/'outputs/stand_evaluation.json'
        if args.command=='video':args.output=args.output or ROOT/'media/stand.mp4'
    torch.set_num_threads(1)
    if args.command == 'env':
        from dickpole.stand.viewer import manual
        manual(args.seed)
    elif args.command == 'play':
        from dickpole.stand.policy import PolicyLibrary, play
        play(PolicyLibrary(args.model), args.seed, plan=not args.no_plan)
    elif args.command == 'train':
        if args.steps < 1 or args.rounds < 1 or args.population < 4 or args.learning_rate <= 0:
            parser.error('Positive steps/rounds/learning rate and population >= 4 required')
        from dickpole.stand.policy import PolicyLibrary, build_library
        from dickpole.stand.training import train
        from dickpole.stand.refinement import refine
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
        from dickpole.stand.policy import PolicyLibrary, run_evaluation
        records = run_evaluation(PolicyLibrary(args.model), args.episodes, args.seed, 40., plan=not args.no_plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2))
        print(f"Réussites : {sum(r['success'] for r in records)}/{len(records)}")
    else:
        from dickpole.stand.video import render_presentation
        render_presentation(args.model, args.output, args.seed, plan=not args.no_plan)
