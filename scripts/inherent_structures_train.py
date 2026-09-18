# """Run one Langevin-to-inherent-structure trajectory from a Python config."""

# import argparse
# import json

# import torch

# from comparing_dynamics.config import Config
# from comparing_dynamics.experiments.inherent_structures import (
#   InherentStructureConfig,
#   run_inherent_structure,
# )


# def main() -> None:
#   parser = argparse.ArgumentParser(description=__doc__)
#   parser.add_argument("--config", required=True, help="JSON config written by Config.save_to_file")
#   parser.add_argument("--temperature", type=float, required=True)
#   parser.add_argument("--equilibration-steps", type=int, required=True)
#   parser.add_argument("--quench-steps", type=int, required=True)
#   parser.add_argument("--langevin-lr", type=float, required=True)
#   parser.add_argument("--quench-lr", type=float, required=True)
#   parser.add_argument("--device", default="cpu")
#   parser.add_argument("--run-name", help="W&B run name used in checkpoint filenames")
#   parser.add_argument("--save-dir", required=True, help="Directory for named checkpoints")
#   args = parser.parse_args()

#   cfg = Config.load_from_file(args.config)
#   result = run_inherent_structure(
#     cfg,
#     InherentStructureConfig(
#       temperature=args.temperature,
#       equilibration_steps=args.equilibration_steps,
#       quench_steps=args.quench_steps,
#       langevin_lr=args.langevin_lr,
#       quench_lr=args.quench_lr,
#       seed=cfg.initial_seed,
#     ),
#     device=torch.device(args.device),
#     save_dir=args.save_dir,
#     run_name=args.run_name,
#   )
#   print(json.dumps({key: value for key, value in result.items() if isinstance(value, (float, int))}, indent=2))


# if __name__ == "__main__":
#   main()
