"""Command-line entry point for experiment modules."""

import argparse

from comparing_dynamics.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Path to a JSON configuration file produced by Config.save_to_file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config is None:
        raise SystemExit("Pass --config with an experiment configuration file.")

    config = Config.load_from_file(args.config)
    print(config)


if __name__ == "__main__":
    main()
