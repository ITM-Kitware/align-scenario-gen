import argparse

from .config import load_config
from .generate import run


def main():
    parser = argparse.ArgumentParser(
        description="Generate decision scenarios from bloom ideation output"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    for key in ("model", "output"):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    run(config)
