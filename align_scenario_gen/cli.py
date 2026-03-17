import argparse

from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate decision scenarios from bloom ideation output"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--step",
        choices=["convert", "bloom", "generate", "evaluate"],
        help="Run a single pipeline step (default: run all)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    steps = [args.step] if args.step else ["convert", "bloom", "generate"]

    if "convert" in steps:
        from .convert_examples import run_convert

        run_convert(config)

    if "bloom" in steps:
        from .bloom_runner import run_bloom

        run_bloom(config)

    if "generate" in steps:
        from .generate import run_generate

        run_generate(config)

    if "evaluate" in steps:
        from .evaluate import run_evaluate

        run_evaluate(config)
