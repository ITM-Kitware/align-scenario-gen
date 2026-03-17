from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    name = config["behavior"]["name"]
    config.setdefault("scenario_id", "generated")
    config.setdefault("temperature", 0.8)
    config.setdefault("max_tokens", 4000)
    config.setdefault("output", "output/scenarios.json")
    config.setdefault("evaluate", {})
    config["evaluate"].setdefault("input", config["output"])
    config["evaluate"].setdefault("output", "output/eval_results.json")

    config["_derived"] = {
        "examples_dir": f"bloom-data/behaviors/examples/{name}",
        "bloom_results_dir": f"bloom-results/{name}",
        "ideation_file": f"bloom-results/{name}/ideation.json",
        "seed_file": "bloom-data/seed.yaml",
        "config_dir": "bloom-data",
    }

    return config
