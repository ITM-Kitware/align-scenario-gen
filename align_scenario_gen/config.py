from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    if _uses_behavior_library(config):
        config = _resolve_behavior_library(config)
    else:
        config = _normalize_legacy_config(config)

    behavior_name = config["behavior"]["name"]
    behavior_id = config["_resolved"]["behavior_id"]

    config.setdefault("temperature", 0.8)
    config.setdefault("max_tokens", 4000)
    config.setdefault("scenario_id", behavior_id)
    config.setdefault("output", f"output/{behavior_id}/scenarios.json")
    config.setdefault("evaluate", {})
    config["evaluate"].setdefault("input", config["output"])
    config["evaluate"].setdefault("output", f"output/{behavior_id}/eval_results.json")

    config["_config_path"] = str(config_path)
    config["_derived"] = {
        "examples_dir": f"bloom-data/behaviors/examples/{behavior_name}",
        "bloom_results_dir": f"bloom-results/{behavior_name}",
        "ideation_file": f"bloom-results/{behavior_name}/ideation.json",
        "seed_file": "bloom-data/seed.yaml",
        "config_dir": "bloom-data",
    }

    return config


def _uses_behavior_library(config: dict) -> bool:
    return all(key in config for key in ("run", "frames", "behaviors"))


def _normalize_legacy_config(config: dict) -> dict:
    config = deepcopy(config)
    behavior_name = config["behavior"]["name"]
    frame = {
        "prompt_template": "two_patient_triage",
        "action_topology": "choose_between_two_patients",
        "choices": [{"label": label, "action_type": "SITREP"} for label in config["behavior"]["choices"]],
    }
    config["_resolved"] = {
        "behavior_id": behavior_name,
        "behavior": {"id": behavior_name},
        "frame_id": frame["prompt_template"],
        "frame": frame,
    }
    return config


def _resolve_behavior_library(config: dict) -> dict:
    resolved = deepcopy(config)

    behavior_id = resolved["run"]["behavior_id"]
    behavior_spec = deepcopy(resolved["behaviors"][behavior_id])
    frame_id = behavior_spec["frame"]
    frame_spec = _normalize_frame(frame_id, deepcopy(resolved["frames"][frame_id]))

    resolved["behavior"] = {
        "name": behavior_id,
        "choices": [choice["label"] for choice in frame_spec["choices"]],
    }

    if "examples" in behavior_spec:
        resolved["examples_source"] = deepcopy(behavior_spec["examples"])

    ideation_cfg = deepcopy(resolved.get("ideation", {}))
    if "variation_dimensions" in behavior_spec:
        ideation_cfg["variation_dimensions"] = list(behavior_spec["variation_dimensions"])
    resolved["ideation"] = ideation_cfg

    if "scenario_id" in behavior_spec:
        resolved["scenario_id"] = behavior_spec["scenario_id"]
    if "output" in behavior_spec:
        resolved["output"] = behavior_spec["output"]

    evaluate_cfg = deepcopy(resolved.get("evaluate", {}))
    if "evaluate" in behavior_spec:
        evaluate_cfg = _deep_merge(evaluate_cfg, behavior_spec["evaluate"])
    resolved["evaluate"] = evaluate_cfg

    behavior_spec["id"] = behavior_id
    resolved["_resolved"] = {
        "behavior_id": behavior_id,
        "behavior": behavior_spec,
        "frame_id": frame_id,
        "frame": frame_spec,
    }
    return resolved


def _normalize_frame(frame_id: str, frame_spec: dict) -> dict:
    normalized_choices = []
    for index, choice in enumerate(frame_spec.get("choices", [])):
        if isinstance(choice, str):
            normalized = {"label": choice, "action_type": "SITREP"}
        else:
            normalized = deepcopy(choice)
        normalized.setdefault("label", f"Choice {index + 1}")
        normalized.setdefault("action_type", "SITREP")
        if frame_id == "two_patient_triage":
            normalized.setdefault("character_id", f"Patient {chr(65 + index)}")
        normalized_choices.append(normalized)
    frame_spec["choices"] = normalized_choices
    return frame_spec


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
