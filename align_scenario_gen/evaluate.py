"""Run generated scenarios through align-system ADMs."""

import json
from pathlib import Path
from types import SimpleNamespace


def _hydrate(record):
    """Hydrate a scenario record into state + actions for align-system."""
    full_state = record["full_state"]

    state = SimpleNamespace(
        unstructured=full_state["unstructured"],
        scenario_complete=full_state["scenario_complete"],
        meta_info=SimpleNamespace(scene_id=full_state["meta_info"]["scene_id"]),
        characters=[],
    )
    actions = [
        SimpleNamespace(
            action_id=a["action_id"],
            action_type=a.get("action_type", "SITREP"),
            unstructured=a["unstructured"],
            justification=a.get("justification"),
            kdma_association=a.get("kdma_association"),
        )
        for a in record["choices"]
    ]
    return state, actions


def _get_icl_data_paths() -> dict:
    """Get absolute paths to ICL data files, following align-app's pattern."""
    import align_system

    icl_base = Path(align_system.__file__).parent / "resources" / "icl" / "phase2"
    return {
        "medical": str(icl_base / "July2025-MU-train_20250804.json"),
        "affiliation": str(icl_base / "July2025-AF-train_20250804.json"),
        "merit": str(icl_base / "July2025-MF-train_20250804.json"),
        "personal_safety": str(icl_base / "July2025-PS-train_20250804.json"),
        "search": str(icl_base / "July2025-SS-train_20250804.json"),
    }


def _resolve_icl_paths(adm_config: dict) -> dict:
    """Replace relative ICL dataset paths with absolute paths in ADM config."""
    step_defs = adm_config.get("step_definitions", {})
    icl_step = step_defs.get("regression_icl", {})
    icl_gen = icl_step.get("icl_generator_partial", {})
    settings = icl_gen.get("incontext_settings", {})

    if "datasets" in settings:
        settings["datasets"] = _get_icl_data_paths()

    return adm_config


def _load_adm_config(adm_name: str) -> dict:
    """Load ADM config using align-app's Hydra pattern."""
    from omegaconf import OmegaConf

    import align_system
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path(align_system.__file__).parent / "configs")

    hydra_instance = GlobalHydra.instance()
    if hydra_instance.is_initialized():
        hydra_instance.clear()

    with initialize_config_dir(config_dir, version_base=None):
        cfg = compose(config_name=f"adm/{adm_name}")
        result = OmegaConf.to_container(cfg)

    # Hydra compose wraps in extra "adm" key — unwrap it
    adm_config = result.get("adm", result)
    adm_config = _resolve_icl_paths(adm_config)
    return adm_config


def _instantiate_adm(adm_config: dict):
    """Instantiate ADM from config, following align-app's executor pattern."""
    from omegaconf import OmegaConf
    from align_system.utils.hydra_utils import initialize_with_custom_references

    if OmegaConf.has_resolver("ref"):
        OmegaConf.clear_resolver("ref")

    adm = initialize_with_custom_references({"adm": adm_config})["adm"]
    instance = adm.instance
    inference_kwargs = dict(adm.get("inference_kwargs", {}))
    return instance, inference_kwargs


def _load_alignment_target(target):
    """Load an alignment target from name or inline dict."""
    if isinstance(target, dict):
        return target

    import align_system
    import yaml

    config_dir = Path(align_system.__file__).parent / "configs" / "alignment_target"
    for candidate in [config_dir / f"{target}.yaml", config_dir / f"feb2026/{target}.yaml"]:
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f)
    return None


def _run_adm(adm_name, records, alignment_target=None):
    """Run a single ADM on all records, return list of result dicts."""
    print(f"\n{'='*60}")
    print(f"Loading ADM: {adm_name}...")
    print(f"{'='*60}")

    adm_config = _load_adm_config(adm_name)
    instance, inference_kwargs = _instantiate_adm(adm_config)
    choose_fn = getattr(instance, "top_level_choose_action", instance.choose_action)

    results = []
    for i, record in enumerate(records):
        inp = record["input"]
        print(f"  [{adm_name}] Scenario {i + 1}/{len(records)}...")

        state, actions = _hydrate(inp)

        raw_result = choose_fn(
            scenario_state=state,
            available_actions=actions,
            alignment_target=alignment_target,
            **inference_kwargs,
        )

        if isinstance(raw_result, tuple):
            action, choice_info = raw_result
        else:
            action, choice_info = raw_result, {}

        results.append({
            "scene_id": inp["full_state"]["meta_info"]["scene_id"],
            "chosen_action": action.unstructured,
            "justification": getattr(action, "justification", None),
        })

    return results


def run_evaluate(config: dict):
    evaluate_cfg = config.get("evaluate", {})
    scenarios_path = Path(evaluate_cfg.get("input", config["output"]))
    output_path = Path(evaluate_cfg.get("output", "output/eval_results.json"))

    # Support single ADM or list of ADMs, each with optional alignment_target
    adm_cfg = evaluate_cfg.get("adm", "random")
    if isinstance(adm_cfg, list):
        adm_entries = [
            (a["name"], a.get("alignment_target")) if isinstance(a, dict) else (a, None)
            for a in adm_cfg
        ]
    else:
        adm_entries = [(adm_cfg, evaluate_cfg.get("alignment_target"))]

    records = json.loads(scenarios_path.read_text())
    print(f"Loaded {len(records)} scenarios from {scenarios_path}")

    all_results = {}
    for adm_name, target_name in adm_entries:
        target = _load_alignment_target(target_name) if target_name else None
        all_results[adm_name] = _run_adm(adm_name, records, target)

    # Build comparison output
    comparison = []
    for i, record in enumerate(records):
        scene_id = record["input"]["full_state"]["meta_info"]["scene_id"]
        scenario_text = record["input"]["full_state"]["unstructured"]

        entry = {
            "scene_id": scene_id,
            "scenario": scenario_text,
            "adm_results": {},
        }
        for adm_name, _ in adm_entries:
            r = all_results[adm_name][i]
            entry["adm_results"][adm_name] = {
                "chosen_action": r["chosen_action"],
                "justification": r["justification"],
            }
        comparison.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for entry in comparison:
        print(f"\nScene {entry['scene_id']}:")
        for adm_name, _ in adm_entries:
            choice = entry["adm_results"][adm_name]["chosen_action"]
            print(f"  {adm_name:50s} → {choice}")

    print(f"\nWrote {len(comparison)} results to {output_path}")
