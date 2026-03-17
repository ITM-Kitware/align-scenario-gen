"""Run generated scenarios through an align-system ADM."""

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


def _load_adm(evaluate_cfg: dict):
    """Instantiate an ADM from config, defaulting to RandomADM."""
    from omegaconf import OmegaConf
    from align_system.utils.hydra_utils import initialize_with_custom_references

    adm_config_path = evaluate_cfg.get("adm_config")
    if adm_config_path:
        cfg = OmegaConf.load(adm_config_path)
        adm_config = OmegaConf.to_container(cfg, resolve=True)
    else:
        adm_config = {
            "instance": {
                "_target_": "align_system.algorithms.random_adm.RandomADM",
            }
        }

    result = initialize_with_custom_references({"adm": adm_config})
    adm = result["adm"]
    return adm.instance if hasattr(adm, "instance") else adm


def run_evaluate(config: dict):
    evaluate_cfg = config.get("evaluate", {})
    scenarios_path = Path(evaluate_cfg.get("input", config["output"]))
    output_path = Path(evaluate_cfg.get("output", "output/eval_results.json"))

    records = json.loads(scenarios_path.read_text())
    print(f"Loaded {len(records)} scenarios from {scenarios_path}")

    adm = _load_adm(evaluate_cfg)
    choose_fn = getattr(adm, "top_level_choose_action", adm.choose_action)

    alignment_target = evaluate_cfg.get("alignment_target")

    results = []
    for i, record in enumerate(records):
        inp = record["input"]
        print(f"Evaluating scenario {i + 1}/{len(records)}...")

        state, actions = _hydrate(inp)

        raw_result = choose_fn(
            scenario_state=state,
            available_actions=actions,
            alignment_target=alignment_target,
        )

        if isinstance(raw_result, tuple):
            action, choice_info = raw_result
        else:
            action, choice_info = raw_result, {}

        results.append({
            "scenario_id": inp["scenario_id"],
            "scene_id": inp["full_state"]["meta_info"]["scene_id"],
            "chosen_action": action.unstructured,
            "chosen_action_id": action.action_id,
            "justification": getattr(action, "justification", None),
            "choice_info": choice_info if isinstance(choice_info, dict) else {},
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} evaluation results to {output_path}")
