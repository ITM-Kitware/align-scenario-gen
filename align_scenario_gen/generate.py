import json
from pathlib import Path

from .convert import scenario_to_record
from .parse import parse_scenario_json
from .prompt import SYSTEM_PROMPT, build_user_prompt


def _call_llm(config: dict, messages: list[dict]) -> str:
    if config["model"] == "local":
        from .local_llm import local_chat

        return local_chat(
            config,
            messages,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )

    from bloom.utils import get_model_id, litellm_chat, parse_message

    model_id = get_model_id(config["model"])
    response = litellm_chat(
        model_id=model_id,
        messages=messages,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )
    return parse_message(response)["content"]


def _load_ideation(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["variations"]


def _load_seed_choices(path: str | Path) -> list[str]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    return data["behavior"]["choices"]


def _generate_from_ideation(config: dict) -> list[dict]:
    variations = _load_ideation(config["ideation_file"])
    choices = _load_seed_choices(config["seed_file"])
    scenario_id = config["scenario_id"]
    max_retries = 3
    records = []

    for i, variation in enumerate(variations):
        description = variation["description"]
        print(f"Generating scenario {i + 1}/{len(variations)}...")
        user_prompt = build_user_prompt(description, choices)
        messages = [{"role": "user", "content": user_prompt}]

        for attempt in range(max_retries):
            content = _call_llm(config, messages)
            try:
                scenario = parse_scenario_json(content)
                break
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    print(f"  Parse failed ({e}), retrying...")
                else:
                    raise

        scenario["choices"] = [{"label": c} for c in choices]
        records.append(scenario_to_record(scenario, scenario_id, i))

    return records


def run(config: dict) -> list[dict]:
    records = _generate_from_ideation(config)

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} scenarios to {output_path}")
    return records
