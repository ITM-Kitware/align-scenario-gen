import json
import os
from pathlib import Path

from .convert import scenario_to_record
from .parse import parse_scenario_json
from .prompt import SYSTEM_PROMPT, build_user_prompt
from .server import llm_server


def _load_ideation(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["variations"]


def _chat(base_url: str, messages: list[dict], system_prompt: str,
          max_tokens: int, temperature: float) -> str:
    """Chat via the local llama.cpp server using litellm."""
    import litellm

    all_messages = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.extend(messages)

    response = litellm.completion(
        model="openai/local-model",
        messages=all_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key="not-needed",
        api_base=f"{base_url}/v1",
    )
    return response.choices[0].message.content


def run_generate(config: dict) -> list[dict]:
    variations = _load_ideation(config["_derived"]["ideation_file"])
    choices = config["behavior"]["choices"]
    local_model = config["local_model"]
    scenario_id = config["scenario_id"]
    max_retries = 3
    records = []

    with llm_server(local_model) as base_url:
        for i, variation in enumerate(variations):
            description = variation["description"]
            print(f"Generating scenario {i + 1}/{len(variations)}...")
            user_prompt = build_user_prompt(description, choices)
            messages = [{"role": "user", "content": user_prompt}]

            for attempt in range(max_retries):
                content = _chat(
                    base_url,
                    messages,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
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

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} scenarios to {output_path}")
    return records
