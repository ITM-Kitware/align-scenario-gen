"""Convert align-system input_output.json records into bloom example transcripts."""

import json
import shutil
from pathlib import Path


def record_to_transcript(record: dict, kdma_name: str) -> dict | None:
    inp = record.get("input", {})
    out = record.get("output")
    if not out or not inp.get("full_state", {}).get("unstructured"):
        return None

    narrative = inp["full_state"]["unstructured"]
    choices = inp.get("choices", [])
    choice_text = "\n".join(f"- {c['unstructured']}" for c in choices)
    user_content = f"{narrative}\n\nAvailable actions:\n{choice_text}"

    action = out.get("action", {})
    chosen = action.get("unstructured", "")
    justification = action.get("justification", "")
    kdma_scores = action.get("kdma_association", {})
    alignment_val = kdma_scores.get(kdma_name)

    assistant_content = f"I choose: {chosen}\n\nReasoning: {justification}"
    if alignment_val is not None:
        label = kdma_name.replace("_", " ").title()
        assistant_content += f"\n\n[{label} alignment: {alignment_val}]"

    return {
        "conversation": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def run_convert(config: dict):
    examples_source = config["examples_source"]
    output_dir = Path(config["_derived"]["examples_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("example*.json"):
        stale_file.unlink()

    source_type = examples_source.get("source", "experiments")
    if source_type == "manual":
        _copy_manual_examples(config, examples_source, output_dir)
        return
    if source_type != "experiments":
        raise ValueError(f"Unsupported examples source: {source_type}")

    experiments_dir = Path(examples_source["experiments_dir"])
    pattern = examples_source["pattern"]
    max_examples = examples_source.get("max_examples", 6)
    kdma_name = examples_source["kdma"]

    exp_dirs = sorted(experiments_dir.glob(pattern))
    example_num = 0

    for exp_dir in exp_dirs:
        io_file = exp_dir / "input_output.json"
        if not io_file.exists():
            continue

        records = json.loads(io_file.read_text())
        for record in records[:2]:
            transcript = record_to_transcript(record, kdma_name)
            if not transcript:
                continue

            example_num += 1
            out_path = output_dir / f"example{example_num}.json"
            out_path.write_text(json.dumps(transcript, indent=2))

            if example_num >= max_examples:
                print(f"Wrote {example_num} examples to {output_dir}")
                return

    print(f"Wrote {example_num} examples to {output_dir}")


def _copy_manual_examples(config: dict, examples_source: dict, output_dir: Path):
    config_dir = Path(config["_config_path"]).parent
    paths = examples_source.get("paths", [])
    for index, raw_path in enumerate(paths, start=1):
        src = Path(raw_path)
        if not src.is_absolute():
            src = (config_dir / src).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Manual example not found: {src}")
        dst = output_dir / f"example{index}.json"
        shutil.copyfile(src, dst)
    print(f"Copied {len(paths)} manual examples to {output_dir}")
