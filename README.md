# align-scenario-gen

Generates decision-making scenarios for [align-system](https://github.com/ITM-Kitware/align-system) ADM evaluation. Uses [bloom](https://github.com/safety-research/bloom) to ideate diverse scenarios from example data, then an LLM to flesh them out into structured InputOutputFile format.

## Pipeline

```
existing experiment data
  → convert   → bloom example transcripts
  → bloom     → understanding + ideation (scenario ideas with variations)
  → generate  → InputOutputFile JSON (narrative + choices)
```

One config file (`config.yaml`) drives all three steps. One command runs the full pipeline or individual steps.

## Setup

```bash
uv sync
```

## Config

Edit `config.yaml`:

```yaml
behavior:
  name: "merit-based-triage" # used to derive intermediate paths
  choices: # fixed choices reused across all scenarios
    - "Patient A"
    - "Patient B"

examples_source: # where to find existing experiment results
  experiments_dir: /path/to/experiments
  pattern: "phase2_baseline/Feb2026-MF-*"
  max_examples: 6

local_model: # HuggingFace GGUF model, used by all steps
  repo_id: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  filename: "*Q4_K_M.gguf"
  # main_gpu: 0       # which GPU to use (matches nvidia-smi index)
  # n_ctx: 4096

temperature: 0.8
max_tokens: 4000

ideation:
  num_scenarios: 3 # base scenarios (total = num_scenarios × (1 + len(variation_dimensions)))
  variation_dimensions: # each base scenario gets one variation per dimension
    - "conflicting_kdmas"
    - "information_ambiguity"

scenario_id: generated-merit # groups scenarios in align-system
output: output/scenarios.json
```

Paths derived from `behavior.name`:

- Examples: `bloom-data/behaviors/examples/<name>/`
- Bloom results: `bloom-results/<name>/`
- Ideation file: `bloom-results/<name>/ideation.json`

## Usage

Run the full pipeline:

```bash
uv run align-scenario-gen config.yaml
```

Run individual steps:

```bash
uv run align-scenario-gen config.yaml --step convert    # extract examples from experiments
uv run align-scenario-gen config.yaml --step bloom       # understanding + ideation
uv run align-scenario-gen config.yaml --step generate    # generate scenario narratives
```

### What each step does

**convert** — Reads `examples_source` from config, extracts align-system experiment results into bloom example transcripts at `bloom-data/behaviors/examples/<name>/`.

**bloom** — Writes a `bloom-data/seed.yaml` from the unified config (injecting bloom-specific model keys), downloads the model (to `~/.cache/huggingface/`), starts a local llama.cpp server, runs bloom's understanding and ideation stages, then shuts down the server. Output goes to `bloom-results/<name>/`.

**generate** — Reads ideation output, generates vivid narrative scenarios via the local model, and writes align-system InputOutputFile JSON to the configured output path.

Output is an align-system InputOutputFile JSON array, ready to use with the `minimal` hydration domain.
