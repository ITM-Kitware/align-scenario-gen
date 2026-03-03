# align-scenario-gen

Generates decision-making scenarios for [align-system](https://github.com/ITM-Kitware/align-system) ADM evaluation. Uses [bloom](https://github.com/safety-research/bloom) to ideate diverse scenarios from example data, then an LLM to flesh them out into structured InputOutputFile format.

## Pipeline

```
existing experiment data
  → convert_examples.py → bloom example transcripts
  → run_bloom.py         → understanding + ideation (scenario ideas with variations)
  → align-scenario-gen   → InputOutputFile JSON (narrative + choices)
```

## Setup

```bash
uv sync
```

## Step 1: Prepare example data

Convert existing align-system experiment results into bloom example transcripts. Edit `convert_examples.py` to point at your experiment directories:

```python
EXPERIMENTS_DIR = Path("/path/to/your/experiments")
MERIT_DIRS = sorted(EXPERIMENTS_DIR.glob("your-experiment-pattern/*"))
OUTPUT_DIR = Path("bloom-data/behaviors/examples/your-behavior-name")
```

Then run:

```bash
uv run python convert_examples.py
```

This writes example transcripts to `bloom-data/behaviors/examples/<behavior>/`.

## Step 2: Configure the bloom seed

Edit `bloom-data/seed.yaml`:

```yaml
behavior:
  name: "merit-based-triage"       # behavior name (used as output directory)
  choices:                          # fixed choices reused across all scenarios
    - "Patient A"
    - "Patient B"
  examples:                         # transcript files from step 1
    - "example1.json"
    - "example2.json"

understanding:
  model: "openai/local-model"       # or a bloom model name like "claude-sonnet-4"
  max_tokens: 2000

ideation:
  model: "openai/local-model"
  num_scenarios: 3                  # base scenarios (each gets variations)
  variation_dimensions:             # dimensions to systematically vary
    - "conflicting_kdmas"
    - "information_ambiguity"
  max_tokens: 4000
```

Key fields:
- `behavior.name` — identifies the domain
- `behavior.choices` — the fixed action labels for all generated scenarios
- `ideation.num_scenarios` — number of base scenarios (total = num_scenarios × (1 + len(variation_dimensions)))
- `ideation.variation_dimensions` — each base scenario gets one variation per dimension

## Step 3: Run bloom (understanding + ideation)

If using a local model, set the endpoint:

```bash
export OPENAI_API_KEY="not-needed"
export OPENAI_API_BASE="http://localhost:8000/v1"
```

Then run:

```bash
uv run python run_bloom.py
```

Output goes to `bloom-results/<behavior-name>/`:
- `understanding.json` — bloom's analysis of the behavior from examples
- `ideation.json` — scenario ideas with systematic variations

## Step 4: Configure scenario generation

Create a config YAML (or edit `example_config.yaml`):

```yaml
seed_file: bloom-data/seed.yaml
ideation_file: bloom-results/merit-based-triage/ideation.json
model: claude-sonnet-4
scenario_id: generated-merit
temperature: 1.0
max_tokens: 4000
output: output/scenarios.json
```

For local models:

```yaml
seed_file: bloom-data/seed.yaml
ideation_file: bloom-results/merit-based-triage/ideation.json
model: local
local_model:
  repo_id: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  filename: "*Q4_K_M.gguf"
  n_ctx: 4096
  n_gpu_layers: -1    # -1 = all on GPU, 0 = CPU only
  main_gpu: 0
scenario_id: generated-merit
temperature: 0.8
max_tokens: 2000
output: output/scenarios.json
```

Config reference:

| Key | Description |
|---|---|
| `seed_file` | Path to bloom seed.yaml (reads choices from here) |
| `ideation_file` | Path to bloom ideation output |
| `model` | `claude-sonnet-4`, `gpt-4.1`, or `local` for GGUF |
| `scenario_id` | Groups scenarios in align-system |
| `temperature` | LLM creativity (0.0-2.0) |
| `max_tokens` | Max response length |
| `output` | Output JSON path |

## Step 5: Generate scenarios

```bash
uv run align-scenario-gen example_config.yaml
```

CLI overrides:

```bash
uv run align-scenario-gen example_config.yaml --model gpt-4.1 --output output/gpt_scenarios.json
```

Output is an align-system InputOutputFile JSON array, ready to use with the `minimal` hydration domain.
