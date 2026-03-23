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
run:
  behavior_id: merit-vs-medical-two-patient-triage

frames:
  two_patient_triage:
    prompt_template: two_patient_triage
    choices:
      - label: Treat Patient A
      - label: Treat Patient B

behaviors:
  merit-vs-medical-two-patient-triage:
    frame: two_patient_triage
    active_kdmas:
      medical: { favors: Patient A, strength: high }
      merit: { favors: Patient B, strength: high }
    generation_guidance:
      - Patient A should be more medically urgent than Patient B.
      - Patient B should have stronger merit cues than Patient A.
    examples:
      source: experiments
      experiments_dir: /data/shared/phase2_feb2026_results_local/phase2_baseline
      pattern: "Feb2026-MF-*"
      max_examples: 6
      kdma: merit
    variation_dimensions:
      - medical_gap
      - merit_gap

local_model: # HuggingFace GGUF model, used by all steps
  repo_id: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
  filename: "*Q4_K_M.gguf"
  # main_gpu: 0       # which GPU to use (matches nvidia-smi index)
  # n_ctx: 4096

temperature: 0.8
max_tokens: 4000

ideation:
  num_scenarios: 3
```

One run resolves one selected `behavior_id` into:

- a bloom behavior name and description
- a fixed scenario frame
- a fixed intended KDMA tension profile
- a fixed example source
- a fixed variation-dimension set

Paths are derived from the resolved behavior id:

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

**convert** — Reads the selected behavior's `examples` spec, extracts align-system experiment results or copies manual examples into `bloom-data/behaviors/examples/<name>/`.

**bloom** — Writes a `bloom-data/seed.yaml` from the resolved behavior config, downloads the model (to `~/.cache/huggingface/`), starts a local llama.cpp server, runs bloom's understanding and ideation stages, then shuts down the server. Output goes to `bloom-results/<name>/`.

**generate** — Reads ideation output, applies the selected frame and intended KDMA tensions to the local generation prompt, and writes align-system InputOutputFile JSON to the configured output path.

Output is an align-system InputOutputFile JSON array, ready to use with the `minimal` hydration domain.
