"""Run bloom understanding + ideation stages with a local llama-cpp server."""

import os
import subprocess
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

import yaml

from .gpu import current_gpu_visibility, resolve_cuda_visible_devices


def resolve_model_path(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    matches = [f for f in files if fnmatch(f, filename)]
    if not matches:
        print(f"No files matching '{filename}' in {repo_id}", file=sys.stderr)
        print(f"Available files: {files}", file=sys.stderr)
        sys.exit(1)
    return hf_hub_download(repo_id=repo_id, filename=matches[0])


def wait_for_server(url: str, timeout: int = 300):
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/v1/models")
            return
        except Exception:
            time.sleep(1)
    print("Server failed to start", file=sys.stderr)
    sys.exit(1)


def _build_seed(config: dict) -> dict:
    examples_dir = Path(config["_derived"]["examples_dir"])
    behavior_name = config["behavior"]["name"]
    examples = sorted(
        f"{behavior_name}/{f.stem}" for f in examples_dir.glob("example*.json")
    )

    behavior = {**config["behavior"], "examples": examples}
    ideation_cfg = config.get("ideation", {})

    return {
        "behavior": behavior,
        "local_model": config["local_model"],
        "temperature": config.get("temperature", 0.8),
        "evaluator_reasoning_effort": "none",
        "target_reasoning_effort": "none",
        "max_concurrent": 1,
        "debug": True,
        "understanding": {
            "model": "openai/local-model",
            "max_tokens": 2000,
        },
        "ideation": {
            "model": "openai/local-model",
            "num_scenarios": ideation_cfg.get("num_scenarios", 3),
            "variation_dimensions": ideation_cfg.get(
                "variation_dimensions",
                ["conflicting_kdmas", "information_ambiguity"],
            ),
            "max_tokens": config.get("max_tokens", 4000),
            "web_search": False,
        },
    }


def run_bloom(config: dict):
    config_dir = Path(config["_derived"]["config_dir"])
    seed_path = Path(config["_derived"]["seed_file"])

    seed = _build_seed(config)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(yaml.dump(seed, default_flow_style=False, sort_keys=False))

    local_model = config["local_model"]
    repo_id = local_model["repo_id"]
    filename = local_model["filename"]
    n_ctx = local_model.get("n_ctx", 4096)

    print(f"Resolving model {repo_id} ({filename})...")
    model_path = resolve_model_path(repo_id, filename)

    port = 8000
    base_url = f"http://localhost:{port}"

    env = resolve_cuda_visible_devices(local_model, os.environ.copy())

    gpu_info = current_gpu_visibility(env)
    print(f"Starting llama.cpp server on port {port} (CUDA_VISIBLE_DEVICES={gpu_info})...")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            model_path,
            "--n_ctx",
            str(n_ctx),
            "--n_gpu_layers",
            "-1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
    )

    try:
        wait_for_server(base_url)
        print("Server ready.")

        os.environ["OPENAI_API_KEY"] = "not-needed"
        os.environ["OPENAI_API_BASE"] = f"{base_url}/v1"

        from bloom import set_debug_mode, utils
        from bloom.stages.step1_understanding import run_understanding
        from bloom.stages.step2_ideation import run_ideation

        set_debug_mode(True)

        bloom_config = utils.load_config(str(seed_path), config_dir=config_dir)

        print("\n=== Running Understanding Stage ===")
        run_understanding(config=bloom_config, config_dir=config_dir)

        print("\n=== Running Ideation Stage ===")
        run_ideation(config=bloom_config, config_dir=config_dir)

        results_dir = Path(config["_derived"]["bloom_results_dir"])

        # Patch model name in results to reflect actual model
        actual_model = f"{repo_id} ({filename})"
        for json_file in results_dir.glob("*.json"):
            import json
            data = json.loads(json_file.read_text())
            if data.get("model") == "openai/local-model":
                data["model"] = actual_model
                json_file.write_text(json.dumps(data, indent=2))

        print("\n=== Done ===")
        for f in sorted(results_dir.glob("*.json")):
            print(f"Output: {f}")

    finally:
        server.terminate()
        server.wait()
