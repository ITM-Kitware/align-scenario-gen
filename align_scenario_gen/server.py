"""Shared llama.cpp server management."""

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from fnmatch import fnmatch


def _resolve_model_path(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    matches = [f for f in files if fnmatch(f, filename)]
    if not matches:
        print(f"No files matching '{filename}' in {repo_id}", file=sys.stderr)
        sys.exit(1)
    return hf_hub_download(repo_id=repo_id, filename=matches[0])


def _wait_for_server(url: str, timeout: int = 300):
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


@contextmanager
def llm_server(local_model: dict, port: int = 8000):
    """Context manager that starts a llama.cpp server and yields the base URL."""
    repo_id = local_model["repo_id"]
    filename = local_model["filename"]
    n_ctx = local_model.get("n_ctx", 4096)

    print(f"Resolving model {repo_id} ({filename})...")
    model_path = _resolve_model_path(repo_id, filename)

    env = os.environ.copy()
    if "main_gpu" in local_model:
        env["CUDA_VISIBLE_DEVICES"] = str(local_model["main_gpu"])

    gpu_info = env.get("CUDA_VISIBLE_DEVICES", "all")
    base_url = f"http://localhost:{port}"
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
        _wait_for_server(base_url)
        print("Server ready.")
        yield base_url
    finally:
        server.terminate()
        server.wait()
