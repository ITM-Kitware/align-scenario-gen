"""Helpers for respecting externally managed GPU visibility."""

import os


def resolve_cuda_visible_devices(local_model: dict, env: dict | None = None) -> dict:
    """Prefer an existing CUDA_VISIBLE_DEVICES setting over config.main_gpu."""
    resolved_env = dict(os.environ if env is None else env)
    if not resolved_env.get("CUDA_VISIBLE_DEVICES") and "main_gpu" in local_model:
        resolved_env["CUDA_VISIBLE_DEVICES"] = str(local_model["main_gpu"])
    return resolved_env


def current_gpu_visibility(env: dict | None = None) -> str:
    return (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES", "all")
