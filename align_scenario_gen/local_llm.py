import os

from llama_cpp import Llama

_model_cache: dict[str, Llama] = {}

def get_model(local_cfg: dict) -> Llama:
    repo_id = local_cfg["repo_id"]
    filename = local_cfg["filename"]
    key = f"{repo_id}:{filename}"
    if key not in _model_cache:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_cfg.get("main_gpu", 0))
        print(f"Loading model {repo_id} ({filename})...")
        _model_cache[key] = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=local_cfg.get("n_ctx", 4096),
            n_gpu_layers=-1,
            verbose=False,
        )
    return _model_cache[key]


def local_chat(
    local_cfg: dict,
    messages: list[dict],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 1.0,
) -> str:
    model = get_model(local_cfg)

    all_messages = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.extend(messages)

    response = model.create_chat_completion(
        messages=all_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]
