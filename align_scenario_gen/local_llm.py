from llama_cpp import Llama

_model_cache: dict[str, Llama] = {}

def get_model(local_cfg: dict) -> Llama:
    repo_id = local_cfg["repo_id"]
    filename = local_cfg["filename"]
    key = f"{repo_id}:{filename}"
    if key not in _model_cache:
        print(f"Loading model {repo_id} ({filename})...")
        _model_cache[key] = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=local_cfg.get("n_ctx", 4096),
            n_gpu_layers=local_cfg.get("n_gpu_layers", 0),
            main_gpu=local_cfg.get("main_gpu", 0),
            verbose=False,
        )
    return _model_cache[key]


def local_chat(
    config: dict,
    messages: list[dict],
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 1.0,
) -> str:
    local_cfg = config.get("local_model", {})
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
